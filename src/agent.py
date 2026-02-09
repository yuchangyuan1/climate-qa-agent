"""
Router-based Agent for climate QA.

This module implements an autonomous agent that:
1. Classifies user intent using an LLM (JSON output)
2. Routes to RAG, data analysis tools, or Python REPL sandbox
3. Supports iterative multi-step reasoning (refine if initial plan is insufficient)
4. Integrates persistent conversation memory for multi-turn context
5. Synthesizes final answers from tool results
"""

import json
import re
from typing import Dict, Any, List, Callable, Optional

from .llm import generate_from_messages
from .rag import rag_answer
from .tools import inspect_dataset, compute_stat
from .sandbox import run_code
from .memory import ConversationMemory

# Allowed values for routing
ALLOWED_VARS = ["t2m", "d2m", "u10", "v10", "msl", "tp"]
ALLOWED_METRICS = ["mean", "max", "min", "sum"]
ALLOWED_SPATIAL = ["box_mean"]

# Maximum refinement iterations
MAX_REFINEMENT_STEPS = 1


def make_router_prompt(question: str, memory_context: str = "") -> str:
    """
    Create the router prompt for intent classification.

    Args:
        question: User's question.
        memory_context: Summary of previous conversation turns (if any).

    Returns:
        Formatted router prompt string.
    """
    context_block = ""
    if memory_context:
        context_block = f"""
{memory_context}

Use the conversation context above to resolve references like "it", "that variable",
"the same dataset", etc. in the user's new question.

"""

    return f"""
You are a tool router for a climate QA agent.

Return ONLY valid JSON with this schema:
{{
  "plan": [
    {{"tool": "...", "args": {{...}}}}
  ]
}}

Available tools:

1) rag
   args: {{"question": string, "k": integer}}

2) inspect_dataset
   args: {{"variable": one of {ALLOWED_VARS}}}

3) compute_stat
   args: {{
     "variable": one of {ALLOWED_VARS},
     "metric": one of {ALLOWED_METRICS},
     "spatial": one of {ALLOWED_SPATIAL},
     "lat": optional number (None if spatial="box_mean"),
     "lon": optional number (None if spatial="box_mean"),
     "units": optional string ("C" only for t2m/d2m)
   }}

4) python_repl
   args: {{"code": string}}
   The code runs in a sandbox with pre-imported numpy (np), pandas (pd), xarray (xr).
   Pre-loaded data: `data` dict mapping variable names to standardized xr.DataArray
   (keys: {ALLOWED_VARS}). Use print() to output results.
   Use for: complex analysis, custom calculations, time series operations, standard
   deviation, correlations, filtering, or anything the other tools cannot handle.

Routing rules (follow strictly):
- Use rag for: definition, meaning, units, variable description, dataset documentation.
- Use inspect_dataset ONLY for: temporal coverage, available dates, schema/dimensions/coords.
- Use compute_stat for: simple numeric values (mean/max/min/sum), temperature/precip totals.
- Use python_repl for: complex or custom analysis that compute_stat cannot handle
  (e.g., standard deviation, daily trends, correlations, filtering by date range,
  comparing multiple variables, custom aggregations).
- Choose the MINIMAL set of tools.
- If both documentation + numeric value are requested, include BOTH rag and compute_stat.
- If asking temperature/dewpoint numeric value, prefer units="C".
{context_block}
User question:
{question}

JSON:
""".strip()


def extract_json(text: str) -> dict:
    """
    Extract JSON from LLM output.

    Handles:
    - Markdown code fences
    - Extra text before/after JSON
    - Whitespace

    Args:
        text: Raw LLM output.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON found.
    """
    # Remove code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")

    return json.loads(match.group())


def route_question(question: str, memory_context: str = "") -> Dict[str, Any]:
    """
    Route a question to determine which tools to use.

    Args:
        question: User's question.
        memory_context: Conversation history context string.

    Returns:
        Dictionary with 'plan' key containing list of tool calls.
    """
    router_text = make_router_prompt(question, memory_context)

    messages = [
        {"role": "system", "content": "Output strictly valid JSON only. No extra text."},
        {"role": "user", "content": router_text},
    ]

    raw = generate_from_messages(messages, max_new_tokens=512, temperature=0.0)

    try:
        plan = extract_json(raw)
        if (
            "plan" not in plan
            or not isinstance(plan["plan"], list)
            or len(plan["plan"]) == 0
        ):
            raise ValueError("Invalid plan schema")
        return plan
    except Exception:
        # Conservative fallback: RAG-only
        return {"plan": [{"tool": "rag", "args": {"question": question, "k": 3}}]}


def _tool_rag(question: str, k: int = 3) -> Dict[str, Any]:
    """RAG tool wrapper."""
    return rag_answer(question, k=k, show_evidence=True)


def _tool_python_repl(code: str) -> Dict[str, Any]:
    """Python REPL sandbox wrapper."""
    return run_code(code)


# Tool registry
TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "rag": _tool_rag,
    "inspect_dataset": inspect_dataset,
    "compute_stat": compute_stat,
    "python_repl": _tool_python_repl,
}


def run_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute a routing plan.

    Args:
        plan: Dictionary with 'plan' key containing list of tool calls.

    Returns:
        List of trace entries with tool, args, result, and result_preview.
    """
    trace = []

    for step in plan["plan"]:
        tool_name = step["tool"]
        args = step.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            tool_fn = TOOL_REGISTRY[tool_name]
            try:
                result = tool_fn(**args)
            except Exception as e:
                result = {"error": f"Tool execution failed: {str(e)}"}

        # Create result preview
        result_preview = str(result)[:200]

        trace.append(
            {
                "tool": tool_name,
                "args": args,
                "result": result,
                "result_preview": result_preview,
            }
        )

    return trace


def _collect_rag_citations(trace: List[Dict[str, Any]]) -> List[str]:
    """Collect doc_ids from RAG results."""
    for t in trace:
        if t["tool"] == "rag" and isinstance(t["result"], dict):
            ev = t["result"].get("evidence", [])
            doc_ids = []
            for item in ev:
                doc_id = item.get("doc_id")
                if doc_id and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
            return doc_ids
    return []


def _has_citation(text: str) -> bool:
    """Check if text contains citations."""
    return bool(re.search(r"\[[^\[\]]+\]", text))


def _build_tool_summary(trace: List[Dict[str, Any]]) -> str:
    """Build a short summary of tool calls and results for memory storage."""
    parts = []
    for t in trace:
        result_str = str(t["result"])[:100]
        parts.append(f"{t['tool']}({t['args']}) -> {result_str}")
    return "; ".join(parts)


def needs_refinement(
    question: str, trace: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determine if the current tool results sufficiently answer the question.

    Uses the LLM to evaluate whether additional tool calls are needed.

    Args:
        question: Original user question.
        trace: Current execution trace.

    Returns:
        Dictionary with:
        - complete (bool): Whether results are sufficient
        - additional_plan (list): Extra tool calls if needed
    """
    # Build summary of what we have so far
    result_blocks = []
    for t in trace:
        result_blocks.append(
            f"Tool: {t['tool']}, Args: {json.dumps(t['args'], ensure_ascii=False)}, "
            f"Result: {str(t['result'])[:300]}"
        )
    results_text = "\n".join(result_blocks)

    # Check for errors in trace that might need a retry with python_repl
    has_errors = any(
        isinstance(t["result"], dict) and "error" in t["result"]
        for t in trace
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate whether tool results fully answer a user's question.\n"
                "Return ONLY valid JSON:\n"
                '{"complete": true} if the results are sufficient.\n'
                '{"complete": false, "additional_plan": [{"tool": "...", "args": {...}}]} '
                "if more tools are needed.\n"
                f"Available tools: rag, inspect_dataset, compute_stat, python_repl.\n"
                f"Allowed variables: {ALLOWED_VARS}\n"
                "python_repl args: {{\"code\": string}} — sandbox with numpy, pandas, xarray, "
                "and pre-loaded `data` dict of DataArrays.\n"
                "Only request additional tools if the existing results clearly cannot "
                "answer the question. If results have errors, consider using python_repl "
                "as an alternative approach."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Current tool results:\n{results_text}\n\n"
                "Are these results sufficient to answer the question? JSON:"
            ),
        },
    ]

    raw = generate_from_messages(messages, max_new_tokens=512, temperature=0.0)

    try:
        result = extract_json(raw)
        if result.get("complete", True):
            return {"complete": True, "additional_plan": []}
        additional = result.get("additional_plan", [])
        if isinstance(additional, list) and len(additional) > 0:
            return {"complete": False, "additional_plan": additional}
    except Exception:
        pass

    # Default: consider it complete (avoid infinite loops)
    return {"complete": True, "additional_plan": []}


def synthesize_answer(
    question: str,
    trace: List[Dict[str, Any]],
    memory_context: str = "",
) -> str:
    """
    Synthesize a final answer from tool results.

    Args:
        question: Original user question.
        trace: List of tool execution results.
        memory_context: Conversation history context string.

    Returns:
        Final answer string.
    """
    citations = _collect_rag_citations(trace)

    # Build tool output summary
    tool_blocks = []
    for t in trace:
        tool_blocks.append(
            f"Tool: {t['tool']}\nArgs: {json.dumps(t['args'], ensure_ascii=False)}\n"
            f"Result: {json.dumps(t['result'], ensure_ascii=False)[:1500]}"
        )
    tool_text = "\n\n".join(tool_blocks)

    must_cite_rule = ""
    if citations:
        must_cite_rule = (
            f"Available citations: {', '.join([f'[{c}]' for c in citations])}\n"
            "If you use any information from RAG/documentation, you MUST include citations like [doc_id].\n"
            "Your final answer MUST include at least one citation if citations are available.\n"
        )

    context_block = ""
    if memory_context:
        context_block = (
            f"\n{memory_context}\n"
            "Use conversation context to provide coherent, contextual answers.\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful climate assistant.\n"
                "Use ONLY the provided tool outputs.\n"
                "Do NOT invent numbers. Use numeric values only from compute_stat/inspect_dataset/python_repl results.\n"
                + must_cite_rule
                + context_block
                + "Be concise and directly answer the question.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Tool outputs:\n{tool_text}\n\n"
                "Write the final answer:"
            ),
        },
    ]

    answer = generate_from_messages(messages, max_new_tokens=256, temperature=0.2)

    # Enforce citation if RAG was used
    if citations and not _has_citation(answer):
        answer = answer.strip() + f"\n\nSources: [{citations[0]}]"

    return answer


def run_agent(
    question: str,
    memory: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """
    Run the full autonomous agent pipeline.

    1. Load conversation memory context (if available)
    2. Route the question to determine tools
    3. Execute the plan
    4. Check if refinement is needed (iterative reasoning)
    5. Synthesize the final answer
    6. Save turn to persistent memory

    Args:
        question: User's question.
        memory: Optional ConversationMemory instance for persistent context.

    Returns:
        Dictionary with question, plan, trace, and final_answer.
    """
    # 1. Get conversation context from memory
    memory_context = ""
    if memory is not None:
        memory_context = memory.get_context_summary()

    # 2. Route the question
    plan = route_question(question, memory_context)

    # 3. Execute the plan
    trace = run_plan(plan)
    all_plans = list(plan["plan"])

    # 4. Iterative refinement: check if results are sufficient
    for _ in range(MAX_REFINEMENT_STEPS):
        refinement = needs_refinement(question, trace)
        if refinement["complete"]:
            break

        additional_plan = {"plan": refinement["additional_plan"]}
        additional_trace = run_plan(additional_plan)
        trace.extend(additional_trace)
        all_plans.extend(refinement["additional_plan"])

    # 5. Synthesize the final answer
    final_answer = synthesize_answer(question, trace, memory_context)

    # 6. Save to persistent memory
    if memory is not None:
        tool_summary = _build_tool_summary(trace)
        memory.add_turn(
            question=question,
            plan=all_plans,
            answer=final_answer,
            tool_summary=tool_summary,
        )

    return {
        "question": question,
        "plan": all_plans,
        "trace": trace,
        "final_answer": final_answer,
    }
