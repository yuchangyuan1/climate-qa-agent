"""
Conversation memory management.

This module provides:
1. Multi-turn conversation functionality (in-memory history)
2. Persistent ConversationMemory for cross-session state management
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from .llm import get_model_and_tokenizer

# Default path for persistent memory storage
DEFAULT_MEMORY_PATH = "./data/conversation_memory.json"


def hf_chat(
    user_input: str,
    history: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Multi-turn chat function using HuggingFace transformers.

    This function demonstrates the core idea behind conversational memory:
    the model itself is stateless, so all context must be explicitly provided
    as a list of messages.

    Args:
        user_input: The user's input text.
        history: A list of message dictionaries with keys {"role", "content"}.
                 This list is modified in-place to include the new messages.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text.
    """
    model, tokenizer = get_model_and_tokenizer()

    # 1. Add user message to history
    history.append({"role": "user", "content": user_input})

    # 2. Format conversation using chat template
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )

    # 3. Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 4. Decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    # 5. Add assistant response to history
    history.append({"role": "assistant", "content": response})

    return response


def create_chat_session() -> List[Dict[str, str]]:
    """
    Create a new empty chat session.

    Returns:
        An empty list to store conversation history.
    """
    return []


def clear_history(history: List[Dict[str, str]]) -> None:
    """
    Clear the conversation history.

    Args:
        history: The conversation history list to clear.
    """
    history.clear()


# ---------------------------------------------------------------------------
# Persistent Conversation Memory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Persistent conversation memory that maintains state across multi-turn
    analytical queries.

    Stores conversation turns (question, plan, answer, tool_summary) to a
    JSON file on disk so that context is preserved across sessions.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize conversation memory.

        Args:
            path: File path for persistent storage. Defaults to
                  DATA_DIR/conversation_memory.json or ./data/conversation_memory.json.
        """
        if path is None:
            data_dir = os.getenv("DATA_DIR", "./data/")
            path = os.path.join(data_dir, "conversation_memory.json")
        self.path = path
        self.turns: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Load conversation history from disk (if file exists)."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.turns = data.get("turns", [])
            except (json.JSONDecodeError, IOError):
                self.turns = []

    def save(self) -> None:
        """Persist conversation history to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"turns": self.turns}, f, ensure_ascii=False, indent=2)

    def add_turn(
        self,
        question: str,
        plan: List[Dict[str, Any]],
        answer: str,
        tool_summary: str = "",
    ) -> None:
        """
        Record one conversation turn and persist to disk.

        Args:
            question: The user's question.
            plan: The router plan (list of tool calls).
            answer: The final synthesized answer.
            tool_summary: Short summary of tool results for context.
        """
        self.turns.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "plan": plan,
            "answer": answer,
            "tool_summary": tool_summary,
        })
        self.save()

    def get_history(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation turns.

        Args:
            last_n: Number of recent turns to return.

        Returns:
            List of turn dictionaries.
        """
        return self.turns[-last_n:]

    def get_context_summary(self, last_n: int = 3) -> str:
        """
        Generate a concise context summary for the LLM.

        This summary is injected into the router and synthesis prompts so the
        agent is aware of prior conversation turns.

        Args:
            last_n: Number of recent turns to include.

        Returns:
            Formatted string summarizing recent conversation history.
        """
        recent = self.get_history(last_n)
        if not recent:
            return ""

        lines = ["Previous conversation context:"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"  Turn {i}: Q: {turn['question']}")
            lines.append(f"          A: {turn['answer'][:200]}")
            if turn.get("tool_summary"):
                lines.append(f"          Tools: {turn['tool_summary'][:150]}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history and remove the file."""
        self.turns = []
        if os.path.exists(self.path):
            os.remove(self.path)
