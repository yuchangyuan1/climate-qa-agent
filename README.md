# Climate QA Agent

An autonomous agent for climate data analysis and question answering, built with open-source LLMs. It combines Router-based intent classification, a Python REPL sandbox, and persistent conversation memory to solve multi-step climate problems.

## Features

- **Router-based Autonomous Agent**: Classifies user intent via LLM and routes to the optimal combination of tools; supports iterative multi-step reasoning — automatically evaluates results and refines its plan when the initial attempt is insufficient
- **Python REPL Sandbox**: Enables the agent to autonomously generate and debug Python code (xarray / numpy / pandas) to process large-scale ERA5 NetCDF datasets, handling complex analyses beyond predefined tools
- **Persistent Conversation Memory**: Maintains state across multi-turn analytical queries with JSON-based disk persistence, allowing the agent to resolve references ("it", "that variable") and build on prior results across sessions
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant climate documentation from a FAISS vector store and generates answers with source citations
- **Climate Data Analysis Tools**: Built-in tools for dataset inspection and statistical computation over ERA5 reanalysis data

## Architecture

```
User Question
      │
      ▼
┌──────────────────┐
│ Conversation     │  (load persistent memory context)
│ Memory           │
└──────────────────┘
      │
      ▼
┌──────────────────┐
│     Router       │  (LLM classifies intent → JSON plan)
└──────────────────┘
      │
      ├──────────────┬────────────────┬──────────────────┐
      ▼              ▼                ▼                  ▼
┌──────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────────┐
│   RAG    │  │  Inspect   │  │ Compute Stat│  │ Python REPL  │
│  (docs)  │  │  Dataset   │  │   (stats)   │  │  (sandbox)   │
└──────────┘  └────────────┘  └─────────────┘  └──────────────┘
      │              │                │                  │
      └──────────────┴────────────────┴──────────────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │ Needs Refinement?  │  (LLM evaluates completeness)
                   └────────────────────┘
                         │ No ──────────────────────┐
                         │ Yes → re-plan & execute   │
                         ▼                           ▼
                   ┌─────────────────┐    ┌─────────────────┐
                   │   Synthesizer   │    │  Save to Memory │
                   └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       Final Answer
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuchangyuan1/climate-qa-agent.git
cd climate-qa-agent
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your HuggingFace token
```

4. Download the data files (see `data/README.md` for instructions)

## Usage

### As a Python Module

```python
from src.agent import run_agent
from src.memory import ConversationMemory

# Initialize persistent memory (optional, enables multi-turn context)
memory = ConversationMemory()

# Single question
result = run_agent("What does t2m mean and what is its average value?", memory=memory)
print(result["final_answer"])

# Follow-up question (memory provides context)
result = run_agent("How does that compare to d2m?", memory=memory)
print(result["final_answer"])
```

### Using the Python REPL Sandbox Directly

```python
from src.sandbox import run_code

# The sandbox has pre-loaded ERA5 data accessible via `data` dict
result = run_code("""
t2m = data['t2m']
daily_mean = t2m.mean(dim=['lat', 'lon'])
std_celsius = float((daily_mean - 273.15).std())
print(f"Std dev of daily mean t2m: {std_celsius:.2f} °C")
""")
print(result["output"])
```

### Running the Example Script

```bash
python examples/run_agent.py
```

### Using the Demo Notebook

Open `notebooks/demo.ipynb` in Jupyter to explore the system interactively.

## Project Structure

```
climate-qa-agent/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── notebooks/
│   └── demo.ipynb              # Interactive demo notebook
├── src/
│   ├── __init__.py             # Package exports
│   ├── llm.py                  # LLM loading and generation
│   ├── memory.py               # Conversation memory (in-memory + persistent)
│   ├── rag.py                  # RAG retrieval and generation
│   ├── tools.py                # Climate data analysis tools
│   ├── sandbox.py              # Python REPL sandbox for code execution
│   └── agent.py                # Router-based autonomous agent
├── data/
│   ├── climate_corpus.jsonl    # RAG corpus (documentation)
│   └── README.md               # Data download instructions
└── examples/
    └── run_agent.py            # Usage example script
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `llm.py` | Loads Qwen2.5-3B-Instruct and provides single/multi-turn generation |
| `rag.py` | FAISS-based retrieval pipeline with citation support |
| `tools.py` | Predefined climate tools: `inspect_dataset`, `compute_stat` |
| `sandbox.py` | Restricted Python REPL with pre-loaded xarray/numpy/pandas and ERA5 data |
| `memory.py` | In-memory chat history + `ConversationMemory` class with JSON persistence |
| `agent.py` | Router, iterative plan execution, refinement loop, and answer synthesis |

## Tech Stack

- **LLM**: Qwen2.5-3B-Instruct (via HuggingFace Transformers)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Data Processing**: xarray, NetCDF4, NumPy, Pandas
- **Framework**: LangChain (for RAG components)

## Supported Climate Variables

| Variable | Description | Unit |
|----------|-------------|------|
| t2m | 2m air temperature | Kelvin (K) |
| d2m | 2m dewpoint temperature | Kelvin (K) |
| u10 | 10m U wind component | m/s |
| v10 | 10m V wind component | m/s |
| msl | Mean sea level pressure | Pa |
| tp | Total precipitation | m |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Climate data from ERA5 reanalysis (Copernicus Climate Data Store)
- Built for EECS 6895 Course Assignment
