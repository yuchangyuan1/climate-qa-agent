#!/usr/bin/env python
"""
Example script demonstrating the Climate QA Agent.

Usage:
    python examples/run_agent.py

Make sure you have:
1. Set up the environment variables (copy .env.example to .env)
2. Downloaded the required data files (see data/README.md)
3. Installed dependencies (pip install -r requirements.txt)
"""

import os
import sys
import json

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, ".env"))
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")


def main():
    """Run example queries through the agent."""
    # Import after path setup
    from src.agent import run_agent
    from src.memory import ConversationMemory

    print("=" * 80)
    print("Climate QA Agent - Example Run")
    print("=" * 80)
    print()

    # Initialize persistent conversation memory
    memory = ConversationMemory()

    # Example questions demonstrating different capabilities
    questions = [
        # RAG question (documentation)
        "What does t2m mean and what unit is it measured in?",

        # Tool question (computation)
        "What is the average temperature (t2m) in Celsius?",

        # Mixed question (RAG + Tool)
        "What does t2m mean and what is its average value in January?",

        # Tool question (dataset inspection + computation)
        "What is the temporal coverage of the dataset and what is the total precipitation?",

        # Python REPL question (complex analysis)
        "What is the standard deviation of daily mean temperature (t2m) in Celsius?",

        # Multi-turn memory question (references previous context)
        "How does that compare to the dewpoint temperature (d2m)?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}: {question}")
        print("=" * 80)

        try:
            result = run_agent(question, memory=memory)

            print("\nRouter Plan:")
            print(json.dumps(result["plan"], indent=2))

            print("\nFinal Answer:")
            print(result["final_answer"])

        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure you have set up the environment and data files correctly.")

    # Show that memory persists
    print(f"\n{'=' * 80}")
    print("Conversation Memory Summary")
    print("=" * 80)
    print(f"Total turns recorded: {len(memory.turns)}")
    print(f"Memory file: {memory.path}")
    print(memory.get_context_summary())

    print("\n" + "=" * 80)
    print("Example run completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
