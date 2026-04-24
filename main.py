import os
import sys
from pathlib import Path

# ── load .env FIRST before any other import ──────────────────────────────────
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
# ─────────────────────────────────────────────────────────────────────────────

from google.genai.errors import APIError
from graph.graph import build_graph
from graph.state import AgentState, IntentType


def run_agent() -> None:
    print("\n" + "=" * 60)
    print("  AutoStream AI Agent — Powered by LangGraph + Gemini")
    print("=" * 60)
    print("  Type your message and press Enter to chat.")
    print("  Type 'exit' or 'quit' to end the session.\n")

    graph = build_graph()

    state: AgentState = {
        "messages":        [],
        "intent":          IntentType.UNKNOWN,
        "lead_info":       {},
        "lead_captured":   False,
        "collecting_field": None,
        "rag_context":     "",
        "response":        "",
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAgent: Thanks for chatting! Goodbye. 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print("\nAgent: Thanks for chatting with AutoStream! Have a great day. 👋\n")
            break

        state["messages"].append({"role": "user", "content": user_input})

        try:
            result = graph.invoke(state)
        except APIError as e:
            state["messages"].pop()
            if e.code == 429:
                print("\nAgent: I'm getting a lot of requests right now — please wait "
                      "a few seconds and try again.\n")
            elif e.code == 404:
                print("\nAgent: Model not found. Check your GEMINI_API_KEY and model name.\n")
                print(f"[Debug] APIError {e.code}: {e.message}\n")
            elif e.code == 400:
                print("\nAgent: Bad request — check API key or prompt.\n")
                print(f"[Debug] APIError {e.code}: {e.message}\n")
            else:
                print(f"\nAgent: Sorry, I ran into an API issue (code {e.code}).\n")
                print(f"[Debug] {e.message}\n")
            continue
        except Exception as e:
            state["messages"].pop()
            print(f"\nAgent: Sorry, I ran into a technical issue. Please try again.\n")
            print(f"[Debug] {type(e).__name__}: {str(e)[:300]}\n")
            continue

        state = {
            **state,
            "intent":           result.get("intent",           IntentType.UNKNOWN),
            "lead_info":        result.get("lead_info",        {}),
            "lead_captured":    result.get("lead_captured",    False),
            "collecting_field": result.get("collecting_field"),
            "rag_context":      result.get("rag_context",      ""),
            "response":         result.get("response",         ""),
        }

        agent_response = state["response"]
        state["messages"].append({"role": "assistant", "content": agent_response})
        print(f"\nAgent: {agent_response}\n")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("\n[ERROR] GEMINI_API_KEY not found.")
        print(f"Make sure your .env file exists at: {_ENV_PATH}")
        print("It should contain:  GEMINI_API_KEY=your_key_here")
        print("Get a free key at:  https://aistudio.google.com/apikey\n")
        sys.exit(1)

    run_agent()
