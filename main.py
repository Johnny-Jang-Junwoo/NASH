from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from agents.state import AgentState
from physics_engine.solver import BoltzmannSolver


def _call_llm(prompt: str) -> str:
    """Query the Groq-hosted Llama 3 model for theorist reasoning."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "NO_API_KEY: Set GROQ_API_KEY to enable reasoning."

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def reason_node(state: AgentState) -> AgentState:
    """LLM theorist decides whether to invoke the physics solver or respond."""

    if state.get("answer"):
        return state

    question = state["question"]
    history = state.get("history", [])
    prompt = (
        "You are the Theorist in a neuro-symbolic loop. "
        "Decide whether to run a simulation. "
        "Return one of: RUN_SIMULATION or RESPOND with a brief answer.\n"
        f"Question: {question}\n"
        f"History: {history}\n"
    )
    decision = _call_llm(prompt).strip()

    if "RUN_SIMULATION" in decision:
        state["code"] = "run_simulation"
    else:
        state["answer"] = decision or "No response from LLM."

    history.append(f"Theorist decision: {decision}")
    state["history"] = history
    return state


def tool_node(state: AgentState) -> AgentState:
    """Run the symbolic solver and attach results for the theorist to review."""

    if state.get("code") != "run_simulation":
        return state

    solver = BoltzmannSolver()
    result = solver.run_simulation({"question": state["question"]})

    history = state.get("history", [])
    history.append(f"Solver result: {result}")
    state["history"] = history
    state["answer"] = f"Simulation complete. Result: {result}"
    state["code"] = ""
    return state


def should_continue(state: AgentState) -> str:
    """Route control flow based on whether an answer has been produced."""

    return "end" if state.get("answer") else "continue"


def build_graph() -> Any:
    """Construct the LangGraph state machine for the neuro-symbolic loop."""

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("reason")
    graph.add_edge("tool", "reason")
    graph.add_conditional_edges("reason", should_continue, {"continue": "tool", "end": END})
    return graph.compile()


def main() -> None:
    """Initialize configuration and run a short neuro-symbolic reasoning cycle."""

    load_dotenv()

    app = build_graph()
    initial_state: AgentState = {
        "question": "Estimate nanoscale heat flux for a thin film.",
        "history": [],
        "code": "",
        "answer": "",
        "error_logs": [],
    }
    result: Dict[str, Any] = app.invoke(initial_state, config={"recursion_limit": 3})
    print(result["answer"])


if __name__ == "__main__":
    main()
