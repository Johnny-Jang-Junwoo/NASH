from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

# Ensure src is in path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

MAX_STEPS = 6

from agents.state import AgentState
from physics_engine.solver import BoltzmannSolver


def _strip_code_fences(text: str) -> str:
    """Remove fenced code blocks around JSON payloads from the LLM output."""

    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _has_observation(history: List[str]) -> bool:
    """Check whether the solver has already produced a system observation."""

    return any("System Observation:" in entry for entry in history)


def reason_node(state: AgentState) -> AgentState:
    """Generate a JSON action for simulation or auditing the result."""

    if state.get("answer"):
        return state

    steps = state.get("steps", 0) + 1
    state["steps"] = steps
    if steps >= MAX_STEPS:
        state["answer"] = f"Error: Max steps ({MAX_STEPS}) reached without a final answer."
        return state

    question = state["question"]
    history = state.get("history", [])
    has_observation = _has_observation(history)

    system_prompt = (
        "You are NASH (Neuro-symbolic Agent for Scalable Heat). "
        "You are a skeptic. You trust physics equations (The Solver) more than human text.\n\n"
        "YOUR GOAL:\n"
        "1. Identify if the user's question implies a specific value (e.g., "
        "'The paper reports 150 W/mK').\n"
        "2. Use your tool 'BoltzmannSolver' to calculate the THEORETICAL value.\n"
        "3. COMPARE the two numbers.\n"
        "   - If they match (within 25%): Report 'VALIDATION SUCCESSFUL'.\n"
        "   - If they differ (>25%): Report 'DISCREPANCY DETECTED'. explain the difference.\n\n"
        "TOOL PROTOCOL:\n"
        "To run the solver, output JSON: {\"action\": \"simulate\", \"T\": <float>, \"D\": <float>, "
        "\"material_props\": {\"v_s\": <float>, \"Theta_D\": <float>, \"A\": <float>, \"B\": <float>, "
        "\"name\": \"<material>\"}}\n"
        "To answer/critique, output JSON: {\"action\": \"answer\", \"text\": \"<your analysis>\"}\n\n"
        "CRITICAL RULE:\n"
        "Never hallucinate a number. Only use the 'System Observation' from the history."
    )
    if has_observation:
        system_prompt += (
            "\n\nSTOP RULE:\n"
            "A System Observation already exists. You must answer now; do not request another simulation."
        )

    messages_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nHistory: {history}"},
    ]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        state["answer"] = "Error: GROQ_API_KEY not found in .env."
        return state

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages_payload,
        temperature=0.1,
    )

    response = completion.choices[0].message.content.strip()
    response = _strip_code_fences(response)
    state["code"] = response

    history.append(f"Theorist thought: {response}")
    state["history"] = history
    return state


def tool_node(state: AgentState) -> AgentState:
    """Execute the symbolic solver or finalize the answer from Theorist JSON."""

    decision_json = state.get("code", "{}")

    try:
        data = json.loads(decision_json)
    except json.JSONDecodeError as exc:
        history = state.get("history", [])
        history.append("Error: Theorist produced invalid JSON.")
        state["history"] = history
        error_logs = state.get("error_logs", [])
        error_logs.append(f"JSONDecodeError: {exc}")
        state["error_logs"] = error_logs
        return state

    if data.get("action") == "simulate":
        history = state.get("history", [])
        if _has_observation(history):
            history.append("Error: Simulation already performed; refusing to run again.")
            state["history"] = history
            error_logs = state.get("error_logs", [])
            error_logs.append("Simulation blocked after prior observation.")
            state["error_logs"] = error_logs
            state["answer"] = "Error: Simulation already performed; provide the audit instead."
            state["code"] = ""
            return state

        solver = BoltzmannSolver()
        params = {
            "T": data.get("T", 300),
            "D": data.get("D", 100),
            "material_props": data.get("material_props", {}),
        }
        result = solver.run_simulation(params)

        history = state.get("history", [])
        history.append(f"System Observation: {result}")
        state["history"] = history
        state["answer"] = ""
        state["code"] = ""
    elif data.get("action") == "answer":
        state["answer"] = data.get("text", "")
        state["code"] = ""
    else:
        history = state.get("history", [])
        history.append("Error: Theorist returned an unknown action.")
        state["history"] = history
        error_logs = state.get("error_logs", [])
        error_logs.append(f"Unknown action payload: {data}")
        state["error_logs"] = error_logs
        state["answer"] = "Error: Theorist returned an unknown action."
        state["code"] = ""

    return state


def should_continue(state: AgentState) -> str:
    """Route control flow based on whether the final answer is available."""

    if state.get("answer"):
        return "end"
    if state.get("steps", 0) >= MAX_STEPS:
        return "end"
    return "continue"


def build_graph() -> Any:
    """Construct the LangGraph state machine for the neuro-symbolic loop."""

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"continue": "tool", "end": END})
    graph.add_edge("tool", "reason")
    return graph.compile()


def main() -> None:
    """Initialize configuration and run a short neuro-symbolic reasoning cycle."""

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in .env")
        return

    app = build_graph()
    initial_state: AgentState = {
        "question": "Calculate the thermal conductivity of a 22 nm Silicon nanowire at 300 K.",
        "history": [],
        "code": "",
        "answer": "",
        "error_logs": [],
        "steps": 0,
    }
    print(f"User: {initial_state['question']}")
    result: Dict[str, Any] = app.invoke(initial_state, config={"recursion_limit": 10})
    print(f"NASH: {result['answer']}")


if __name__ == "__main__":
    main()
