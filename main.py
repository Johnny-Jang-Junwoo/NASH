from __future__ import annotations

import os
import sys
import json  # Added json for parameter parsing
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

# Ensure src is in path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from agents.state import AgentState
from physics_engine.solver import BoltzmannSolver

# 1. SETUP LLM
def _call_llm(prompt: str) -> str:
    """Query Groq. We lower temperature to 0.1 for precise JSON output."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "NO_API_KEY"

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, # Keep it strict
    )
    return completion.choices[0].message.content or ""

# 2. THE THEORIST (Brain)
def reason_node(state: AgentState) -> AgentState:
    if state.get("answer"):
        return state

    question = state["question"]
    history = state.get("history", [])
    
    # We force the LLM to output structured JSON so we can extract T and D
    prompt = (
        "You are NASH, a Neuro-symbolic Physicist. "
        "Your goal is to answer questions about Nanoscale Heat Transfer.\n"
        "You have a tool: BoltzmannSolver(T, D).\n"
        "INSTRUCTIONS:\n"
        "1. If you need to run a simulation, return JSON: {\"action\": \"simulate\", \"T\": <float>, \"D\": <float>}\n"
        "2. If you have the result or can answer directly, return JSON: {\"action\": \"answer\", \"text\": \"<explanation>\"}\n"
        "3. Output ONLY valid JSON. No markdown.\n\n"
        f"Question: {question}\n"
        f"History: {history}\n"
    )
    
    response = _call_llm(prompt).strip()
    
    # Basic cleaning if LLM adds markdown blocks
    if response.startswith("```json"):
        response = response.replace("```json", "").replace("```", "")
    
    state["code"] = response # Store the raw JSON decision
    
    # Log thoughts
    history.append(f"Theorist thought: {response}")
    state["history"] = history
    return state

# 3. THE EXPERIMENTER (Hands)
def tool_node(state: AgentState) -> AgentState:
    decision_json = state.get("code", "{}")
    
    try:
        data = json.loads(decision_json)
        
        # Check if the Theorist wants to simulate
        if data.get("action") == "simulate":
            solver = BoltzmannSolver()
            
            # EXTRACT PARAMETERS (The Fix)
            params = {
                "T": data.get("T", 300), 
                "D": data.get("D", 100)
            }
            
            # Run the physics engine
            result = solver.run_simulation(params)
            
            # Update history with the observation
            state["history"].append(f"System Observation: {result}")
            
            # We explicitly clear 'answer' so the loop continues back to reason_node
            state["answer"] = "" 
            
        elif data.get("action") == "answer":
            state["answer"] = data.get("text")
            
    except json.JSONDecodeError:
        state["history"].append("Error: Theorist produced invalid JSON.")
        
    return state

# 4. ROUTING LOGIC
def should_continue(state: AgentState) -> str:
    """If we have a final text answer, end. Otherwise loop."""
    if state.get("answer"):
        return "end"
    return "continue"

# 5. BUILD GRAPH
def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("tool", tool_node)
    
    graph.set_entry_point("reason")
    
    # Conditional edge: After reasoning, either act (tool) or end
    graph.add_conditional_edges(
        "reason", 
        should_continue, 
        {"continue": "tool", "end": END}
    )
    
    # Normal edge: After tool, always go back to reason to interpret results
    graph.add_edge("tool", "reason")
    
    return graph.compile()

def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in .env")
        return

    app = build_graph()
    
    # Test Question: Specific check for Li & Shi paper logic (22nm wire)
    initial_state: AgentState = {
        "question": "Calculate the thermal conductivity of a 22 nm Silicon nanowire at 300 K.",
        "history": [],
        "code": "",
        "answer": "",
        "error_logs": [],
    }
    
    print(f"User: {initial_state['question']}")
    result = app.invoke(initial_state, config={"recursion_limit": 5})
    print(f"NASH: {result['answer']}")

if __name__ == "__main__":
    main()