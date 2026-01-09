from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Ensure src is in path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from main import AgentState, build_graph
from physics_engine.solver import BoltzmannSolver


def _extract_observation(history: List[str]) -> Optional[Dict[str, Any]]:
    """Pull the most recent solver observation from history for plotting."""

    for entry in reversed(history):
        if "System Observation:" in entry:
            payload = entry.split("System Observation:", 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                return None
    return None


def _plot_observation(observation: Dict[str, Any]) -> None:
    """Render a plot that visualizes the latest solver output."""

    data = observation.get("data", {})
    input_data = observation.get("input", {})
    temperature_profile = data.get("temperature_profile")

    if isinstance(temperature_profile, list) and temperature_profile:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(temperature_profile))),
                y=temperature_profile,
                mode="lines+markers",
                name="Temperature Profile",
            )
        )
        fig.update_layout(
            title="Boltzmann Solver Output",
            xaxis_title="Position Index",
            yaxis_title="Temperature (K)",
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    k_val = data.get("k_wmk") or observation.get("k_wmk")
    d_val = data.get("D_nm") or input_data.get("D") or observation.get("D_nm")
    if k_val is None or d_val is None:
        st.info("Simulation completed. No plottable series available yet.")
        st.json(observation)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[d_val],
            y=[k_val],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="NASH Simulation",
        )
    )
    fig.update_layout(
        title=f"Simulation Result: {k_val} W/mK",
        xaxis_title="Diameter (nm)",
        yaxis_title="Thermal Conductivity (W/mK)",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Launch the NASH dashboard for interactive simulations and auditing."""

    load_dotenv()

    st.set_page_config(page_title="NASH: Neuro-symbolic Scientist", layout="wide")
    st.title("NASH: Neuro-symbolic Agent for Scalable Heat")
    st.markdown("### The Autonomous Physics Auditor")

    with st.sidebar:
        st.header("Experiment Controls")
        st.subheader("Material Properties")
        material_choice = st.selectbox(
            "Select Material",
            ["Silicon", "Germanium", "MXene (Ti3C2Tx)", "Custom"],
        )
        material_db = {
            "Silicon": {
                "v_s": 8433.0,
                "Theta_D": 645.0,
                "A": 1.32e-45,
                "B": 1.73e-24,
                "name": "Silicon",
            },
            "Germanium": {
                "v_s": 5400.0,
                "Theta_D": 374.0,
                "A": 1.0e-44,
                "B": 2.0e-23,
                "name": "Germanium",
            },
            "MXene (Ti3C2Tx)": {
                "v_s": 6200.0,
                "Theta_D": 500.0,
                "A": 1.0e-43,
                "B": 1.0e-23,
                "name": "MXene (Ti3C2Tx)",
            },
        }
        selected_props = material_db.get(material_choice, {})
        if material_choice == "Custom":
            v_s = st.number_input("Sound Velocity (m/s)", value=5000.0)
            theta_d = st.number_input("Debye Temp (K)", value=400.0)
            selected_props = {"v_s": v_s, "Theta_D": theta_d, "name": "Custom Material"}

        temp = st.slider("Temperature (K)", 100, 500, 300)
        diameter = st.slider("Diameter (nm)", 10, 200, 22)

        if st.button("Run Manual Simulation"):
            solver = BoltzmannSolver()
            result = solver.run_simulation(
                {"T": temp, "D": diameter, "material_props": selected_props}
            )
            st.success("Simulation complete.")
            st.json(result)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history_log" not in st.session_state:
        st.session_state.history_log = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not os.getenv("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY not found. Add it to your .env to enable reasoning.")
        return

    user_input = st.chat_input("Ask a physics question...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("NASH is thinking and simulating...")

        app = build_graph()
        inputs: AgentState = {
            "question": user_input,
            "history": list(st.session_state.history_log),
            "code": "",
            "answer": "",
            "error_logs": [],
            "steps": 0,
        }

        try:
            final_state = app.invoke(inputs, config={"recursion_limit": 10})
        except Exception as exc:
            message_placeholder.error(f"Error: {exc}")
            return

        final_answer = final_state.get("answer", "")
        history_log = final_state.get("history", [])
        st.session_state.history_log = history_log

        message_placeholder.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        observation = _extract_observation(history_log)
        if observation:
            with st.expander("Experimental Data (Generated by BTE Solver)", expanded=True):
                _plot_observation(observation)


if __name__ == "__main__":
    main()
