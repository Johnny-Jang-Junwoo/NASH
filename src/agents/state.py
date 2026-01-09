from __future__ import annotations

from typing import List, TypedDict


class AgentState(TypedDict):
    """Shared state passed between the neuro-symbolic reasoning and solver steps."""

    question: str
    history: List[str]
    code: str
    answer: str
    error_logs: List[str]
    steps: int
