"""Agent state definition."""

from typing import Annotated

import numpy as np
from langgraph.graph import MessagesState
import operator


def _to_native(obj: Annotated) -> Annotated:
    """Recursively convert numpy/pandas types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _merge_dict(existing: dict, new: dict) -> dict:
    merged = dict(existing)
    merged.update(new)
    return merged


class AgentState(MessagesState):
    """State for the network management agent.

    Extends MessagesState with agent-specific fields:
    - network: Currently selected entity IDs (accumulated)
    - summary: Running summary of agent progress
    """
    network: Annotated[list[str], operator.add] = []
    summary: str = ""
    county_specialty_thresholds: dict[str, dict[str, float]] = {}
    sandbox_cache: Annotated[dict, _merge_dict] = {}


