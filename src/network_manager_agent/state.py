"""Agent state definition."""

from typing import Annotated

from langgraph.graph import MessagesState
import operator


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


