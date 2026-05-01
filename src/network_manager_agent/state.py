"""Agent state definition."""

from typing import Annotated

from langgraph.graph import MessagesState
import operator


class AgentState(MessagesState):
    """State for the network management agent.
 
    Extends MessagesState with agent-specific fields:
    - candidates: Available provider candidates
    - members: Member locations
    - network: Currently selected providers (accumulated)
    - summary: Running summary of agent progress
    - entity_summaries: Pre-computed aggregated entity summaries (cached)
    - schema_profile: Pre-computed JSON schema profile string for system prompt
    """
    candidates: list[dict] = []
    members: list[dict] = []
    network: Annotated[list[dict], operator.add] = []
    summary: str = ""
    county_specialty_thresholds: dict[str, dict[str, float]] = {}
    entity_summaries: list[dict] = []
    schema_profile: str = ""

