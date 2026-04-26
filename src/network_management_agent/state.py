"""Agent state definition."""

from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
import operator


class AgentState(MessagesState):
    """State for the network management agent.

    Extends MessagesState with agent-specific fields:
    - candidates: Available provider candidates
    - members: Member locations
    - network: Currently selected providers (accumulated)
    - summary: Running summary of agent progress
    - original_message: The user's initial request
    """
    candidates: list[dict] = []
    members: list[dict] = []
    network: Annotated[list[dict], operator.add] = []
    summary: str = ""
    original_message: str = ""
