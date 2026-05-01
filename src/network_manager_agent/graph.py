"""Graph construction for the network management agent."""

from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    network_manager,
    update_state,
    summarize_messages,
    should_summarize,
    execute_tools,
)


def build_agent(llm: ChatOpenAI):
    """Build and compile the network management agent graph.

    Args:
        llm: Configured ChatOpenAI instance.

    Returns:
        Compiled LangGraph agent ready for execution.
    """
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("network_manager", lambda state: network_manager(state, llm))
    builder.add_node("tools", execute_tools)
    builder.add_node("update_state", update_state)
    builder.add_node("summarize_messages", lambda state: summarize_messages(state, llm))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "network_manager")

    builder.add_conditional_edges(
        "network_manager",
        tools_condition,
    )

    builder.add_conditional_edges(
        "update_state",
        should_summarize,
        {
            "summarize": "summarize_messages",
            "continue": "network_manager",
        }
    )

    builder.add_edge("tools", "update_state")
    builder.add_edge("summarize_messages", "network_manager")

    memory = MemorySaver()
    agent = builder.compile(checkpointer=memory)

    return agent
