"""Tests for graph.py — graph construction and edge routing."""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import tools_condition

from network_manager_agent.graph import build_agent


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------

class TestBuildAgent:
    def test_returns_compiled_graph(self, clean_data_manager):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="OK")

        agent = build_agent(mock_llm)
        assert agent is not None

    def test_graph_has_expected_nodes(self, clean_data_manager):
        mock_llm = MagicMock()
        agent = build_agent(mock_llm)

        node_names = list(agent.get_graph().nodes.keys())
        assert "network_manager" in node_names
        assert "tools" in node_names
        assert "update_state" in node_names
        assert "summarize_messages" in node_names

    def test_start_edges_to_network_manager(self, clean_data_manager):
        mock_llm = MagicMock()
        agent = build_agent(mock_llm)

        graph = agent.get_graph()
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("__start__", "network_manager") in edge_pairs

    def test_tools_to_update_state_edge(self, clean_data_manager):
        mock_llm = MagicMock()
        agent = build_agent(mock_llm)

        graph = agent.get_graph()
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("tools", "update_state") in edge_pairs

    def test_summarize_to_network_manager_edge(self, clean_data_manager):
        mock_llm = MagicMock()
        agent = build_agent(mock_llm)

        graph = agent.get_graph()
        edge_pairs = [(e.source, e.target) for e in graph.edges]
        assert ("summarize_messages", "network_manager") in edge_pairs

    def test_checkpointer_passed_through(self, clean_data_manager):
        from langgraph.checkpoint.memory import MemorySaver

        mock_llm = MagicMock()
        checkpointer = MemorySaver()

        agent = build_agent(mock_llm, checkpointer=checkpointer)
        assert agent.checkpointer is checkpointer


# ---------------------------------------------------------------------------
# tools_condition routing
# ---------------------------------------------------------------------------

class TestToolsCondition:
    def test_routes_to_tools_when_tool_calls_present(self):
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{"id": "1", "name": "run_code", "args": {}}]),
            ]
        }
        assert tools_condition(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Here's my response"),
            ]
        }
        assert tools_condition(state) == "__end__"

    def test_routes_to_end_on_empty_messages(self):
        state = {"messages": []}
        with pytest.raises(ValueError):
            tools_condition(state)
