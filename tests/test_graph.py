"""Tests for graph construction in graph.py."""

from unittest.mock import MagicMock

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver

from network_manager_agent.graph import build_agent


def _make_mock_llm():
    """Create a mocked ChatOpenAI instance."""
    llm = MagicMock(spec=ChatOpenAI)
    return llm


class TestBuildAgent:
    """Tests for the build_agent function."""

    def test_compiles_without_error(self):
        llm = _make_mock_llm()
        agent = build_agent(llm)
        assert agent is not None

    def test_has_expected_nodes(self):
        llm = _make_mock_llm()
        agent = build_agent(llm)
        graph = agent.get_graph()
        node_ids = list(graph.nodes)

        assert "network_manager" in node_ids
        assert "tools" in node_ids
        assert "update_state" in node_ids
        assert "summarize_messages" in node_ids

    def test_has_checkpointer(self):
        llm = _make_mock_llm()
        checkpointer = MemorySaver()
        agent = build_agent(llm, checkpointer=checkpointer)
        # The compiled agent should have a checkpointer configured
        assert agent.checkpointer is not None
