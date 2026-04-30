"""Tests for node functions in nodes.py."""

from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from network_manager_agent.nodes import (
    network_manager,
    update_state,
    summarize_messages,
    should_summarize,
)


def _make_llm(return_content="default response", return_tool_calls=None):
    """Create a mocked ChatOpenAI instance."""
    llm = MagicMock()
    if return_tool_calls:
        mock_msg = AIMessage(content=return_content, tool_calls=return_tool_calls)
    else:
        mock_msg = AIMessage(content=return_content)
    llm.bind_tools.return_value.invoke.return_value = mock_msg
    llm.invoke.return_value = AIMessage(content=return_content)
    return llm


def _make_state(messages=None, summary="", original_message="", schema_profile="",
                candidates=None, entity_summaries=None, county_specialty_thresholds=None):
    """Create an AgentState dict."""
    state = {
        "messages": messages or [],
        "summary": summary,
        "original_message": original_message,
        "schema_profile": schema_profile,
        "candidates": candidates or [],
        "entity_summaries": entity_summaries or [],
        "county_specialty_thresholds": county_specialty_thresholds or {},
    }
    return state


class TestNetworkManager:
    """Tests for the network_manager node."""

    def test_injects_schema_profile(self):
        state = _make_state(
            messages=[HumanMessage(content="test prompt")],
            schema_profile='{"avg_effectiveness": {"type": "float", "min": 1.0, "max": 5.0}}',
        )
        llm = _make_llm()
        result = network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        system_content = call_args[0][0][0].content
        assert "avg_effectiveness" in system_content

    def test_injects_scope_section(self):
        state = _make_state(
            messages=[HumanMessage(content="test prompt")],
            county_specialty_thresholds={"wayne": {"hospital": 10.0, "pcp": 5.0}},
        )
        llm = _make_llm()
        network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        system_content = call_args[0][0][0].content
        assert "wayne" in system_content
        assert "hospital" in system_content
        assert "10.0mi" in system_content

    def test_fallback_schema_when_empty(self):
        state = _make_state(
            messages=[HumanMessage(content="test prompt")],
            schema_profile="",
            candidates=[{"id": 1, "specialty": "hospital", "effectiveness": 5, "Primary Contract Entity": "A"}],
        )
        llm = _make_llm()
        network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        system_content = call_args[0][0][0].content
        assert "No schema available" in system_content or "avg_effectiveness" in system_content

    def test_uses_anchor_from_human_message(self):
        state = _make_state(
            messages=[HumanMessage(content="find me a hospital")],
        )
        llm = _make_llm()
        network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        # The anchor message should be the HumanMessage content
        human_msg = call_args[0][0][1]
        assert isinstance(human_msg, HumanMessage)
        assert "find me a hospital" in human_msg.content

    def test_anchor_fallback_when_no_human_message(self):
        state = _make_state(
            messages=[AIMessage(content="tool response")],
        )
        llm = _make_llm()
        network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        # Should fall back to "Please continue."
        human_msg = call_args[0][0][1]
        assert "Please continue" in human_msg.content

    def test_includes_summary_in_system_prompt(self):
        state = _make_state(
            messages=[HumanMessage(content="test prompt")],
            summary="Objective: Add hospitals\nProgress: Checked candidates",
        )
        llm = _make_llm()
        network_manager(state, llm)

        call_args = llm.bind_tools.return_value.invoke.call_args
        system_content = call_args[0][0][0].content
        assert "SUMMARY SO FAR" in system_content
        assert "Add hospitals" in system_content

    def test_returns_ai_message(self):
        state = _make_state(messages=[HumanMessage(content="test prompt")])
        llm = _make_llm(return_tool_calls=[{"name": "get_candidates", "args": {}, "id": "1"}])
        result = network_manager(state, llm)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_does_not_return_original_message(self):
        state = _make_state(
            messages=[HumanMessage(content="test prompt")],
            original_message="old prompt",
        )
        llm = _make_llm()
        result = network_manager(state, llm)

        assert "original_message" not in result


class TestUpdateState:
    """Tests for the update_state node."""

    def test_extracts_providers_from_tool_message(self):
        ai_msg = AIMessage(content="", tool_calls=[{"name": "add_contract_entity", "args": {"entity_ids": ["A"]}, "id": "1"}])
        tool_msg = ToolMessage(content='{"added_providers": [{"id": 1, "Name": "Entity A", "Latitude": 42.0, "Longitude": -83.0, "Specialty": "hospital", "Primary Contract Entity": "Entity A"}]}', tool_call_id="1", name="add_contract_entity")
        state = _make_state(messages=[ai_msg, tool_msg])

        result = update_state(state)
        assert "network" in result
        assert len(result["network"]) == 1
        assert result["network"][0]["id"] == 1
        assert result["network"][0]["Name"] == "Entity A"
        assert result["network"][0]["Latitude"] == 42.0
        assert result["network"][0]["Longitude"] == -83.0
        assert result["network"][0]["Specialty"] == "hospital"
        assert result["network"][0]["Primary Contract Entity"] == "Entity A"

    def test_skips_skip_response(self):
        ai_msg = AIMessage(content="", tool_calls=[{"name": "add_contract_entity", "args": {"entity_ids": ["A"]}, "id": "1"}])
        tool_msg = ToolMessage(content="Skip: Entity already in network.", tool_call_id="1", name="add_contract_entity")
        state = _make_state(messages=[ai_msg, tool_msg])

        result = update_state(state)
        assert "network" not in result or result.get("network", []) == []

    def test_returns_empty_when_no_add_contract_entity(self):
        ai_msg = AIMessage(content="", tool_calls=[{"name": "get_candidates", "args": {}, "id": "1"}])
        tool_msg = ToolMessage(content="[]", tool_call_id="1", name="get_candidates")
        state = _make_state(messages=[ai_msg, tool_msg])

        result = update_state(state)
        assert result == {}

    def test_handles_malformed_json(self):
        ai_msg = AIMessage(content="", tool_calls=[{"name": "add_contract_entity", "args": {"entity_ids": ["A"]}, "id": "1"}])
        tool_msg = ToolMessage(content="not valid json", tool_call_id="1", name="add_contract_entity")
        state = _make_state(messages=[ai_msg, tool_msg])

        result = update_state(state)
        # Should not crash, just skip the malformed message
        assert "network" not in result or result.get("network", []) == []

    def test_returns_empty_when_no_messages(self):
        state = _make_state(messages=[])
        result = update_state(state)
        assert result == {}


class TestSummarizeMessages:
    """Tests for the summarize_messages node."""

    def test_updates_summary(self):
        messages = [
            HumanMessage(content="add a hospital"),
            AIMessage(content="checking candidates"),
            ToolMessage(content="found 5 entities", tool_call_id="1", name="get_candidates"),
        ]
        state = _make_state(messages=messages)
        llm = _make_llm(return_content="New summary: adding hospital")

        result = summarize_messages(state, llm)
        assert result["summary"] == "New summary: adding hospital"

    def test_removes_old_messages(self):
        messages = [
            HumanMessage(content="add a hospital", id="msg1"),
            AIMessage(content="checking", id="msg2"),
            ToolMessage(content="done", id="msg3", tool_call_id="1", name="get_candidates"),
        ]
        state = _make_state(messages=messages)
        llm = _make_llm()

        result = summarize_messages(state, llm)
        remove_ids = {m.id for m in result["messages"]}
        assert "msg1" in remove_ids
        assert "msg2" in remove_ids
        assert "msg3" in remove_ids

    def test_preserves_existing_summary_on_empty_llm_response(self):
        messages = [HumanMessage(content="test")]
        state = _make_state(messages=messages, summary="existing summary")
        llm = _make_llm(return_content="")

        result = summarize_messages(state, llm)
        assert result["summary"] == "existing summary"

    def test_falls_back_to_unavailable_when_no_existing_summary(self):
        messages = [HumanMessage(content="test")]
        state = _make_state(messages=messages, summary="")
        llm = _make_llm(return_content="")

        result = summarize_messages(state, llm)
        assert result["summary"] == "Summary unavailable."

    def test_all_messages_removed(self):
        messages = [HumanMessage(content="test", id="msg1")]
        state = _make_state(messages=messages)
        llm = _make_llm()

        result = summarize_messages(state, llm)
        remove_ids = {m.id for m in result["messages"]}
        assert "msg1" in remove_ids


class TestShouldSummarize:
    """Tests for the should_summarize routing function."""

    def test_returns_summarize_above_threshold(self):
        messages = [HumanMessage(content=f"msg{i}") for i in range(15)]
        state = _make_state(messages=messages)
        assert should_summarize(state) == "summarize"

    def test_returns_continue_at_threshold(self):
        messages = [HumanMessage(content=f"msg{i}") for i in range(14)]
        state = _make_state(messages=messages)
        assert should_summarize(state) == "continue"

    def test_returns_continue_below_threshold(self):
        messages = [HumanMessage(content=f"msg{i}") for i in range(5)]
        state = _make_state(messages=messages)
        assert should_summarize(state) == "continue"
