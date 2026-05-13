"""Tests for nodes.py — anchor message, content extraction, summarization, state update, tool execution."""

import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    RemoveMessage,
)

from network_manager_agent.nodes import (
    _get_anchor_message,
    _get_content,
    should_summarize,
    update_state,
    network_manager,
    summarize_messages,
    execute_tools,
)


# ---------------------------------------------------------------------------
# _get_anchor_message
# ---------------------------------------------------------------------------

class TestGetAnchorMessage:
    def test_finds_last_human_message(self):
        state = {
            "messages": [
                HumanMessage(content="First request"),
                AIMessage(content="OK"),
                HumanMessage(content="Second request"),
            ]
        }
        assert _get_anchor_message(state) == "Second request"

    def test_returns_default_when_no_human_msg(self):
        state = {
            "messages": [
                AIMessage(content="Hello"),
                ToolMessage(content="Result", tool_call_id="1"),
            ]
        }
        assert _get_anchor_message(state) == "Please continue."

    def test_skips_ai_and_tool_messages(self):
        state = {
            "messages": [
                AIMessage(content="AI response"),
                ToolMessage(content="Tool result", tool_call_id="1"),
                HumanMessage(content="My question"),
            ]
        }
        assert _get_anchor_message(state) == "My question"

    def test_empty_messages_list(self):
        state = {"messages": []}
        assert _get_anchor_message(state) == "Please continue."


# ---------------------------------------------------------------------------
# _get_content
# ---------------------------------------------------------------------------

class TestGetContent:
    def test_string_content_passthrough(self):
        msg = HumanMessage(content="hello")
        assert _get_content(msg) == "hello"

    def test_list_content_joins(self):
        msg = AIMessage(content=["part1", "part2", "part3"])
        assert _get_content(msg) == "part1 part2 part3"

    def test_empty_content(self):
        msg = HumanMessage(content="")
        assert _get_content(msg) == ""


# ---------------------------------------------------------------------------
# should_summarize
# ---------------------------------------------------------------------------

class TestShouldSummarize:
    def test_returns_summarize_when_over_threshold(self):
        state = {"messages": [MagicMock() for _ in range(15)]}
        assert should_summarize(state) == "summarize"

    def test_returns_continue_when_under_threshold(self):
        state = {"messages": [MagicMock() for _ in range(10)]}
        assert should_summarize(state) == "continue"

    def test_returns_continue_at_exact_threshold(self):
        state = {"messages": [MagicMock() for _ in range(14)]}
        assert should_summarize(state) == "continue"


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------

class TestUpdateState:
    def test_extracts_added_entities(self):
        tool_msg = ToolMessage(
            content=json.dumps({"added_entities": ["Entity A", "Entity B"]}),
            tool_call_id="1",
            name="add_contract_entity",
        )
        state = {"messages": [tool_msg]}
        result = update_state(state)
        assert result["network"] == ["Entity A", "Entity B"]

    def test_skips_non_add_contract_entity_tools(self):
        tool_msg = ToolMessage(
            content='{"added_entities": ["Entity A"]}',
            tool_call_id="1",
            name="other_tool",
        )
        state = {"messages": [tool_msg]}
        result = update_state(state)
        assert result == {}

    def test_handles_multiple_add_messages(self):
        msg1 = ToolMessage(
            content=json.dumps({"added_entities": ["Entity A"]}),
            tool_call_id="1",
            name="add_contract_entity",
        )
        msg2 = ToolMessage(
            content=json.dumps({"added_entities": ["Entity B"]}),
            tool_call_id="2",
            name="add_contract_entity",
        )
        state = {"messages": [msg1, msg2]}
        result = update_state(state)
        assert result["network"] == ["Entity A", "Entity B"]

    def test_skips_skip_in_content(self):
        tool_msg = ToolMessage(
            content="Skip: already in network",
            tool_call_id="1",
            name="add_contract_entity",
        )
        state = {"messages": [tool_msg]}
        result = update_state(state)
        assert result == {}

    def test_handles_json_parse_error(self):
        tool_msg = ToolMessage(
            content="not valid json {",
            tool_call_id="1",
            name="add_contract_entity",
        )
        state = {"messages": [tool_msg]}
        result = update_state(state)
        assert result == {}

    def test_empty_messages_returns_empty(self):
        state = {"messages": []}
        result = update_state(state)
        assert result == {}

    def test_extracts_sandbox_cache_from_run_code(self):
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {"rankings": [1, 2, 3]}
        tool_msg = ToolMessage(
            content="some output",
            tool_call_id="1",
            name="run_code",
        )
        state = {"messages": [tool_msg], "sandbox_cache": {}}
        result = update_state(state)
        assert "sandbox_cache" in result
        assert result["sandbox_cache"]["rankings"] == [1, 2, 3]

    def test_cache_shallow_merges_with_existing(self):
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {"new_key": "new_val"}
        tool_msg = ToolMessage(
            content="output",
            tool_call_id="1",
            name="run_code",
        )
        state = {
            "messages": [tool_msg],
            "sandbox_cache": {"existing": "data"},
        }
        result = update_state(state)
        assert result["sandbox_cache"]["existing"] == "data"
        assert result["sandbox_cache"]["new_key"] == "new_val"

    def test_cache_overwrites_existing_keys(self):
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {"key": "updated"}
        tool_msg = ToolMessage(
            content="output",
            tool_call_id="1",
            name="run_code",
        )
        state = {
            "messages": [tool_msg],
            "sandbox_cache": {"key": "old"},
        }
        result = update_state(state)
        assert result["sandbox_cache"]["key"] == "updated"

    def test_no_cache_marker_returns_empty(self):
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {}
        tool_msg = ToolMessage(
            content="just normal output",
            tool_call_id="1",
            name="run_code",
        )
        state = {"messages": [tool_msg], "sandbox_cache": {}}
        result = update_state(state)
        assert result == {}

    def test_cache_invalid_json_returns_empty(self):
        """Empty _last_sandbox_cache means no cache update."""
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {}
        tool_msg = ToolMessage(
            content="output",
            tool_call_id="1",
            name="run_code",
        )
        state = {"messages": [tool_msg], "sandbox_cache": {}}
        result = update_state(state)
        assert result == {}

    def test_entities_and_cache_extracted_together(self):
        from network_manager_agent import tools as tools_mod
        tools_mod._last_sandbox_cache = {"data": [1, 2]}
        entity_msg = ToolMessage(
            content=json.dumps({"added_entities": ["Entity A"]}),
            tool_call_id="1",
            name="add_contract_entity",
        )
        code_msg = ToolMessage(
            content="output",
            tool_call_id="2",
            name="run_code",
        )
        state = {"messages": [entity_msg, code_msg], "sandbox_cache": {}}
        result = update_state(state)
        assert result["network"] == ["Entity A"]
        assert result["sandbox_cache"]["data"] == [1, 2]


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_produces_summary_and_remove_messages(self):
        messages = [
            HumanMessage(content="Do something", id=f"msg_{i}")
            for i in range(10)
        ]
        state = {"messages": messages, "summary": ""}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Objective: Do something\nProgress: Done."

        result = summarize_messages(state, mock_llm)
        assert "summary" in result
        assert result["summary"] == "Objective: Do something\nProgress: Done."
        assert len(result["messages"]) > 0
        assert all(isinstance(m, RemoveMessage) for m in result["messages"])

    def test_preserves_existing_summary_in_prompt(self):
        messages = [
            HumanMessage(content="New request", id=f"msg_{i}")
            for i in range(10)
        ]
        state = {"messages": messages, "summary": "Previous work done."}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Updated summary."

        summarize_messages(state, mock_llm)
        prompt = mock_llm.invoke.call_args[0][0][0].content
        assert "Previous work done." in prompt

    def test_handles_list_type_llm_response(self):
        messages = [
            HumanMessage(content="Request", id=f"msg_{i}")
            for i in range(10)
        ]
        state = {"messages": messages, "summary": ""}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = [{"text": "Part1"}, {"text": "Part2"}]

        result = summarize_messages(state, mock_llm)
        assert "Part1 Part2" in result["summary"]

    def test_empty_llm_response_falls_back(self):
        messages = [
            HumanMessage(content="Request", id=f"msg_{i}")
            for i in range(10)
        ]
        state = {"messages": messages, "summary": "Existing summary."}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = ""

        result = summarize_messages(state, mock_llm)
        assert result["summary"] == "Existing summary."


# ---------------------------------------------------------------------------
# network_manager
# ---------------------------------------------------------------------------

class TestNetworkManager:
    def test_system_prompt_contains_scope(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="OK")

        state = {
            "messages": [HumanMessage(content="Start")],
            "county_specialty_thresholds": {"MI": {"washtenaw": {"cardiology": 10.0}}},
            "summary": "",
        }

        with patch("network_manager_agent.nodes.DataManager") as MockDM:
            mock_dm = MagicMock()
            MockDM.return_value = mock_dm

            network_manager(state, mock_llm)

        bound_tools = mock_llm.bind_tools.call_args[0][0]
        assert len(bound_tools) == 2

    def test_system_prompt_contains_columns(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="OK")

        state = {
            "messages": [HumanMessage(content="Start")],
            "county_specialty_thresholds": {},
            "summary": "",
        }

        with patch("network_manager_agent.nodes.DataManager") as MockDM:
            mock_dm = MagicMock()
            mock_dm.get_candidates_df.return_value.columns = ["entity", "specialty", "lat", "lon"]
            mock_dm.get_members_df.return_value.columns = ["lat", "lon", "county", "state"]
            MockDM.return_value = mock_dm

            network_manager(state, mock_llm)

        call_args = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "entity" in system_msg.content
        assert "county" in system_msg.content

    def test_includes_summary_in_prompt(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="OK")

        state = {
            "messages": [HumanMessage(content="Continue")],
            "county_specialty_thresholds": {},
            "summary": "Previous progress here.",
        }

        with patch("network_manager_agent.nodes.DataManager") as MockDM:
            mock_dm = MagicMock()
            MockDM.return_value = mock_dm

            network_manager(state, mock_llm)

        call_args = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "Previous progress here." in system_msg.content

    def test_system_prompt_requires_compute_coverage_validation(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content="OK")

        state = {
            "messages": [HumanMessage(content="Start")],
            "county_specialty_thresholds": {},
            "summary": "",
        }

        with patch("network_manager_agent.nodes.DataManager") as MockDM:
            mock_dm = MagicMock()
            MockDM.return_value = mock_dm

            network_manager(state, mock_llm)

        call_args = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "compute_coverage" in system_msg.content
        assert "definitive" in system_msg.content or "authoritative" in system_msg.content

    def test_returns_ai_response_in_messages(self):
        mock_llm = MagicMock()
        ai_resp = AIMessage(content="Response", id="ai_1")
        mock_llm.bind_tools.return_value.invoke.return_value = ai_resp

        state = {
            "messages": [HumanMessage(content="Start")],
            "county_specialty_thresholds": {},
            "summary": "",
        }

        with patch("network_manager_agent.nodes.DataManager") as MockDM:
            mock_dm = MagicMock()
            MockDM.return_value = mock_dm

            result = network_manager(state, mock_llm)

        assert len(result["messages"]) == 1
        assert result["messages"][0] is ai_resp


# ---------------------------------------------------------------------------
# execute_tools
# ---------------------------------------------------------------------------

class TestExecuteTools:
    def test_catches_error_returns_tool_message(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    id="ai_1",
                    tool_calls=[{"id": "tc_1", "name": "run_code", "args": {}}],
                )
            ]
        }

        with patch("network_manager_agent.nodes.ToolNode") as MockToolNode:
            mock_node = MagicMock()
            mock_node.invoke.side_effect = RuntimeError("sandbox crash")
            MockToolNode.return_value = mock_node

            result = execute_tools(state)

        assert len(result["messages"]) == 1
        tool_msg = result["messages"][0]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "tc_1"
        assert "sandbox crash" in tool_msg.content

    def test_handles_multiple_tool_calls_on_error(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    id="ai_1",
                    tool_calls=[
                        {"id": "tc_1", "name": "add_contract_entity", "args": {}},
                        {"id": "tc_2", "name": "run_code", "args": {}},
                    ],
                )
            ]
        }

        with patch("network_manager_agent.nodes.ToolNode") as MockToolNode:
            mock_node = MagicMock()
            mock_node.invoke.side_effect = RuntimeError("crash")
            MockToolNode.return_value = mock_node

            result = execute_tools(state)

        assert len(result["messages"]) == 2
        assert result["messages"][0].tool_call_id == "tc_1"
        assert result["messages"][1].tool_call_id == "tc_2"

    def test_handles_missing_tool_call_id(self):
        state = {
            "messages": [
                AIMessage(content="No tool calls", id="ai_1", tool_calls=[])
            ]
        }

        with patch("network_manager_agent.nodes.ToolNode") as MockToolNode:
            mock_node = MagicMock()
            mock_node.invoke.side_effect = RuntimeError("crash")
            MockToolNode.return_value = mock_node

            result = execute_tools(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_call_id == "unknown"
