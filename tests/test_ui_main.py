"""Tests for ui.py and main.py — agent runner, CLI entry point."""

import json
import argparse
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from io import StringIO

import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from network_manager_agent.ui import run_agent
from network_manager_agent.main import run_agent_session


# ---------------------------------------------------------------------------
# run_agent (ui.py)
# ---------------------------------------------------------------------------

class TestRunAgent:
    def test_streams_output_and_writes_log(self, tmp_path, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            {"network_manager": {"messages": [AIMessage(content="Analysis", id="ai_1")]}},
            {"tools": {"messages": [ToolMessage(content='{"added_entities": ["Entity A"]}', tool_call_id="1", name="add_contract_entity")]}},
        ]

        inputs = {"messages": [HumanMessage(content="Test prompt")]}
        config = {"configurable": {"thread_id": "test"}}

        with patch("network_manager_agent.ui.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20240101_120000"
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                run_agent(mock_agent, inputs, config, output_dir=tmp_path)

        assert "Starting Agent Execution" in stdout.getvalue()
        assert "NODE: network_manager" in stdout.getvalue()
        assert "NODE: tools" in stdout.getvalue()
        log_dir = tmp_path / "thread_test"
        log_files = list(log_dir.glob("*.txt"))
        assert len(log_files) == 1
        log_content = log_files[0].read_text()
        assert "HUMAN INPUT" in log_content
        assert "Test prompt" in log_content
        assert "add_contract_entity" in log_content

    def test_handles_empty_messages(self, tmp_path, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            {"network_manager": {"messages": [AIMessage(content="OK", id="ai_1")]}},
        ]

        inputs = {"messages": []}
        config = {"configurable": {"thread_id": "test"}}

        with patch("network_manager_agent.ui.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20240101_120000"
            with patch("sys.stdout", new_callable=StringIO):
                run_agent(mock_agent, inputs, config, output_dir=tmp_path)

        log_dir = tmp_path / "thread_test"
        log_files = list(log_dir.glob("*.txt"))
        assert len(log_files) == 1

    def test_logs_tool_calls_in_file(self, tmp_path, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            {
                "network_manager": {
                    "messages": [
                        AIMessage(
                            content="",
                            id="ai_1",
                            tool_calls=[{"id": "tc_1", "name": "run_code", "args": {"code": "x=1"}}],
                        )
                    ]
                }
            },
        ]

        inputs = {"messages": [HumanMessage(content="Run code")]}
        config = {"configurable": {"thread_id": "test"}}

        with patch("network_manager_agent.ui.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20240101_120000"
            with patch("sys.stdout", new_callable=StringIO):
                run_agent(mock_agent, inputs, config, output_dir=tmp_path)

        log_dir = tmp_path / "thread_test"
        log_files = list(log_dir.glob("*.txt"))
        log_content = log_files[0].read_text()
        assert "TOOL CALL" in log_content
        assert "run_code" in log_content


# ---------------------------------------------------------------------------
# run_agent_session (main.py)
# ---------------------------------------------------------------------------

class TestRunAgentSession:
    def test_builds_correct_inputs(self, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        mock_state = MagicMock()
        mock_state.values = {"summary": "Done."}
        mock_agent.get_state.return_value = mock_state

        with patch("network_manager_agent.main.run_agent") as mock_run:
            with patch("sys.stdout", new_callable=StringIO):
                run_agent_session(
                    mock_agent,
                    "Add cardiology providers",
                    {"MI": {"washtenaw": {"cardiology": 10.0}}},
                    {"configurable": {"thread_id": "1"}},
                )

        mock_run.assert_called_once()
        inputs = mock_run.call_args[0][1]
        assert len(inputs["messages"]) == 1
        assert isinstance(inputs["messages"][0], HumanMessage)
        assert inputs["messages"][0].content == "Add cardiology providers"
        assert "county_specialty_thresholds" in inputs

    def test_prints_summary_when_available(self, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        mock_state = MagicMock()
        mock_state.values = {"summary": "Added 3 entities."}
        mock_agent.get_state.return_value = mock_state

        with patch("network_manager_agent.main.run_agent"):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                run_agent_session(
                    mock_agent,
                    "Test",
                    {},
                    {"configurable": {"thread_id": "1"}},
                )

        assert "Added 3 entities." in stdout.getvalue()

    def test_skips_summary_when_empty(self, clean_data_manager):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        mock_state = MagicMock()
        mock_state.values = {"summary": ""}
        mock_agent.get_state.return_value = mock_state

        with patch("network_manager_agent.main.run_agent"):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                run_agent_session(
                    mock_agent,
                    "Test",
                    {},
                    {"configurable": {"thread_id": "1"}},
                )

        assert "--- Summary ---" not in stdout.getvalue()


# ---------------------------------------------------------------------------
# CLI argument parsing (main.py)
# ---------------------------------------------------------------------------

class TestMainArgParsing:
    def test_default_args(self):
        from network_manager_agent.main import argparse as ap
        parser = ap.ArgumentParser()
        parser.add_argument("prompt", nargs="?", default=None)
        parser.add_argument("--thread-id", type=str, default="1")
        parser.add_argument("--list-threads", action="store_true")
        parser.add_argument("--clear-all", action="store_true")

        args = parser.parse_args([])
        assert args.prompt is None
        assert args.thread_id == "1"
        assert not args.list_threads
        assert not args.clear_all

    def test_prompt_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("prompt", nargs="?", default=None)
        args = parser.parse_args(["Add providers"])
        assert args.prompt == "Add providers"

    def test_thread_id_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--thread-id", type=str, default="1")
        args = parser.parse_args(["--thread-id", "42"])
        assert args.thread_id == "42"
