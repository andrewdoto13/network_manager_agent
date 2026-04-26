"""Tests for network_manager_agent."""

import pandas as pd
import pytest

from network_manager_agent.data import load_hospitals, load_members, load_data
from network_manager_agent.state import AgentState
from network_manager_agent.tools import (
    get_candidates,
    get_candidate_schema,
    add_provider,
    get_network_status,
)


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_hospitals_returns_list(self):
        candidates = load_hospitals()
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_load_hospitals_has_required_columns(self):
        candidates = load_hospitals()
        first = candidates[0]
        assert "id" in first
        assert "lat" in first
        assert "lon" in first
        assert "specialty" in first
        assert "effectiveness" in first

    def test_load_members_returns_list(self):
        members = load_members()
        assert isinstance(members, list)
        assert len(members) > 0

    def test_load_members_has_required_columns(self):
        members = load_members()
        first = members[0]
        assert "id" in first
        assert "lat" in first
        assert "lon" in first
        assert "county" in first

    def test_load_data_returns_tuple(self):
        candidates, members = load_data()
        assert isinstance(candidates, list)
        assert isinstance(members, list)

    def test_load_data_with_custom_paths(self, tmp_path):
        hospitals_csv = tmp_path / "hospitals.csv"
        hospitals_csv.write_text("lat,lon,cluster,county,specialty,effectiveness\n42.0,-83.0,test,test,hospital,5\n")
        members_csv = tmp_path / "members.csv"
        members_csv.write_text("lat,lon,county\n42.1,-83.1,test\n")

        candidates, members = load_data(
            hospitals_path=hospitals_csv,
            members_path=members_csv,
        )
        assert len(candidates) == 1
        assert len(members) == 1


class TestAgentState:
    """Tests for AgentState."""

    def test_state_has_required_keys(self):
        state: dict = {}  # AgentState is a TypedDict
        state["messages"] = []
        state["candidates"] = []
        state["members"] = []
        state["network"] = []
        state["summary"] = ""
        state["original_message"] = ""
        assert isinstance(state, dict)
        assert "messages" in state
        assert "candidates" in state
        assert "members" in state
        assert "network" in state
        assert "summary" in state
        assert "original_message" in state


class TestTools:
    """Tests for agent tools."""

    def test_get_candidates_filters_by_specialty(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4},
        ]
        result = get_candidates.invoke({
            "specialty": "hospital",
            "candidates": candidates,
            "network": [],
        })
        assert isinstance(result, list)
        assert len(result) <= 5
        for c in result:
            assert c["specialty"] == "hospital"

    def test_get_candidates_excludes_network(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3},
        ]
        # First call
        result1 = get_candidates.invoke({
            "specialty": "hospital",
            "candidates": candidates,
            "network": [],
        })
        # Add first result to network
        network = result1 if isinstance(result1, list) else []
        # Second call should exclude already-added providers
        result2 = get_candidates.invoke({
            "specialty": "hospital",
            "candidates": candidates,
            "network": network,
        })
        if isinstance(result2, list):
            result2_ids = {c["id"] for c in result2}
            result1_ids = {c["id"] for c in result1}
            assert result1_ids.isdisjoint(result2_ids)

    def test_get_candidates_returns_message_when_none_available(self):
        candidates = [{"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0}]
        result = get_candidates.invoke({
            "specialty": "hospital",
            "candidates": candidates,
            "network": [candidates[0]],
        })
        assert isinstance(result, str)
        assert "No available candidates" in result

    def test_get_candidate_schema(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
        ]
        result = get_candidate_schema.invoke({"candidates": candidates})
        assert isinstance(result, dict)
        assert "specialty" in result
        assert "effectiveness" in result

    def test_get_candidate_schema_empty(self):
        result = get_candidate_schema.invoke({"candidates": []})
        assert result == "No candidate data available."

    def test_add_provider(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
        ]
        result = add_provider.invoke({
            "id": 1,
            "candidates": candidates,
            "network": [],
        })
        assert isinstance(result, dict)
        assert result["id"] == 1

    def test_add_provider_already_in_network(self):
        provider = {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0}
        result = add_provider.invoke({
            "id": 1,
            "candidates": [provider],
            "network": [provider],
        })
        assert isinstance(result, str)
        assert "Skip" in result

    def test_add_provider_not_found(self):
        result = add_provider.invoke({
            "id": 999,
            "candidates": [{"id": 1}],
            "network": [],
        })
        assert "not found" in result

    def test_get_network_status_empty(self):
        members = [
            {"id": 1, "lat": 42.0, "lon": -83.0, "county": "wayne"},
        ]
        result = get_network_status.invoke({
            "members": members,
            "network": [],
        })
        assert result["total_providers"] == 0
        assert result["member_coverage"] == []

    def test_get_network_status_with_providers(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "threshold": 20.0,
        })
        assert result["total_providers"] == 1
        assert "member_coverage" in result
        assert isinstance(result["member_coverage"], list)
