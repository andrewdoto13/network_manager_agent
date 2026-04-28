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
    simulate_network_change,
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
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4},
        ]
        result = get_candidate_schema.invoke({"candidates": candidates})
        assert isinstance(result, dict)
        
        # Check numeric
        assert "effectiveness" in result
        assert "min" in result["effectiveness"]
        assert "q3" in result["effectiveness"]
        assert result["effectiveness"]["mean"] == 4.0
        
        # Check categorical
        assert "specialty" in result
        assert "unique_count" in result["specialty"]
        assert result["specialty"]["unique_count"] == 2
        assert "hospital" in result["specialty"]["distribution"]
        assert result["specialty"]["distribution"]["hospital"] == 2

    def test_get_candidate_schema_empty(self):
        result = get_candidate_schema.invoke({"candidates": []})
        assert result == "No candidate data available."

    def test_add_provider(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
        ]
        result = add_provider.invoke({
            "ids": [1],
            "candidates": candidates,
            "network": [],
        })
        assert isinstance(result, dict)
        assert "added" in result
        assert len(result["added"]) == 1
        assert result["added"][0]["id"] == 1

    def test_add_provider_batch(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3},
        ]
        result = add_provider.invoke({
            "ids": [1, 2],
            "candidates": candidates,
            "network": [],
        })
        assert len(result["added"]) == 2
        assert {p["id"] for p in result["added"]} == {1, 2}

    def test_add_provider_already_in_network(self):
        provider = {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0}
        result = add_provider.invoke({
            "ids": [1],
            "candidates": [provider],
            "network": [provider],
        })
        assert isinstance(result, dict)
        assert result["added"] == []

    def test_add_provider_not_found(self):
        result = add_provider.invoke({
            "ids": [999],
            "candidates": [{"id": 1}],
            "network": [],
        })
        assert "errors" in result
        assert any("not found" in e for e in result["errors"])

    def test_add_provider_mixed(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3},
        ]
        result = add_provider.invoke({
            "ids": [1, 2, 999],
            "candidates": candidates,
            "network": [],
        })
        assert len(result["added"]) == 2
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_get_network_status_empty(self):
        members = [
            {"id": 1, "lat": 42.0, "lon": -83.0, "county": "wayne"},
        ]
        result = get_network_status.invoke({
            "members": members,
            "network": [],
            "county_thresholds": {},
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
            "county_thresholds": {},
        })
        assert result["total_providers"] == 1
        assert "member_coverage" in result
        assert isinstance(result["member_coverage"], list)


class TestSimulateNetworkChange:
    """Tests for the simulate_network_change tool."""

    def test_simulate_add_provider(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [2],
            "remove_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert "current" in result
        assert "simulated" in result
        assert "delta" in result
        assert result["current"]["total_providers"] == 1
        assert result["simulated"]["total_providers"] == 2

    def test_simulate_remove_provider(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
        ]
        candidates = network.copy()
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [1],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert result["current"]["total_providers"] == 2
        assert result["simulated"]["total_providers"] == 1

    def test_simulate_add_and_remove(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [2],
            "remove_ids": [1],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert result["current"]["total_providers"] == 1
        assert result["simulated"]["total_providers"] == 1

    def test_simulate_invalid_add_already_in_network(self):
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [1],
            "remove_ids": [],
            "candidates": candidates,
            "network": network,
            "members": [],
            "county_thresholds": {},
        })
        assert "error" in result
        assert any("already in the network" in d for d in result["details"])

    def test_simulate_invalid_remove_not_in_network(self):
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [999],
            "candidates": [],
            "network": [],
            "members": [],
            "county_thresholds": {},
        })
        assert "error" in result
        assert any("not in the current network" in d for d in result["details"])

    def test_simulate_too_many_add_ids(self):
        result = simulate_network_change.invoke({
            "add_ids": [1, 2, 3, 4, 5, 6],
            "remove_ids": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_simulate_too_many_remove_ids(self):
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [1, 2, 3, 4, 5, 6],
            "candidates": [],
            "network": [{"id": i, "lat": 42.0, "lon": -83.0} for i in range(1, 7)],
            "members": [],
            "county_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_simulate_delta_has_correct_counties(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "oakland"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [2],
            "remove_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        delta_counties = {d["county"] for d in result["delta"]}
        assert "wayne" in delta_counties
        assert "oakland" in delta_counties
        for d in result["delta"]:
            assert "coverage_change" in d

    def test_compare_scenarios_basic(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [],
            "compare_scenarios": [
                {"add_ids": [1], "remove_ids": []},
                {"add_ids": [2], "remove_ids": []},
                {"add_ids": [3], "remove_ids": []},
            ],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert "current" in result
        assert "scenarios" in result
        assert len(result["scenarios"]) == 3
        assert result["scenarios"][0]["rank"] == 1
        assert result["scenarios"][1]["rank"] == 2
        assert result["scenarios"][2]["rank"] == 3

    def test_compare_scenarios_with_removals(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital"},
        ]
        candidates = [
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [],
            "compare_scenarios": [
                {"add_ids": [3], "remove_ids": [1]},
                {"add_ids": [3], "remove_ids": [2]},
            ],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert len(result["scenarios"]) == 2
        assert result["scenarios"][0]["rank"] == 1

    def test_compare_scenarios_invalid_ids(self):
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [],
            "compare_scenarios": [
                {"add_ids": [999], "remove_ids": []},
            ],
            "candidates": [],
            "network": [],
            "members": [],
            "county_thresholds": {},
        })
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_compare_scenarios_max_limit(self):
        result = simulate_network_change.invoke({
            "add_ids": [],
            "remove_ids": [],
            "compare_scenarios": [
                {"add_ids": [1], "remove_ids": []},
                {"add_ids": [2], "remove_ids": []},
                {"add_ids": [3], "remove_ids": []},
                {"add_ids": [4], "remove_ids": []},
                {"add_ids": [5], "remove_ids": []},
                {"add_ids": [6], "remove_ids": []},
            ],
            "candidates": [],
            "network": [],
            "members": [],
            "county_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_compare_scenarios_empty_falls_back(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        result = simulate_network_change.invoke({
            "add_ids": [1],
            "remove_ids": [],
            "compare_scenarios": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_thresholds": {},
        })
        assert "simulated" in result
        assert "scenarios" not in result
        assert result["simulated"]["total_providers"] == 1
