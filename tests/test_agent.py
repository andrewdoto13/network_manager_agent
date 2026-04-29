"""Tests for network_manager_agent."""

import pandas as pd
import pytest

from network_manager_agent.data import load_candidates, load_members, load_data
from network_manager_agent.state import AgentState
from network_manager_agent.tools import (
    get_candidates,
    get_candidate_schema,
    add_contract_entity,
    get_network_status,
    simulate_network_change,
)


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_candidates_returns_list(self):
        candidates = load_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_load_candidates_has_required_columns(self):
        candidates = load_candidates()
        first = candidates[0]
        assert "id" in first
        # Check for either lat or Latitude
        assert any(col in first for col in ["lat", "Latitude"])
        assert any(col in first for col in ["lon", "Longitude"])
        assert any(col in first for col in ["specialty", "Specialty"])
        assert any(col in first for col in ["effectiveness", "Effectiveness"])

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
        candidates_csv = tmp_path / "candidates.csv"
        candidates_csv.write_text("lat,lon,cluster,county,specialty,effectiveness\n42.0,-83.0,test,test,hospital,5\n")
        members_csv = tmp_path / "members.csv"
        members_csv.write_text("lat,lon,county\n42.1,-83.1,test\n")
    
        candidates, members = load_data(
            candidates_path=candidates_csv,
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
        state["county_specialty_thresholds"] = {}
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
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "Primary Contract Entity": "Entity B"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "Primary Contract Entity": "Entity A"},
        ]
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
        })
        assert isinstance(result, list)
        assert len(result) <= 5
        for e in result:
            assert "entity_id" in e
            assert "hospital" in e["capabilities"]["specialties"]

    def test_get_candidates_excludes_network(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "Primary Contract Entity": "Entity B"},
        ]
        # First call
        result1 = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
        })
        # Add first result to network (simulate adding one of the entities)
        # result1 is a list of entity summaries
        network_entity = result1[0]["entity_id"]
        # To simulate the entity being in network, we add its providers
        network = [p for p in candidates if p["Primary Contract Entity"] == network_entity]
        
        # Second call should exclude already-added entities
        result2 = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": network,
        })
        if isinstance(result2, list):
            result2_ids = {e["entity_id"] for e in result2}
            assert network_entity not in result2_ids

    def test_get_candidates_returns_message_when_none_available(self):
        candidates = [{"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "Primary Contract Entity": "Entity A"}]
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [candidates[0]],
        })
        assert isinstance(result, str)
        assert "No available entities" in result

    def test_get_candidate_schema(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "Primary Contract Entity": "Entity B"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "Primary Contract Entity": "Entity A"},
        ]
        result = get_candidate_schema.invoke({"candidates": candidates})
        assert isinstance(result, dict)
        
        # Check entity-level numeric
        assert "avg_effectiveness" in result
        assert "min" in result["avg_effectiveness"]
        assert "q3" in result["avg_effectiveness"]
        
        # Check entity size
        assert "provider_count" in result
        assert "unique_count" not in result["provider_count"] # It's numeric
        assert "min" in result["provider_count"]
        
        # Check categorical (specialties is now a list per entity, so it's handled as 'object' in the profile)
        assert "specialties" in result
        assert "unique_count" in result["specialties"]

    def test_get_candidate_schema_empty(self):
        result = get_candidate_schema.invoke({"candidates": []})
        assert result == "No candidate data available."

    def test_add_contract_entity(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 4, "Primary Contract Entity": "Entity A"},
        ]
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity A"],
            "candidates": candidates,
            "network": [],
        })
        assert isinstance(result, dict)
        assert "added_providers" in result
        assert len(result["added_providers"]) == 2
        assert {p["id"] for p in result["added_providers"]} == {1, 2}

    def test_add_contract_entity_batch(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "Primary Contract Entity": "Entity B"},
        ]
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity A", "Entity B"],
            "candidates": candidates,
            "network": [],
        })
        assert len(result["added_providers"]) == 2
        assert {p["id"] for p in result["added_providers"]} == {1, 2}

    def test_add_contract_entity_already_in_network(self):
        provider = {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "Primary Contract Entity": "Entity A"}
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity A"],
            "candidates": [provider],
            "network": [provider],
        })
        assert isinstance(result, dict)
        assert result["added_providers"] == []

    def test_add_contract_entity_not_found(self):
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity Z"],
            "candidates": [{"id": 1, "Primary Contract Entity": "Entity A"}],
            "network": [],
        })
        assert "errors" in result
        assert any("not found" in e for e in result["errors"])

    def test_add_contract_entity_mixed(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "Primary Contract Entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "Primary Contract Entity": "Entity B"},
        ]
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity A", "Entity B", "Entity Z"],
            "candidates": candidates,
            "network": [],
        })
        assert len(result["added_providers"]) == 2
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_get_network_status_empty(self):
        members = [
            {"id": 1, "lat": 42.0, "lon": -83.0, "county": "wayne"},
        ]
        result = get_network_status.invoke({
            "members": members,
            "network": [],
            "county_specialty_thresholds": {},
            "candidates": [],
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
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0}},
            "candidates": network,
        })
        assert result["total_providers"] == 1
        assert "member_coverage" in result
        assert isinstance(result["member_coverage"], list)
        assert len(result["member_coverage"]) > 0
        for cov in result["member_coverage"]:
            assert "county" in cov
            assert "specialty" in cov
            assert "members_with_access" in cov
            assert "total_members" in cov
            assert "coverage_percentage" in cov


class TestSimulateNetworkChange:
    """Tests for the simulate_network_change tool."""

    def test_simulate_add_entity(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
        })
        assert "current" in result
        assert "simulated" in result
        assert "delta" in result
        assert result["current"]["total_providers"] == 1
        assert result["simulated"]["total_providers"] == 2

    def test_simulate_remove_entity(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        candidates = network.copy()
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": ["Entity A"],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
        })
        assert result["current"]["total_providers"] == 2
        assert result["simulated"]["total_providers"] == 1

    def test_simulate_add_and_remove(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": ["Entity A"],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
        })
        assert result["current"]["total_providers"] == 1
        assert result["simulated"]["total_providers"] == 1

    def test_simulate_invalid_add_already_in_network(self):
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity A"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert any("already in the network" in d for d in result["details"])

    def test_simulate_invalid_remove_not_in_network(self):
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": ["Entity Z"],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert any("not in the current network" in d for d in result["details"])

    def test_simulate_too_many_add_ids(self):
        result = simulate_network_change.invoke({
            "add_entity_ids": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "remove_entity_ids": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_simulate_too_many_remove_ids(self):
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "candidates": [],
            "network": [{"id": i, "lat": 42.0, "lon": -83.0, "Primary Contract Entity": f"E{i}"} for i in range(1, 7)],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_simulate_delta_has_correct_counties(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "oakland"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0}, "oakland": {"hospital": 20.0}},
        })
        delta_counties = {d["county"] for d in result["delta"]}
        delta_specialties = {d["specialty"] for d in result["delta"]}
        assert "wayne" in delta_counties
        assert "oakland" in delta_counties
        assert "hospital" in delta_specialties
        for d in result["delta"]:
            assert "coverage_change" in d

    def test_compare_scenarios_basic(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital", "Primary Contract Entity": "Entity C"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": [],
            "compare_scenarios": [
                {"add_entity_ids": ["Entity A"], "remove_entity_ids": []},
                {"add_entity_ids": ["Entity B"], "remove_entity_ids": []},
                {"add_entity_ids": ["Entity C"], "remove_entity_ids": []},
            ],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
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
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        candidates = [
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital", "Primary Contract Entity": "Entity C"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": [],
            "compare_scenarios": [
                {"add_entity_ids": ["Entity C"], "remove_entity_ids": ["Entity A"]},
                {"add_entity_ids": ["Entity C"], "remove_entity_ids": ["Entity B"]},
            ],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
        })
        assert len(result["scenarios"]) == 2
        assert result["scenarios"][0]["rank"] == 1

    def test_compare_scenarios_invalid_ids(self):
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": [],
            "compare_scenarios": [
                {"add_entity_ids": ["Entity Z"], "remove_entity_ids": []},
            ],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_compare_scenarios_max_limit(self):
        result = simulate_network_change.invoke({
            "add_entity_ids": [],
            "remove_entity_ids": [],
            "compare_scenarios": [
                {"add_entity_ids": ["E1"], "remove_entity_ids": []},
                {"add_entity_ids": ["E2"], "remove_entity_ids": []},
                {"add_entity_ids": ["E3"], "remove_entity_ids": []},
                {"add_entity_ids": ["E4"], "remove_entity_ids": []},
                {"add_entity_ids": ["E5"], "remove_entity_ids": []},
                {"add_entity_ids": ["E6"], "remove_entity_ids": []},
            ],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_compare_scenarios_empty_falls_back(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity A"],
            "remove_entity_ids": [],
            "compare_scenarios": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {},
        })
        assert "simulated" in result
        assert "scenarios" not in result
        assert result["simulated"]["total_providers"] == 1

    def test_get_network_status_per_county_specialty(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne"},
            {"id": 3, "lat": 42.5, "lon": -83.6, "county": "oakland"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.31, "lon": -83.48, "specialty": "cardiologist"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0, "cardiologist": 20.0}, "oakland": {"hospital": 20.0}},
            "candidates": candidates,
        })
        assert result["total_providers"] == 2
        coverage = result["member_coverage"]
        assert len(coverage) == 3
        coverage_map = {(c["county"], c["specialty"]): c for c in coverage}
        assert ("wayne", "hospital") in coverage_map
        assert ("wayne", "cardiologist") in coverage_map
        assert ("oakland", "hospital") in coverage_map

    def test_get_network_status_validation_errors(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0, "neurologist": 10.0}},
            "candidates": candidates,
        })
        assert "validation_errors" in result
        assert len(result["validation_errors"]) > 0
        assert any("neurologist" in e for e in result["validation_errors"])

    def test_simulate_network_change_validation_errors(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "Primary Contract Entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "Primary Contract Entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0, "psychiatrist": 15.0}},
        })
        assert "validation_errors" in result
        assert any("psychiatrist" in e for e in result["validation_errors"])

    def test_compute_coverage_no_specialty_in_candidates(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49},
        ]
        candidates = []
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"wayne": {"hospital": 20.0}},
            "candidates": candidates,
        })
        assert result["total_providers"] == 1
        assert len(result["member_coverage"]) == 1
        assert result["member_coverage"][0]["coverage_percentage"] == 0.0

    def test_compute_coverage_zero_coverage_when_no_network_providers(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"wayne": {"cardiologist": 20.0}},
            "candidates": candidates,
        })
        assert len(result["member_coverage"]) == 1
        assert result["member_coverage"][0]["specialty"] == "cardiologist"
        assert result["member_coverage"][0]["coverage_percentage"] == 0.0
