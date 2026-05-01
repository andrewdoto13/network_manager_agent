"""Tests for network_manager_agent."""

import pytest

from network_manager_agent.data import load_candidates, load_members, load_data
from network_manager_agent.state import AgentState
from network_manager_agent.tools import (
    get_candidates,
    add_contract_entity,
    get_network_status,
    simulate_network_change,
    _filter_by_service_area,
    precompute_entity_summaries,
    get_candidate_schema_profile,
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
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A", "Primary Institutional Affiliation": "Health System A", "new_patient_claims": "Yes", "claims_volume": "Core", "City": "City A"},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B", "Primary Institutional Affiliation": "Health System B", "new_patient_claims": "No", "claims_volume": "Standard", "City": "City B"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "entity": "Entity A", "Primary Institutional Affiliation": "Health System A", "new_patient_claims": "Yes", "claims_volume": "Standard", "City": "City A"},
        ]
        entity_summaries = precompute_entity_summaries(candidates)
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
            "entity_summaries": entity_summaries,
        })
        assert isinstance(result, list)
        assert len(result) <= 5
        for e in result:
            assert "entity_id" in e
            assert "hospital" in e["capabilities"]["specialties"]
            # Verify new metrics
            assert "new_patient_rate" in e["metrics"]
            assert "geographic_reach" in e["metrics"]


    def test_get_candidates_excludes_network(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B"},
        ]
        entity_summaries = precompute_entity_summaries(candidates)
        # First call
        result1 = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
            "entity_summaries": entity_summaries,
        })
        # Add first result to network (simulate adding one of the entities)
        # result1 is a list of entity summaries
        network_entity = result1[0]["entity_id"]
        # To simulate the entity being in network, we add its providers
        network = [p for p in candidates if p["entity"] == network_entity]

        # Second call should exclude already-added entities
        result2 = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": network,
            "entity_summaries": entity_summaries,
        })
        if isinstance(result2, list):
            result2_ids = {e["entity_id"] for e in result2}
            assert network_entity not in result2_ids

    def test_get_candidates_returns_message_when_none_available(self):
        candidates = [{"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "entity": "Entity A"}]
        entity_summaries = precompute_entity_summaries(candidates)
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [candidates[0]],
            "entity_summaries": entity_summaries,
        })
        assert isinstance(result, str)
        assert "No available entities" in result

    def test_add_contract_entity(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 4, "entity": "Entity A"},
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
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B"},
        ]
        result = add_contract_entity.invoke({
            "entity_ids": ["Entity A", "Entity B"],
            "candidates": candidates,
            "network": [],
        })
        assert len(result["added_providers"]) == 2
        assert {p["id"] for p in result["added_providers"]} == {1, 2}

    def test_add_contract_entity_already_in_network(self):
        provider = {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "entity": "Entity A"}
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
            "candidates": [{"id": 1, "entity": "Entity A"}],
            "network": [],
        })
        assert "errors" in result
        assert any("not found" in e for e in result["errors"])

    def test_add_contract_entity_mixed(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0}}},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
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
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
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
            "network": [{"id": i, "lat": 42.0, "lon": -83.0, "entity": f"E{i}"} for i in range(1, 7)],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "error" in result
        assert "maximum 5" in result["error"]

    def test_simulate_delta_has_correct_counties(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "oakland", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0}, "oakland": {"hospital": 20.0}}},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne", "state": "mi"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital", "entity": "Entity C"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
        ]
        candidates = [
            {"id": 3, "lat": 42.0, "lon": -83.0, "specialty": "hospital", "entity": "Entity C"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = []
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
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
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "county": "wayne", "state": "mi"},
            {"id": 3, "lat": 42.5, "lon": -83.6, "county": "oakland", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
            {"id": 2, "lat": 42.31, "lon": -83.48, "specialty": "cardiologist"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0, "cardiologist": 20.0}, "oakland": {"hospital": 20.0}}},
            "candidates": candidates,
        })
        assert result["total_providers"] == 2
        coverage = result["member_coverage"]
        assert len(coverage) == 3
        coverage_map = {(c.get("state", ""), c["county"], c["specialty"]): c for c in coverage}
        assert ("mi", "wayne", "hospital") in coverage_map
        assert ("mi", "wayne", "cardiologist") in coverage_map
        assert ("mi", "oakland", "hospital") in coverage_map

    def test_get_network_status_validation_errors(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0, "neurologist": 10.0}}},
            "candidates": candidates,
        })
        assert "validation_errors" in result
        assert len(result["validation_errors"]) > 0
        assert any("neurologist" in e for e in result["validation_errors"])

    def test_simulate_network_change_validation_errors(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
        ]
        candidates = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital", "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.3, "specialty": "hospital", "entity": "Entity B"},
        ]
        result = simulate_network_change.invoke({
            "add_entity_ids": ["Entity B"],
            "remove_entity_ids": [],
            "candidates": candidates,
            "network": network,
            "members": members,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0, "psychiatrist": 15.0}}},
        })
        assert "validation_errors" in result
        assert any("psychiatrist" in e for e in result["validation_errors"])

    def test_compute_coverage_no_specialty_in_candidates(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49},
        ]
        candidates = []
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"mi": {"wayne": {"hospital": 20.0}}},
            "candidates": candidates,
        })
        assert result["total_providers"] == 1
        assert len(result["member_coverage"]) == 1
        assert result["member_coverage"][0]["coverage_percentage"] == 0.0

    def test_compute_coverage_zero_coverage_when_no_network_providers(self):
        members = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "county": "wayne", "state": "mi"},
        ]
        network = [
            {"id": 1, "lat": 42.32, "lon": -83.49, "specialty": "hospital"},
        ]
        candidates = network.copy()
        result = get_network_status.invoke({
            "members": members,
            "network": network,
            "county_specialty_thresholds": {"mi": {"wayne": {"cardiologist": 20.0}}},
            "candidates": candidates,
        })
        assert len(result["member_coverage"]) == 1
        assert result["member_coverage"][0]["specialty"] == "cardiologist"
        assert result["member_coverage"][0]["coverage_percentage"] == 0.0


class TestServiceAreaFiltering:
    """Tests for _filter_by_service_area."""

    def test_empty_thresholds_returns_all(self):
        candidates = [
            {"id": 1, "lat": 42.0, "lon": -83.0, "entity": "A"},
            {"id": 2, "lat": 45.0, "lon": -90.0, "entity": "B"},
        ]
        members = [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}]
        result, _ = _filter_by_service_area(candidates, members, {})
        assert len(result) == 2

    def test_entity_retained_when_one_provider_in_bounds(self):
        candidates = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "entity": "Entity A"},
            {"id": 2, "lat": 47.0, "lon": -85.0, "entity": "Entity A"},
            {"id": 3, "lat": 45.0, "lon": -90.0, "entity": "Entity B"},
        ]
        members = [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}]
        thresholds = {"mi": {"wayne": {"hospital": 10.0}}}
        result, _ = _filter_by_service_area(candidates, members, thresholds)
        result_ids = {r["id"] for r in result}
        assert 1 in result_ids
        assert 2 in result_ids
        assert 3 not in result_ids

    def test_entity_excluded_when_no_providers_in_bounds(self):
        candidates = [
            {"id": 1, "lat": 47.0, "lon": -85.0, "entity": "Entity A"},
            {"id": 2, "lat": 48.0, "lon": -86.0, "entity": "Entity A"},
        ]
        members = [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}]
        thresholds = {"mi": {"wayne": {"hospital": 10.0}}}
        result, _ = _filter_by_service_area(candidates, members, thresholds)
        assert len(result) == 0

    def test_empty_candidates_returns_empty(self):
        result, _ = _filter_by_service_area([], [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}], {"mi": {"wayne": {"hospital": 10.0}}})
        assert result == []

    def test_empty_members_returns_all(self):
        candidates = [{"id": 1, "lat": 42.0, "lon": -83.0, "entity": "A"}]
        result, _ = _filter_by_service_area(candidates, [], {"mi": {"wayne": {"hospital": 10.0}}})
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["entity"] == "A"

    def test_scaled_coordinates_normalized(self):
        candidates = [
            {"id": 1, "Latitude": 42300000, "Longitude": 83500000, "entity": "Entity A"},
            {"id": 2, "Latitude": 47000000, "Longitude": 85000000, "entity": "Entity B"},
        ]
        members = [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}]
        thresholds = {"mi": {"wayne": {"hospital": 10.0}}}
        result, _ = _filter_by_service_area(candidates, members, thresholds)
        result_ids = {r["id"] for r in result}
        assert 1 in result_ids
        assert 2 not in result_ids


class TestPrecomputeFunctions:
    """Tests for precompute_entity_summaries and get_candidate_schema_profile."""

    def test_precompute_entity_summaries_basic(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity A"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "entity": "Entity B"},
        ]
        result = precompute_entity_summaries(candidates)
        assert isinstance(result, list)
        assert len(result) == 2
        entity_map = {r["entity"]: r for r in result}
        assert entity_map["Entity A"]["provider_count"] == 2
        assert entity_map["Entity B"]["provider_count"] == 1

    def test_precompute_entity_summaries_empty(self):
        assert precompute_entity_summaries([]) == []

    def test_get_candidate_schema_profile_with_summaries(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
        ]
        summaries = precompute_entity_summaries(candidates)
        profile = get_candidate_schema_profile(entity_summaries=summaries)
        assert isinstance(profile, dict)
        assert "avg_effectiveness" in profile
        assert "provider_count" in profile

    def test_get_candidate_schema_profile_fallback_candidates(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
        ]
        profile = get_candidate_schema_profile(candidates=candidates)
        assert isinstance(profile, dict)
        assert "avg_effectiveness" in profile


class TestCachedAggregation:
    """Tests that get_candidates uses pre-computed entity_summaries."""

    def test_get_candidates_uses_cached_summaries(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B"},
        ]
        entity_summaries = precompute_entity_summaries(candidates)
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
            "entity_summaries": entity_summaries,
        })
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_candidates_fallback_without_summaries(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "entity": "Entity B"},
        ]
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
            "entity_summaries": [],
        })
        assert isinstance(result, list)
        assert len(result) == 2

    def test_full_pipeline_filter_then_aggregate(self):
        candidates = [
            {"id": 1, "lat": 42.3, "lon": -83.5, "specialty": "hospital", "effectiveness": 5, "entity": "Entity A"},
            {"id": 2, "lat": 42.4, "lon": -83.4, "specialty": "hospital", "effectiveness": 3, "entity": "Entity A"},
            {"id": 3, "lat": 47.0, "lon": -85.0, "specialty": "hospital", "effectiveness": 4, "entity": "Entity B"},
        ]
        members = [{"id": 1, "lat": 42.3, "lon": -83.5, "state": "mi", "county": "wayne"}]
        thresholds = {"mi": {"wayne": {"hospital": 10.0}}}

        filtered, _ = _filter_by_service_area(candidates, members, thresholds)
        summaries = precompute_entity_summaries(filtered)
        profile = get_candidate_schema_profile(candidates=filtered)

        assert len(filtered) == 2
        assert len(summaries) == 1
        assert summaries[0]["entity"] == "Entity A"
        assert "avg_effectiveness" in profile

    def test_total_claims_amount_aggregation(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "total_claims_amount": 100.0, "medicare_total_claims_amount": 50.0, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "total_claims_amount": 200.0, "medicare_total_claims_amount": 80.0, "entity": "Entity A"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "total_claims_amount": 500.0, "medicare_total_claims_amount": 250.0, "entity": "Entity B"},
        ]
        summaries = precompute_entity_summaries(candidates)
        entity_map = {r["entity"]: r for r in summaries}
        assert entity_map["Entity A"]["total_claims_amount"] == 300.0
        assert entity_map["Entity A"]["avg_total_claims_amount"] == 150.0
        assert entity_map["Entity A"]["total_medicare_claims_amount"] == 130.0
        assert entity_map["Entity B"]["total_claims_amount"] == 500.0
        assert entity_map["Entity B"]["avg_total_claims_amount"] == 500.0

    def test_get_candidates_sort_by_total_claims(self):
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5, "total_claims_amount": 100.0, "entity": "Entity A"},
            {"id": 2, "specialty": "hospital", "lat": 42.1, "lon": -83.1, "effectiveness": 3, "total_claims_amount": 200.0, "entity": "Entity A"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 4, "total_claims_amount": 500.0, "entity": "Entity B"},
        ]
        entity_summaries = precompute_entity_summaries(candidates)
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": candidates,
            "network": [],
            "entity_summaries": entity_summaries,
            "sort_by": "total_claims_amount",
            "ascending": False,
        })
        assert isinstance(result, list)
        assert result[0]["entity_id"] == "Entity B"
        assert result[1]["entity_id"] == "Entity A"
        assert result[0]["metrics"]["total_claims"] == 500.0
        assert result[1]["metrics"]["total_claims"] == 300.0
