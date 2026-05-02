"""Tests for run_code tool and expanded get_candidates functionality."""

import pytest

from network_manager_agent.tools import (
    get_candidates,
    run_code,
    precompute_entity_summaries,
    get_raw_candidate_schema_profile,
)


class TestGetCandidatesExpanded:
    """Tests for expanded get_candidates functionality."""

    @pytest.fixture
    def mock_candidates(self):
        return [
            {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5.0, "entity": "Entity A", "city": "Detroit"},
            {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 2.0, "entity": "Entity B", "city": "Ann Arbor"},
            {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 1.0, "entity": "Entity C", "city": "Flint"},
            {"id": 4, "specialty": "clinic", "lat": 42.3, "lon": -83.3, "effectiveness": 4.0, "entity": "Entity D", "city": "Dearborn"},
            {"id": 5, "specialty": "hospital", "lat": 42.4, "lon": -83.4, "effectiveness": 3.0, "entity": "Entity E", "city": "Lansing"},
            {"id": 6, "specialty": "hospital", "lat": 42.5, "lon": -83.5, "effectiveness": 4.5, "entity": "Entity F", "city": "Grand Rapids"},
            {"id": 7, "specialty": "hospital", "lat": 42.6, "lon": -83.6, "effectiveness": 2.5, "entity": "Entity G", "city": "Kalamazoo"},
            {"id": 8, "specialty": "hospital", "lat": 42.7, "lon": -83.7, "effectiveness": 3.5, "entity": "Entity H", "city": "Saginaw"},
        ]

    def test_get_candidates_specialty_discovery(self, mock_candidates):
        """Call with no specialties → returns available_specialties dict."""
        result = get_candidates.invoke({
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": [],
        })
        assert isinstance(result, dict)
        assert "available_specialties" in result
        assert "hospital" in result["available_specialties"]
        assert "clinic" in result["available_specialties"]
        assert result["total_entities"] == 8

    def test_get_candidates_search_by_entity_name(self, mock_candidates):
        """Search by entity name fragment."""
        result = get_candidates.invoke({
            "search": "Entity A",
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": [],
        })
        assert isinstance(result, dict)
        assert "entities" in result
        assert len(result["entities"]) >= 1
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert "Entity A" in entity_ids

    def test_get_candidates_search_by_city(self, mock_candidates):
        """Search by city name."""
        result = get_candidates.invoke({
            "search": "Detroit",
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": [],
        })
        assert isinstance(result, dict)
        assert "entities" in result
        assert len(result["entities"]) >= 1
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert "Entity A" in entity_ids

    def test_get_candidates_pagination(self, mock_candidates):
        """Test pagination with offset and limit."""
        page1 = get_candidates.invoke({
            "specialties": ["hospital"],
            "limit": 2,
            "offset": 0,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        page2 = get_candidates.invoke({
            "specialties": ["hospital"],
            "limit": 2,
            "offset": 2,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert isinstance(page1, dict)
        assert "pagination" in page1
        assert page1["pagination"]["total_matching"] == 6  # 6 hospital entities
        assert page1["pagination"]["offset"] == 0
        assert page1["pagination"]["returned"] == 2
        assert page2["pagination"]["offset"] == 2
        # Pages should have different entities
        page1_ids = {e["entity_id"] for e in page1["entities"]}
        page2_ids = {e["entity_id"] for e in page2["entities"]}
        assert page1_ids.isdisjoint(page2_ids)

    def test_get_candidates_min_effectiveness(self, mock_candidates):
        """Filter by minimum effectiveness."""
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "min_effectiveness": 4.0,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert isinstance(result, dict)
        for e in result["entities"]:
            assert e["metrics"]["avg_effectiveness"] >= 4.0

    def test_get_candidates_exclude_entity_ids(self, mock_candidates):
        """Exclude specific entities."""
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "exclude_entity_ids": ["Entity A", "Entity E"],
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert isinstance(result, dict)
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert "Entity A" not in entity_ids
        assert "Entity E" not in entity_ids

    def test_get_candidates_include_not_in_network(self, mock_candidates):
        """See all entities including already-contracted ones."""
        network = [mock_candidates[0]]  # Entity A is in network
        # Default: exclude network entities
        result_excluded = get_candidates.invoke({
            "specialties": ["hospital"],
            "candidates": mock_candidates,
            "network": network,
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        excluded_ids = [e["entity_id"] for e in result_excluded["entities"]]
        assert "Entity A" not in excluded_ids

        # With include_not_in_network=False: include network entities
        result_included = get_candidates.invoke({
            "specialties": ["hospital"],
            "include_not_in_network": False,
            "limit": 100,
            "candidates": mock_candidates,
            "network": network,
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        included_ids = [e["entity_id"] for e in result_included["entities"]]
        assert "Entity A" in included_ids

    def test_get_candidates_pagination_metadata(self, mock_candidates):
        """Test pagination metadata is correct."""
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "limit": 3,
            "offset": 1,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert result["pagination"]["total_matching"] == 6  # 6 hospital entities
        assert result["pagination"]["offset"] == 1
        assert result["pagination"]["limit"] == 3
        assert result["pagination"]["returned"] == 3

    def test_get_candidates_min_provider_count(self, mock_candidates):
        """Filter by minimum provider count."""
        result = get_candidates.invoke({
            "specialties": ["hospital"],
            "min_provider_count": 2,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert isinstance(result, dict)
        for e in result["entities"]:
            assert e["metrics"]["provider_count"] >= 2

    def test_get_candidates_multi_specialty_with_filters(self, mock_candidates):
        """Multi-specialty with min_effectiveness filter."""
        result = get_candidates.invoke({
            "specialties": ["hospital", "clinic"],
            "min_effectiveness": 3.0,
            "candidates": mock_candidates,
            "network": [],
            "entity_summaries": precompute_entity_summaries(mock_candidates),
        })
        assert isinstance(result, dict)
        assert "entities" in result
        for e in result["entities"]:
            assert e["metrics"]["avg_effectiveness"] >= 3.0


class TestRunCodeTool:
    """Tests for the run_code tool."""

    def test_run_code_basic_entity_filter(self):
        """Filter entity summaries by effectiveness."""
        entity_summaries = [
            {"entity": "A", "avg_effectiveness": 4.5, "provider_count": 10},
            {"entity": "B", "avg_effectiveness": 2.0, "provider_count": 5},
            {"entity": "C", "avg_effectiveness": 3.5, "provider_count": 8},
        ]
        result = run_code.invoke({
            "code": "result = entity_summaries_df[entity_summaries_df['avg_effectiveness'] > 3.0]",
            "entity_summaries": entity_summaries,
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 2
        entity_names = [e["entity"] for e in result]
        assert "A" in entity_names
        assert "C" in entity_names
        assert "B" not in entity_names

    def test_run_code_basic_candidate_filter(self):
        """Filter raw candidates by specialty."""
        candidates = [
            {"id": 1, "specialty": "hospital", "entity": "A"},
            {"id": 2, "specialty": "clinic", "entity": "B"},
            {"id": 3, "specialty": "hospital", "entity": "C"},
        ]
        result = run_code.invoke({
            "code": "result = candidates_df[candidates_df['specialty'] == 'hospital']",
            "candidates": candidates,
            "entity_summaries": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 2
        entities = [e["entity"] for e in result]
        assert "A" in entities
        assert "C" in entities

    def test_run_code_sorting(self):
        """Sort entity summaries."""
        entity_summaries = [
            {"entity": "A", "avg_effectiveness": 3.0},
            {"entity": "B", "avg_effectiveness": 5.0},
            {"entity": "C", "avg_effectiveness": 1.0},
        ]
        result = run_code.invoke({
            "code": "result = entity_summaries_df.sort_values('avg_effectiveness', ascending=False).head(2)",
            "entity_summaries": entity_summaries,
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["entity"] == "B"
        assert result[1]["entity"] == "A"

    def test_run_code_custom_aggregation(self):
        """Custom groupby aggregation on raw candidates."""
        candidates = [
            {"id": 1, "specialty": "hospital", "effectiveness": 4.0, "entity": "A"},
            {"id": 2, "specialty": "hospital", "effectiveness": 6.0, "entity": "A"},
            {"id": 3, "specialty": "clinic", "effectiveness": 3.0, "entity": "B"},
        ]
        result = run_code.invoke({
            "code": "result = candidates_df.groupby('entity').agg(avg_eff=('effectiveness', 'mean'), count=('id', 'count')).reset_index()",
            "candidates": candidates,
            "entity_summaries": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 2
        entity_map = {e["entity"]: e for e in result}
        assert entity_map["A"]["avg_eff"] == 5.0
        assert entity_map["A"]["count"] == 2
        assert entity_map["B"]["avg_eff"] == 3.0

    def test_run_code_error_handling(self):
        """Invalid code should return error message."""
        result = run_code.invoke({
            "code": "result = undefined_variable_name",
            "entity_summaries": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "Error" in result or "NameError" in result

    def test_run_code_blocked_operations(self):
        """Should block open(), eval(), __import__."""
        result = run_code.invoke({
            "code": "result = open('/etc/passwd').read()",
            "entity_summaries": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert "Error" in result

    def test_run_code_with_network_df(self):
        """Access network_df in sandbox."""
        network = [{"id": 1, "entity": "InNetwork"}]
        result = run_code.invoke({
            "code": "result = network_df[network_df['entity'] == 'InNetwork']",
            "network": network,
            "candidates": [],
            "entity_summaries": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["entity"] == "InNetwork"

    def test_run_code_with_members_df(self):
        """Access members_df in sandbox."""
        members = [{"id": 1, "county": "wayne", "lat": 42.0, "lon": -83.0}]
        result = run_code.invoke({
            "code": "result = members_df[members_df['county'] == 'wayne']",
            "members": members,
            "candidates": [],
            "entity_summaries": [],
            "network": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["county"] == "wayne"

    def test_run_code_empty_dataframes(self):
        """Should handle empty input data gracefully."""
        result = run_code.invoke({
            "code": "result = entity_summaries_df.head(0)",
            "entity_summaries": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert isinstance(result, list)
        assert len(result) == 0

    def test_run_code_json_import(self):
        """Should allow json module."""
        result = run_code.invoke({
            "code": "result = json.dumps({'key': 'value'})",
            "entity_summaries": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert result == '{"key": "value"}'

    def test_run_code_math_import(self):
        """Should allow math module."""
        result = run_code.invoke({
            "code": "result = math.sqrt(16)",
            "entity_summaries": [],
            "candidates": [],
            "network": [],
            "members": [],
            "county_specialty_thresholds": {},
        })
        assert result == 4.0

    def test_run_code_coverage_simulation(self):
        """Test that run_code can compute coverage using the helper."""
        members = [{"id": 1, "county": "wayne", "lat": 42.33, "lon": -83.04}]
        network = []
        candidates = [
            {"id": 1, "specialty": "hospital", "lat": 42.33, "lon": -83.04, "entity": "Entity A"},
        ]
        thresholds = {"mi": {"wayne": {"hospital": 1.0}}}
        
        # Simulate adding Entity A
        code = """
sim_net = pd.concat([network_df, candidates_df[candidates_df['entity'] == 'Entity A']])
result, errs = compute_coverage(sim_net, members_df, thresholds, candidates_df)
"""
        result = run_code.invoke({
            "code": code,
            "members": members,
            "network": network,
            "candidates": candidates,
            "entity_summaries": [],
            "county_specialty_thresholds": thresholds,
        })
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["coverage_percentage"] == 100.0


class TestRawCandidateSchemaProfile:
    """Tests for get_raw_candidate_schema_profile."""

    def test_returns_dict_with_columns(self):
        """Returns dict with column names as keys."""
        candidates = [
            {"id": 1, "specialty": "hospital", "effectiveness": 5.0, "entity": "A"},
            {"id": 2, "specialty": "clinic", "effectiveness": 3.0, "entity": "B"},
        ]
        profile = get_raw_candidate_schema_profile(candidates)
        assert isinstance(profile, dict)
        assert "specialty" in profile
        assert "effectiveness" in profile
        assert "entity" in profile

    def test_numeric_column_has_stats(self):
        """Numeric columns have min, max, mean, etc."""
        candidates = [
            {"id": 1, "effectiveness": 4.0, "entity": "A"},
            {"id": 2, "effectiveness": 6.0, "entity": "B"},
        ]
        profile = get_raw_candidate_schema_profile(candidates)
        assert "min" in profile["effectiveness"]
        assert "max" in profile["effectiveness"]
        assert "mean" in profile["effectiveness"]
        assert profile["effectiveness"]["min"] == 4.0
        assert profile["effectiveness"]["max"] == 6.0

    def test_string_column_has_unique_values(self):
        """String columns have unique_values and distribution."""
        candidates = [
            {"id": 1, "specialty": "hospital", "entity": "A"},
            {"id": 2, "specialty": "clinic", "entity": "B"},
        ]
        profile = get_raw_candidate_schema_profile(candidates)
        assert "unique_values" in profile["specialty"]
        assert "hospital" in profile["specialty"]["unique_values"]
        assert "clinic" in profile["specialty"]["unique_values"]

    def test_empty_candidates_returns_empty_dict(self):
        """Empty input returns empty dict."""
        profile = get_raw_candidate_schema_profile([])
        assert profile == {}
