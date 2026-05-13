"""Tests for tools.py — coverage computation, tools, and helpers."""

import pandas as pd
import pytest

from network_manager_agent.tools import (
    compute_coverage,
    add_contract_entity,
    run_code,
    _deg2rad,
    _miles_to_radians,
    TOOLS,
)


# ---------------------------------------------------------------------------
# compute_coverage (unified)
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    def test_with_dataframe(self, seeded_data_manager, mock_candidates_df, mock_members_df, mock_thresholds):
        coverage, errors = compute_coverage(
            mock_candidates_df, mock_members_df, mock_thresholds, mock_candidates_df
        )
        assert isinstance(coverage, list)
        assert len(coverage) == 2

    def test_with_entity_ids(self, seeded_data_manager, mock_members_df, mock_thresholds):
        coverage, errors = compute_coverage(
            ["health system a"], mock_members_df, mock_thresholds
        )
        assert isinstance(coverage, list)
        for entry in coverage:
            assert "coverage_percentage" in entry

    def test_with_provider_dicts(self, seeded_data_manager, mock_candidates_df, mock_members_df, mock_thresholds):
        providers = mock_candidates_df.to_dict(orient="records")
        coverage, errors = compute_coverage(
            providers, mock_members_df, mock_thresholds, mock_candidates_df
        )
        assert isinstance(coverage, list)

    def test_empty_list(self, seeded_data_manager, mock_members_df, mock_thresholds):
        coverage, errors = compute_coverage([], mock_members_df, mock_thresholds)
        assert len(coverage) == 2

    def test_none_input(self, seeded_data_manager, mock_members_df, mock_thresholds):
        coverage, errors = compute_coverage(None, mock_members_df, mock_thresholds)
        assert len(coverage) == 2

    def test_empty_members_returns_empty(self, seeded_data_manager, mock_thresholds):
        coverage, errors = compute_coverage(
            pd.DataFrame(), pd.DataFrame(), mock_thresholds
        )
        assert coverage == []
        assert errors == []

    def test_coverage_filters_by_county(self, seeded_data_manager):
        """Members in other counties must be excluded from coverage calculation."""
        members = pd.DataFrame([
            {"lat": 42.00, "lon": -83.00, "state": "mi", "county": "washtenaw"},
            {"lat": 42.01, "lon": -83.00, "state": "mi", "county": "washtenaw"},
            {"lat": 43.00, "lon": -83.00, "state": "mi", "county": "oakland"},
        ])
        network = pd.DataFrame([
            {"entity": "clinic", "specialty": "cardiology", "lat": 42.00, "lon": -83.00},
        ])
        thresholds = {"mi": {"washtenaw": {"cardiology": 20.0}}}
        coverage, errors = compute_coverage(network, members, thresholds, network)
        assert len(errors) == 0
        card = [c for c in coverage if c["specialty"] == "cardiology"][0]
        assert card["total_members"] == 2, "Should only count washtenaw members"
        assert card["members_with_access"] == 2
        assert card["coverage_percentage"] == 100.0

    def test_coverage_exact_haversine_distance(self, seeded_data_manager):
        """Hand-computed: at ~42deg lat, 0.0725 deg lat ~ 5mi, 0.2174 deg lat ~ 15mi."""
        members = pd.DataFrame([
            {"lat": 42.0000, "lon": -83.00, "state": "mi", "county": "washtenaw"},
            {"lat": 42.0725, "lon": -83.00, "state": "mi", "county": "washtenaw"},
            {"lat": 42.2174, "lon": -83.00, "state": "mi", "county": "washtenaw"},
            {"lat": 43.0000, "lon": -83.00, "state": "mi", "county": "oakland"},
        ])
        network = pd.DataFrame([
            {"entity": "clinic", "specialty": "cardiology", "lat": 42.0000, "lon": -83.00},
        ])
        thresholds = {"mi": {"washtenaw": {"cardiology": 10.0}}}
        coverage, errors = compute_coverage(network, members, thresholds, network)
        assert len(errors) == 0
        card = [c for c in coverage if c["specialty"] == "cardiology"][0]
        assert card["total_members"] == 3, "Only washtenaw members; oakland excluded"
        assert card["members_with_access"] == 2, "0mi and ~5mi covered; ~15mi not"
        assert card["coverage_percentage"] == 66.67


# ---------------------------------------------------------------------------
# add_contract_entity
# ---------------------------------------------------------------------------

class TestAddContractEntity:
    def test_add_valid_entity(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["health system a"],
            "network": [],
        })
        assert "health system a" in result["added_entities"]

    def test_skip_duplicate(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["health system a"],
            "network": ["health system a"],
        })
        assert result["added_entities"] == []

    def test_invalid_entity_error(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["nonexistent entity"],
            "network": [],
        })
        assert any("nonexistent entity" in e for e in result["errors"])

    def test_mixed_valid_invalid(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["health system a", "nonexistent"],
            "network": [],
        })
        assert "health system a" in result["added_entities"]
        assert len(result["errors"]) == 1

    def test_multiple_entities(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["health system a", "medcare b", "regional clinic c"],
            "network": [],
        })
        assert len(result["added_entities"]) == 3

    def test_skip_duplicate_case_insensitive(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A"],
            "network": ["health system a"],
        })
        assert result["added_entities"] == []
        assert "Health System A" in result["skipped_entities"]

    def test_stores_lowercase_canonical_name(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A"],
            "network": [],
        })
        assert result["added_entities"] == ["health system a"]

    def test_skipped_and_added_together(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A", "Medcare B"],
            "network": ["health system a"],
        })
        assert result["added_entities"] == ["medcare b"]
        assert "Health System A" in result["skipped_entities"]


# ---------------------------------------------------------------------------
# run_code
# ---------------------------------------------------------------------------

class TestRunCode:
    def test_simple_expression(self, seeded_data_manager, mock_thresholds):
        import network_manager_agent.tools as tools_mod
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "sandbox_cache['answer'] = 2 + 2",
        })
        assert tools_mod._last_sandbox_cache["answer"] == 4

    def test_pandas_query(self, seeded_data_manager, mock_thresholds):
        import network_manager_agent.tools as tools_mod
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "sandbox_cache['count'] = candidates_df.shape[0]",
        })
        assert tools_mod._last_sandbox_cache["count"] == 6

    def test_dataframe_result(self, seeded_data_manager, mock_thresholds):
        import network_manager_agent.tools as tools_mod
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "sandbox_cache['data'] = candidates_df[['entity', 'specialty']].head(2).to_dict('records')",
        })
        data = tools_mod._last_sandbox_cache["data"]
        assert isinstance(data, list)
        assert len(data) == 2

    def test_error_handling(self, seeded_data_manager, mock_thresholds):
        output = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "sandbox_cache['err'] = undefined_variable + 1",
        })
        assert isinstance(output, str)
        assert "Error:" in output

    def test_compute_coverage_in_sandbox(self, seeded_data_manager, mock_thresholds):
        import network_manager_agent.tools as tools_mod
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "cov, errs = compute_coverage(network_df, members_df, thresholds, candidates_df)\nsandbox_cache['count'] = cov.__len__()",
        })
        assert tools_mod._last_sandbox_cache["count"] == 2

    def test_import_returns_error(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "import pandas as pd\nresult = 1",
        })
        assert isinstance(result, str)
        assert "Error: Import statements are disabled" in result

    def test_from_import_returns_error(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "from itertools import combinations\nresult = 1",
        })
        assert isinstance(result, str)
        assert "Error: Import statements are disabled" in result

    def test_indented_import_returns_error(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "if True:\n    import json\n    result = 1",
        })
        assert isinstance(result, str)
        assert "Error: Import statements are disabled" in result

    def test_multiple_imports_return_error(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "import pandas as pd\nimport numpy as np\nresult = candidates_df.shape[0]",
        })
        assert isinstance(result, str)
        assert "Error: Import statements are disabled" in result

    def test_sandbox_cache_injected(self, seeded_data_manager, mock_thresholds):
        """sandbox_cache should be available in the sandbox."""
        import network_manager_agent.tools as tools_mod
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {"mykey": "myvalue"},
            "code": "sandbox_cache['retrieved'] = sandbox_cache.get('mykey')",
        })
        assert tools_mod._last_sandbox_cache["retrieved"] == "myvalue"

    def test_cache_appended_to_output(self, seeded_data_manager, mock_thresholds):
        """Cache is persisted via _last_sandbox_cache, not embedded in tool output."""
        import network_manager_agent.tools as tools_mod
        # Call func directly to bypass InjectedState filtering in @tool decorator
        run_code.func(
            network=[],
            county_specialty_thresholds=mock_thresholds,
            sandbox_cache={"rankings": [1, 2, 3]},
            code="sandbox_cache['status'] = 'done'",
        )
        assert tools_mod._last_sandbox_cache["rankings"] == [1, 2, 3]
        assert tools_mod._last_sandbox_cache["status"] == "done"

    def test_cache_modified_in_sandbox_persists(self, seeded_data_manager, mock_thresholds):
        """Agent can modify sandbox_cache and changes persist via _last_sandbox_cache."""
        import network_manager_agent.tools as tools_mod
        # Call func directly to bypass InjectedState filtering in @tool decorator
        run_code.func(
            network=[],
            county_specialty_thresholds=mock_thresholds,
            sandbox_cache={},
            code="sandbox_cache['computed'] = 99",
        )
        assert tools_mod._last_sandbox_cache["computed"] == 99

 


# ---------------------------------------------------------------------------
# Geo utilities
# ---------------------------------------------------------------------------

class TestGeoUtilities:
    def test_deg2rad(self):
        df = pd.DataFrame({"lat": [90.0], "lon": [180.0]})
        rad = _deg2rad(df)
        import math
        assert abs(rad[0][0] - math.pi / 2) < 0.001
        assert abs(rad[0][1] - math.pi) < 0.001

    def test_miles_to_radians(self):
        rad = _miles_to_radians(3958.8)
        assert abs(rad - 1.0) < 0.001


# ---------------------------------------------------------------------------
# TOOLS list
# ---------------------------------------------------------------------------

class TestToolsList:
    def test_tools_registered(self):
        assert add_contract_entity in TOOLS
        assert run_code in TOOLS

    def test_tools_count(self):
        assert len(TOOLS) == 2

    def test_run_code_docstring_emphasizes_compute_coverage(self):
        """run_code docstring must identify compute_coverage as definitive."""
        doc = run_code.description
        assert "compute_coverage" in doc
        assert "definitive" in doc or "authoritative" in doc
