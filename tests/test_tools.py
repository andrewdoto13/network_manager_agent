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
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = 2 + 2",
        })
        assert "---CACHE---" in result
        assert "4" in result

    def test_pandas_query(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = candidates_df.shape[0]",
        })
        assert "---CACHE---" in result
        assert "6" in result

    def test_dataframe_result(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = candidates_df[['entity', 'specialty']].head(2)",
        })
        assert "---CACHE---" in result

    def test_error_handling(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = undefined_variable + 1",
        })
        assert isinstance(result, str)
        assert "Error:" in result

    def test_compute_coverage_in_sandbox(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "cov, errs = compute_coverage(network_df, members_df, thresholds, candidates_df)\nresult = cov.__len__()",
        })
        assert "---CACHE---" in result
        assert "2" in result

    def test_import_stripped_with_warning(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "import pandas as pd\nresult = 1",
        })
        assert isinstance(result, str)
        assert "[sandbox] Stripped 1 import(s)" in result
        assert "---CACHE---" in result

    def test_from_import_stripped_with_warning(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "from itertools import combinations\nresult = 1",
        })
        assert isinstance(result, str)
        assert "[sandbox] Stripped 1 import(s)" in result
        assert "---CACHE---" in result

    def test_indented_import_stripped(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "if True:\n    import json\n    result = 1",
        })
        assert isinstance(result, str)
        assert "[sandbox] Stripped 1 import(s)" in result
        assert "---CACHE---" in result

    def test_multiple_imports_stripped(self, seeded_data_manager, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "import pandas as pd\nimport numpy as np\nresult = candidates_df.shape[0]",
        })
        assert isinstance(result, str)
        assert "[sandbox] Stripped 2 import(s)" in result
        assert "---CACHE---" in result

    def test_prev_result_none_on_first_call(self, seeded_data_manager, mock_thresholds):
        """prev_result should be None on first call."""
        import network_manager_agent.tools as tools_mod
        tools_mod._prev_result = None

        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = prev_result",
        })
        # Output includes cache suffix; prev_result=None means the main output is "None"
        assert "---CACHE---" in result
        assert result.split("---CACHE---")[0].strip() == "None"

    def test_prev_result_holds_last_result(self, seeded_data_manager, mock_thresholds):
        """prev_result should hold the result from the previous call."""
        import network_manager_agent.tools as tools_mod
        tools_mod._prev_result = None

        # First call sets result
        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = 42",
        })

        # Second call reads prev_result
        second = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = prev_result",
        })
        # Result includes cache suffix
        assert isinstance(second, str)
        assert "42" in second

    def test_sandbox_cache_injected(self, seeded_data_manager, mock_thresholds):
        """sandbox_cache should be available in the sandbox."""
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {"mykey": "myvalue"},
            "code": "result = sandbox_cache.get('mykey')",
        })
        assert isinstance(result, str)
        assert "myvalue" in result

    def test_cache_appended_to_output(self, seeded_data_manager, mock_thresholds):
        """---CACHE--- should appear in output with sandbox_cache JSON."""
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {"rankings": [1, 2, 3]},
            "code": "result = 'done'",
        })
        assert isinstance(result, str)
        assert "---CACHE---" in result
        import json as pyjson
        cache_part = result.split("---CACHE---")[1].strip()
        parsed = pyjson.loads(cache_part)
        assert parsed["rankings"] == [1, 2, 3]

    def test_cache_modified_in_sandbox_persists(self, seeded_data_manager, mock_thresholds):
        """Agent can modify sandbox_cache and changes appear in output."""
        result = run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "sandbox_cache['computed'] = 99\nresult = 'ok'",
        })
        assert isinstance(result, str)
        cache_part = result.split("---CACHE---")[1].strip()
        import json as pyjson
        parsed = pyjson.loads(cache_part)
        assert parsed["computed"] == 99

    def test_prev_result_dataframe_serialized(self, seeded_data_manager, mock_thresholds):
        """DataFrame result should be serialized to list[dict] for prev_result."""
        import network_manager_agent.tools as tools_mod
        tools_mod._prev_result = None

        run_code.invoke({
            "network": [],
            "county_specialty_thresholds": mock_thresholds,
            "sandbox_cache": {},
            "code": "result = candidates_df[['entity', 'specialty']].head(2)",
        })
        assert isinstance(tools_mod._prev_result, list)
        assert len(tools_mod._prev_result) == 2


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
