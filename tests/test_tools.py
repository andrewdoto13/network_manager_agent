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
            ["Health System A"], mock_members_df, mock_thresholds
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


# ---------------------------------------------------------------------------
# add_contract_entity
# ---------------------------------------------------------------------------

class TestAddContractEntity:
    def test_add_valid_entity(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A"],
            "network": [],
        })
        assert "Health System A" in result["added_entities"]

    def test_skip_duplicate(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A"],
            "network": ["Health System A"],
        })
        assert result["added_entities"] == []

    def test_invalid_entity_error(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Nonexistent Entity"],
            "network": [],
        })
        assert any("Nonexistent Entity" in e for e in result["errors"])

    def test_mixed_valid_invalid(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A", "Nonexistent"],
            "network": [],
        })
        assert "Health System A" in result["added_entities"]
        assert len(result["errors"]) == 1

    def test_multiple_entities(self, seeded_data_manager):
        result = add_contract_entity.invoke({
            "entity_ids": ["Health System A", "MedCare B", "Regional Clinic C"],
            "network": [],
        })
        assert len(result["added_entities"]) == 3


# ---------------------------------------------------------------------------
# run_code
# ---------------------------------------------------------------------------

class TestRunCode:
    def test_simple_expression(self, seeded_data_manager, mock_entity_summaries, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "entity_summaries": mock_entity_summaries,
            "county_specialty_thresholds": mock_thresholds,
            "code": "result = 2 + 2",
        })
        assert result == 4

    def test_pandas_query(self, seeded_data_manager, mock_entity_summaries, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "entity_summaries": mock_entity_summaries,
            "county_specialty_thresholds": mock_thresholds,
            "code": "result = candidates_df.shape[0]",
        })
        assert result == 6

    def test_dataframe_result(self, seeded_data_manager, mock_entity_summaries, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "entity_summaries": mock_entity_summaries,
            "county_specialty_thresholds": mock_thresholds,
            "code": "result = candidates_df[['entity', 'specialty']].head(2)",
        })
        assert isinstance(result, list)
        assert len(result) == 2

    def test_error_handling(self, seeded_data_manager, mock_entity_summaries, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "entity_summaries": mock_entity_summaries,
            "county_specialty_thresholds": mock_thresholds,
            "code": "result = undefined_variable + 1",
        })
        assert isinstance(result, str)
        assert "Error:" in result

    def test_compute_coverage_in_sandbox(self, seeded_data_manager, mock_entity_summaries, mock_thresholds):
        result = run_code.invoke({
            "network": [],
            "entity_summaries": mock_entity_summaries,
            "county_specialty_thresholds": mock_thresholds,
            "code": "cov, errs = compute_coverage(network_df, members_df, thresholds, candidates_df)\nresult = cov.__len__()",
        })
        assert result == 2


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
