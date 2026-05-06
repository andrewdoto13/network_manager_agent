"""Tests for data.py — column normalization, aggregation, profiling, filtering, DataManager."""

import json
import pandas as pd
import numpy as np
import pytest

from network_manager_agent.data import DataManager


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

class TestResolveColumn:
    def test_exact_match(self):
        assert DataManager._resolve_column(["lat", "lon", "entity"], "lat") == "lat"

    def test_synonym_match(self):
        assert DataManager._resolve_column(["latitude", "longitude"], "lat") == "latitude"

    def test_case_insensitive(self):
        assert DataManager._resolve_column(["Latitude", "LATITUDE"], "lat") == "Latitude"

    def test_not_found(self):
        assert DataManager._resolve_column(["foo", "bar"], "lat") is None

    def test_primary_contract_entity_synonym(self):
        assert DataManager._resolve_column(["primary contract entity", "specialty"], "entity") == "primary contract entity"

    def test_countyname_synonym(self):
        assert DataManager._resolve_column(["countyname", "state"], "county") == "countyname"


class TestNormalizeColumns:
    def test_rename_variants_to_canonical(self):
        df = pd.DataFrame({
            "latitude": [42.0],
            "longitude": [-83.0],
            "primary contract entity": ["A"],
            "countyname": ["washtenaw"],
        })
        result = DataManager._normalize_columns(df)
        assert set(result.columns) >= {"lat", "lon", "entity", "county"}

    def test_already_canonical_no_change(self):
        df = pd.DataFrame({"lat": [42.0], "lon": [-83.0], "entity": ["A"]})
        result = DataManager._normalize_columns(df)
        assert set(result.columns) >= {"lat", "lon", "entity"}

    def test_preserves_unknown_columns(self):
        df = pd.DataFrame({"lat": [42.0], "custom_col": ["x"]})
        result = DataManager._normalize_columns(df)
        assert "custom_col" in result.columns


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------

class TestNormalizeCoordinates:
    def test_detects_and_normalizes_scaled_integers(self):
        df = pd.DataFrame({"lat": [42331000], "lon": [83055000]})
        DataManager._normalize_coordinates(df)
        assert abs(df["lat"].iloc[0] - 42.331) < 0.01
        assert abs(df["lon"].iloc[0] - (-83.055)) < 0.01

    def test_skips_already_normalized(self):
        df = pd.DataFrame({"lat": [42.33], "lon": [-83.05]})
        DataManager._normalize_coordinates(df)
        assert df["lat"].iloc[0] == 42.33
        assert df["lon"].iloc[0] == -83.05

    def test_no_op_when_columns_missing(self):
        df = pd.DataFrame({"entity": ["A"]})
        DataManager._normalize_coordinates(df)

    def test_flips_positive_longitude(self):
        df = pd.DataFrame({"lat": [42331000], "lon": [83055000]})
        DataManager._normalize_coordinates(df)
        assert df["lon"].iloc[0] < 0


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

class TestNormalizeStrings:
    def test_strips_whitespace(self):
        df = pd.DataFrame({"entity": [" Corewell Health "], "city": [" Ann Arbor "]})
        DataManager._normalize_strings(df)
        assert df["entity"].iloc[0] == "corewell health"
        assert df["city"].iloc[0] == "ann arbor"

    def test_lowercases_all_string_columns(self):
        df = pd.DataFrame({"entity": ["Health System A"], "state": ["MI"], "specialty": ["Cardiology"]})
        DataManager._normalize_strings(df)
        assert df["entity"].iloc[0] == "health system a"
        assert df["state"].iloc[0] == "mi"
        assert df["specialty"].iloc[0] == "cardiology"

    def test_preserves_nan_values(self):
        import numpy as np
        df = pd.DataFrame({"entity": ["A", None, "B"]})
        DataManager._normalize_strings(df)
        assert df["entity"].iloc[0] == "a"
        assert pd.isna(df["entity"].iloc[1])
        assert df["entity"].iloc[2] == "b"

    def test_leaves_numeric_columns_untouched(self):
        df = pd.DataFrame({"entity": ["A"], "effectiveness": [4.5], "lat": [42.33]})
        DataManager._normalize_strings(df)
        assert df["effectiveness"].iloc[0] == 4.5
        assert df["lat"].iloc[0] == 42.33

    def test_no_op_on_empty_dataframe(self):
        df = pd.DataFrame()
        result = DataManager._normalize_strings(df)
        assert result.empty


# ---------------------------------------------------------------------------
# Entity aggregation
# ---------------------------------------------------------------------------

class TestAggregateEntities:
    def test_basic_aggregation(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        assert len(result) == 3
        assert "provider_count" in result.columns
        assert "avg_effectiveness" in result.columns
        assert "specialties" in result.columns

    def test_provider_counts(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        assert result.loc["health system a", "provider_count"] == 3
        assert result.loc["medcare b", "provider_count"] == 2
        assert result.loc["regional clinic c", "provider_count"] == 1

    def test_avg_effectiveness(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        assert abs(result.loc["health system a", "avg_effectiveness"] - 4.23) < 0.01

    def test_specialties_sorted_unique(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        specs = result.loc["health system a", "specialties"]
        assert specs == ["cardiology", "general practice"]

    def test_geographic_reach(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        assert result.loc["health system a", "geographic_reach"] == 2

    def test_empty_input_list(self):
        result = DataManager.aggregate_entities([])
        assert result.empty

    def test_empty_input_dataframe(self):
        result = DataManager.aggregate_entities(pd.DataFrame())
        assert result.empty

    def test_dataframe_input(self, mock_candidates_df):
        result = DataManager.aggregate_entities(mock_candidates_df)
        assert len(result) == 3

    def test_missing_optional_columns(self):
        df = pd.DataFrame({
            "entity": ["A", "A"],
            "specialty": ["cardiology", "general practice"],
        })
        result = DataManager.aggregate_entities(df)
        assert len(result) == 1
        assert "provider_count" in result.columns
        assert "avg_effectiveness" not in result.columns


# ---------------------------------------------------------------------------
# Schema profiling
# ---------------------------------------------------------------------------

class TestBuildSchemaProfile:
    def test_numeric_column_profile(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5], "entity": ["a", "b", "c", "d", "e"]})
        profile = DataManager.build_schema_profile(df)
        assert profile["value"]["min"] == 1.0
        assert profile["value"]["max"] == 5.0
        assert profile["value"]["mean"] == 3.0
        assert profile["value"]["median"] == 3.0

    def test_string_column_profile(self):
        df = pd.DataFrame({"city": ["ann arbor", "ann arbor", "ypsilanti"], "entity": ["a", "b", "c"]})
        profile = DataManager.build_schema_profile(df)
        assert profile["city"]["unique_count"] == 2
        assert "ann arbor" in profile["city"]["unique_values"]

    def test_list_column_profile(self):
        df = pd.DataFrame({
            "specialties": [["cardiology", "general practice"], ["cardiology"], ["general practice"]],
            "entity": ["a", "b", "c"],
        })
        profile = DataManager.build_schema_profile(df)
        assert profile["specialties"]["unique_count"] == 2
        assert "cardiology" in profile["specialties"]["unique_values"]

    def test_dict_column_profile(self):
        df = pd.DataFrame({
            "dist": [{"a": 3, "b": 2}, {"a": 1}],
            "entity": ["a", "b"],
        })
        profile = DataManager.build_schema_profile(df)
        assert "a" in profile["dist"]["unique_values"]
        assert "b" in profile["dist"]["unique_values"]

    def test_excludes_entity_column(self):
        df = pd.DataFrame({"entity": ["A", "B"], "value": [1, 2]})
        profile = DataManager.build_schema_profile(df)
        assert "entity" not in profile
        assert "value" in profile

    def test_excludes_entity_id_column(self):
        df = pd.DataFrame({"entity_id": ["A", "B"], "value": [1, 2]})
        profile = DataManager.build_schema_profile(df)
        assert "entity_id" not in profile

    def test_empty_dataframe(self):
        profile = DataManager.build_schema_profile(pd.DataFrame())
        assert profile == {}


# ---------------------------------------------------------------------------
# Service area filtering
# ---------------------------------------------------------------------------

class TestFilterByServiceArea:
    def test_filters_members_by_state_county(self, mock_candidates_df, mock_members_df, mock_thresholds):
        cdf, mdf = DataManager._filter_by_service_area(mock_candidates_df, mock_members_df, mock_thresholds)
        assert len(mdf) == len(mock_members_df)

    def test_filters_candidates_by_bounding_box(self, mock_candidates_df, mock_members_df, mock_thresholds):
        cdf, mdf = DataManager._filter_by_service_area(mock_candidates_df, mock_members_df, mock_thresholds)
        assert len(cdf) == len(mock_candidates_df)

    def test_returns_all_entities_with_at_least_one_provider_in_bounds(self, mock_members_df, mock_thresholds):
        cdf = pd.DataFrame([
            {"entity": "near entity", "specialty": "cardiology", "lat": 42.33, "lon": -83.05},
            {"entity": "split entity", "specialty": "cardiology", "lat": 42.34, "lon": -83.06},
            {"entity": "split entity", "specialty": "cardiology", "lat": 45.00, "lon": -90.00},
        ])
        filtered_cdf, _ = DataManager._filter_by_service_area(cdf, mock_members_df, mock_thresholds)
        entities = filtered_cdf["entity"].unique().tolist()
        assert "split entity" in entities
        assert len(filtered_cdf[filtered_cdf["entity"] == "split entity"]) == 2

    def test_empty_members_returns_empty_candidates(self, mock_candidates_df, mock_thresholds):
        mdf = pd.DataFrame(columns=["lat", "lon", "state", "county"])
        cdf, filtered_mdf = DataManager._filter_by_service_area(mock_candidates_df, mdf, mock_thresholds)
        assert cdf.empty


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------

class TestDataManager:
    def test_singleton_behavior(self, clean_data_manager):
        dm1 = DataManager()
        dm2 = DataManager()
        assert dm1 is dm2

    def test_reset_clears_singleton(self, clean_data_manager):
        DataManager()
        DataManager.reset()
        assert DataManager._instance is None

    def test_loads_real_data(self, clean_data_manager):
        dm = DataManager()
        assert not dm.get_candidates_df().empty
        assert not dm.get_members_df().empty
        assert len(dm.get_entity_summaries()) > 0

    def test_get_providers_by_entity(self, clean_data_manager):
        dm = DataManager()
        entities = dm.get_candidates_df()["entity"].dropna().unique()
        if len(entities) > 0:
            providers = dm.get_providers_by_entity(entities[0])
            assert not providers.empty
            assert (providers["entity"] == entities[0]).all()

    def test_reinit_noop(self, clean_data_manager):
        dm = DataManager()
        original_len = len(dm.get_candidates_df())
        DataManager()
        assert len(dm.get_candidates_df()) == original_len

    def test_schema_profile_structure(self, clean_data_manager):
        dm = DataManager()
        profile = dm.get_schema_profile()
        assert isinstance(profile, dict)
        if profile:
            for col, info in profile.items():
                assert "type" in info
