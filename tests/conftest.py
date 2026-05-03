"""Shared test fixtures for the network management agent."""

import pytest
import pandas as pd

from network_manager_agent.data import DataManager


@pytest.fixture
def mock_candidates():
    """Sample candidate provider records."""
    return [
        {"entity": "Health System A", "specialty": "cardiology", "effectiveness": 4.5, "efficiency": 0.85, "lat": 42.33, "lon": -83.05, "city": "Ann Arbor", "state": "MI", "county": "washtenaw"},
        {"entity": "Health System A", "specialty": "general practice", "effectiveness": 4.2, "efficiency": 0.90, "lat": 42.34, "lon": -83.06, "city": "Ann Arbor", "state": "MI", "county": "washtenaw"},
        {"entity": "Health System A", "specialty": "cardiology", "effectiveness": 4.0, "efficiency": 0.80, "lat": 42.35, "lon": -83.07, "city": "Ypsilanti", "state": "MI", "county": "washtenaw"},
        {"entity": "MedCare B", "specialty": "general practice", "effectiveness": 3.8, "efficiency": 0.75, "lat": 42.32, "lon": -83.04, "city": "Ann Arbor", "state": "MI", "county": "washtenaw"},
        {"entity": "MedCare B", "specialty": "cardiology", "effectiveness": 3.5, "efficiency": 0.70, "lat": 42.36, "lon": -83.08, "city": "Ypsilanti", "state": "MI", "county": "washtenaw"},
        {"entity": "Regional Clinic C", "specialty": "general practice", "effectiveness": 4.8, "efficiency": 0.95, "lat": 42.37, "lon": -83.09, "city": "Saline", "state": "MI", "county": "washtenaw"},
    ]


@pytest.fixture
def mock_members():
    """Sample member records."""
    return [
        {"lat": 42.331, "lon": -83.046, "state": "MI", "county": "washtenaw"},
        {"lat": 42.340, "lon": -83.055, "state": "MI", "county": "washtenaw"},
        {"lat": 42.350, "lon": -83.065, "state": "MI", "county": "washtenaw"},
        {"lat": 42.360, "lon": -83.075, "state": "MI", "county": "washtenaw"},
        {"lat": 42.370, "lon": -83.085, "state": "MI", "county": "washtenaw"},
    ]


@pytest.fixture
def mock_thresholds():
    """Sample county-specialty thresholds."""
    return {
        "MI": {
            "washtenaw": {
                "cardiology": 10.0,
                "general practice": 20.0,
            }
        }
    }


@pytest.fixture
def mock_candidates_df(mock_candidates):
    """Pandas DataFrame of mock candidates."""
    return pd.DataFrame(mock_candidates)


@pytest.fixture
def mock_members_df(mock_members):
    """Pandas DataFrame of mock members."""
    return pd.DataFrame(mock_members)


@pytest.fixture
def mock_entity_summaries(mock_candidates_df):
    """Pre-aggregated entity summaries from mock candidates."""
    return DataManager.aggregate_entities(mock_candidates_df).reset_index().to_dict(orient="records")


@pytest.fixture
def mock_state(mock_entity_summaries, mock_thresholds):
    """AgentState with all required fields populated."""
    return {
        "messages": [],
        "network": [],
        "summary": "",
        "county_specialty_thresholds": mock_thresholds,
        "entity_summaries": mock_entity_summaries,
        "schema_profile": "",
    }


@pytest.fixture
def clean_data_manager():
    """Reset DataManager singleton before and after each test."""
    DataManager.reset()
    yield
    DataManager.reset()


@pytest.fixture
def seeded_data_manager(clean_data_manager, mock_candidates_df, mock_members_df, mock_entity_summaries, mock_thresholds):
    """Seed DataManager with mock data for testing."""
    dm = DataManager.__new__(DataManager)
    dm.candidates_df = mock_candidates_df
    dm.members_df = mock_members_df
    dm.entity_summaries_df = DataManager.aggregate_entities(mock_candidates_df)
    dm.entity_summaries = mock_entity_summaries
    dm.schema_profile = {}
    dm.raw_candidate_schema = {}
    dm.thresholds = mock_thresholds
    dm._initialized = True
    DataManager._instance = dm
    return dm
