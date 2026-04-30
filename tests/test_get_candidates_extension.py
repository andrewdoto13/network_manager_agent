import pytest
import pandas as pd
from network_manager_agent.tools import get_candidates, precompute_entity_summaries

@pytest.fixture
def mock_candidates():
    return [
        {"id": 1, "specialty": "hospital", "lat": 42.0, "lon": -83.0, "effectiveness": 5.0, "efficiency": 4.0, "Primary Contract Entity": "Entity A", "provider_count": 10},
        {"id": 2, "specialty": "clinic", "lat": 42.1, "lon": -83.1, "effectiveness": 2.0, "efficiency": 2.0, "Primary Contract Entity": "Entity B", "provider_count": 2},
        {"id": 3, "specialty": "hospital", "lat": 42.2, "lon": -83.2, "effectiveness": 1.0, "efficiency": 1.0, "Primary Contract Entity": "Entity C", "provider_count": 1},
        {"id": 4, "specialty": "clinic", "lat": 42.3, "lon": -83.3, "effectiveness": 4.0, "efficiency": 4.0, "Primary Contract Entity": "Entity D", "provider_count": 5},
        {"id": 5, "specialty": "hospital", "lat": 42.4, "lon": -83.4, "effectiveness": 3.0, "efficiency": 3.0, "Primary Contract Entity": "Entity E", "provider_count": 20},
    ]

@pytest.fixture
def mock_network():
    return []

def test_get_candidates_multi_specialty(mock_candidates, mock_network):
    entity_summaries = precompute_entity_summaries(mock_candidates)
    result = get_candidates.invoke({
        "specialties": ["hospital", "clinic"],
        "candidates": mock_candidates,
        "network": mock_network,
        "entity_summaries": entity_summaries,
    })
    assert isinstance(result, list)
    # Should find entities from both
    entity_ids = {e["entity_id"] for e in result}
    assert "Entity A" in entity_ids
    assert "Entity B" in entity_ids
    assert "Entity D" in entity_ids

def test_get_candidates_weighted_metrics(mock_candidates, mock_network):
    entity_summaries = precompute_entity_summaries(mock_candidates)
    result = get_candidates.invoke({
        "specialties": ["hospital"],
        "candidates": mock_candidates,
        "network": mock_network,
        "entity_summaries": entity_summaries,
        "weighted_metrics": {"avg_effectiveness": 1.0, "provider_count": 0.0},
        "ascending": False
    })
    
    assert result[0]["entity_id"] == "Entity A"
    assert result[1]["entity_id"] == "Entity E"
    assert result[2]["entity_id"] == "Entity C"

def test_get_candidates_weighted_metrics_mixed(mock_candidates, mock_network):
    entity_summaries = precompute_entity_summaries(mock_candidates)
    result = get_candidates.invoke({
        "specialties": ["hospital"],
        "candidates": mock_candidates,
        "network": mock_network,
        "entity_summaries": entity_summaries,
        "weighted_metrics": {"avg_effectiveness": 0.1, "provider_count": 1.0},
        "ascending": False
    })
    
    assert result[0]["entity_id"] == "Entity A"
    assert result[1]["entity_id"] == "Entity E"
    assert result[2]["entity_id"] == "Entity C"

def test_get_candidates_limit(mock_candidates, mock_network):
    entity_summaries = precompute_entity_summaries(mock_candidates)
    result = get_candidates.invoke({
        "specialties": ["hospital"],
        "candidates": mock_candidates,
        "network": mock_network,
        "entity_summaries": entity_summaries,
        "limit": 2
    })
    assert len(result) == 2

def test_get_candidates_no_specialties(mock_candidates, mock_network):
    entity_summaries = precompute_entity_summaries(mock_candidates)
    result = get_candidates.invoke({
        "candidates": mock_candidates,
        "network": mock_network,
        "entity_summaries": entity_summaries,
    })
    assert result == "No specialties provided."
