# Plan: Enhance Candidate Data Utilization in `get_candidates` Tool

## Objective
Improve the agent's decision-making capabilities by exposing more rich data from the candidate dataset (mi_market_data.csv) that is currently ignored or discarded during aggregation.

## 1. Expand `_aggregate_entities` Aggregation Logic
Update the aggregation process to capture the following missing insights:
- **Affiliations**: Aggregate `Primary Institutional Affiliation` into a unique list of health systems associated with the entity.
- **Patient Access**: Calculate the `new_patient_rate` (percentage of providers within the entity who accept new patients) using the `Medicare New Patient Claims` column.
- **Scale & Volume**: 
    - Include `avg_total_claims_amount` and `avg_medicare_total_claims_amount` (already computed but currently discarded in `get_candidates`).
    - Create a `claims_volume_dist` (count of providers in "Core", "Standard", "Peripheral", "Ghost" categories) using the `Total Claims Volume` column.
- **Geographic Footprint**: Count unique cities or zip codes where the entity has providers to determine its regional reach.
- **Reliability**: Include the `location_confidence_dist` (already computed but currently discarded in `get_candidates`).

## 2. Update `get_candidates` Tool Output
Modify the return object so the LLM receives these metrics:
- **`metrics` dict**: Add `avg_total_claims`, `avg_medicare_claims`, `new_patient_rate`, `geographic_reach`, and `location_confidence`.
- **`capabilities` dict**: Add `affiliations`.
- **`summary` string**: Enhance the auto-generated summary to highlight scale, affiliation, or accessibility (e.g., "Entity X is affiliated with Corewell Health and has a high new-patient acceptance rate").

## 3. Update LLM Guidance (Docstrings)
Update tool docstrings to inform the LLM about the new sorting and weighting capabilities:
- Update `sort_by` examples to include `avg_total_claims_amount` or `new_patient_rate`.
- Update `weighted_metrics` examples to show how to balance quality (effectiveness) with scale (claims volume) and accessibility.

## 4. Verification & Testing
- **Unit Tests**: Update `tests/test_agent.py` to verify that the new fields are present and correctly calculated.
- **Integration Test**: Execute prompts requiring the agent to prioritize specific attributes, such as:
    - "Find the most effective cardiologist entities that are also accepting new patients."
    - "Find entities with high claims volume and strong affiliations with major health systems."
