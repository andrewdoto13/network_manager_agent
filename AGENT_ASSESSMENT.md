# Agent Assessment

## Current State
- 37 tests passing
- Agent built on LangGraph (ReAct pattern)
- Tools: `get_candidate_schema`, `get_candidates`, `add_contract_entity`, `get_network_status`, `simulate_network_change`
- Coverage computed via BallTree with haversine distance
- Context summarization triggers after 14 messages, archiving first 7

## Strengths

### 1. Scope Injection (Network Scope)
The system prompt now dynamically injects the `county_specialty_thresholds` from state, telling the agent exactly which county-specialty combinations to evaluate. This replaced the agent guessing random specialties (e.g., "Clinical Social Work", "Outpatient Behavioral Health") with focused queries on the priority specialties (Cardiology, Internal Medicine, Psychiatry, Neurology).

### 2. Recommendation vs. Action Mode
Rule 11 in the system prompt correctly distinguishes between recommendation requests (evaluate + report, don't add) and action requests (explicit "add", "commit", "go ahead" triggers entity addition). The agent respects this boundary.

### 3. Simulation-Based Comparison
The agent uses `simulate_network_change` with `compare_scenarios` to evaluate multiple 3-entity combinations in a single call, ranking them by total coverage delta. This is efficient and avoids oscillation.

### 4. Coverage Calculation
The BallTree + haversine distance calculation is verified correct. Example: ABELARDO G CONTRERAS MD PC (3 neurology providers in Detroit/West Bloomfield/Keego Harbor) legitimately achieves 87.6% Wayne County coverage within 20 miles.

### 5. Enhanced Schema
`get_candidate_schema` now returns 7 columns instead of 4, including `avg_total_claims_amount`, `avg_medicare_total_claims_amount`, and `location_confidence_dist`. This gives the agent more data for future filtering/sorting decisions.

## Areas of Weakness

### 1. Ranking Metric Favors Heavy Hitters
The agent ranks scenarios by **total coverage delta** (sum of all specialty deltas). This means a combo with 87.6% + 25.45% + 8.15% = 121.2 total delta beats a balanced combo with lower per-specialty numbers. The agent gravitates toward massive multi-specialty entities (e.g., Lincare with 38 providers) rather than targeted, balanced picks.

**Fix:** Consider a weighted or balanced ranking metric that penalizes 0% coverage in any priority specialty, or allows the user to specify coverage balance requirements.

### 2. No Geographic Sanity Check
The agent doesn't verify whether a large national entity (e.g., Lincare, 38 providers across many specialties) is a realistic contract partner. The simulation accounts for distance, but the agent doesn't flag entities whose providers are spread across the country vs. concentrated in the target county.

**Fix:** Add a check that compares provider distribution to member location. Flag entities where most providers are far from the target county.

### 3. Entity Quality Signal Not Used
The schema exposes `location_confidence_dist` (e.g., "Very Low", "Low", "Medium", "High") but the agent doesn't use it. Many providers have "Very Low" confidence scores, which could indicate unreliable data. The agent picks entities without considering data quality.

**Fix:** Add a rule or tool that filters by confidence score, or include confidence in the ranking metric.

### 4. Summary Aggressiveness
Summarization triggers after 14 messages, archiving the first 7. The summary prompt says "summarizing the tools that you called" but doesn't explicitly say "preserve key numerical results." The LLM summarizer likely drops coverage percentages, delta values, and provider counts, which could be lost if the agent needs to reference them later in longer sessions.

**Fix:** Update the summarization prompt to explicitly say "preserve key numerical results from tool outputs (coverage percentages, deltas, provider counts)" or increase `MESSAGES_TO_ARCHIVE`.

### 5. No Diversity Across Specialties
The ranking metric doesn't penalize 0% coverage in one specialty if others are high. In the last test, Psychiatry stayed at 0% across the top-ranked combo. A user might want balanced coverage across all priority specialties.

**Fix:** Adjust the ranking to include a minimum coverage floor per specialty, or offer the user a "balanced coverage" option.

## Testing Notes

### Prompts Tested
1. `"I need to add one contract entity. Find one with high effectiveness."` → Agent picked a distant entity (Regional West Medical Center, South Bend, IN, 160 miles away). No coverage impact.
2. `"I need to add one contract entity near Wayne County, Michigan..."` → Agent correctly compared options and picked Stamford Health (Detroit area, 23.25% Internal Medicine coverage).
3. `"I need 3 contract entity recommendations..."` (with "add all 3") → Agent added 3 entities, 45.6% Internal Medicine coverage. Correctly auto-added because prompt said "add."
4. `"Give me 3 contract entity recommendations..."` (no "add" language) → Agent evaluated 5 combos, reported results, did NOT add. Recommended Lincare + Pulmonary Specialists + Abuelardo Contreras (87.6% Neurology, 25.45% Internal Medicine, 8.15% Cardiology, 0% Psychiatry).

### Data
- Members: 2,000 in Wayne County (all)
- Candidates: ~81,607 providers, 13,844 entities, 46 specialties
- Thresholds: Cardiology (10mi), Internal Medicine (5mi), Psychiatry (15mi), Neurology (20mi)
