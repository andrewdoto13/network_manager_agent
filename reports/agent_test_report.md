# Network Manager Agent — Test Run Report

## Overview

This report documents a test run of the Network Manager Agent, an AI-powered provider network optimization tool built on LangGraph. The agent uses a ReAct (Reasoning + Acting) loop to analyze candidate provider data, simulate network changes via sandboxed code execution, and recommend entities for contracting.

## LLM Configuration

| Property | Value |
|----------|-------|
| Model | Qwen 3.6 27B |
| Quantization | Q4_K_XL (4-bit) |
| Context window | 262,144 tokens |

The Qwen 3.6 27B model was run locally with Q4_K_XL quantization, a 4-bit quantization scheme that balances memory efficiency and quality. The 262K context window provided ample headroom for the agent's multi-step reasoning loop, which includes injected schema profiles, DataFrames, code execution output, and conversation history. The context threshold for automatic summarization is set at 14 messages; when exceeded, older messages are compressed into a running summary to manage context usage.

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 7 9800X3D |
| GPU | NVIDIA RTX 5090 |
| RAM | 64GB DDR5-6000 CL30 |

The model was run locally on this workstation. The ~6 minute total execution time reflects inference on the RTX 5090 combined with the pandas/BallTree code execution running on the Ryzen 9800X3D. The agent took 36 steps — double the previous run — due to more thorough batch simulation of multiple candidates before committing to a decision.

## Prompt & Scope

**Prompt given to the agent:**

> "Find an entity that balances effectiveness, new patient acceptance rate, and network coverage, and add it. Then find the next best one to recommend to me, and report back how it would help the network."

**Scope:**
- **Service area:** Washtenaw County and Wayne County, Michigan
- **Specialties tracked:** Cardiology, General Practice, Orthopedic Surgery
- **Coverage thresholds:** Cardiology 10 mi, General Practice 20 mi, Orthopedic Surgery 15 mi
- **Data:** 124,263 candidate providers filtered to service area; 12,051 scoped members (1,794 Washtenaw + 10,257 Wayne)
- **Candidates in service area:** ~13,650 unique entities
- **Max steps allowed:** 50 (agent completed in 36 steps)

## Data Quality Fixes (Between Run 1 and Run 2)

Two critical data quality issues were discovered and fixed between the first and second test run. Both bugs silently degraded agent performance by eliminating viable candidates and inflating dataset sizes.

### Corewell Trailing Space Bug

**Root cause:** The entity name `"Corewell Health "` had a trailing space in the raw data (213 providers affected). When `add_contract_entity` or `compute_coverage` matched against `"Corewell Health"` (no space), the entity was never found — resulting in 0% coverage across all categories.

**Impact:** Corewell Health is Michigan's largest provider system (8,493 providers, 953 cardiologists, 284 orthopedic surgeons). With the bug, the agent had no way to discover that Corewell could provide 75-100% coverage in the target counties. The agent's batch simulation confirmed 0% coverage, leading it to correctly but _prematurely_ eliminate Corewell from consideration.

**Fix:** Added `_normalize_strings()` method to `DataManager` that strips whitespace and lowercases all object-type columns. `"Corewell Health "` → `"corewell health"`, `"Corewell Health"` → `"corewell health"`. All 213 whitespace-affected entities now match correctly.

**Before vs After:**

| County/Specialty | Run 1 (bug) | Run 2 (fixed) |
|-----------------|-------------|---------------|
| Washtenaw Cardiology | 0.0% | 75.7% |
| Washtenaw GP | 0.0% | 62.37% |
| Wayne Cardiology | 0.0% | 98.38% |
| Wayne GP | 0.0% | 100.0% |
| Wayne Orthopedic | 0.0% | 100.0% |

### Service Area Member Filter Bug

**Root cause:** The `_filter_by_service_area()` method at `data.py:267` filtered members by matching state OR county — keeping all members in any state that appeared in the thresholds config. With Michigan thresholds, all 65,000 MI members were retained instead of just those in Washtenaw/Wayne counties.

**Impact:** The agent's coverage calculations were diluted by 53,000 irrelevant members spread across Michigan. While the percentage results were not mathematically wrong (the BallTree queries still used per-county member subsets), the inflated dataset increased computation time and could confuse the agent's analysis.

**Fix:** Changed the member filter to require **both** state AND county match. Members now correctly scoped to 12,051 (1,794 Washtenaw + 10,257 Wayne).

### String Normalization

All string columns in both candidate and member data are now stripped and lowercased. This prevents future mismatches from:
- Leading/trailing whitespace (213 entities affected)
- Mixed case (e.g., `"MyMichigan Health"` vs `"mymichigan health"`)
- Inconsistent specialty naming

## Execution Summary

### Run 1 (Pre-Fix) — 2026-05-03

| Metric | Value |
|--------|-------|
| LLM | Qwen 3.6 27B (Q4_K_XL, 262K context) |
| Start time | 2026-05-03 21:52:03 EDT |
| End time | ~2026-05-03 21:58:00 EDT |
| Duration | ~6 min |
| Agent steps | 36 (20 code executions + 1 state update + 3 summarizations) |
| Tool calls | 20 `run_code` invocations + 1 `add_contract_entity` |
| Entities added to network | 1 (Henry Ford Health) |
| Entities recommended | 1 (MyMichigan Health) |

### Run 2 (Post-Fix) — 2026-05-04

| Metric | Value |
|--------|-------|
| LLM | Qwen 3.6 27B (Q4_K_XL, 262K context) |
| Start time | 2026-05-04 08:49:16 EDT |
| Duration | ~3 min |
| Agent steps | 26 (14 code executions + 1 state update + 2 summarizations) |
| Tool calls | 14 `run_code` invocations + 1 `add_contract_entity` |
| Entities added to network | 1 (MyMichigan Health) |
| Entities recommended | 1 (Corewell Health) |

### Side-by-Side Comparison

| Metric | Run 1 (pre-fix) | Run 2 (post-fix) |
|--------|----------------|-----------------|
| Agent steps | 36 | 26 (-28%) |
| Code executions | 20 | 14 (-30%) |
| Summarizations | 3 | 2 |
| First entity added | Henry Ford Health | MyMichigan Health |
| Second entity recommended | MyMichigan Health | Corewell Health |
| Corewell batch sim result | 0% (bug) | 75-100% |
| Geographic analysis | Yes (bounding boxes + Euclidean distance) | Skipped |
| Error encountered | KeyError, NameError | TypeError (new_patient_claims string) |
| Final 2-entity coverage | 97-100% | 97-100% |

## How the Agent Works

The agent operates as a ReAct loop with exactly two tools: `run_code` (sandboxed Python execution) and `add_contract_entity` (commits entities to the network). Each step follows this pattern:

1. **Reason:** The LLM reviews the current state (network, coverage gaps, candidate data) and decides what to investigate.
2. **Act:** The LLM writes Python code that runs in a sandboxed environment with access to `pandas`, `numpy`, `json`, and a `compute_coverage` helper (uses BallTree for haversine distance queries).
3. **Observe:** The code output is fed back to the LLM as context for the next step.
4. **Repeat** until the agent has enough information to produce a final answer.

The agent's working memory is managed via LangGraph's `SqliteSaver` checkpoint system. When the message count exceeds a threshold (14), older messages are archived into a running summary to preserve context while managing token limits.

### How `run_code` and `compute_coverage` Work Together

The `run_code` tool is the agent's primary analytical instrument. It executes arbitrary Python code in a sandboxed environment with a 20-second timeout. The sandbox is pre-populated with these variables:

- **`candidates_df`** — Raw provider-level candidate data (DataFrame)
- **`entity_summaries_df`** — Pre-aggregated entity summaries (DataFrame)
- **`network_df`** — Currently contracted providers (DataFrame)
- **`members_df`** — Member locations (DataFrame)
- **`thresholds`** — Service area configuration (dict)
- **`compute_coverage`** — Predefined coverage computation function

Critically, `compute_coverage` is **not a tool** — it is a plain Python function injected directly into the `run_code` sandbox by the framework. The agent cannot call it as a standalone tool; it can only invoke it from within code it writes. This means every coverage query is embedded inside a `run_code` call that the agent composes.

The `compute_coverage` function itself:
- Iterates over each state/county/specialty combination in the thresholds configuration
- Filters members to the target county
- Filters the network to providers of the target specialty
- Builds a BallTree index over provider coordinates using haversine distance
- For each member, queries whether any qualifying provider falls within the threshold radius
- Returns per-county/specialty coverage: `members_with_access`, `total_members`, `coverage_percentage`

This design means the agent's entire analytical capability flows through one channel: writing Python code. The agent discovers data, filters candidates, simulates network changes, and measures coverage impact — all by composing `run_code` calls that leverage the injected data frames and the `compute_coverage` helper.

## Step-by-Step Breakdown

### Phase 1: Baseline Assessment & Initial Filtering (Steps 1–4)

The agent's first action was to establish a baseline by running `compute_coverage()` on the empty network:

```python
base_cov, _ = compute_coverage(network_df, members_df, thresholds, candidates_df)
```

**Result:** 0.0% coverage across all 5 county/specialty combinations — no contracted providers yet.

The agent then filtered entities by required specialties (cardiology, general practice, orthopedic surgery) and ranked them by effectiveness, new patient rate, and provider count. The top-ranked entities were small specialists with perfect scores: SOUTHFIELD CITY URGENT CARE PC (37 providers, GP-focused), MID MICHIGAN DIAGNOSTIC CORP (4 providers, cardiology), and others.

### Phase 2: Geographic Intelligence Gathering & Strategy Pivot (Steps 5–10)

Rather than immediately simulating the top-ranked small entities, the agent took a broader approach. It inspected provider locations for the top candidates, then queried MI provider counts by entity for each required specialty:

```python
# Find cardiology providers in MI
cardio_mi = candidates_df[(candidates_df["specialty"].str.lower() == "cardiology") &
                          (candidates_df["state"].str.lower() == "mi")]
print(cardio_mi.groupby("entity").size().sort_values(ascending=False).head(15))
# Repeat for orthopedic surgery and general practice
```

**The output triggered a strategic pivot.** The agent saw:

```
Cardiology providers in MI: 3581
entity
Corewell Health                       953
MyMichigan Health                     621
Henry Ford Health                     363
Trinity Health                        172
Mclaren Health                        132

Orthopedic surgery providers in MI: 2228
entity
Corewell Health                       284
Henry Ford Health                     207
MyMichigan Health                     204
```

The top-ranked entities from Phase 1 had at most **37 providers** (SOUTHFIELD CITY URGENT CARE PC). These large systems had **hundreds to thousands** per specialty — a 10x to 300x difference. The agent recognized that covering 12,051 members across 2 counties with a 37-provider entity would leave most gaps unfilled, while a 5,000+ provider system could potentially cover everything in one contract.

**This was not prompted by the system instructions.** The agent self-corrected its strategy based purely on the data: the initial ranking by effectiveness and new patient rate was misleading because it favored small entities with few data points (perfect scores are easy with 1–4 providers). The volume data told a different story.

The agent then examined 5 major health systems — Corewell Health, Henry Ford Health, MyMichigan Health, Trinity Health, and Mclaren Health — checking their MI provider counts, specialty coverage, and geographic proximity to both counties. The geographic analysis that followed was particularly sophisticated, as described in the next section.

### Phase 2.5: Geographic Analysis — Two-Phase Spatial Reasoning (Steps 5–10)

This section details how the agent performed geographic analysis to identify which health systems would cover the target counties — an impressive chain of reasoning that spanned 6 steps and used two complementary techniques: a **fast Euclidean distance proxy** for initial filtering, followed by **precise BallTree haversine simulation** for final validation.

#### Member Distribution Mapping (Step 7)

The agent first quantified the scale of the coverage problem:

```
Wayne members: 10257
Washtenaw members: 1794
```

This 5.7x ratio informed the agent's later strategy: prioritize Wayne County coverage first (larger population), then fill Washtenaw gaps with a complementary entity.

#### Bounding Box Computation (Step 9)

With no county boundary data available in the sandbox, the agent **invented** a geographic proxy: compute the mean latitude/longitude of members per county to establish a "county center," then measure provider distance to that center. The agent's code:

```python
wayne_members = members_df[members_df["county"].str.lower() == "wayne"]
washtenaw_members = members_df[members_df["county"].str.lower() == "washtenaw"]

print("Wayne county member location range:")
print(f"  lat: {wayne_members['lat'].min():.2f} to {wayne_members['lat'].max():.2f}")
print(f"  lon: {wayne_members['lon'].min():.2f} to {wayne_members['lon'].max():.2f}")
print(f"  center: ({wayne_members['lat'].mean():.2f}, {wayne_members['lon'].mean():.2f})")

print("\nWashtenaw county member location range:")
print(f"  lat: {washtenaw_members['lat'].min():.2f} to {washtenaw_members['lat'].max():.2f}")
print(f"  lon: {washtenaw_members['lon'].min():.2f} to {washtenaw_members['lon'].max():.2f}")
print(f"  center: ({washtenaw_members['lat'].mean():.2f}, {washtenaw_members['lon'].mean():.2f})")
```

**Output:**

| County | Lat Range | Lon Range | Computed Center |
|--------|-----------|-----------|----------------|
| Wayne | 42.05 – 42.45 | -83.55 – -82.87 | (42.33, -83.23) |
| Washtenaw | 42.08 – 42.43 | -84.13 – -83.54 | (42.26, -83.74) |

The agent computed these bounding boxes from the member data itself — not from external geographic data — then used the mean as a county centroid for distance calculations.

#### Provider-to-County Distance Calculation (Step 10)

The agent then calculated the Euclidean distance from each provider to both county centers. The relevant code:

```python
for system in large_systems:
    sys_provs = candidates_df[candidates_df["entity"] == system]
    if not sys_provs.empty:
        # ... specialty checks ...
        mi_provs = sys_provs[sys_provs["state"].str.lower() == "mi"]
        for _, row in mi_provs.head(5).iterrows():
            dist_wayne = np.sqrt((row["lat"] - wayne_center_lat)**2 + (row["lon"] - wayne_center_lon)**2) * 69
            dist_washtenaw = np.sqrt((row["lat"] - washtenaw_center_lat)**2 + (row["lon"] - washtenaw_center_lon)**2) * 69
            print(f"    {row['city']}, MI - {row['specialty']} (dist to Wayne: {dist_wayne:.1f}mi, to Washtenaw: {dist_washtenaw:.1f}mi)")
```

The agent multiplied the Euclidean distance in degrees by 69 (approximate miles per degree of latitude) to convert to miles. This is a rough approximation but sufficient for relative comparison at this geographic scale.

**Full distance output for all 5 health systems:**

| Entity | City | Specialty | → Wayne Center | → Washtenaw Center |
|--------|------|-----------|---------------|-------------------|
| **Henry Ford Health** | West Bloomfield | Internal Medicine | 19.0 mi | 30.2 mi |
| | Detroit | Internal Medicine | 10.7 mi | 45.8 mi |
| | Plymouth | Family Practice | 19.4 mi | 18.1 mi |
| | Southfield | Family Practice | 9.2 mi | 39.2 mi |
| | Algonac | Family Practice | 52.3 mi | 86.7 mi |
| **MyMichigan Health** | Midland | Internal Medicine | 114.8 mi | 101.8 mi |
| | Ann Arbor | Internal Medicine | 35.7 mi | **1.0 mi** |
| | Ann Arbor | Internal Medicine | 30.5 mi | 5.7 mi |
| | Ann Arbor | Internal Medicine | 32.9 mi | 3.2 mi |
| | Dexter | Family Practice | 44.3 mi | 10.8 mi |
| **Trinity Health** | Pontiac | Family Practice | 20.1 mi | 40.5 mi |
| | Ypsilanti | Internal Medicine | 29.0 mi | 6.2 mi |
| | Pinckney | Family Practice | 48.9 mi | 19.2 mi |
| | Caledonia | Internal Medicine | 159.7 mi | 127.1 mi |

#### Key Geographic Insights from Distance Analysis

The distance data revealed three clear patterns:

1. **Henry Ford = Wayne-dominant:** Detroit (10.7 mi to Wayne center), Southfield (9.2 mi), West Bloomfield (19.0 mi) — all within 20 mi of Wayne center, but 30-46 mi from Washtenaw. The only Washtenaw-adjacent provider is Plymouth at 18.1 mi.
2. **MyMichigan = Washtenaw-dominant:** Three Ann Arbor providers at 1.0, 3.2, and 5.7 mi from Washtenaw center — essentially inside the county. But 30-36 mi from Wayne center, and Midland providers are 100+ mi from both.
3. **Trinity = Distributed:** Pontiac (Wayne-adjacent at 20 mi) and Ypsilanti (Washtenaw-adjacent at 6.2 mi), suggesting coverage of both counties but with fewer concentrated providers.

These distance-based insights **pre-filtered** the candidates before the expensive batch simulation: Henry Ford for Wayne, MyMichigan for Washtenaw, Trinity as a wildcard for both.

#### Why This Approach Is Notable

- **Invented heuristic:** The agent created the distance-to-county-center proxy without being prompted. No county boundary data, no zip-to-county lookup table was available — the agent used member mean coordinates as a stand-in.
- **Two-phase strategy:** Fast Euclidean filtering (Step 10) identified likely candidates, then precise BallTree haversine simulation (Step 14) validated the findings. This saved ~10+ simulation steps by avoiding blind testing of geographically irrelevant entities.
- **Approximation awareness:** The agent used Euclidean distance on lat/lon with a 69 mi/degree multiplier — knowing it's approximate but sufficient for relative ranking at this scale.

### Phase 3: Batch Simulation — The Key Difference (Steps 11–14)

This is where the agent's approach diverged from previous runs. Instead of simulating candidates one at a time, the agent **batch-simulated all 5 health systems simultaneously**:

```python
candidates = ["Corewell Health", "Henry Ford Health", "MyMichigan Health",
              "Trinity Health", "Mclaren Health"]
for ent in candidates:
    provs = candidates_df[candidates_df["entity"] == ent]
    sim_net = pd.concat([network_df, provs])
    sim_cov, _ = compute_coverage(sim_net, members_df, thresholds, candidates_df)
    # ... compute delta_pp for each county/specialty
```

**Results (side-by-side comparison):**

| Entity | Washtenaw Cardio | Washtenaw GP | Wayne Cardio | Wayne GP | Wayne Ortho |
|--------|-----------------|--------------|-------------|----------|-------------|
| Corewell Health | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Henry Ford Health** | 89.3% | 9.2% | **99.4%** | 97.9% | **99.9%** |
| **MyMichigan Health** | 96.1% | 99.8% | 81.3% | 95.9% | 84.7% |
| Trinity Health | 91.8% | 100.0% | 55.0% | 99.4% | 92.8% |
| Mclaren Health | 78.1% | 0.0% | 78.5% | 85.9% | 70.6% |

Corewell Health showed 0% coverage across all categories — its providers are concentrated in West Michigan (Grand Rapids area), far from both target counties. This single batch simulation eliminated Corewell and Mclaren from consideration and surfaced Henry Ford and MyMichigan as the top candidates.

### Phase 4: Paired Simulation & Trade-off Analysis (Steps 16–26)

The agent then drilled deeper, simulating **combinations** of entities. First, it tested Henry Ford + MyMichigan Health together:

```python
sim_net2 = pd.concat([network_df, henry_ford_provs, mymichigan_provs])
```

**Result:** Near-perfect coverage across all 5 combinations — Washtenaw GP jumped from 9.2% to 100%, Washtenaw Cardio from 89.3% to 97.3%, and Wayne specialties all reached 99.6%+.

Then the agent ran a **head-to-head comparison** of two potential second-entity choices:

```python
# Henry Ford + MyMichigan vs Henry Ford + Trinity
merged = pd.merge(sim_df_mm, sim_df_trinity, on=["state", "county", "specialty"],
                  suffixes=("_my_michigan", "_trinity"))
```

| County/Specialty | Henry Ford + MyMichigan | Henry Ford + Trinity | Winner |
|-----------------|------------------------|---------------------|--------|
| Washtenaw Cardio | 97.32% | 93.48% | MyMichigan (+3.84pp) |
| Washtenaw GP | 100.00% | 100.00% | Tie |
| Wayne Cardio | 99.63% | 99.91% | Trinity (+0.28pp) |
| Wayne GP | 100.00% | 100.00% | Tie |
| Wayne Ortho | 99.97% | 99.97% | Tie |

MyMichigan won decisively on Washtenaw Cardiology (+3.84pp), while Trinity had a marginal edge on Wayne Cardiology (+0.28pp). The agent also verified MyMichigan's specialty composition: 204 orthopedic surgeons (89 in Ann Arbor), 622 cardiologists, 19 GP providers (11 in Ann Arbor), confirming it could fill the Washtenaw gaps.

### Phase 5: Commit & Final Verification (Steps 27–36)

The agent committed **Henry Ford Health** to the network (5,135 providers), then ran a final verification simulation to confirm the Henry Ford + MyMichigan combination:

```python
# Verify current network coverage after adding Henry Ford
cov, _ = compute_coverage(network_df, members_df, thresholds, candidates_df)
# Washtenaw Cardio: 89.30%, Washtenaw GP: 9.20%
# Wayne Cardio: 99.40%, Wayne GP: 97.88%, Wayne Ortho: 99.88%

# Simulate adding MyMichigan on top
sim_cov, _ = compute_coverage(sim_net, members_df, thresholds, candidates_df)
# Washtenaw GP: 9.20% → 100.00% (+90.80pp)
```

The agent delivered its final recommendation with a detailed coverage impact table and rationale.

## Test Run 2 (Post-Fix) — Step-by-Step Breakdown

The second test run used the identical prompt, thresholds, and LLM configuration. With the data quality fixes applied, the agent followed a different reasoning path and reached a different (but equally valid) conclusion.

**Action log:** `react_agent_actions_20260504_084916.txt`

### Phase 1: Baseline & Entity Landscape (Steps 1–6)

The agent checked baseline coverage (0% across all categories) and explored `entity_summaries_df`. It then searched for entities with all 3 required specialties:

```
Cardiology entities: 242
General Practice entities: 166
Orthopedic Surgery entities: 233

Entities with both cardiology and general practice: 21
Entities with all 3 specialties: 9
```

The 9 entities with full specialty coverage were small-to-mid-sized: CARE CONNECTION PLUS (995 providers), CARELINC MEDICAL EQUIPMENT (147), BINSONS HOSPITAL SUPPLIES (74), and others. All showed `new_patient_rate: 0.0` — a data quality artifact.

### Phase 1.5: Data Type Discovery & Error Recovery (Steps 11–16)

The agent attempted to aggregate MI providers by entity using `(x > 0).mean()` on `new_patient_claims`, which triggered a `TypeError` because the column stores strings (`'yes'`, `'no'`, `'insufficient'`), not numbers. The agent diagnosed the issue:

```
new_patient_claims dtype: str
Sample new_patient_claims: ['yes', 'yes', 'insufficient', 'yes', 'no', ...]
```

Then fixed the aggregation with `.str.lower() == "yes"` and retried. This produced a correct ranking of MI entities with all 3 specialties, sorted by provider count:

| Entity | Providers | Effectiveness | New Patient Rate |
|--------|-----------|--------------|-----------------|
| MyMichigan Health | 8,675 | 3.21 | 88.8% |
| Corewell Health | 8,493 | 3.13 | 93.4% |
| Henry Ford Health | 5,120 | 2.81 | 83.4% |
| Trinity Health | 3,759 | 2.90 | 91.4% |
| Mclaren Health | 1,278 | 2.56 | 88.4% |

### Phase 2: Batch Simulation (Steps 18–19)

**The agent skipped geographic analysis entirely.** In Run 1, the agent spent 6 steps computing bounding boxes and Euclidean distances to county centers. In Run 2, the agent went straight to batch simulation of 6 candidates (adding Care Connection Plus and CVS Health alongside the top 4).

**Batch simulation results:**

| Entity | Washtenaw Cardio | Washtenaw GP | Wayne Cardio | Wayne GP | Wayne Ortho |
|--------|-----------------|--------------|-------------|----------|-------------|
| Care Connection Plus | 79.6% | 95.0% | 75.6% | 100.0% | 99.4% |
| **Corewell Health** | 75.7% | 62.4% | **98.4%** | **100.0%** | **100.0%** |
| CVS Health | 0.0% | 100.0% | 19.8% | 100.0% | 91.8% |
| Henry Ford Health | 89.3% | 9.2% | 99.4% | 97.9% | 99.9% |
| **MyMichigan Health** | **96.1%** | **99.8%** | 81.3% | 95.9% | 84.7% |
| Trinity Health | 91.8% | 100.0% | 55.0% | 99.4% | 92.8% |

Corewell Health now shows real coverage (75-100%) instead of 0%. The agent identified MyMichigan as the best first entity (highest Washtenaw coverage, best effectiveness/new patient metrics) and committed it immediately.

### Phase 3: Second-Entity Simulation (Steps 23–24)

With MyMichigan committed, the agent simulated 4 potential second entities on top:

| Entity | Wayne Cardio Δ | Wayne GP Δ | Wayne Ortho Δ |
|--------|---------------|-----------|--------------|
| Care Connection Plus | +18.28pp | +4.11pp | +14.73pp |
| **Corewell Health** | **+18.37pp** | **+4.11pp** | **+15.35pp** |
| Henry Ford Health | +18.34pp | +4.11pp | +15.32pp |
| Trinity Health | +0.43pp | +4.05pp | +11.99pp |

Corewell and Henry Ford were nearly tied on Wayne improvement. Corewell won on Orthopedic (+15.35pp vs +15.32pp) and had a higher new patient acceptance rate (93.4% vs 83.4%).

### Phase 4: Final Output (Step 26)

The agent delivered its recommendation: MyMichigan Health added (8,675 providers), Corewell Health recommended as second entity. The two-entity combination achieves 96-100% coverage across all categories.

**Coverage progression (Run 2):**

| County | Specialty | Start | After MyMichigan | After + Corewell |
|--------|-----------|-------|-----------------|-----------------|
| Washtenaw | Cardiology | 0% | 96.10% | 96.93% |
| Washtenaw | General Practice | 0% | 99.78% | 99.78% |
| Wayne | Cardiology | 0% | 81.29% | **99.66%** |
| Wayne | General Practice | 0% | 95.89% | **100.00%** |
| Wayne | Orthopedic Surgery | 0% | 84.65% | **100.00%** |

## Cross-Run Comparison

Both runs used the identical prompt, thresholds, and LLM. The only difference was the data quality fixes applied between runs. Yet the agent followed different reasoning paths and chose different entities.

### Why Different Decisions?

**Primary driver: Corewell fix.** In Run 1, Corewell showed 0% coverage and was eliminated. Henry Ford was the only entity that could cover Wayne County at scale, making it the necessary first choice. In Run 2, Corewell's real coverage (75-100%) made it a viable Wayne-dominant alternative. With Corewell available, the agent could prioritize MyMichigan first (best metrics) and fill Wayne gaps with Corewell second.

**Different reasoning paths:**

| Aspect | Run 1 (36 steps) | Run 2 (26 steps) |
|--------|-----------------|-----------------|
| Initial filter | Ranked by effectiveness/new patient rate | Searched for all-3-specialty overlap |
| Large system discovery | Queried MI provider counts per specialty | Queried MI entities with all 3 specialties |
| Geographic analysis | Yes — bounding boxes + Euclidean distance (6 steps) | **Skipped entirely** |
| Candidate selection | 5 systems (Corewell dead) | 6 systems (Corewell viable) |
| First entity rationale | Wayne-dominant (bigger population first) | Best metrics + Washtenaw-dominant |
| Second entity rationale | MyMichigan fills Washtenaw gaps | Corewell fills Wayne gaps |
| Paired comparison | Henry Ford + MyMichigan vs Henry Ford + Trinity | MyMichigan + each of 4 candidates |
| Error type | KeyError, NameError | TypeError (string data) |

**Why Run 2 skipped geographic analysis:** The agent's initial query in Run 2 directly targeted MI entities with all 3 required specialties, surfacing the large health systems immediately. In Run 1, the agent first found small entities with perfect scores, then had to discover large systems through a separate provider-count query, which triggered the geographic curiosity. Run 2's more direct path to the large systems made the geographic pre-filter unnecessary — the agent went straight to batch simulation.

**Both combinations achieve equivalent final coverage:**

| Combo | Washtenaw Cardio | Washtenaw GP | Wayne Cardio | Wayne GP | Wayne Ortho |
|-------|-----------------|--------------|-------------|----------|-------------|
| Henry Ford + MyMichigan | 97.32% | 100.00% | 99.63% | 100.00% | 99.97% |
| MyMichigan + Corewell | 96.93% | 99.78% | 99.66% | 100.00% | 100.00% |

The difference is within 0.4 percentage points across all categories — functionally equivalent.

### What This Shows

1. **Data quality bugs silently prune the search space.** The Corewell trailing space eliminated Michigan's largest provider system from consideration. The agent's reasoning was correct _given the data it saw_ — but the data was wrong.
2. **Non-deterministic reasoning paths.** Even with identical prompt, config, and LLM, the agent explored the data differently across runs. Run 1's path was more exploratory (geo analysis, paired comparison). Run 2's path was more direct (straight to batch sim). Both reached equivalent outcomes.
3. **Cleaner data enables simpler reasoning.** Run 2 completed in 26 steps (28% fewer) because the agent didn't need the geographic pre-filter — the batch simulation alone was sufficient to identify the right candidates.

## Reasoning & Decision-Making Analysis (Run 1)

Run 1's approach was notable for its geographic pre-filtering and strategic "biggest bang first" reasoning. (Run 2 followed a more direct path — see Cross-Run Comparison above.)

### 1. Batch Simulation Over Ranked-Shortlist

**Previous approach:** Rank entities by metrics → pick the top one → simulate → repeat. This led to selecting small entities (SOUTHFIELD CITY URGENT CARE PC with 37 providers) that scored well on individual metrics but had limited coverage impact.

**This run:** The agent first identified that large health systems dominate MI provider counts (Corewell: 953 cardiologists, MyMichigan: 621, Henry Ford: 363), then **batch-simulated all 5 major systems simultaneously**. This revealed that Henry Ford Health alone achieves near-total Wayne County coverage (99.4% cardiology, 99.9% orthopedic) — a finding impossible to discover through ranked-shortlist evaluation.

### 2. Geographic Reasoning — Two-Phase Spatial Analysis

The agent performed a sophisticated two-phase geographic analysis (detailed in Phase 2.5 above). First, it computed member bounding boxes per county to establish county centers, then calculated Euclidean distance from each provider to both county centers as a fast proxy for geographic fit. This revealed Henry Ford's Wayne-dominant profile (9-20 mi to Wayne center) and MyMichigan's Washtenaw-dominant profile (1-11 mi to Washtenaw center). Second, it validated these findings through precise BallTree haversine simulation in Phase 3. The agent invented the distance-to-county-center heuristic without being prompted, using member mean coordinates as a proxy for county boundaries — no external geographic data was available in the sandbox.

### 3. Paired Simulation for Trade-off Analysis

After identifying Henry Ford as the first entity, the agent didn't just pick the next best candidate — it **simulated two potential second entities against the same baseline** (Henry Ford + MyMichigan vs Henry Ford + Trinity). This head-to-head comparison revealed that MyMichigan's Washtenaw Cardiology advantage (+3.84pp) outweighed Trinity's marginal Wayne Cardiology edge (+0.28pp), and that both combinations achieved identical 100% GP and 99.97% orthopedic coverage.

### 4. Error Recovery

The agent encountered two errors during execution:
- **Step 12:** `KeyError` when trying to merge entity summary columns that didn't exist in the merged DataFrame. The agent retried without the problematic columns (Step 13).
- **Step 23:** `NameError` because sandboxed code execution doesn't persist variables between calls. The agent retried by redefining all variables fresh (Step 25).

Both errors were recovered from autonomously without user intervention.

### 5. Why Henry Ford Over MyMichigan First? (Run 1)

Henry Ford was chosen as the first entity despite MyMichigan having better metrics (effectiveness 3.21 vs 2.81, new patient rate 88.8% vs 83.4%). The reasoning was strategic: Henry Ford provides dominant Wayne County coverage (99.4% cardiology, 99.9% orthopedic) covering 10,257 members — the larger population. MyMichigan then fills the Washtenaw gaps (1,794 members). This "biggest bang first" approach maximizes immediate coverage impact.

**Run 2 chose MyMichigan first** because Corewell was now viable as the Wayne-dominant partner. With Corewell available, the agent could prioritize the entity with the best metrics (MyMichigan: 3.21 effectiveness, 88.8% new patient rate) and fill Wayne gaps with Corewell second.

## Coverage Progression

### Run 1: Henry Ford → MyMichigan

| County | Specialty | Start | After Henry Ford | After + MyMichigan |
|--------|-----------|-------|-----------------|-------------------|
| Washtenaw | Cardiology | 0% | 89.30% | **97.32%** |
| Washtenaw | General Practice | 0% | 9.20% | **100.00%** |
| Wayne | Cardiology | 0% | 99.40% | **99.63%** |
| Wayne | General Practice | 0% | 97.88% | **100.00%** |
| Wayne | Orthopedic Surgery | 0% | 99.88% | **99.97%** |

### Run 2: MyMichigan → Corewell

| County | Specialty | Start | After MyMichigan | After + Corewell |
|--------|-----------|-------|-----------------|-----------------|
| Washtenaw | Cardiology | 0% | 96.10% | **96.93%** |
| Washtenaw | General Practice | 0% | 99.78% | **99.78%** |
| Wayne | Cardiology | 0% | 81.29% | **99.66%** |
| Wayne | General Practice | 0% | 95.89% | **100.00%** |
| Wayne | Orthopedic Surgery | 0% | 84.65% | **100.00%** |

## Key Observations

### From Run 1 (Pre-Fix)

1. **Data-driven strategy pivot:** The agent's initial ranking favored small entities with perfect effectiveness scores (5.0) and 100% new patient rates. But when it queried MI provider counts by specialty, it saw that the top-ranked entity had at most 37 providers, while large systems had hundreds to thousands per specialty — a 10x to 300x difference. The agent self-corrected its strategy based on this data, shifting from ranked-shortlist evaluation of small entities to batch simulation of major health systems. This pivot was not prompted by system instructions; it emerged from the agent's own analysis of the data.

2. **Two-phase geographic analysis guides simulation:** Before running expensive batch simulations, the agent performed fast Euclidean distance calculations from providers to county centers (Phase 2.5). This pre-filtering identified Henry Ford as Wayne-dominant and MyMichigan as Washtenaw-dominant, directing the batch simulation toward the right candidates. Corewell Health's 0% coverage in the batch simulation then confirmed what the distance analysis already suggested: its West Michigan location is irrelevant to both target counties. (In hindsight, Corewell's 0% was a data bug, not a geographic fact.)

3. **Batch simulation produces better decisions:** The side-by-side comparison of 5 health systems in a single simulation revealed patterns invisible to sequential evaluation. Henry Ford's Wayne-dominant profile (99.4% Wayne Cardio, 99.9% Wayne Ortho) stood out against MyMichigan's Washtenaw strength (96.1% Washtenaw Cardio, 99.8% Washtenaw GP).

4. **Paired simulation enables trade-off reasoning:** By simulating Henry Ford + MyMichigan vs Henry Ford + Trinity, the agent could quantify the exact coverage trade-off (MyMichigan wins Washtenaw Cardio by +3.84pp, Trinity wins Wayne Cardio by +0.28pp) and make an evidence-based recommendation.

5. **Large systems beat small specialists on coverage:** The previous run selected SOUTHFIELD CITY URGENT CARE PC (37 providers) which achieved 100% Wayne GP but 0% everywhere else. This run selected Henry Ford Health (5,135 providers) which achieved 89-100% coverage across 4 of 5 categories immediately. The batch simulation made this trade-off explicit.

6. **Two-entity network achieves near-perfect coverage:** The Henry Ford + MyMichigan combination covers 97-100% of all members across all specialties — a comprehensive network built from just 2 contracted entities.

7. **Context management at scale:** The agent triggered summarization 3 times (Steps 15, 24, 34) as its message history grew. The compressed summaries preserved all key findings (batch simulation results, pairwise comparisons, entity metrics) and the agent continued reasoning without losing context.

8. **Code as reasoning remains the core pattern:** Every analytical step — filtering candidates, geographic analysis, batch simulation, pairwise comparison — flows through `run_code`. The agent's decision-making is expressed as composable Python snippets, each producing concrete numerical results that inform the next step.

### From Run 2 (Post-Fix)

9. **Data quality bugs silently prune the search space:** The Corewell trailing space bug eliminated Michigan's largest provider system from consideration. The agent's reasoning was correct _given the data it saw_ — but the data was wrong. After the fix, Corewell emerged as a viable Wayne-dominant alternative, changing the agent's entire decision path. This underscores the importance of data normalization before agent execution.

10. **Non-deterministic reasoning paths:** Even with identical prompt, config, and LLM, the agent explored the data differently across runs. Run 1 was more exploratory (geographic analysis, paired comparison, 36 steps). Run 2 was more direct (straight to batch simulation, 26 steps). Both reached equivalent outcomes (97-100% coverage). The agent's reasoning path depends on what data it encounters first — not a fixed algorithm.

11. **Cleaner data enables simpler reasoning:** Run 2 completed in 26 steps (28% fewer than Run 1) because the agent didn't need the geographic pre-filter. With Corewell viable, the batch simulation alone was sufficient to identify the right candidates. The geographic analysis in Run 1 was valuable _because_ Corewell was dead — it helped the agent understand why Henry Ford and MyMichigan were complementary. With Corewell available, that analysis was unnecessary.

12. **`new_patient_claims` string data type needs handling:** Run 2 exposed that `new_patient_claims` stores `'yes'`/`'no'`/`'insufficient'` strings, not numeric values. The agent discovered this through a `TypeError` and self-corrected. This is a data schema issue that should be documented or pre-processed.

### Cross-Run

13. **Both two-entity combos achieve equivalent coverage:** Henry Ford + MyMichigan and MyMichigan + Corewell both reach 97-100% across all categories. The choice between them depends on entity-level preferences (effectiveness, new patient rate, contracting complexity) rather than coverage outcomes.

14. **Geographic analysis is not always necessary:** Run 1's sophisticated bounding box + Euclidean distance analysis (Phase 2.5) was impressive but ultimately redundant — the batch simulation alone identified the right candidates. The geographic analysis was valuable in Run 1 because it helped the agent understand _why_ Corewell showed 0% (West Michigan location). In Run 2, with Corewell viable, the geographic analysis was skipped entirely.
