# Agent Run Evaluation Report

**Threads:** `test_notebook2` & `test_notebook3` &nbsp;|&nbsp; **Date:** May 13, 2026

---

## Executive Summary

Two runs of the same prompt produced markedly different approaches. **Run #2** (`test_notebook2`) was a deep, self-driven investigation (118s, 8 steps) that discovered coverage gaps, expanded its candidate pool, and produced a nuanced 7-entity ranking. **Run #3** (`test_notebook3`) started as a fast, linear execution (51s, 6 steps) but evolved across 4 user-driven sessions into an iterative refinement process, ultimately converging on a similar 3-factor scoring methodology. The notebook3 run also exposed a `numpy.float64` serialization bug in the checkpoint layer (now fixed in `nodes.py`).

---

## Task Prompt

> Find 5 large health systems (show in the entities column) that have high geographic proximity to our members. Then calculate the actual coverage they would provide to the network. Then create a balanced score between that and effectiveness. Rank them and present back to me in table format.

---

## Run #2: `test_notebook2` — Deep Investigation

**Duration:** 118s &nbsp;|&nbsp; **Steps:** 8 (7 tool calls + 1 final output)

### Execution Phases

The agent's work unfolded across **3 distinct phases**. Each phase represents a shift in strategy based on what the agent learned from prior results.

### Phase 1: Data Exploration & Proximity Ranking (Steps 1–3)

**Goal:** Understand the data landscape and rank entities by geographic proximity to members.

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 1 | 12:09:08 | 6.8s | Aggregated 8,609 entities by provider count, effectiveness, coordinates. Computed member centroids per county. | Discovered 2 counties: Washtenaw (1,794 members), Wayne (10,257 members). Saved entity stats to cache. |
| 2 | 12:09:16 | 7.6s | Wrote haversine function with `from math import ...` → **FAILED** (sandbox blocks imports). | Error: "Import statements are disabled." |
| 3 | 12:09:21 | 6.1s | Rewrote haversine using pre-loaded `math` module. Computed weighted average distance for all 8,609 entities to member centroids. Filtered for &ge;500 providers. | Top 5 by proximity: Wayne State (9.4 mi), Henry Ford (12.1 mi), Michigan Healthcare Prof. (16.9 mi), MyMichigan (30.8 mi), Great Lakes Psychology (35.1 mi). |

**Decision point:** The agent now had a proximity ranking. The next logical step was to check whether proximity translates to actual member coverage.

### Phase 2: Coverage Validation & Problem Discovery (Steps 4–6)

**Goal:** Simulate actual network coverage for the top proximity-ranked entities, then investigate and fix gaps.

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 4 | 12:09:27 | 5.8s | Ran `compute_coverage()` on each of the top 5 proximity entities individually. | **Problem found:** Wayne State only 40% coverage (cardiology only, no GP). Great Lakes Psychology 0% coverage (psychology-only). Henry Ford 76.8%, MyMichigan 78.9% are solid. |
| 5 | 12:09:36 | 8.7s | Investigated _why_ coverage was 0%: examined specialty lists for top 5 and next 5 candidates. Checked for "general practice" and "cardiology" presence. | Root cause: specialty mismatch. Wayne State lacks GP. Great Lakes Psychology has neither GP nor cardiology. Trinity, Promedica, Corewell have both. |
| 6 | 12:09:43 | 7.1s | Scanned entities in positions 5–15 for GP + cardiology. Tested coverage for 3 new candidates: Trinity (83.0%), Sparrow (0.0%), McLaren (64.6%). | Trinity has best balanced coverage. Sparrow has right specialties but providers too far from members. McLaren strong on cardiology, weak on GP. |

**Key strategic shift:** At this point, the agent moved from a passive "rank the top 5" approach to an active "find the best candidates" approach. It recognized that the prompt's request for "5 large health systems with high proximity" needed to be interpreted as "5 that are both close AND useful."

### Phase 3: Scoring, Ranking & Final Output (Steps 7–8)

**Goal:** Combine proximity, coverage, and effectiveness into a single balanced score, then present ranked results.

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 7 | 12:09:56 | 12.6s | Tested Corewell Health coverage (91.2% — highest). Built comparison table of 7 candidates. Applied min-max normalization to coverage, distance, and effectiveness. Computed weighted balanced score (40/30/30). | Final ranking computed. MyMichigan #1 (81.98), Henry Ford #2 (69.26), Corewell #3 (66.31). |
| 8 | 12:10:59 | 63s | Formatted results into markdown table with key insights, weighting methodology, and offer for next steps. | Task complete. |

**Note:** Step 8's 63s duration is LLM reasoning time for composing the final response — no tool calls were made.

### Timeline

```
12:09:01  ── Human input received
          │
12:09:08  ── PHASE 1: Exploration
          │  Step 1: Data aggregation (6.8s)
          │  Step 2: Import error (7.6s) ← FAILED
          │  Step 3: Fix + proximity calc (6.1s) ← RECOVERED
          │
12:09:27  ── PHASE 2: Validation & Investigation
          │  Step 4: Coverage simulation → 0% coverage discovered (5.8s)
          │  Step 5: Specialty root-cause analysis (8.7s)
          │  Step 6: Expanded candidate testing (7.1s)
          │
12:09:56  ── PHASE 3: Scoring & Delivery
          │  Step 7: Balanced score computation (12.6s)
          │  Step 8: Final formatted output (63s)
          │
12:10:59  ── COMPLETE (118s total)
```

The three-phase structure reveals the agent's problem-solving pattern: **explore → validate → refine → deliver**. Each phase's output triggered the next phase's strategy, rather than following a rigid pre-planned sequence.

### Reasoning Highlights

**1. Self-Correction: Sandbox Import Error Recovery**

In step 2, the agent wrote a haversine distance function using standard Python imports (`from math import radians, sin, cos, sqrt, atan2`). This failed because the sandbox blocks `import` statements. The agent immediately recognized the error and rewrote the function using the pre-loaded `math` module (`math.radians`, `math.sin`, etc.) and succeeded on retry.

**2. Discovery: Proximity Does Not Equal Coverage**

After ranking entities by proximity, the agent ran `compute_coverage()` on the top 5 individually and found that 2 of them provided minimal or zero coverage:

| Entity | Distance | Coverage |
|--------|----------|----------|
| Wayne State | 9.4 mi (closest) | 40.0% |
| Great Lakes Psychology | 35.1 mi | **0.0%** |

The agent recognized this gap and initiated a root-cause investigation rather than simply reporting the flawed results.

**3. Root-Cause Analysis: Specialty Investigation**

The agent examined specialty lists to understand the 0% coverage, then scanned 10 additional entities for GP and cardiology availability. This led to discovering Trinity Health (83.0% coverage) and Corewell Health (91.2% coverage).

**4. Balanced Score Methodology**

The agent designed a weighted scoring model with min-max normalization: 40% coverage + 30% proximity (inverted distance) + 30% effectiveness.

### Final Results

| Rank | Entity | Providers | Coverage | Distance (mi) | Effectiveness | Balanced Score |
|------|--------|-----------|----------|---------------|---------------|----------------|
| 1 | **MyMichigan Health** | 8,684 | 78.9% | 30.79 | 3.21 | **81.98** |
| 2 | **Henry Ford Health** | 5,135 | 76.8% | 12.07 | 2.81 | **69.26** |
| 3 | **Corewell Health** | 8,506 | 91.2% | 85.45 | 3.13 | **66.31** |
| 4 | **Trinity Health** | 3,890 | 83.0% | 55.80 | 2.89 | **60.55** |
| 5 | **Wayne State Univ. School of Medicine** | 650 | 40.0% | 9.38 | 3.15 | **57.23** |

**Key insight:** MyMichigan Health ranks #1 despite being 4th closest, because it achieves strong coverage across all 4 county-specialty buckets while maintaining top effectiveness. Corewell has the highest raw coverage (91.2%) but is penalized for distance (85.5 mi).

### Run #2 Metrics

| Metric | Value |
|--------|-------|
| Total duration | 118 seconds |
| Steps | 8 (7 tool calls + 1 final output) |
| Tool calls | 7 (`run_code`) |
| Errors encountered | 1 (import blocked) |
| Error recovery | 1 step (immediate fix) |
| Entities evaluated | 17+ (8,609 for proximity, 10+ for coverage) |
| Candidates in final ranking | 7 |

---

## Run #3: `test_notebook3` — Linear Start, Iterative Refinement

**Total duration:** ~191s across 4 sessions &nbsp;|&nbsp; **Total steps:** 18 (12 tool calls)

### Session 1: Initial Execution (51s, 6 steps)

The agent's initial approach was more linear than run #2, with less self-driven iterative refinement.

#### Phase 1: Data Exploration & Proximity Ranking (Steps 1–3)

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 1 | 16:11:22 | 8.0s | Aggregated 8,609 entities by provider count, effectiveness, coordinates. | Discovered 2 counties: Washtenaw (1,794), Wayne (10,257). Saved entity stats to cache. |
| 2 | 16:11:30 | 7.9s | Wrote BallTree proximity code with `from scipy.spatial import distance` → **FAILED** (sandbox blocks imports). | Error: "Import statements are disabled." |
| 3 | 16:11:36 | 13.6s | Rewrote without imports. Used BallTree per-provider haversine queries for 147 entities (&ge;50 providers). Computed avg/median distance, members within 10/15 mi. | Top 5 by proximity: Henry Ford (1.30 mi), VAMC (1.39 mi), Great Lakes Psychology (1.82 mi), Care Connection Plus (1.86 mi), Corewell (1.99 mi). |

**Key difference from run #2:** The agent used BallTree for per-provider haversine queries rather than centroid-to-centroid haversine. The distances are notably lower (1.30–1.99 mi vs 9.4–35.1 mi), reflecting per-provider proximity rather than entity-centroid distance.

#### Phase 2: Coverage Validation (Step 4)

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 4 | 16:11:49 | 5.6s | Combined all providers from top 5 entities (16,949 total). Ran `compute_coverage()` once on the combined set. | Near-complete coverage: Wayne cardiology 100%, Wayne GP 99.1%, Washtenaw cardiology 99.7%, Washtenaw GP 79.7%. |

**Weakness:** The agent tested the combined set only. It did not discover that Great Lakes Psychology Group contributes 0% individually (psychology-only specialties), nor did it scan for additional candidates.

#### Phase 3: Scoring & Final Output (Steps 5–6)

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 5 | 16:11:57 | 7.6s | Applied min-max normalization to proximity (inverted distance) and effectiveness. Computed 50/50 weighted balanced score. | Final ranking: Care Connection Plus #1 (59.4), VAMC #2 (53.8), Henry Ford #3 (50.0). |
| 6 | 16:12:06 | 8.4s | Formatted results into markdown table with key insights and coverage breakdown. | Task complete. |

#### Session 1 Results

| Rank | Entity | Providers | Avg Distance (mi) | Effectiveness | Proximity Score | Effectiveness Score | Balanced Score |
|------|--------|-----------|-------------------|---------------|-----------------|---------------------|----------------|
| 1 | **Care Connection Plus** | 995 | 1.86 | 4.20 | 18.8 | 100.0 | **59.4** |
| 2 | **VAMC (All)** | 1,751 | 1.39 | 3.10 | 87.0 | 20.6 | **53.8** |
| 3 | **Henry Ford Health** | 5,135 | 1.30 | 2.81 | 100.0 | 0.0 | **50.0** |
| 4 | **Great Lakes Psychology Group** | 562 | 1.82 | 2.82 | 24.6 | 1.2 | **12.9** |
| 5 | **Corewell Health** | 8,506 | 1.99 | 3.13 | 0.0 | 23.0 | **11.5** |

### Session 2: Individual Coverage Request (20:33:58, 4 steps)

User asked: *"I wanted to see how each of the top 5 provided coverage to the network individually."*

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 1 | 20:34:10 | 12.3s | Computed individual coverage for each of the top 5 entities. | Discovered Great Lakes Psychology at 0% coverage, Henry Ford at 0% Washtenaw GP. VAMC 88.5%, Care Connection 83.5%, Corewell 72.2%. |
| 2 | 20:35:10 | 60.0s | Attempted to merge entity_stats with proximity_df and individual coverage. | **FAILED:** `KeyError: 'avg_effectiveness'` — merged dataframe lost column after filter. |
| 3 | 20:35:12 | 2.0s | Inspected column names of both cached dataframes for debugging. | Found `entity_stats` has the column but `proximity_df` also has it — the merge on filtered subset lost it. |
| 4 | 20:35:19 | 7.0s | Used `proximity_df` (which already had effectiveness) as the base, merged only with individual coverage. Recomputed balanced score. | Success. Same ranking as session 1, but now with individual coverage columns. |

**Bug exposed:** After step 4, the `update_state` node stored `sandbox_cache` containing `numpy.float64` values into agent state. When LangGraph tried to checkpoint to SQLite via msgpack serialization, it crashed: `TypeError: Type is not msgpack serializable: numpy.float64`. This was fixed by adding `_to_native()` conversion in `nodes.py`.

### Session 3: Finish Request (20:41:47, 1 step)

User asked: *"Ok, I think you have what you need, finish your job."*

The agent produced a final table combining individual coverage metrics with balanced scores:

| Rank | Entity | Balanced Score | Avg Coverage | Wayne GP | Wayne Cardio | Washtenaw GP | Washtenaw Cardio |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Care Connection Plus** | **59.42** | **83.46%** | 76.56% | 91.12% | 74.92% | 91.25% |
| **2** | **VAMC (All)** | **53.79** | **88.52%** | 88.97% | 99.87% | 72.91% | 92.31% |
| **3** | **Henry Ford Health** | **50.00** | **65.69%** | 63.05% | 100.00% | 0.00% | 99.72% |
| **4** | **Great Lakes Psychology Group** | **12.92** | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% |
| **5** | **Corewell Health** | **11.48** | **72.24%** | 98.33% | 100.00% | 1.34% | 89.30% |

### Session 4: Coverage-Inclusive Scoring (20:46:41, 3 steps)

User asked: *"I want you to include the coverage scores into the balanced score calculation. Then rank by that new balanced score and report back."*

| Step | Time | Duration | Action | Outcome |
|------|------|----------|--------|---------|
| 1 | 20:47:01 | 19.7s | Attempted to recalculate with `import pandas as pd` / `import numpy as np` → **FAILED** (sandbox blocks imports). | Error: "Import statements are disabled." |
| 2 | 20:47:42 | 41.4s | Rewrote without imports. Computed new 3-way balanced score: 33.3% proximity + 33.3% effectiveness + 33.3% coverage. | New ranking: Care Connection Plus (67.43), VAMC (65.37), Henry Ford (55.23), Corewell (31.73), Great Lakes Psychology (8.62). |
| 3 | 20:47:53 | 10.9s | Formatted final output with updated methodology explanation and ranking table. | Task complete. |

#### Session 4 Final Results

| Rank | Entity | Providers | Avg Dist (mi) | Avg Coverage | Proximity Score | Effectiveness Score | **New Balanced Score** |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Care Connection Plus** | 995 | 1.86 | 83.46% | 18.84 | 100.00 | **67.43** |
| 2 | **VAMC (All)** | 1,751 | 1.39 | 88.52% | 86.96 | 20.62 | **65.37** |
| 3 | **Henry Ford Health** | 5,135 | 1.30 | 65.69% | 100.00 | 0.00 | **55.23** |
| 4 | **Corewell Health** | 8,506 | 1.99 | 72.24% | 0.00 | 22.96 | **31.73** |
| 5 | **Great Lakes Psychology Group** | 562 | 1.82 | 0.00% | 24.64 | 1.21 | **8.62** |

Including coverage in the score shifted Corewell above Great Lakes Psychology, and tightened the gap between #1 and #2.

### Run #3 Metrics (All Sessions)

| Metric | Session 1 | Sessions 2–4 | Total |
|--------|-----------|--------------|-------|
| Duration | 51s | ~140s | ~191s |
| Steps | 6 | 12 | 18 |
| Tool calls | 5 | 7 | 12 |
| Errors encountered | 1 (import) | 3 (import x2, KeyError) | 4 |
| Error recovery | 1 step | 3 steps | 4 |
| Bug exposed | — | numpy serialization crash | 1 |

---

## Cross-Run Comparison

| Aspect | Run #2 (`test_notebook2`) | Run #3 (`test_notebook3`) |
|--------|---------------------------|---------------------------|
| **Session 1 duration** | 118s | 51s |
| **Total duration (all sessions)** | 118s (single session) | ~191s (4 sessions) |
| **Steps** | 8 | 18 |
| **Tool calls** | 7 | 12 |
| **Proximity method** | Centroid haversine (all 8,609 entities) | BallTree per-provider (147 entities &ge;50 providers) |
| **Coverage testing** | Individual per entity, self-driven | Combined initially; individual only after user request |
| **0% coverage discovered** | Yes, agent-initiated | Yes, but only after user asked for individual coverage |
| **Candidate expansion** | Yes, tested 10+ additional entities | No, stayed within initial top 5 |
| **Balanced score factors (final)** | Coverage (40%) + Proximity (30%) + Effectiveness (30%) | Proximity (33.3%) + Effectiveness (33.3%) + Coverage (33.3%) |
| **Final candidates ranked** | 7 | 5 |
| **Top-ranked entity** | MyMichigan Health (81.98) | Care Connection Plus (67.43) |
| **Errors encountered** | 1 (import blocked) | 4 (import x2, KeyError, numpy serialization) |
| **Bug exposed** | None | `numpy.float64` msgpack serialization crash |

### Methodology Convergence

Both runs started with different approaches but converged on similar methodology:

| Factor | Run #2 | Run #3 (Session 4) |
|--------|--------|--------------------|
| Coverage weight | 40% | 33.3% |
| Proximity weight | 30% | 33.3% |
| Effectiveness weight | 30% | 33.3% |
| Normalization | Min-max | Min-max |
| Individual coverage | Yes | Yes |

### Key Differences

**Run #2 advantages:**
- Self-driven discovery of coverage gaps without user prompting
- Expanded candidate pool beyond initial top 5
- Root-cause investigation (specialty analysis) triggered by data, not user
- Larger candidate pool (7 entities) provides more options
- BallTree-based `compute_coverage` used for individual entity testing

**Run #3 advantages:**
- More precise proximity measurement (BallTree per-provider vs centroid approximation)
- Faster initial execution (51s vs 118s)
- Individual coverage breakdown per county-specialty bucket
- User-guided refinement produced a more detailed final table
- Exposed the numpy serialization bug (now fixed)

---

## Bugs Discovered

### `numpy.float64` Serialization Crash (Run #3, Session 2)

**Error:** `TypeError: Type is not msgpack serializable: numpy.float64`

**Cause:** `sandbox_cache` values from `pd.DataFrame.to_dict("records")` contain `numpy.float64` types. When `update_state()` merged the cache into agent state, the numpy types were stored directly. LangGraph's checkpoint layer uses msgpack serialization, which cannot serialize numpy types.

**Fix:** Added `_to_native()` helper in `nodes.py` that recursively converts numpy types (`np.integer`, `np.floating`, `np.bool_`, `np.ndarray`) to native Python equivalents before storing in state. All 111 tests pass.

---

## Overall Assessment

Both runs demonstrate the agent's core capabilities: error recovery from sandbox constraints, effective use of `sandbox_cache` for cross-step state, and transparent methodology. The runs highlight a tradeoff in the agent's behavior:

- **Run #2** shows the agent at its most proactive — discovering problems, investigating root causes, and expanding its search without prompting. This produced a more robust result but took longer.
- **Run #3** shows the agent at its most efficient — completing the task as stated in fewer steps. But it required user follow-up to surface the same insights that run #2 found on its own.

The agent's behavior appears sensitive to how it interprets the prompt. When it interprets "find 5 large health systems" as a screening exercise, it expands the search. When it interprets it as a ranking exercise, it stays within the top 5. Both interpretations are valid; the former produces more robust results.
