# Agent Testing Report - ChatDeepSeek Migration

## Context
- Commit: `bbab015` - refactor: migrate from ChatOpenAIWithReasoning to ChatDeepSeek
- Key changes: Removed `aggregate_entities`, `build_schema_profile`, `entity_summaries_df` from sandbox
- LLM: `llm` @ `http://localhost:8080/v1`
- Date: 2026-05-10

## Thresholds
```json
{"mi": {"wayne": {"general practice": 10.0, "cardiology": 15.0}, "washtenaw": {"general practice": 10.0, "cardiology": 15.0}}}
```

---

## Test 1: Multi-objective Optimization
**Thread:** `test1_pareto_v3`
**Prompt:** "Find 3 entities that, when added together, maximize total coverage while minimizing the number of providers needed. Show the Pareto-optimal set."
**Status:** ✅ SUCCESS | 12 steps

**Result:** Pareto-optimal 3-entity combos identified. Top recommendation: `trinity health` + `vamc (all)` + `cvs health` (5,931 providers, 389.9% total coverage, min 93.2%). Most efficient: `vamc (all)` + `care connection plus` + `cvs health` (3,036 providers, 383.0% coverage, min 92.9%). Maximum coverage: `trinity health` + `corewell health` + `mymichigan health` (21,080 providers, 396.4% coverage, min 97.0%).

**Observations:**
- `TypeError` on `mean` for string column (`new_patient_claims`) — worked around by dropping the column
- Case-sensitivity trap: state values are lowercase `'mi'`, not `'MI'` — returned 0 results, had to debug and retry
- No import errors — agent avoided the `import is DISABLED` pitfall
- Evaluated 50+ combinations across 1/2/3-entity sets to build Pareto frontier
- **Summarization bug persists**: summary says "No 3-entity combinations have been simulated yet" when 30+ were evaluated

---

## Test 2: Gap Analysis + Before/After
**Thread:** `test2_gap_analysis_v2`
**Prompt:** "Identify the geographic areas (counties) with the lowest coverage, then for each gap find the single best entity to fill it. Show a before/after comparison."
**Status:** ✅ SUCCESS | 7 steps

**Result:** Best entity per gap (4 entities, 115 providers total):
| Gap | Best Entity | Coverage |
|-----|-------------|----------|
| wayne / general practice | asnaa | 60.8% |
| wayne / cardiology | option care health | 99.6% |
| washtenaw / general practice | rwjbarnabas health | 78.1% |
| washtenaw / cardiology | los angeles county dept of health services | 99.5% |

**Observations:**
- 1 import error (tried `from math import radians, cos, sin, asin, sqrt` instead of pre-loaded `math`) — recovered on retry
- **No case-sensitivity issue** — agent used lowercase values throughout (new system prompt note may have helped)
- Used haversine distance heuristic to rank entities, then validated with `compute_coverage`
- 2 entities from heuristic ranking not found in candidates (`family medical center pc`, `urgent care associates inc`)
- No summarization triggered (7 steps, under threshold of 14)
- Faster than v1 (7 steps vs 10) — simpler approach, no combinatorial explosion

---

## Test 3: Trade-off Analysis
**Thread:** `test3_tradeoff_v2`
**Prompt:** "Compare adding the entity with the highest effectiveness vs the entity with the broadest specialty coverage. Which gives better coverage per provider, and why?"
**Status:** ✅ SUCCESS | 8 steps

**Result:**
| Entity | Strategy | Relevant Providers | Members Covered | Coverage/provider |
|--------|----------|-------------------|-----------------|-------------------|
| vamc (all) (eff 5.0) | Highest effectiveness | 41 | 22,334 | **544.7** |
| Henry Ford Health (46 specialties) | Broadest coverage | 379 | 18,513 | 48.8 |

Conclusion: Highest-effectiveness entity delivers **11x more coverage per provider**. Strategic concentration beats geographic redundancy.

**Observations:**
- Zero errors — clean run
- Agent pivoted from literal reading (chiropractor with eff 5.0) → filtered for GP/cardiology relevance → found `vamc (all)` as best high-eff entity
- Summarization triggered at step 8 — summary captured state accurately this time
- More thorough than v1 (8 steps vs 3) — explored more entities before concluding

---

## Test 4: Stress Test (Removal + Replacement)
**Thread:** `test4_stress_v2`
**Prompt:** "Simulate removing all currently contracted entities one by one. Which single entity's removal would cause the biggest coverage drop? Then suggest the best replacement."
**Status:** ⚠️ PARTIAL | 3 steps

**Result:** Empty network (0 contracted entities) — removal is a no-op. Agent correctly identified the edge case and asked for clarification (build network first, start with specific entities, or other).

**Observations:**
- 1 error: `KeyError: 'specialty'` on members_df (members don't have a specialty column) — recovered on retry
- No import errors, no case-sensitivity issues
- Cleanest stress test run — agent identified empty network in 3 steps vs 8 in v1
- No summarization triggered (under threshold)
- Agent didn't pivot to alternative analysis (unlike v1) — instead asked for clarification, which is arguably more correct behavior

---

## Test 5: Edge Case (Out-of-scope Specialties)
**Thread:** `test5_edge_case_v2`
**Prompt:** "Find entities that offer a specialty NOT in our current network scope. Could any of them be useful anyway? Rank them by how many members they'd cover if we expanded our scope."
**Status:** ✅ SUCCESS | 12 steps

**Result:** Top out-of-scope entities by member coverage (15mi threshold):
| Rank | Entity | Specialty | Coverage | Providers |
|------|--------|-----------|----------|-----------|
| 1 | care connection plus | gastroenterology | 96.5% | 9 |
| 2 | binsons hospital supplies inc | internal medicine | 93.2% | 20 |
| 3 | southfield city urgent care pc | internal medicine | 85.2% | 9 |
| 4 | drs harris birkhill wang songe | internal medicine | 83.1% | 9 |
| 5 | pinkus dermatopathology lab | dermatology | 81.3% | 4 |

**Observations:**
- `KeyError: 'specialty'` on members_df (members don't have specialty column) — recovered
- **3 `NameError` retries** from fresh sandbox losing variables (`result_df`, `promising_specs`, `results_df`) — agent had to rebuild entire pipeline each time
- `NameError: name 'next' is not defined` — `next()` not in sandbox builtins, agent switched to manual loop
- Eventually succeeded by rebuilding everything self-contained in one call
- **Summarization bug**: summary says "Exact coverage percentages are pending simulation execution" when they were computed
- 12 steps is expensive — most were retries from sandbox state loss

---

## Test 6: Build from Scratch (Greedy 90% Coverage)
**Thread:** `test6_greedy_v2`
**Prompt:** "Build a network from scratch (ignore current network). Starting with zero entities, greedily add entities one at a time based on marginal coverage gain. Stop when we hit 90%+ coverage everywhere. How many entities did it take?"
**Status:** ✅ SUCCESS | 11 steps

**Result:** Only **1 entity** needed — Henry Ford Health alone achieves ≥90% on all 10 coverable specialties:
| Specialty | Coverage |
|-----------|----------|
| Cardiology | 99.4% |
| Dermatology | 100.0% |
| Endocrinology | 90.3% |
| Gastroenterology | 100.0% |
| General Practice | 97.9% |
| Neurology | 99.9% |
| Orthopedic Surgery | 99.9% |
| Psychiatry | 99.9% |
| Pulmonology | 100.0% |
| Urology | 99.7% |

3 specialties (oncology, otolaryngology, radiology) have no exact matches in candidate data and remain at 0%.

**Observations:**
- `NameError: name 'coverable_thresholds' is not defined` — sandbox state loss, recovered on retry
- Henry Ford Health (5,135 providers) dominates Wayne County — greedy algorithm terminates at step 1
- Agent correctly identified the 3 uncoverable specialties and proceeded with the 10 coverable ones
- Summarization triggered at step 10, final answer at step 11
- Clean result — no case-sensitivity or dtype errors
