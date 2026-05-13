# Agent Testing Report

## Thresholds
```json
{"mi": {"wayne": {"general practice": 10.0, "cardiology": 15.0}, "washtenaw": {"general practice": 10.0, "cardiology": 15.0}}}
```

---

## Test 1: Multi-objective Optimization
**Thread:** `test1_pareto`
**Prompt:** "Find 3 entities that, when added together, maximize total coverage while minimizing the number of providers needed. Show the Pareto-optimal set."
**Status:** ✅ SUCCESS | 15 steps

**Result:** Pareto-optimal 3-entity combos identified. Best: `corewell health` + `trinity health` + `cvs health` (3,330 relevant providers, 99.5% coverage, 23,987/24,102 members).

**Pareto Frontier (6 solutions):**
| Rank | Entities | Coverage | Relevant Providers |
|------|----------|----------|-------------------|
| 1 | corewell health + trinity health + cvs health | 99.5% | 3,330 |
| 2 | vamc (all) + trinity health + corewell health | 99.4% | 3,357 |
| 3 | vamc (all) + corewell health + cvs health | 99.0% | 2,466 |
| 4 | corewell health + care connection plus + cvs health | 98.8% | 2,676 |
| 5 | henry ford health + trinity health + cvs health | 96.9% | 2,322 |
| 6 | vamc (all) + care connection plus + cvs health | 96.3% | 690 |

**Observations:**
- 1 import error (tried `import math`) — recovered on retry
- Summarization triggered at step 15 — summary captured state accurately
- Agent correctly identified 395 entities with relevant providers from 8,609 total
- Evaluated 26+ unique combinations using BallTree heuristic + compute_coverage validation

---

## Test 2: Gap Analysis + Before/After
**Thread:** `test2_gap_analysis`
**Prompt:** "Identify the geographic areas (counties) with the lowest coverage, then for each gap find the single best entity to fill it. Show a before/after comparison."
**Status:** ✅ SUCCESS | 11 steps

**Result:** 2 entities filled all 4 gaps:
| Gap | Best Entity | Coverage |
|-----|-------------|----------|
| wayne / general practice | corewell health | 99.4% |
| wayne / cardiology | corewell health | 100.0% |
| washtenaw / general practice | cvs health | 95.5% |
| washtenaw / cardiology | corewell health | 89.3% |

**Observations:**
- 1 import error (tried `import math`) — recovered on retry
- 1 `NameError: name 'result' is not defined` — sandbox state loss, recovered on retry
- 1 `TypeError: the JSON object must be str, bytes or bytearray, not dict` — recovered on retry
- Agent identified 111 Wayne GP, 27 Washtenaw GP, 159 Wayne Cardiology, 90 Washtenaw Cardiology nearby entities
- corewell health + cvs health covered all 4 gaps with 2 entities

---

## Test 3: Trade-off Analysis
**Thread:** `test3_tradeoff`
**Prompt:** "Compare adding the entity with the highest effectiveness vs the entity with the broadest specialty coverage. Which gives better coverage per provider, and why?"
**Status:** ✅ SUCCESS | 4 steps

**Result:**
| Entity | Strategy | Providers | Members Covered | Coverage/Provider |
|--------|----------|-----------|-----------------|-------------------|
| wise transition llc (eff 5.0) | Highest effectiveness | 1 | 0 | **0** |
| henry ford health (46 specialties) | Broadest coverage | 5,135 | 18,513 | 3.6 |

Conclusion: Henry Ford Health is vastly superior. The "highest effectiveness" entity (wise transition llc) has 1 provider in outpatient behavioral health — not in our target specialties. Effectiveness scores are meaningless without specialty alignment.

**Observations:**
- Zero errors — cleanest run
- Agent immediately identified the mismatch and explained why
- Fastest test (4 steps) — straightforward comparison

---

## Test 4: Stress Test (Removal + Replacement)
**Thread:** `test4_stress`
**Prompt:** "Simulate removing all currently contracted entities one by one. Which single entity's removal would cause the biggest coverage drop? Then suggest the best replacement."
**Status:** ⚠️ PARTIAL | 3 steps

**Result:** Empty network (0 contracted entities) — removal is a no-op. Agent correctly identified the edge case and asked for clarification (build network first, start with specific entities, or other).

**Observations:**
- Zero errors — clean run
- Agent identified empty network in 3 steps
- Asked for clarification rather than pivoting — correct behavior for edge case

---

## Test 5: Edge Case (Out-of-scope Specialties)
**Thread:** `test5_edge_case`
**Prompt:** "Find entities that offer a specialty NOT in our current network scope. Could any of them be useful anyway? Rank them by how many members they'd cover if we expanded our scope."
**Status:** ✅ SUCCESS | 10 steps

**Result:** Top out-of-scope specialties by member coverage (10mi threshold):
| Rank | Specialty | Entities | Coverage |
|------|-----------|----------|----------|
| 1 | clinical social work | 1,904 | 100.0% |
| 2 | outpatient behavioral health | 1,381 | 100.0% |
| 3 | clinical psychology | 1,053 | 100.0% |
| 4 | chiropractor | 914 | 100.0% |
| 5 | family practice | 913 | 100.0% |

Top 5 entities per high-value specialty (e.g. OB/GYN: 99.15%, General Surgery: 88.46%, Psychiatry: 72.14%).

**Observations:**
- 1 import error (tried `import json`) — recovered on retry
- 1 `NameError: name 'out_of_scope_specialties' is not defined` — sandbox state loss, recovered on retry
- 290 entities offer BOTH in-scope and out-of-scope specialties (natural expansion candidates)
- Behavioral health specialties dominate the high-coverage tier
- Summarization triggered at step 10

---

## Test 6: Build from Scratch (Greedy 90% Coverage)
**Thread:** `test6_greedy`
**Prompt:** "Build a network from scratch (ignore current network). Starting with zero entities, greedily add entities one at a time based on marginal coverage gain. Stop when we hit 90%+ coverage everywhere. How many entities did it take?"
**Status:** ✅ SUCCESS | 14 steps

**Result:** **9 entities** achieved ≥90% on all 4 targets:
| Entity | Providers |
|--------|-----------|
| mymichigan health | 8,684 |
| corewell health | 8,506 |
| trinity health | 3,890 |
| henry ford health | 5,135 |
| cvs health | 290 |
| halo medical group pllc | 28 |
| advanced heart & vascular care pllc | 19 |
| eastlake cardiovascular pc | 55 |
| promedica health | 1,579 |

**Final coverage:**
| Specialty | Coverage |
|-----------|----------|
| wayne / general practice | 99.4% |
| wayne / cardiology | 100.0% |
| washtenaw / general practice | 97.9% |
| washtenaw / cardiology | 100.0% |

**Observations:**
- 1 import error (tried `import json`, `from scipy.spatial import cKDTree`, `from sklearn.neighbors import BallTree`) — recovered on retry
- 1 `NameError: name 'hasattr' is not defined` — `hasattr()` not in sandbox builtins, recovered on retry
- Agent pre-computed BallTree coverage for all 8,609 entities, then ran greedy selection
- Checkpoint save crashed with `TypeError: Type is not msgpack serializable: numpy.int64` — entities were still added successfully
- Summarization triggered at step 14
