# Network Manager Agent — Multi-Path Coverage Analysis

## The Setup

**Goal:** Reach 90% General Practice coverage in Wayne County using two distinct strategies.

**The Prompt:**

> "I need to reach 90% General Practice coverage in Wayne County. Show me 2 different paths to get there — one that minimizes the number of entities added, and one that prioritizes high-effectiveness providers. Compare them and recommend which approach is better."

---

## What the Agent Did

The agent ran a **6-step reasoning loop** across multiple iterations:

### Phase 1 — Explore the landscape

Fetched candidates sorted by **effectiveness** (small clinics, 5.0 scores) and by **total claims** (massive health systems like Corewell, Henry Ford, Trinity) to understand the trade-off space.

### Phase 2 — Test hypotheses

Ran `compare_scenarios` to evaluate:

- Corewell Health alone → **88.35%**
- Corewell + Southfield Urgent Care → **89.8%**
- 5 high-effectiveness entities → **70.05%**

### Phase 3 — Iterate toward the target

Added a 3rd entity to the Corewell + Southfield base:

- + NEXTCARE → **90.8%**
- + Motor City Medical → **90.3%**

### Phase 4 — Hit a tool limit and adapt

Tried testing 10 small entities at once for Path 2 — hit the `simulate_network_change` tool's 5-entity-per-scenario limit, got rejected, then backed off and refined. This is a tool constraint, not a reasoning failure.

---

## The Results

| | **Path 1: Fewest Entities** | **Path 2: High Effectiveness Only** |
|---|---|---|
| **Entities** | 3 | 5+ |
| **Coverage** | **90.8%** (1,816 members) | **70.05%** (1,401 members) |
| **Avg Effectiveness** | Mixed (2.77–5.0) | 5.0 across the board |
| **Verdict** | Achieves 90% target | Falls short |

**Winner: Path 1** — Corewell Health (8,506 providers) + Southfield Urgent Care (37 providers) + NEXTCARE (4 providers).

**Key insight:** High-effectiveness clinics are too small and geographically limited to reach 90% on their own. A hybrid approach — one massive network for base coverage, two perfect-score clinics to fill gaps — is the only viable path.

---

## Assessment

### What Worked Well

- The agent independently explored both strategies before comparing — it didn't just pick one and run with it.
- It discovered that Path 2 was fundamentally infeasible and reported it honestly, rather than forcing a false conclusion.
- It iterated incrementally toward the 90% target, adding entities one at a time to find the minimum count.
- The final recommendation was nuanced — acknowledging the trade-off between quality and scale.

### Where It Stumbled

- Took 6 tool calls to converge — one attempt was blocked by the `simulate_network_change` tool's 5-entity-per-scenario limit. The agent tried to test 10 entities at once to be thorough; the tool rejected it, and the agent adapted. This is a tool constraint, not a reasoning failure.
- It didn't anticipate the Path 2 constraint earlier; it could have estimated coverage from provider counts before simulating.
- The "high effectiveness" path was tested with only one batch of 5 entities, not enough to conclusively prove it can't reach 90% (though the 70% result is suggestive).

### Overall

Solid multi-step reasoning for a non-trivial optimization problem. The agent showed genuine exploration, hypothesis testing, and course correction — not just a linear tool-calling pattern.
