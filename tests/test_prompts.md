# Agent Test Prompt Results

## Bugs Fixed During Testing

### `ui.py:43` — `None` output crash
- **Symptom**: `TypeError: argument of type 'NoneType' is not iterable` when `update_state` returns `{}`
- **Fix**: Added `if output is None: continue` guard in stream loop

### `tools.py:208` — Missing sandbox builtins
- **Symptom**: `NameError: name 'sorted' is not defined`, `NameError: name 'len' is not defined`
- **Root cause**: `__builtins__` was set to `{}` in the `run_code` sandbox
- **Fix**: Populated `__builtins__` with essential functions: `len`, `sorted`, `range`, `str`, `int`, `float`, `bool`, `set`, `list`, `dict`, `tuple`, `enumerate`, `zip`, `map`, `filter`, `isinstance`, `type`, `print`, `abs`, `round`, `min`, `max`, `sum`, `any`, `all`, common exceptions

### `tools.py:208` — Missing `__import__` in sandbox
- **Symptom**: `ImportError: __import__ not found` when agent tries `import pandas as pd`
- **Fix**: Added `__import__` to `__builtins__`

### `tools.py` — `print()` returns `None` / `null`
- **Symptom**: Agent uses `print()` for exploration, tool returns `null`, forcing retries
- **Fix**: Capture `sys.stdout` in a `StringIO` buffer, prepend output as `[stdout]...[/stdout]` to tool result

### `nodes.py` — System prompt missing canonical column info
- **Symptom**: Agent used `Primary Institutional Affiliation` instead of `entity`, tried city-matching instead of county
- **Fix**: Added explicit "CANONICAL COLUMNS" section listing all columns in `candidates_df` and `members_df`, with notes about what's NOT available (e.g., `county` not in candidates, `city` not in members)

### `nodes.py` — Agent re-explores schema despite it being in the prompt
- **Symptom**: Agent wastes tool calls checking column names that are already documented
- **Fix**: Added Rule 9: "The schema is provided above — do NOT waste tool calls exploring column names or data structure."

### `nodes.py` — `run_code` state doesn't persist between calls
- **Symptom**: Agent tried to reuse `both_entities` variable from a previous `run_code` call → `NameError`
- **Root cause**: Each `run_code` invocation is a fresh sandbox; no cross-call state
- **Fix**: Added Rule 10: "each run_code call is a fresh sandbox. Variables from a previous run_code call are NOT available in the next call."

### `tools.py` — 20s timeout too short for complex geographic queries
- **Symptom**: Agent timed out during legitimate multi-step optimization (BallTree on 65k members)
- **Decision**: Not a prompt issue — agent was doing productive exploration. Expanded timeout from 20s → 60s.

### `nodes.py` — Empty LLM response causes silent __end__
- **Symptom**: Model returned empty AI message after tool calls → `tools_condition` routed to `__end__`
- **Fix**: Added retry loop (up to 3 attempts) in `network_manager` node with nudge messages on empty response

### `data.py:135` — `specialties` column too verbose (2000+ chars per entity)
- **Symptom**: Entity summaries include ALL specialties per entity as full list → massive context noise
- **Fix**: Changed aggregation to top 10 by frequency + "... and N more" suffix. ~220 chars vs ~2000 before.

### `tools.py:293` — No stdout cap in run_code sandbox
- **Symptom**: Agent can dump entire DataFrames to stdout, flooding context
- **Fix**: Truncate stdout at 2000 chars with `"... [output truncated, N more chars]"` suffix

### `nodes.py:197` — Summarization loses quantitative results
- **Symptom**: Archived messages lose exact coverage %, entity names, best combinations
- **Fix**: Enhanced summarization instruction with Key Findings section, explicit rules to preserve numbers, entity metrics, and best combinations

---

## Level 1 — Basic Exploration

**Thresholds**: `{"MI": {"washtenaw": {"cardiology": 10.0, "general practice": 20.0}}}`

### Prompt 1: "What are the top 5 entities by provider count?"
- **Status**: ✅ Success (after ui.py fix)
- **Tool calls**: 1 (`run_code` → `entity_summaries_df.nlargest(5, 'provider_count')`)
- **Response cycles**: 1 tool call → 1 final response
- **Observations**:
  - Used correct data source (`entity_summaries_df`)
  - Noted relevance to configured thresholds (cardiology/general practice)
  - Clean table presentation with key observations
  - No redundant reasoning

### Prompt 2: "What specialties are available in the candidate data?"
- **Status**: ✅ Success (after sandbox builtins fix)
- **Tool calls**: 1 (`run_code` → `candidates_df['specialty'].unique()`)
- **Response cycles**: 1 tool call → 1 final response
- **Observations**:
  - Required 2 retries before builtins fix (tried `sorted()`, then `len()`, then `.__len__()`)
  - After fix: clean single-attempt execution
  - Returned all 46 specialties, noted alignment with configured thresholds

---

## Level 2 — Single-Goal Analysis

### Prompt 1: "Find the top 5 entities with cardiology providers in Washtenaw county, ranked by average effectiveness."
- **Status**: ✅ Success (after multiple fixes)
- **Tool calls**: 3 (after fixes)
- **Response cycles**: 3 tool calls → 1 final response
- **Observations**:
  - Initial attempt: Used wrong column (`Primary Institutional Affiliation`), tried city-matching instead of county → 6 tool calls, timeout
  - Second attempt: Used centroid distance approach → worked but inefficient (5 tool calls)
  - Final attempt: Used zip code matching between members_df and candidates_df → correct approach (3 tool calls)
  - Agent still wastes calls: First tried `city` on `members_df` (doesn't exist), then re-explored columns
  - Result is reasonable but approach (zip matching) is lossy — providers in Washtenaw zips ≠ providers physically in Washtenaw county
  - Final answer: 457 cardiology providers, 26 entities, top 5 all at 5.0 or 4.67 effectiveness

### Prompt 2: "Which entities have the highest new patient rate?"
- **Status**: ⬜ Pending

---

## Level 3 — Multi-Step Reasoning

### Prompt 1: "Simulate adding the top 3 cardiology entities (using a weighted score balanced for effectiveness and efficiency) to the network and measure the coverage improvement for cardiology in Washtenaw county."
- **Status**: ✅ Success
- **Tool calls**: 2 (identify top 3 + simulate coverage, then check baseline)
- **Response cycles**: 2 tool calls → 1 final response
- **Observations**:
  - Correctly used composite scoring (effectiveness × efficiency × log(provider_count))
  - Used `compute_coverage` with simulated network (concat existing + new candidates)
  - Compared baseline (0%) → simulated (66.84%) correctly
  - Clean presentation with improvement metrics

### Prompt 2: "Compare the coverage impact of adding Covenant Healthcare vs Corewell Health."
- **Status**: ✅ Success
- **Tool calls**: 1 (single simulation with both entities)
- **Response cycles**: 1 tool call → 1 final response
- **Observations**:
  - Efficiently simulated both entities in one tool call
  - Handled whitespace in entity names (`.str.strip()`)
  - Provided clear comparison table with actionable recommendation
  - Noted diminishing returns of adding both

---

## Level 4 — Network-Aware Optimization

### Prompt 1: "Recommend an entity that would provide the best coverage to the network with an avg effectiveness of at least 3.0."
- **Status**: ✅ Success (run 2, with thresholds)
- **Tool calls**: 3 (baseline + entity list → specialty filter → simulation loop)
- **Response cycles**: 3 tool calls → 1 final response
- **Observations**:
  - Correctly checked existing network first (empty baseline, 0% across all 5 county/specialty slots)
  - Filtered 4019 entities with eff >= 3.0, identified 30 top candidates with required specialties
  - Simulated 8 entities in one tool call (MyMichigan, Corewell, VAMC, Care Connection Plus, Wayne State, Concentra, CVS, Community Health)
  - Used `pd.merge()` for coverage delta comparison — correct pattern
  - Recommended **Corewell Health** (eff 3.13, 8506 providers, +263.77 coverage points)
  - Provided runner-ups with comparison data
  - Did NOT auto-add — followed "recommend only" instruction
  - Total coverage: cardio 51.1%, GP 52.4%, ortho 56.7% across both counties

---

### BallTree Benchmark Results
- `bench_balltree.py` — measures BallTree performance on full vs filtered member sets
- **Full set** (2.2M members): avg query ~1.9s per entity, build ~2ms
- **Filtered set** (65.8K members, washtenaw + wayne): avg query ~1.1s per entity, build ~2ms
- **Conclusion**: BallTree is fast. With 6 specialty queries per `compute_coverage` call (~6s total), well within 60s sandbox timeout. Agent can do ~10 coverage calls per execution.

### Prompt 2: "Build a network of no more than 5 entities that maximizes coverage for the network. Check what's already in the network first, and only consider entities with average effectiveness >= 3.0."
- **Status**: ⬜ Pending
- **Tests**: Network awareness, multi-specialty discovery, entity limit, effectiveness floor, incremental optimization

---

### Legacy L4 Prompts (superseded)

#### Prompt L1: "Build a network that covers both cardiology and general practice in Washtenaw county using no more than 5 entities, prioritizing effectiveness above 4.0."
- **Status**: ✅ Success (after empty-response guard + summarization fix)
- **Tool calls**: 11 (including 1 retry on empty response, 1 BallTree API error)
- **Response cycles**: 11 tool calls → 1 add_contract_entity → 1 verification → final summary
- **Observations**:
  - Agent followed systematic approach: explore → identify candidates → simulate → optimize → commit
  - Found 4 entities with BOTH specialties near Washtenaw (Henry Ford, MyMichigan, Trinity, VAMC)
  - Systematically tested 5th entity options (10 candidates), selected McLaren Health (70.0% cardio, 69.3% GP)
  - Summarization fired twice (at 14 messages) — preserved key data with enhanced instructions
  - Empty response guard worked: model returned empty once, retry succeeded
  - BallTree `query()` API error (wrong `r` param) — agent adapted with `query_radius()` on retry
  - Final result: 5 entities, 20,741 providers, 70.0% cardiology, 69.3% general practice
  - Note: summary at end still showed outdated "best combo" from pre-summarization (CVS Health → 67.5%/79.6%)
  - Trailing spaces in entity names ("Corewell Health ") handled correctly by agent

#### Prompt L2: "Find the minimum number of entities needed to reach 80% cardiology coverage in Washtenaw, and add them."
- **Status**: ⬜ Pending

---

## Level 5 — Adversarial / Edge Cases

### Prompt 1: "Add the entity 'Nonexistent Health System 12345' to the network."
- **Status**: ⬜ Pending

### Prompt 2: "What's the coverage for a specialty that doesn't exist in the data, like 'astrology'?"
- **Status**: ⬜ Pending

---

## Level 6 — Full Workflow

### Prompt 1: "Analyze the current coverage gaps, recommend entities to fill them, add your top recommendation, then verify the new coverage."
- **Status**: ⬜ Pending
