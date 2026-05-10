# Agent Instructions: Network Manager Agent

## Developer Commands
- **Activate venv**: `source .venv/bin/activate`
- **Run Agent**: `run-agent` (or `python -m network_manager_agent.main`)
- **CLI Options**:
    - `--base-url <url>`: Override LLM base URL
    - `--model <name>`: Override LLM model name
    - `--candidates <path>`: Custom candidates CSV path
    - `--members <path>`: Custom members CSV path
    - `--county-specialty-thresholds '<json>'`: REQUIRED — JSON threshold configuration
    - `--max-steps <N>`: Stop after N agent steps (useful for long-running prompts)
- **Session Management**:
    - List all saved threads: `python -m network_manager_agent.main --list-threads`
    - Run specific thread: `python -m network_manager_agent.main --thread-id <id>`
    - Clear specific thread: `python -m network_manager_agent.main --clear-thread <id>`
    - Reset all persistence: `python -m network_manager_agent.main --clear-all`
- **Run Tests**: `pytest`
- **Lint**: `ruff check .`
- **Type Check**: `mypy src/network_manager_agent`
- **Install**: `pip install -e ".[dev]"`

## Architecture & Key Files
- **Framework**: LangGraph (ReAct agent).
- **Persistence**: Uses `SqliteSaver` to persist agent state in `checkpoints.sqlite` via `thread_id`.
- **Core Logic**:
    - `src/network_manager_agent/__init__.py`: Package exports (`build_agent`, `AgentState`, `LLMConfig`, `create_llm`), version `0.1.0`.
    - `src/network_manager_agent/__main__.py`: Enables `python -m network_manager_agent` invocation.
    - `src/network_manager_agent/config.py`: `LLMConfig` dataclass (env-var configurable), constants (`SUMMARIZE_THRESHOLD=14`, `MESSAGES_TO_ARCHIVE=7`, `SERVICE_AREA_BUFFER_MILES=20`).
    - `src/network_manager_agent/data.py`: `DataManager` singleton with column normalization, coordinate normalization, entity aggregation, schema profiling, and service area filtering.
    - `src/network_manager_agent/graph.py`: Graph orchestration and flow.
    - `src/network_manager_agent/nodes.py`: Node implementations (LLM reasoning, tool execution, state updates, summarization).
    - `src/network_manager_agent/state.py`: Agent state definition (`AgentState` extends `MessagesState`).
    - `src/network_manager_agent/tools.py`: Tool definitions and logic.
    - `src/network_manager_agent/ui.py`: Streaming output to console and timestamped action log files.
- **Entry Point**: `src/network_manager_agent/main.py`.
- **Data**: Market data files in `data/raw/mi_market_data.csv` (candidates) and `data/raw/MedicareSampleCensus2023Q4.csv` (members).
- **Thresholds**: Nested JSON format `{"state": {"county": {"specialty": threshold_miles}}}`. Specialties and counties are case-insensitive. Example: `{"mi": {"wayne": {"general practice": 20.0, "cardiology": 10.0}}}`.
- **Interactive Dev**: `notebooks/react_agent.ipynb`.

## Key Logic & Patterns
- **DataManager Singleton**: Centralized data loading with `reset()` for testing. Handles column name synonym resolution (15 canonical columns), coordinate normalization (detects scaled integers, flips positive longitude), and service area geographic filtering via bounding box.
- **Entity-Level Aggregation**: Provider-level candidate data is aggregated into entity-level summaries (e.g., `provider_count`, `avg_effectiveness`, `avg_efficiency`, `specialties`, `new_patient_rate`, `geographic_reach`, claims volume distributions) before being presented to the agent.
- **System Prompt Structure**: `nodes.py` injects: ROLE, NETWORK SCOPE (dynamic thresholds), DATA (DataFrame column names), SANDBOX LIBRARIES (pre-imported modules + builtins), SANDBOX FUNCTIONS (`compute_coverage` signature/return schema), GUIDANCE (4 concise tips), and RULES (6 guardrails).
- **Sandbox Simulation**: The agent uses the `run_code` tool to evaluate network changes. It can manipulate `candidates_df`, `network_df`, `members_df` using pandas and compute coverage using the injected `compute_coverage` helper (uses `BallTree` for haversine distance queries).
- **Guidance**: 3 concise tips in the system prompt: (1) BallTree for haversine proximity checks, (2) compute_coverage for coverage simulation, (3) pandas filtering and aggregation patterns.
- **Core Tools**:
    - `run_code`: The primary tool for discovery, custom filtering, data analysis, and "what-if" network simulations. Injected state: `network`, `county_specialty_thresholds`. Allowed modules: pandas, numpy, json, math, functools, itertools, collections, sklearn.neighbors.BallTree. 60-second timeout.
    - `add_contract_entity`: Commits entities to the network. Uses `InjectedState("network")` to read current network. Case-insensitive matching, stores lowercase canonical name. Returns `added_entities`, `skipped_entities`, `errors`.
- **Graph Flow**:
    ```
    START -> network_manager
      -> (tools_condition) -> tools OR __end__
    tools -> update_state
      -> (should_summarize) -> summarize_messages OR network_manager
    summarize_messages -> network_manager
    ```
- **Summarization**: When message count exceeds `SUMMARIZE_THRESHOLD` (14), the `summarize_messages` node archives old messages into a running summary to manage context.
- **Streaming UI**: `ui.py` provides real-time console output and writes action logs (`logs/thread_<id>/log.txt` and `logs/thread_<id>/log.jsonl`) with per-step details. Default log directory uses `PROJECT_ROOT / "logs"` so logs are consistent regardless of CWD (CLI vs notebook).

## Important Notes
- **State**: `AgentState` extends LangGraph's `MessagesState` (not a plain `TypedDict`). Fields: `network` (accumulated entity IDs), `summary` (running summary string), `county_specialty_thresholds`.
- **Tools**: Only 2 tools exist: `run_code` and `add_contract_entity`. `compute_coverage` is a plain function injected into the `run_code` sandbox, not a standalone tool.
- **Data**: Data loading is centralized in `data.py` via the `DataManager` singleton. Accessors: `get_candidates_df()`, `get_members_df()`, `get_providers_by_entity()`.
- **Tests**: Comprehensive test suite across 6 files (113 tests): `test_data.py`, `test_graph.py`, `test_nodes.py`, `test_tools.py`, `test_ui_main.py`, with shared fixtures in `conftest.py`.
