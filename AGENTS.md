# Agent Instructions: Network Manager Agent

## Developer Commands
- **Activate venv**: `source .venv/bin/activate`
- **Run Agent**: `run-agent` (or `python -m network_manager_agent.main`)
- **Session Management**:
    - List all saved threads: `python -m network_manager_agent.main --list-threads`
    - Run specific thread: `python -m network_manager_agent.main --thread-id <id>`
    - Clear specific thread: `python -m network_manager_agent.main --clear-thread <id>`
    - Reset all persistence: `python -m network_manager_agent.main --clear-all`
- **Run Tests**: `pytest`
- **Install**: `pip install -e ".[dev]"`

## Architecture & Key Files
- **Framework**: LangGraph (ReAct agent).
- **Persistence**: Uses `SqliteSaver` to persist agent state in `checkpoints.sqlite` via `thread_id`.
- **Core Logic**:
    - `src/network_manager_agent/graph.py`: Graph orchestration and flow.
    - `src/network_manager_agent/nodes.py`: Node implementations (LLM reasoning, tool execution).
    - `src/network_manager_agent/state.py`: Agent state definition (`TypedDict`).
    - `src/network_manager_agent/tools.py`: Tool definitions and logic.
    - `src/network_manager_agent/data.py`: Data loading utilities.
- **Configuration**: `src/network_manager_agent/config.py`.
- **Entry Point**: `src/network_manager_agent/main.py`.
- **Data**: Market data files in `data/raw/mi_market_data.csv` (candidates) and `data/raw/MedicareSampleCensus2023Q4.csv` (members).
- **Thresholds**: Nested JSON format `{"state": {"county": {"specialty": threshold_miles}}}`. Specialties and counties are case-insensitive. Example: `{"mi": {"wayne": {"general practice": 20.0, "cardiology": 10.0}}}`.
- **Interactive Dev**: `notebooks/react_agent.ipynb`.

## Key Logic & Patterns
- **Entity-Level Aggregation**: Provider-level candidate data is aggregated into entity-level summaries (e.g., `avg_effectiveness`, `new_patient_rate`, `geographic_reach`) before being presented to the agent.
- **Schema Injection**: To reduce tool-call overhead, a statistical profile of the candidate data is computed via `get_candidate_schema_profile` and injected directly into the system prompt in `nodes.py`.
- **Sandbox Simulation**: The agent uses the `run_code` tool to evaluate network changes. It can manipulate `candidates_df` and `network_df` using pandas and compute coverage using the injected `compute_coverage` helper.
- **Core Tools**:
    - `run_code`: The primary tool for discovery, custom filtering, data analysis, and "what-if" network simulations.
    - `add_contract_entity`: Commits entities to the network.
    - `get_network_status`: Source of truth for current member coverage.
    - `get_candidates`: (Internal helper) Filters and retrieves high-quality candidate entities.

## Important Notes
- **State**: The agent state is a `TypedDict` defined in `state.py`.
- **Tools**: Tools are defined in `tools.py` and invoked within nodes in `nodes.py`.
- **Data**: Data loading is centralized in `data.py`.
