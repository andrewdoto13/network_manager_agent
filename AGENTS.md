# Agent Instructions: Network Manager Agent

## Developer Commands
- **Activate venv**: `source .venv/bin/activate`
- **Run Agent**: `run-agent` (or `python -m network_manager_agent.main`)
- **Generate Data**: `generate-data` (or `python -m scripts.generate_data`)
- **Run Tests**: `pytest`
- **Install**: `pip install -e ".[dev]"`

## Architecture & Key Files
- **Framework**: LangGraph (ReAct agent).
- **Core Logic**:
    - `src/network_manager_agent/graph.py`: Graph orchestration and flow.
    - `src/network_manager_agent/nodes.py`: Node implementations (LLM reasoning, tool execution).
    - `src/network_manager_agent/state.py`: Agent state definition (`TypedDict`).
    - `src/network_manager_agent/tools.py`: Tool definitions and logic.
    - `src/network_manager_agent/data.py`: Data loading utilities.
- **Configuration**: `src/network_manager_agent/config.py`.
- **Entry Point**: `src/network_manager_agent/main.py`.
- **Data**: Market data files in `data/raw/mi_market_data.csv` (candidates) and `data/raw/members.csv` (members).
- **Interactive Dev**: `notebooks/react_agent.ipynb`.

## Key Logic & Patterns
- **Entity-Level Aggregation**: Provider-level candidate data is aggregated into entity-level summaries (e.g., `avg_effectiveness`, `new_patient_rate`, `geographic_reach`) before being presented to the agent via `get_candidates`.
- **Schema Injection**: To reduce tool-call overhead, a statistical profile of the candidate data is computed via `get_candidate_schema_profile` and injected directly into the system prompt in `nodes.py`.
- **Simulation-First Workflow**: The agent is encouraged to use `simulate_network_change` with `compare_scenarios` to evaluate and rank potential additions/removals before using `add_contract_entity`.
- **Core Tools**:
    - `get_candidates`: Filters and retrieves high-quality candidate entities.
    - `add_contract_entity`: Commits entities to the network.
    - `get_network_status`: Source of truth for current member coverage.
    - `simulate_network_change`: Evaluates potential changes without modifying state.

## Important Notes
- **State**: The agent state is a `TypedDict` defined in `state.py`.
- **Tools**: Tools are defined in `tools.py` and invoked within nodes in `nodes.py`.
- **Data**: Data loading is centralized in `data.py`.
