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
- **Data**: Market data files in `data/raw/` (candidates) and `data/raw/members.csv` (members).
- **Interactive Dev**: `notebooks/react_agent.ipynb`.

## Important Notes
- **State**: The agent state is a `TypedDict` defined in `state.py`.
- **Tools**: Tools are defined in `tools.py` and invoked within nodes in `nodes.py`.
- **Data**: Data loading is centralized in `data.py`.
