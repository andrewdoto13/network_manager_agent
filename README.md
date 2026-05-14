# Network Manager Agent

AI Agent that performs provider network management and optimization using LangGraph.

## Overview

This project implements a ReAct (Reasoning + Acting) agent that manages a healthcare provider network. The agent:

- Loads candidate providers and member location data with robust column/coordinate normalization.
- Uses an LLM to reason about which providers to add to the network based on coverage, quality, and accessibility.
- Tracks network coverage of members within configurable distance thresholds.
- Performs complex data analysis and "what-if" network simulations using a sandboxed pandas execution environment.
- Persists intermediate results across code calls via `sandbox_cache` for multi-step analysis workflows.
- Streams real-time output to the console and writes timestamped action logs.

## Architecture

The agent is built with [LangGraph](https://langchain-ai.github.io/langgraph/) and consists of:

- **State**: `AgentState` extends `MessagesState` with fields for `network` (accumulated entity IDs), `summary`, `county_specialty_thresholds`, and `sandbox_cache` (persistent dict for cross-call data).
- **Persistence**: Uses `SqliteSaver` to persist session state in `checkpoints.sqlite`, allowing conversations to be resumed via `thread_id`.
- **Tools**:
    - `run_code`: A pandas sandbox for discovery, custom filtering, and simulating coverage impact. Injects `candidates_df`, `network_df`, `members_df`, `thresholds`, `sandbox_cache`, and `compute_coverage`. Supports pandas, numpy, sklearn.neighbors.BallTree, defaultdict, functools, and standard library modules. 60-second timeout. Each call is a fresh sandbox — only `sandbox_cache` persists between calls.
    - `add_contract_entity`: Commits validated entities to the network, skipping duplicates.
- **Nodes**: `network_manager` (LLM reasoning with 3x retry on empty responses), `tools` (tool execution via `execute_tools` wrapper), `update_state` (extracts added entity IDs + merges sandbox_cache), `summarize_messages` (context management).
- **Graph**: START -> network_manager -> [tools -> update_state -> {summarize_messages | continue}] -> END.
- **Summarization**: Automatically triggers when message count exceeds 14, archiving older messages to preserve context. Uses `_get_anchor_message` to resume conversations after summarization.
- **UI**: `ui.py` provides streaming console output and timestamped action log files (`logs/thread_<id>/log.txt` and `logs/thread_<id>/log.jsonl`).

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Usage

### CLI

The agent can be run in interactive mode or with a direct prompt:

```bash
# Interactive mode
python -m network_manager_agent.main

# Direct prompt
python -m network_manager_agent.main "Find candidates in Wayne county with effectiveness > 80%"

# With custom options
python -m network_manager_agent.main --model gpt-4 --thread-id my_session --county-specialty-thresholds '{"mi": {"wayne": {"general practice": 20.0}}}' "..."
```

### CLI Options

| Flag | Description |
|---|---|
| `--base-url <url>` | Override LLM base URL |
| `--model <name>` | Override LLM model name |
| `--candidates <path>` | Custom candidates CSV path |
| `--members <path>` | Custom members CSV path |
| `--county-specialty-thresholds '<json>'` | JSON threshold configuration |
| `--thread-id <id>` | Conversation thread ID (default: "1") |
| `--list-threads` | List all existing threads |
| `--clear-thread <id>` | Delete a specific thread |
| `--clear-all` | Wipe the entire persistence database |
| `--max-steps <N>` | Stop after N agent steps |

### Session Management

The agent supports persistent threads. You can resume a previous conversation or start a fresh one:

```bash
# Run a specific thread (or create it if it doesn't exist)
python -m network_manager_agent.main --thread-id my_test_session "..."

# List all existing threads in the database
python -m network_manager_agent.main --list-threads

# Clear a specific thread
python -m network_manager_agent.main --clear-thread my_test_session

# Wipe the entire persistence database
python -m network_manager_agent.main --clear-all
```

### Jupyter Notebook

Open the interactive notebook for exploration and testing:

```bash
cd notebooks
jupyter lab react_agent.ipynb
```

## Data Format

### Candidates (`data/raw/mi_market_data.csv`)
Provider-level data with metrics such as `Effectiveness`, `Efficiency`, `Medicare New Patient Claims`, and `Total Claims Volume`. The `DataManager` normalizes column names via synonym resolution, handles scaled integer coordinates, and aggregates provider-level data into entity-level summaries with 13+ aggregate metrics.

### Members (`data/raw/MedicareSampleCensus2023Q4.csv`)
Contains member location data with coordinates, county (`countyname`), and state information.

### Thresholds
Nested JSON format: `{"state": {"county": {"specialty": threshold_miles}}}`. Specialties and counties are case-insensitive. Example: `{"mi": {"wayne": {"general practice": 20.0, "cardiology": 10.0}}}`.

## Sandbox Cache

The `sandbox_cache` is a persistent dictionary that survives across `run_code` calls. Use it to save intermediate results, cache expensive computations, or pass data between analysis steps:

```python
# Call 1: compute and save
entity_stats = candidates_df.groupby("entity").agg(
    count=("entity", "count"),
    avg_eff=("effectiveness", "mean")
).reset_index()
sandbox_cache["entity_stats"] = entity_stats.to_dict("records")

# Call 2: retrieve and use
entity_stats = pd.DataFrame(sandbox_cache.get("entity_stats", []))
top_entities = entity_stats.nlargest(5, "avg_eff")["entity"].tolist()
```

Only JSON-serializable types are supported. The cache is merged into agent state after each `run_code` call.

## Development

```bash
# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src/network_manager_agent
```

## Project Structure

```
.
├── pyproject.toml          # Project configuration, dependencies, ruff/mypy settings
├── README.md               # This file
├── AGENTS.md               # Agent instructions for CLI agents
├── checkpoints.sqlite      # Persistence database (generated at runtime)
├── src/
│   └── network_manager_agent/
│       ├── __init__.py     # Package exports: build_agent, AgentState, LLMConfig, create_llm
│       ├── __main__.py     # python -m support
│       ├── config.py       # LLMConfig, thresholds, constants
│       ├── data.py         # DataManager singleton, aggregation, schema profiling
│       ├── graph.py        # LangGraph workflow construction
│       ├── main.py         # CLI entry point
│       ├── nodes.py        # Node implementations (reasoning, tools, summarization)
│       ├── state.py        # AgentState definition (extends MessagesState), _to_native helper
│       ├── tools.py        # Tool definitions: run_code, add_contract_entity
│       └── ui.py           # Streaming output and action log files
├── notebooks/
│   └── react_agent.ipynb   # Interactive development notebook
├── data/
│   └── raw/
│       ├── mi_market_data.csv
│       └── MedicareSampleCensus2023Q4.csv
├── logs/                   # Per-thread action logs (log.txt, log.jsonl)
├── reports/                # Agent run evaluation reports
├── scripts/                # Utility scripts (reserved)
└── tests/
    ├── conftest.py         # Shared test fixtures
    ├── test_data.py        # DataManager, aggregation, schema tests
    ├── test_graph.py       # Graph construction and routing tests
    ├── test_nodes.py       # Node function tests
    ├── test_tools.py       # Tool and coverage computation tests
    └── test_ui_main.py     # UI streaming and CLI tests
```
