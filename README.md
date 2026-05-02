# Network Manager Agent

AI Agent that performs provider network management and optimization using LangGraph.

## Overview

This project implements a ReAct (Reasoning + Acting) agent that manages a healthcare provider network. The agent:

- Loads candidate providers and member location data.
- Uses an LLM to reason about which providers to add to the network based on coverage, quality, and accessibility.
- Tracks network coverage of members within a distance threshold.
- Performs complex data analysis and "what-if" network simulations using a sandboxed pandas execution environment.

## Architecture

The agent is built with [LangGraph](https://langchain-ai.github.io/langgraph/) and consists of:

- **State**: Defines the agent's state including candidates, members, current network, and conversation history.
- **Persistence**: Uses `SqliteSaver` to persist session state in `checkpoints.sqlite`, allowing conversations to be resumed via `thread_id`.
- **Tools**: 
    - `run_code`: A powerful pandas sandbox used for discovery, custom filtering, and simulating coverage impact.
    - `add_contract_entity`: Commits entities to the network.
    - `get_network_status`: Source of truth for current member coverage.
- **Nodes**: `network_manager` (LLM reasoning), `tools` (tool execution), `update_state`, `summarize_messages` (context management).
- **Graph**: START → network_manager → [tools → update_state → {summarize_messages | continue}] → END.

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
```

### Session Management

The agent supports persistent threads. You can resume a previous conversation or start a fresh one using the following flags:

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
Includes provider-level metrics such as `Effectiveness`, `Efficiency`, `Medicare New Patient Claims`, and `Total Claims Volume`. These are aggregated at the entity level for the agent's decision-making.

### Members (`data/raw/MedicareSampleCensus2023Q4.csv`)
Contains Medicare provider location data with coordinates, county (`countyname`), and state information.

## Project Structure

```
.
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── AGENTS.md               # Agent instructions
├── src/                    # Source code
│   └── network_manager_agent/
│       ├── config.py       # LLM and agent configuration
│       ├── data.py         # Data loading utilities
│       ├── graph.py        # Graph orchestration and flow
│       ├── main.py         # Entry point
│       ├── nodes.py        # Node implementations
│       ├── state.py        # Agent state definition
│       └── tools.py        # Tool definitions and logic
│       ├── checkpoints.sqlite # Persistence database (generated at runtime)
├── notebooks/              # Interactive development notebooks
│   └── react_agent.ipynb
├── data/                   # Data files
│   └── raw/
│       ├── mi_market_data.csv
│       └── MedicareSampleCensus2023Q4.csv
└── tests/                  # Test suite
    └── test_agent.py
```
