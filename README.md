# Network Manager Agent

AI Agent that performs provider network management and optimization using LangGraph.

## Overview

This project implements a ReAct (Reasoning + Acting) agent that manages a healthcare provider network. The agent:

- Loads candidate providers and member location data.
- Uses an LLM to reason about which providers to add to the network based on coverage, quality, and accessibility.
- Tracks network coverage of members within a distance threshold.
- Simulates network changes to evaluate the impact of adding or removing entities before committing.

## Architecture

The agent is built with [LangGraph](https://langchain-ai.github.io/langgraph/) and consists of:

- **State**: Defines the agent's state including candidates, members, current network, and conversation history.
- **Tools**: `get_candidates`, `add_contract_entity`, `get_network_status`, `simulate_network_change`.
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

### Jupyter Notebook

Open the interactive notebook for exploration and testing:

```bash
cd notebooks
jupyter lab react_agent.ipynb
```

### CLI

```bash
python -m network_manager_agent.main
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
├── notebooks/              # Interactive development notebooks
│   └── react_agent.ipynb
├── data/                   # Data files
│   └── raw/
│       ├── mi_market_data.csv
│       └── MedicareSampleCensus2023Q4.csv
└── tests/                  # Test suite
    └── test_agent.py
```
