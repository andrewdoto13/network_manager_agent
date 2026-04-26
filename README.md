# Network Manager Agent

AI Agent that performs provider network management and optimization using LangGraph.

## Overview

This project implements a ReAct (Reasoning + Acting) agent that manages a healthcare provider network. The agent:

- Loads candidate providers (hospitals) and member location data
- Uses an LLM to reason about which providers to add to the network
- Tracks network coverage of members within a distance threshold
- Optimizes provider selection based on criteria like effectiveness ratings

## Architecture

The agent is built with [LangGraph](https://langchain-ai.github.io/langgraph/) and consists of:

- **State**: Defines the agent's state including candidates, members, current network, and conversation history
- **Tools**: `get_candidates`, `get_candidate_schema`, `add_provider`, `get_network_status`
- **Nodes**: `network_manager` (LLM reasoning), `tools` (tool execution), `update_state`, `summarize_messages` (context management)
- **Graph**: START → network_manager → [tools → update_state → {summarize_messages | continue}] → END

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"
```

## Usage

### Jupyter Notebook

Open the interactive notebook for exploration and testing:

```bash
cd notebooks
jupyter lab react_agent.ipynb
```

### CLI (after extracting to source)

```bash
python -m network_manager_agent.main
```

## Data Format

### `data/raw/hospitals.csv`
| Column | Type | Description |
|--------|------|-------------|
| id | int | Provider ID |
| lat | float | Latitude |
| lon | float | Longitude |
| cluster | str | Geographic cluster name |
| county | str | County |
| specialty | str | Provider specialty |
| effectiveness | int | Quality rating (1-5) |

### `data/raw/members.csv`
| Column | Type | Description |
|--------|------|-------------|
| lat | float | Member latitude |
| lon | float | Member longitude |
| county | str | Member county |

## Project Structure

```
.
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # This file
├── .gitignore
├── src/                    # Source code (importable package)
│           └── network_manager_agent/
│       ├── __init__.py
│       ├── agent.py        # State, tools, graph definition
│       ├── config.py       # LLM and agent configuration
│       └── main.py         # Entry point
├── notebooks/              # Interactive development notebooks
│   └── react_agent.ipynb
├── data/                   # Data files
│   └── raw/
│       ├── hospitals.csv
│       └── members.csv
└── tests/                  # Test suite
    └── test_agent.py
```
