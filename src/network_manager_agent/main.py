"""CLI entry point for the network management agent."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from .config import LLMConfig, create_llm
from .data import DataManager
from .graph import build_agent
from .ui import run_agent
from langgraph.checkpoint.sqlite import SqliteSaver


def run_agent_session(agent, prompt, thresholds, summaries, profile, thread_config):
    """Helper to encapsulate agent execution and summary printing."""
    messages = [HumanMessage(content=prompt)]
    inputs = {
        "messages": messages,
        "county_specialty_thresholds": thresholds,
        "entity_summaries": summaries,
        "schema_profile": profile,
    }
    run_agent(agent, inputs, thread_config)
    
    state = agent.get_state(thread_config)
    summary = state.values.get("summary", "")
    if summary:
        print(f"\n--- Summary ---\n{summary}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Network Manager Agent - AI agent for provider network optimization"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to send to the agent. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM base URL (overrides config default)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (overrides config default)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to candidates CSV",
    )
    parser.add_argument(
        "--members",
        type=Path,
        default=None,
        help="Path to members CSV",
    )
    parser.add_argument(
        "--county-specialty-thresholds",
        type=str,
        default=None,
        help="JSON string mapping state->county->specialty->threshold (e.g. '{\"mi\": {\"wayne\": {\"general practice\": 20.0, \"cardiology\": 10.0}}'). Default threshold is 20.0 miles.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="1",
        help="The ID of the conversation thread to load/create. Defaults to '1'.",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        help="List all unique threads in the persistence database.",
    )
    parser.add_argument(
        "--clear-thread",
        type=str,
        metavar="THREAD_ID",
        help="Delete all checkpoints for the specified thread ID.",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Wipe the entire persistence database.",
    )

    args = parser.parse_args()

    db_path = "checkpoints.sqlite"

    # Handle management commands before loading data/agent
    if args.clear_all:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Wiped persistence database: {db_path}")
        sys.exit(0)

    if args.list_threads:
        if not Path(db_path).exists():
            print("No persistence database found.")
            sys.exit(0)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = cursor.fetchall()
            if threads:
                print("Available Threads:")
                for t in threads:
                    print(f" - {t[0]}")
            else:
                print("No threads found in database.")
        sys.exit(0)

    if args.clear_thread:
        if not Path(db_path).exists():
            print("No persistence database found.")
            sys.exit(0)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (args.clear_thread,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (args.clear_thread,))
            conn.commit()
            print(f"Cleared all state for thread: {args.clear_thread}")
        sys.exit(0)

    # Create LLM
    config = LLMConfig()
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model

    llm = create_llm(config)

    county_specialty_thresholds = {}
    if args.county_specialty_thresholds:
        try:
            county_specialty_thresholds = json.loads(args.county_specialty_thresholds)
        except json.JSONDecodeError:
            print("Error: --county-specialty-thresholds must be a valid JSON string.")
            sys.exit(1)

    # Initialize DataManager: handles loading and filtering internally
    dm = DataManager(
        candidates_path=args.candidates,
        members_path=args.members,
        county_specialty_thresholds=county_specialty_thresholds,
    )
    
    entity_summaries = dm.get_entity_summaries()
    schema_profile = json.dumps(dm.get_schema_profile(), indent=2)

    print(
        f"Loaded candidates and members -> "
        f"{len(dm.get_candidates_df())} candidates in service area, "
        f"{len(dm.get_members_df())} scoped members "
        f"-> {len(entity_summaries)} entities"
    )

    # Build agent
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        agent = build_agent(llm, checkpointer=checkpointer)

        # Create thread config
        thread_config = {"configurable": {"thread_id": args.thread_id}}

        if args.prompt:
            prompt = args.prompt
        else:
            print("Network Management Agent")
            print("Enter your prompt (or 'quit' to exit):")
            prompt = input("> ").strip()
            while prompt.lower() not in ("quit", "exit", "q"):
                    if prompt:
                        run_agent_session(
                            agent, prompt, 
                            county_specialty_thresholds, entity_summaries, 
                            schema_profile, thread_config
                        )
            
            
                        prompt = input("\n> ").strip()

        if args.prompt:
            run_agent_session(
                agent, args.prompt, 
                county_specialty_thresholds, entity_summaries, 
                schema_profile, thread_config
            )


    print("Goodbye!")


if __name__ == "__main__":
    main()
