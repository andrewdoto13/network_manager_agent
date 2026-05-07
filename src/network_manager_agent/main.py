"""CLI entry point for the network management agent."""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import LLMConfig, create_llm
from .data import DataManager
from .graph import build_agent
from .ui import run_agent


def _get_db_threads(db_path: str) -> set[str]:
    """Get thread IDs from SQLite checkpoints database."""
    if not Path(db_path).exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return {row[0] for row in cursor.fetchall()}


def _get_log_threads(log_dir: Path) -> set[str]:
    """Get thread IDs from logs directory."""
    if not log_dir.exists():
        return set()
    threads = set()
    for d in log_dir.iterdir():
        if d.is_dir() and d.name.startswith("thread_"):
            threads.add(d.name.replace("thread_", "", 1))
    return threads


def _clear_thread_state(thread_id: str, db_path: str, log_dir: Path) -> None:
    """Delete checkpoint state and log directory for a thread."""
    db_cleared = False
    if Path(db_path).exists():
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
        db_cleared = True

    log_cleared = False
    thread_log = log_dir / f"thread_{thread_id}"
    if thread_log.exists():
        shutil.rmtree(thread_log)
        log_cleared = True

    if db_cleared:
        print(f"Cleared checkpoint state for thread: {thread_id}")
    if log_cleared:
        print(f"Deleted log directory: {thread_log}")
    if not db_cleared and not log_cleared:
        print(f"No state or logs found for thread: {thread_id}")


def run_agent_session(
    agent, prompt, thresholds, summaries, profile, thread_config, max_steps=None
):
    """Helper to encapsulate agent execution and summary printing."""
    messages = [HumanMessage(content=prompt)]
    inputs = {
        "messages": messages,
        "county_specialty_thresholds": thresholds,
        "entity_summaries": summaries,
        "schema_profile": profile,
    }
    run_agent(agent, inputs, thread_config, max_steps=max_steps)

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
        required=False,
        help="REQUIRED: JSON string mapping state->county->specialty->threshold (e.g. '{\"mi\": {\"wayne\": {\"general practice\": 20.0, \"cardiology\": 10.0}}'). Default threshold is 20.0 miles.",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of agent steps before auto-stopping. None = unlimited.",
    )

    args = parser.parse_args()

    db_path = "checkpoints.sqlite"
    log_dir = Path("logs")

    # Handle management commands before loading data/agent
    if args.clear_all:
        db_cleared = False
        if Path(db_path).exists():
            Path(db_path).unlink()
            db_cleared = True
        log_cleared = False
        if log_dir.exists():
            shutil.rmtree(log_dir)
            log_cleared = True
        if db_cleared:
            print(f"Wiped persistence database: {db_path}")
        if log_cleared:
            print(f"Deleted log directory: {log_dir}")
        if not db_cleared and not log_cleared:
            print("No persistence database or logs found.")
        sys.exit(0)

    if args.list_threads:
        db_threads = _get_db_threads(db_path)
        log_threads = _get_log_threads(log_dir)
        all_threads = sorted(db_threads | log_threads)
        if all_threads:
            print("Available Threads:")
            for t in all_threads:
                tags = ""
                if t in db_threads:
                    tags += " [checkpoint]"
                if t in log_threads:
                    tags += " [log]"
                print(f" - {t}{tags}")
        else:
            print("No threads found.")
        sys.exit(0)

    if args.clear_thread:
        _clear_thread_state(args.clear_thread, db_path, log_dir)
        sys.exit(0)

    # Validate required args (management commands bypass this check above)
    if not args.county_specialty_thresholds:
        parser.error(
            "the following argument is required: --county-specialty-thresholds\n"
            "Example: --county-specialty-thresholds '{\"mi\": {\"washtenaw\": {\"cardiology\": 10.0, \"general practice\": 20.0}}}'"
        )

    # Create LLM
    config = LLMConfig()
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model

    llm = create_llm(config)

    try:
        county_specialty_thresholds = json.loads(args.county_specialty_thresholds)
    except json.JSONDecodeError:
        print("Error: --county-specialty-thresholds must be a valid JSON string.")
        sys.exit(1)

    if not county_specialty_thresholds:
        print("Error: --county-specialty-thresholds must not be empty. Provide at least one state/county/specialty threshold.")
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
            run_agent_session(
                agent, args.prompt,
                county_specialty_thresholds, entity_summaries,
                schema_profile, thread_config,
                max_steps=args.max_steps,
            )
        else:
            print("Network Management Agent")
            print("Enter your prompt (or 'quit' to exit):")
            try:
                prompt = input("> ").strip()
                while prompt.lower() not in ("quit", "exit", "q"):
                    if prompt:
                        run_agent_session(
                            agent, prompt,
                            county_specialty_thresholds, entity_summaries,
                            schema_profile, thread_config,
                            max_steps=args.max_steps,
                        )
                        prompt = input("\n> ").strip()
            except EOFError:
                pass


    print("Goodbye!")


if __name__ == "__main__":
    main()
