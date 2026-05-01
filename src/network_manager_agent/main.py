"""CLI entry point for the network management agent."""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from .config import LLMConfig, create_llm
from .data import load_data
from .tools import (
    _filter_by_service_area,
    precompute_entity_summaries,
    get_candidate_schema_profile,
)
from .graph import build_agent
from .ui import run_agent


def run_agent_session(agent, prompt, candidates, members, thresholds, summaries, profile, thread_config):
    """Helper to encapsulate agent execution and summary printing."""
    messages = [HumanMessage(content=prompt)]
    inputs = {
        "messages": messages,
        "candidates": candidates,
        "members": members,
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

    args = parser.parse_args()

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

    # Load data
    candidates, members = load_data(
        candidates_path=args.candidates,
        members_path=args.members,
    )

    # Pre-compute: filter by service area, aggregate entities, build schema
    filtered_candidates, filtered_members = _filter_by_service_area(
        candidates, members, county_specialty_thresholds
    )

    entity_summaries = precompute_entity_summaries(filtered_candidates)
    schema_profile = get_candidate_schema_profile(candidates=filtered_candidates)


    print(
        f"Loaded {len(candidates)} raw candidates, {len(members)} raw members "
        f"-> {len(filtered_candidates)} in service area, {len(filtered_members)} scoped members "
        f"-> {len(entity_summaries)} entities"
    )

    # Build agent
    agent = build_agent(llm)

    # Create thread config
    thread_config = {"configurable": {"thread_id": "1"}}

    if args.prompt:
        prompt = args.prompt
    else:
        print("Network Management Agent")
        print("Enter your prompt (or 'quit' to exit):")
        prompt = input("> ").strip()
        while prompt.lower() not in ("quit", "exit", "q"):
                if prompt:
                    run_agent_session(
                        agent, prompt, filtered_candidates, filtered_members, 
                        county_specialty_thresholds, entity_summaries, 
                        schema_profile, thread_config
                    )
 
 
                    prompt = input("\n> ").strip()

    if args.prompt:
        run_agent_session(
            agent, args.prompt, filtered_candidates, filtered_members, 
            county_specialty_thresholds, entity_summaries, 
            schema_profile, thread_config
        )


    print("Goodbye!")


if __name__ == "__main__":
    main()
