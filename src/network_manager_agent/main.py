"""CLI entry point for the network management agent."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from .config import LLMConfig, create_llm
from .data import load_data
from .graph import build_agent


def run_agent(agent, inputs: dict, config: dict):
    """Run the agent and stream output to console.

    Args:
        agent: Compiled LangGraph agent.
        inputs: Agent input dict with 'messages', 'candidates', 'members'.
        config: LangGraph config dict with thread_id.
    """
    print("Starting Agent Execution...\n")

    for chunk in agent.stream(inputs, stream_mode="updates", config=config):
        for node_name, output in chunk.items():
            print(f"\n>>>> NODE: {node_name} <<<<")
            if node_name in ("network_manager", "tools"):
                if "messages" in output:
                    for m in output["messages"]:
                        print("   --- CURRENT CONTENT ---")
                        m.pretty_print()

                        if hasattr(m, 'tool_calls') and m.tool_calls:
                            print("   --- TOOL CALL ---")
                            for tc in m.tool_calls:
                                print(f"   -> {tc['name']}({json.dumps(tc['args'], default=str)})")

    # Save action log
    history = list(agent.get_state_history(config))
    history.reverse()

    last_message_id = None
    filename = f"react_agent_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== AGENT ACTION LOG ===\n\n")

        for i, state in enumerate(history):
            messages = state.values.get("messages", [])
            if not messages:
                continue

            m = messages[-1]

            if getattr(m, "id", None) == last_message_id:
                node_type = state.metadata.get("node", "update_state")
                if node_type in ("update_state", "summarize_messages"):
                    continue
                f.write(f"[STEP {i}] STATE UPDATE\n")
                f.write(f"  Node executed: {node_type}\n\n")
                continue

            last_message_id = getattr(m, "id", None)

            from langchain_core.messages import AIMessage, ToolMessage, HumanMessage as HM

            if isinstance(m, AIMessage) and m.tool_calls:
                f.write(f"[STEP {i}] AI -> TOOL CALL\n")
                for tc in m.tool_calls:
                    f.write(f"  -> {tc['name']}({json.dumps(tc['args'], default=str)})\n")
                f.write("\n")
                continue

            if isinstance(m, ToolMessage):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                f.write(f"[STEP {i}] TOOL RESULT ({m.name})\n")
                f.write(f"  {content.strip()}\n\n")
                continue

            if isinstance(m, AIMessage):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                if content.strip():
                    f.write(f"[STEP {i}] AI OUTPUT\n")
                    f.write(f"  {content.strip()}\n\n")
                continue

            if isinstance(m, HM):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                f.write(f"[STEP {i}] HUMAN INPUT\n")
                f.write(f"  {content.strip()}\n\n")
                continue

    print(f"\nExecution Complete. Detailed history saved to '{filename}'")


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
        "--hospitals",
        type=Path,
        default=None,
        help="Path to hospitals CSV",
    )
    parser.add_argument(
        "--members",
        type=Path,
        default=None,
        help="Path to members CSV",
    )

    args = parser.parse_args()

    # Create LLM
    config = LLMConfig()
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model

    llm = create_llm(config)

    # Load data
    candidates, members = load_data(
        hospitals_path=args.hospitals,
        members_path=args.members,
    )

    # Build agent
    agent = build_agent(llm, candidates, members)

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
                messages = [HumanMessage(content=prompt)]
                inputs = {
                    "messages": messages,
                    "candidates": candidates,
                    "members": members,
                }
                run_agent(agent, inputs, thread_config)

                # Show summary
                state = agent.get_state(thread_config)
                summary = state.values.get("summary", "")
                if summary:
                    print(f"\n--- Summary ---\n{summary}")

            prompt = input("\n> ").strip()

    if args.prompt:
        messages = [HumanMessage(content=prompt)]
        inputs = {
            "messages": messages,
            "candidates": candidates,
            "members": members,
        }
        run_agent(agent, inputs, thread_config)

        state = agent.get_state(thread_config)
        summary = state.values.get("summary", "")
        if summary:
            print(f"\n--- Summary ---\n{summary}")

    print("Goodbye!")


if __name__ == "__main__":
    main()
