"""UI utilities for running and displaying agent output."""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


def run_agent(agent, inputs: dict, config: dict, output_dir: Path | None = None):
    """Run the agent and stream output to console.

    Args:
        agent: Compiled LangGraph agent.
        inputs: Agent input dict with 'messages', 'candidates', 'members'.
        config: LangGraph config dict with thread_id.
        output_dir: Directory to save action log. Defaults to current working directory.
    """
    print("Starting Agent Execution...\n")

    for chunk in agent.stream(inputs, stream_mode="updates", config=config):
        for node_name, output in chunk.items():
            print(f"\n>>>> NODE: {node_name} <<<<")
            if node_name in ("network_manager", "tools"):
                if "messages" in output:
                    for m in output["messages"]:
                        print("   --- CURRENT CONTENT ---")
                        
                        if isinstance(m, ToolMessage) and m.name == "add_contract_entity":
                            content = m.content
                            if isinstance(content, str):
                                try:
                                    parsed = json.loads(content)
                                    added = parsed.get("added_providers", [])
                                    errors = parsed.get("errors", [])
                                    print(f"   Added {len(added)} providers")
                                    if errors:
                                        print(f"   Errors: {', '.join(errors)}")
                                except json.JSONDecodeError:
                                    m.pretty_print()
                            else:
                                m.pretty_print()
                        else:
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
    log_path = (output_dir or Path.cwd()) / filename

    with open(log_path, "w", encoding="utf-8") as f:
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

            if isinstance(m, HumanMessage):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                f.write(f"[STEP {i}] HUMAN INPUT\n")
                f.write(f"  {content.strip()}\n\n")
                continue

    print(f"\nExecution Complete. Detailed history saved to '{log_path}'")
