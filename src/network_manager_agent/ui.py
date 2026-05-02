"""UI utilities for running and displaying agent output."""

import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


def run_agent(agent, inputs: dict, config: dict, output_dir: Path | None = None):
    """Run the agent and stream output to console and a log file in real-time.

    Args:
        agent: Compiled LangGraph agent.
        inputs: Agent input dict with 'messages', 'candidates', 'members'.
        config: LangGraph config dict with thread_id.
        output_dir: Directory to save action log. Defaults to current working directory.
    """
    print("Starting Agent Execution...\n")

    # Initialize log file
    filename = f"react_agent_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = (output_dir or Path.cwd()) / filename
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== AGENT ACTION LOG (REAL-TIME) ===\n\n")
        
        # Log initial human input
        human_msgs = [m for m in inputs.get("messages", []) if isinstance(m, HumanMessage)]
        if human_msgs:
            f.write("[STEP 0] HUMAN INPUT\n")
            f.write(f"  {human_msgs[-1].content.strip()}\n\n")

        step_count = 1
        for chunk in agent.stream(inputs, stream_mode="updates", config=config):
            for node_name, output in chunk.items():
                # 1. Console Output
                print(f"\n>>>> NODE: {node_name} <<<<")
                
                # 2. File Log Output
                f.write(f"[STEP {step_count}] NODE: {node_name}\n")
                
                if "messages" in output:
                    for m in output["messages"]:
                        # Console: Pretty print
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

                        # File: Log message content
                        if isinstance(m, AIMessage) and m.tool_calls:
                            f.write("  AI -> TOOL CALL\n")
                            for tc in m.tool_calls:
                                f.write(f"    -> {tc['name']}({json.dumps(tc['args'], default=str)})\n")
                        elif isinstance(m, ToolMessage):
                            content = m.content
                            if isinstance(content, list):
                                content = " ".join(str(c) for c in content)
                            f.write(f"  TOOL RESULT ({m.name})\n")
                            f.write(f"    {content.strip()}\n")
                        elif isinstance(m, AIMessage):
                            content = m.content
                            if isinstance(content, list):
                                content = " ".join(str(c) for c in content)
                            if content.strip():
                                f.write(f"  AI OUTPUT\n")
                                f.write(f"    {content.strip()}\n")
                        elif isinstance(m, HumanMessage):
                            f.write(f"  HUMAN INPUT\n")
                            f.write(f"    {m.content.strip()}\n")
                        
                        f.write("\n")
                
                f.write("\n")
                step_count += 1

    print(f"\nExecution Complete. Detailed history saved to '{log_path}'")
