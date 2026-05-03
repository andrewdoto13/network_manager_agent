"""Node functions for the network management agent graph."""

import json
from typing import Any

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from .config import SUMMARIZE_THRESHOLD, MESSAGES_TO_ARCHIVE
from .data import DataManager
from .state import AgentState
from .tools import TOOLS


def _get_anchor_message(state: AgentState) -> str:
    """Get the most recent human message to use as an anchor."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return "Please continue."


def network_manager(state: AgentState, llm: ChatOpenAI):
    """The main LLM reasoning node that decides which tools to call."""
    dm = DataManager()
    county_specialty_thresholds = state.get("county_specialty_thresholds", {})

    scope_lines = []
    for state_val, counties in county_specialty_thresholds.items():
        for county, specialties in counties.items():
            spec_str = ", ".join(f"{spec} ({threshold}mi)" for spec, threshold in specialties.items())
            scope_lines.append(f"- {state_val}/{county}: {spec_str}")
    scope_section = "\n".join(scope_lines) if scope_lines else ""

    # Inject candidate schema into system prompt (pre-computed or fallback)
    schema_section = state.get("schema_profile", "")
    if not schema_section:
        schema_profile = dm.get_schema_profile()
        if isinstance(schema_profile, dict) and schema_profile:
            schema_section = json.dumps(schema_profile, indent=2)
        else:
            schema_section = "No schema available."

    # Build concise raw candidate schema for run_code tool
    raw_schema = dm.get_raw_candidate_schema_profile()
    # Only include column names and types, not full distributions
    raw_schema_section = ""
    if raw_schema:
        for col, info in raw_schema.items():
            col_type = info.get("type", "unknown")
            unique_count = info.get("unique_count", "")
            if unique_count:
                raw_schema_section += f"- {col}: {col_type} (unique_count={unique_count})\n"
            else:
                raw_schema_section += f"- {col}: {col_type}\n"

    system_message_content = f'''
You are an assistant responsible for managing a healthcare provider network.
You MUST use the available tools to update the network state.

NETWORK SCOPE
You are evaluating member coverage for these county-specialty combinations:
{scope_section}

CANDIDATE DATA SCHEMA
Entity summaries (aggregated):
{schema_section}

Raw provider-level columns:
{raw_schema_section}

Follow all rules below exactly.

RULES
---------------
1. You may ONLY call add_contract_entity with valid entity IDs that you discover through your analysis.
2. Never invent entities, providers, specialties, counts, or network state. Use ONLY the data returned by tools.
3. Use run_code with compute_coverage(network_df, members_df, thresholds, candidates_df) to check current network status or simulate changes. Create a temporary network DataFrame (e.g., by concatenating existing network with new candidates) and call compute_coverage(sim_net, members_df, thresholds, candidates_df) to assess the coverage delta.
4. If required information is missing, ask the user for clarification instead of guessing.
5. You MUST always write a response in your final message. Never return an empty response. Always summarize what was accomplished when the task is complete.
6. Be decisive. After gathering sufficient data, present your findings. Do not repeat the same reasoning or simulations.
7. If the user asks for recommendations, analysis, or evaluation — present your findings and stop. Factor in the user's stated preferences. You may suggest entities or ask if the user wants to proceed, but do NOT call add_contract_entity in the same response.
8. Your run_code tool executes pandas code. Variables available: candidates_df, entity_summaries_df, network_df, members_df, thresholds, and compute_coverage. Assign your result to 'result'. Timeout is 20 seconds. Allowed modules: pandas, numpy, json, math, functools, itertools, collections, sklearn.neighbors.BallTree. Use the raw provider-level columns above to write effective queries — for example, filter by location_confidence or effectiveness at the provider level, then groupby('entity') for entity-level results. Use the entity summaries schema for quick entity-level queries.
'''

    messages_history = state.get("messages", [])

    summary = state.get("summary", "")
    anchor = _get_anchor_message(state)

    if summary:
        messages = [
            SystemMessage(content=system_message_content + f"\n\nSUMMARY SO FAR:\n{summary}"),
            HumanMessage(content=anchor),
        ] + messages_history
    else:
        has_human = any(isinstance(m, HumanMessage) for m in messages_history)
        if has_human:
            messages = [SystemMessage(content=system_message_content)] + messages_history
        else:
            messages = [
                SystemMessage(content=system_message_content),
                HumanMessage(content=anchor),
            ] + messages_history

    tools_list = TOOLS
    response = llm.bind_tools(tools_list).invoke(messages)

    return {
        "messages": [response],
    }


def update_state(state: AgentState):
    """Extract new entity IDs from tool results and update the network state."""
    messages = state["messages"]
    new_entities = []

    batch = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        batch.append(msg)

    batch.reverse()

    for msg in batch:
        if msg.name == "add_contract_entity":
            raw_output = msg.content
            if isinstance(raw_output, str) and "Skip" in raw_output:
                continue
            try:
                parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                if isinstance(parsed_output, dict):
                    added = parsed_output.get("added_entities", [])
                    if isinstance(added, list):
                        new_entities.extend(added)
                    else:
                        new_entities.append(parsed_output)
            except (json.JSONDecodeError, ValueError):
                continue

    return {"network": new_entities} if new_entities else {}


def _get_content(m: Any) -> str:
    """Extract string content from a message, handling list-type content."""
    if isinstance(m.content, list):
        return " ".join(str(c) for c in m.content)
    return str(m.content)


def summarize_messages(state: AgentState, llm: ChatOpenAI):
    """Summarize old messages to manage context window size."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")
 
    last_message_to_summarize = MESSAGES_TO_ARCHIVE - 1
 
    if (last_message_to_summarize + 1 < len(messages) and
        isinstance(messages[last_message_to_summarize], AIMessage) and
        isinstance(messages[last_message_to_summarize + 1], ToolMessage)):
        last_message_to_summarize += 1
 
    to_summarize = messages[:last_message_to_summarize + 1]
 
    instruction = f"""You are a task summarizer. Update the existing summary based on the new history provided below.
 
    EXISTING SUMMARY:
    {existing_summary if existing_summary else "No previous summary."}
 
    STRUCTURE:
    Objective: [Recap of the user requests]
    Progress: [High-level status of progress, summarizing the tools that you called]
 
    RULES:
    1. Use ONLY English.
    2. Incorporate new info into the existing summary; do not just append.
    3. Be specific with the objective because details matter here.
    4. Always preserve the user's original constraints and preferences verbatim in the Objective section, no matter what actually happened during execution.
    """
 
    history_text = "\n".join([f"{m.type}: {_get_content(m)}" for m in to_summarize])
    final_prompt = f"{instruction}\n\nHISTORY TO SUMMARIZE:\n{history_text}\n\nSummary:"
 
    response = llm.invoke([HumanMessage(content=final_prompt)])
 
    updated_summary = response.content
    if isinstance(updated_summary, list):
        updated_summary = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in updated_summary
        ).strip()
    else:
        updated_summary = updated_summary.strip() if updated_summary else ""
 
    if not updated_summary:
        updated_summary = existing_summary or "Summary unavailable."
 
    messages_to_remove = [RemoveMessage(id=m.id) for m in to_summarize]
 
    return {
        "summary": updated_summary,
        "messages": messages_to_remove,
    }
 
 
def should_summarize(state: AgentState):
    """Routing function: decide whether to summarize messages."""
    messages = state["messages"]
    if len(messages) > SUMMARIZE_THRESHOLD:
        return "summarize"
    else:
        return "continue"


def execute_tools(state: AgentState):
    """Custom tool execution node that catches errors and returns them as ToolMessages."""
    tool_node = ToolNode(TOOLS)
    try:
        return tool_node.invoke(state)
    except Exception as e:
        messages = state.get("messages", [])
        last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai_msg and last_ai_msg.tool_calls:
            error_messages = []
            for tool_call in last_ai_msg.tool_calls:
                error_messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Error executing tool {tool_call['name']}: {str(e)}",
                    )
                )
            return {"messages": error_messages}
        return {
            "messages": [
                ToolMessage(
                    tool_call_id="unknown",
                    content=f"Unexpected error during tool execution: {str(e)}",
                )
            ]
        }

