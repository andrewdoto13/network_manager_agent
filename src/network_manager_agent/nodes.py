"""Node functions for the network management agent graph."""

import json
from typing import Any

import numpy as np
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode

from . import tools
from .config import SUMMARIZE_THRESHOLD
from .data import DataManager
from .state import AgentState


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _get_anchor_message(state: AgentState) -> str:
    """Get the most recent human message to use as an anchor."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return "Please continue."


def network_manager(state: AgentState, llm: ChatDeepSeek):
    """The main LLM reasoning node that decides which tools to call."""
    dm = DataManager()
    county_specialty_thresholds = state.get("county_specialty_thresholds", {})

    scope_lines = []
    for state_val, counties in county_specialty_thresholds.items():
        for county, specialties in counties.items():
            spec_str = ", ".join(f"{spec} ({threshold}mi)" for spec, threshold in specialties.items())
            scope_lines.append(f"- {state_val}/{county}: {spec_str}")
    scope_section = "\n".join(scope_lines) if scope_lines else ""

    cand_cols = ", ".join(dm.get_candidates_df().columns)
    mem_cols = ", ".join(dm.get_members_df().columns)

    system_message_content = f'''# ROLE
You are a healthcare provider network management assistant. Analyze candidate provider data,
simulate network changes, and recommend or commit contract entities to improve member coverage.

# SCOPE
You are evaluating member coverage for these county-specialty combinations:
{scope_section}

# DATA
candidates_df (provider-level): {cand_cols}
members_df (member-level): {mem_cols}
network_df (currently contracted): filtered from candidates_df by the network state
NOTE: All string values are lowercased and whitespace-stripped.

# SANDBOX
Pre-loaded modules (use directly, no declaration needed): pd, np, json, math, itertools, collections, defaultdict, functools, BallTree
Available builtins: len, sorted, range, str, int, float, bool, set, list, dict, tuple, enumerate, zip, map, filter, isinstance, type, print, abs, round, min, max, sum, any, all

  Cross-call state:
   - sandbox_cache: persistent dict. Save with sandbox_cache["key"] = value. Retrieve with sandbox_cache.get("key"). Only JSON-serializable types.

compute_coverage(network_df, members_df, thresholds, candidates_df) -> (list[dict], list[str])
  Each dict: {{state, county, specialty, members_with_access, total_members, coverage_percentage}}

## GUIDANCE
- BallTree gives fast proximity heuristics but is approximate.
- compute_coverage() is authoritative: it evaluates coverage per member, not per county or aggregate.
- Use pandas boolean indexing, isin(), groupby().agg(). Remember candidates_df is provider-level - group by entity for entity-level summaries.
- Each run_code call is a fresh sandbox — only sandbox_cache persists. Write self-contained code.

## EXAMPLE
  # --- Call 1: compute once, cache for later ---
  entity_stats = candidates_df.groupby("entity").agg(
      count=("entity", "count"),
      avg_eff=("effectiveness", "mean")
  ).reset_index()
  sandbox_cache["entity_stats"] = entity_stats.to_dict("records")

  # --- Call 2: read from cache, use in simulation ---
  entity_stats = pd.DataFrame(sandbox_cache.get("entity_stats", []))
  top_entities = entity_stats.nlargest(5, "avg_eff")["entity"].tolist()
  sim_df = candidates_df[candidates_df["entity"].isin(top_entities)]
  coverage, errors = compute_coverage(sim_df, members_df, thresholds, candidates_df)
  sandbox_cache["coverage"] = coverage

## RULES
1. compute_coverage() is the definitive coverage calculator. Report only results validated by compute_coverage().
2. Only call add_contract_entity when the user explicitly asks to add or commit entities. Analysis, simulation, and comparison prompts should report findings and stop.
3. Only add entity names that exist in candidates_df. Verify names before calling add_contract_entity.
4. When data is missing or ambiguous, ask the user for clarification.
5. Present your final answer with coverage numbers and stop. Do not repeat simulations that already ran.'''

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

    tools_list = tools.TOOLS

    # Retry on empty response (up to 3 attempts)
    for attempt in range(3):
        response = llm.bind_tools(tools_list).invoke(messages)
        content = response.content if hasattr(response, 'content') else ""
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        if content.strip() or tool_calls:
            break
        # Empty response — append a nudge and retry
        messages = messages + [
            AIMessage(content="I need to continue analyzing the data."),
            HumanMessage(content="Continue. Use run_code to analyze, then add_contract_entity to commit entities. Provide a final answer when done."),
        ]
    else:
        # All retries failed — force a meaningful response
        response = AIMessage(
            content="I've analyzed the data. Let me commit the best entities found to the network.",
            tool_calls=[{
                "name": "run_code",
                "args": {"code": "# Final analysis\ncoverage_results, errors = compute_coverage(network_df, members_df, thresholds, candidates_df)\nprint('Coverage:', coverage_results)\nprint('Errors:', errors)"},
                "id": "final_check",
            }],
        )

    return {
        "messages": [response],
    }


def update_state(state: AgentState):
    """Extract new entity IDs from tool results and update the network state."""
    messages = state["messages"]
    new_entities = []
    new_cache = {}

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
                        new_entities.extend([str(e) for e in added])
            except (json.JSONDecodeError, ValueError):
                continue

        if msg.name == "run_code":
            new_cache = tools._last_sandbox_cache

    result = {}
    if new_entities:
        result["network"] = new_entities
    if new_cache:
        native_cache = _to_native(new_cache)
        existing_cache = state.get("sandbox_cache", {})
        merged = {**existing_cache, **native_cache}
        result["sandbox_cache"] = merged
    return result


def _get_content(m: Any) -> str:
    """Extract string content from a message, handling list-type content."""
    if isinstance(m.content, list):
        return " ".join(str(c) for c in m.content)
    return str(m.content)


def _find_safe_boundary(messages: list, keep_count: int) -> tuple:
    """Find safe boundary that doesn't split AI+Tool pairs.

    Returns (to_summarize, to_keep) where to_keep contains at least keep_count
    messages, plus any AI+Tool pairs that would be orphaned at the boundary.
    """
    if len(messages) <= keep_count:
        return (messages, [])

    to_summarize = list(messages[:-keep_count])
    to_keep = list(messages[-keep_count:])

    # Check if last message in to_summarize is an AI with tool calls
    # If so, move it and its Tool responses to kept set to avoid orphaning
    while to_summarize and isinstance(to_summarize[-1], AIMessage):
        last_ai = to_summarize[-1]
        if not last_ai.tool_calls:
            break
        # Move this AI message to kept set
        to_summarize.pop()
        to_keep.insert(0, last_ai)
        # Also move any ToolMessages that follow (they should already be in to_keep)
        break

    return (to_summarize, to_keep)


def summarize_messages(state: AgentState, llm: ChatDeepSeek):
    """Summarize old messages to manage context window size."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # Keep the last MESSAGES_TO_KEEP messages in context (user prompt + recent tool results).
    # Use safe boundary to avoid splitting AI+Tool pairs.
    MESSAGES_TO_KEEP = 3
    to_summarize, to_keep = _find_safe_boundary(messages, MESSAGES_TO_KEEP)

    instruction = f"""You are a task summarizer. Update the existing summary based on the new history provided below.

    EXISTING SUMMARY:
    {existing_summary if existing_summary else "No previous summary."}

    STRUCTURE:
    Objective: [User's original request and constraints — preserve verbatim]
    Progress: [What tools were called and what was accomplished]
    Key Findings: [Quantitative results: coverage percentages, entity names with provider counts, best combinations found, geographic constraints]
    Remaining: [What still needs to be done to complete the task]

    RULES:
    1. Use ONLY English.
    2. Incorporate new info into the existing summary; do not just append.
    3. Be specific with the objective because details matter here.
    4. Always preserve the user's original constraints and preferences verbatim in the Objective section.
    5. Preserve ALL quantitative results from tool output: exact coverage percentages, provider counts, entity names, distances, rankings. Do not replace numbers with vague descriptions.
    6. When listing entities, include their key metrics (e.g., "MyMichigan Health: 124 cardio providers, eff 5.0").
    7. Always include the best combination or result found so far, with exact numbers.
    8. Preserve the sequence of tool calls and outcomes as a reasoning chain. Note which simulations built on which prior results.
    """

    # Summarize ALL messages (not just deleted ones) for complete picture
    history_text = "\n".join([f"{m.type}: {_get_content(m)}" for m in messages])
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

    # Only remove messages before the safe boundary (not orphaned pairs)
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
    tool_node = ToolNode(tools.TOOLS)
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

