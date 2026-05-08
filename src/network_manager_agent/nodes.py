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

    cand_cols = ", ".join(dm.get_candidates_df().columns)
    mem_cols = ", ".join(dm.get_members_df().columns)
    ent_cols = ", ".join(dm.entity_summaries_df.columns)

    system_message_content = f'''ROLE
You are a healthcare provider network management assistant. Your job is to analyze
candidate provider data, simulate network changes, and recommend or commit contract
entities to improve member coverage. You MUST use the available tools to analyze data
and update the network.

NETWORK SCOPE
You are evaluating member coverage for these county-specialty combinations:
{scope_section}

DATA
Three DataFrames available in run_code:

candidates_df (provider-level): {cand_cols}
members_df (member-level): {mem_cols}
entity_summaries_df (per-entity): {ent_cols}

SANDBOX LIBRARIES
The following are ALREADY AVAILABLE in run_code — DO NOT use import statements:
  pd (pandas), np (numpy), json, math, functools, itertools, collections, BallTree
Builtins: len, sorted, range, str, int, float, bool, set, list, dict, tuple, enumerate, zip, map, filter, isinstance, type, print, abs, round, min, max, sum, any, all
** import is DISABLED and will raise ImportError. **

SANDBOX FUNCTIONS
compute_coverage(network_df, members_df, thresholds, candidates_df)
  → (list[dict], list[str]) — list of {{state, county, specialty, members_with_access, total_members, coverage_percentage}} + validation errors

COMMON PROCEDURES
Self-contained run_code blocks for common tasks. Each is a complete, copy-paste starting point.

1) Rank multi-specialty candidates by marginal coverage:
   # Derive required specialties from thresholds
   required_specs = set()
   for state_val, counties in thresholds.items():
       for county_val, specs in counties.items():
           required_specs.update(specs.keys())
   entity_specs = candidates_df.groupby("entity")["specialty"].apply(
       lambda x: set(x.str.lower().unique()))
   candidate_entities = entity_specs[entity_specs.apply(lambda s: required_specs.issubset(s))].index.tolist()
   # First, extract uncovered members per county/specialty so we can rank
   # by which entity covers the most uncovered members (faster than full simulation)
   uncovered_by_spec = {{}}
   for state_val, counties in thresholds.items():
       for county_val, specs in counties.items():
           county_members = members_df[members_df["county"].str.lower() == county_val.lower()]
           if "state" in members_df.columns:
               county_members = county_members[county_members["state"].str.lower() == state_val.lower()]
           for spec, threshold_miles in specs.items():
               key = (state_val, county_val, spec)
               spec_providers = network_df[network_df["specialty"].str.lower() == spec.lower()]
               radius_rad = threshold_miles / 3958.8
               if not spec_providers.empty:
                   tree = BallTree(np.deg2rad(spec_providers[["lat", "lon"]]), metric="haversine")
                   nearby = tree.query_radius(np.deg2rad(county_members[["lat", "lon"]]), r=radius_rad)
                   uncovered = county_members.iloc[[i for i, n in enumerate(nearby) if len(n) == 0]]
               else:
                   uncovered = county_members
               uncovered_by_spec[key] = uncovered
   # Now rank each candidate by how many uncovered members it would cover
   rankings = []
   for ent in candidate_entities:
       ent_providers = candidates_df[candidates_df["entity"].str.lower() == ent.lower()]
       total_newly_covered = 0
       for state_val, counties in thresholds.items():
           for county_val, specs in counties.items():
               for spec, threshold_miles in specs.items():
                   key = (state_val, county_val, spec)
                   uncovered = uncovered_by_spec.get(key)
                   if uncovered is None or uncovered.empty:
                       continue
                   ent_spec = ent_providers[ent_providers["specialty"].str.lower() == spec.lower()]
                   if ent_spec.empty:
                       continue
                   radius_rad = threshold_miles / 3958.8
                   tree = BallTree(np.deg2rad(ent_spec[["lat", "lon"]]), metric="haversine")
                   nearby = tree.query_radius(np.deg2rad(uncovered[["lat", "lon"]]), r=radius_rad)
                   total_newly_covered += sum(len(n) > 0 for n in nearby)
       rankings.append({{"entity": ent, "newly_covered": total_newly_covered}})
   result = sorted(rankings, key=lambda x: x["newly_covered"], reverse=True)

2) Simulate adding entities — test before committing:
   new_entities = ["Entity A", "Entity B"]
   new_providers = candidates_df[candidates_df["entity"].str.lower().isin([e.lower() for e in new_entities])]
   sim_net = pd.concat([network_df, new_providers])
   coverage, errors = compute_coverage(sim_net, members_df, thresholds, candidates_df)
   result = pd.DataFrame(coverage)

3) Compare coverage delta — baseline vs simulated for one entity:
   base_cov, _ = compute_coverage(network_df, members_df, thresholds, candidates_df)
   base_df = pd.DataFrame(base_cov)
   new_providers = candidates_df[candidates_df["entity"].str.lower() == "new entity"]
   sim_cov, _ = compute_coverage(pd.concat([network_df, new_providers]), members_df, thresholds, candidates_df)
   sim_df = pd.DataFrame(sim_cov)
   merged = pd.merge(base_df, sim_df, on=["state", "county", "specialty"], suffixes=("_base", "_sim"))
   merged["delta_pp"] = merged["coverage_percentage_sim"] - merged["coverage_percentage_base"]
   result = merged[["county", "specialty", "coverage_percentage_base", "coverage_percentage_sim", "delta_pp"]]

RULES
1. You may ONLY call add_contract_entity with valid entity names found in the data. Never invent entities, providers, or metrics.
2. If required information is missing, ask the user for clarification instead of guessing.
3. Always write a response in your final message. Never return an empty response. Summarize what was accomplished when complete.
4. Be decisive. Present your best result with coverage numbers and stop. Do not repeat the same simulations or keep exploring after finding a viable answer.
5. If the user asks for analysis or recommendations — present your findings and stop. Do NOT call add_contract_entity in the same response.
6. run_code mechanics: each call is a fresh sandbox — variables from a previous call are NOT available. NO import statements — all libraries are pre-injected. The schema is documented above — do NOT waste calls exploring column names. Assign your result to 'result'. Timeout is 60s.
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

