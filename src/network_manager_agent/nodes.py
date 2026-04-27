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

from .config import SUMMARIZE_THRESHOLD, MESSAGES_TO_ARCHIVE
from .state import AgentState
from .tools import (
    get_candidates,
    get_candidate_schema,
    add_provider,
    get_network_status,
    simulate_network_change,
)


def _get_anchor_message(state: AgentState) -> str:
    """Get the most recent human message to use as an anchor."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state.get("original_message") or "Please continue."


def network_manager(state: AgentState, llm: ChatOpenAI):
    """The main LLM reasoning node that decides which tools to call."""
    specialties = list(set(c["specialty"] for c in state.get("candidates", [])))

    system_message_content = f'''
You are an assistant responsible for managing a healthcare provider network.
You MUST use the available tools to update the network state.
Follow all rules below exactly.

RULES
---------------
1. You may ONLY call add_provider with valid provider IDs that appear in get_candidates result.
2. Never invent providers, specialties, counts, or network state. Use ONLY the data returned by tools.
3. If get_candidates returns an empty list, that means no candidates remain for that specialty.
   Do NOT retry unless the user explicitly requests it.
4. The get_network_status function is THE source of truth for the network.
5. Factor in the user's stated preferences when deciding whether to call tools.
6. If required information is missing, ask the user for clarification instead of guessing.
7. You MUST always write a response in your final message. Never return an empty response.
   Always summarize what was accomplished when the task is complete.
8. When choosing between candidates, use simulate_network_change with compare_scenarios
   to evaluate all options in a single call. Once you have simulation data, commit to
   the best option - do not oscillate between simulating and deciding.
9. Be decisive. After gathering sufficient data, take action. Avoid repeating the same
   reasoning or simulation multiple times.

CANDIDATE SPECIALTIES
---------------
{specialties}
'''
    messages_history = state.get("messages", [])
    original_message = state.get("original_message") or ""

    if not original_message:
        for msg in messages_history:
            if isinstance(msg, HumanMessage):
                original_message = msg.content
                break

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

    tools_list = [get_candidates, get_candidate_schema, add_provider, get_network_status, simulate_network_change]
    response = llm.bind_tools(tools_list).invoke(messages)

    return {
        "messages": [response],
        "original_message": original_message,
    }


def update_state(state: AgentState):
    """Extract new providers from tool results and update the network state."""
    messages = state["messages"]
    new_providers = []

    batch = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        batch.append(msg)

    batch.reverse()

    for msg in batch:
        if msg.name == "add_provider":
            raw_output = msg.content
            if isinstance(raw_output, str) and "Skip" in raw_output:
                continue
            try:
                parsed_output = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                if isinstance(parsed_output, dict):
                    added = parsed_output.get("added", [])
                    if isinstance(added, list):
                        new_providers.extend(added)
                    else:
                        new_providers.append(parsed_output)
            except (json.JSONDecodeError, ValueError):
                continue

    return {"network": new_providers} if new_providers else {}


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
