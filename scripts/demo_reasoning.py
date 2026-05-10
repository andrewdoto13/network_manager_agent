"""Demo: ChatDeepSeek reasoning_content in multi-turn conversations.

Run with:
    python scripts/demo_reasoning.py

Requires a local OpenAI-compatible server (llama.cpp, Ollama, etc.)
running on http://localhost:8080/v1 with a reasoning-capable model.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from network_manager_agent.config import LLMConfig, create_llm


def print_separator(char="=", width=72):
    print(char * width)


def demo_single_turn():
    print_separator()
    print("SINGLE-TURN: reasoning_content in response")
    print_separator()

    llm = create_llm()

    messages = [
        SystemMessage(content="You are a helpful assistant. Think step by step."),
        HumanMessage(content="What is 47 * 26? Show your work."),
    ]

    response = llm.invoke(messages)

    print(f"\nModel: {llm.model_name}")
    print(f"Response type: {type(response).__name__}")
    print()

    # Extract reasoning_content
    reasoning = response.additional_kwargs.get("reasoning_content", "")
    content = response.content if isinstance(response.content, str) else str(response.content)

    print("--- REASONING ---")
    print(reasoning if reasoning else "(no reasoning_content in response)")
    print()

    print("--- FINAL ANSWER ---")
    print(content)
    print()

    # Show the full message structure
    print("--- MESSAGE STRUCTURE ---")
    print(f"  response.content: {type(response.content).__name__}")
    print(f"  response.additional_kwargs keys: {list(response.additional_kwargs.keys())}")
    if reasoning:
        print(f"  reasoning_content length: {len(reasoning)} chars")
    print()

    return response


def demo_multi_turn():
    print_separator()
    print("MULTI-TURN: reasoning_content persists in conversation history")
    print_separator()

    llm = create_llm()

    # Conversation history — this is what LangGraph tracks
    history: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content="You are a helpful math tutor. Think step by step.")
    ]

    # --- Turn 1 ---
    print("\n--- TURN 1 ---")
    turn1_input = HumanMessage(content="What is the square root of 841?")
    history.append(turn1_input)
    print(f"Human: {turn1_input.content}")

    response1 = llm.invoke(history)
    history.append(response1)

    reasoning1 = response1.additional_kwargs.get("reasoning_content", "")
    print(f"AI reasoning: {reasoning1[:120]}..." if reasoning1 else "AI reasoning: (none)")
    print(f"AI answer: {response1.content[:120]}...")

    # --- Turn 2 — reference previous answer ---
    print("\n--- TURN 2 ---")
    turn2_input = HumanMessage(content="Now square that result. Does it match the original?")
    history.append(turn2_input)
    print(f"Human: {turn2_input.content}")

    # The LLM sees the full history including previous reasoning_content
    response2 = llm.invoke(history)
    history.append(response2)

    reasoning2 = response2.additional_kwargs.get("reasoning_content", "")
    print(f"AI reasoning: {reasoning2[:120]}..." if reasoning2 else "AI reasoning: (none)")
    print(f"AI answer: {response2.content[:120]}...")

    # --- Turn 3 — ask about the conversation ---
    print("\n--- TURN 3 ---")
    turn3_input = HumanMessage(content="Summarize what we computed.")
    history.append(turn3_input)
    print(f"Human: {turn3_input.content}")

    response3 = llm.invoke(history)
    history.append(response3)

    reasoning3 = response3.additional_kwargs.get("reasoning_content", "")
    print(f"AI reasoning: {reasoning3[:120]}..." if reasoning3 else "AI reasoning: (none)")
    print(f"AI answer: {response3.content[:120]}...")

    # --- Show full history structure ---
    print("\n--- FULL CONVERSATION HISTORY ---")
    for i, msg in enumerate(history):
        kind = msg.type.upper()
        content_preview = str(msg.content)[:60].replace("\n", " ")

        if isinstance(msg, AIMessage):
            rc = msg.additional_kwargs.get("reasoning_content", "")
            rc_preview = f" [reasoning: {len(rc)} chars]" if rc else ""
            print(f"  [{i}] {kind}: {content_preview}...{rc_preview}")
        else:
            print(f"  [{i}] {kind}: {content_preview}...")

    print()
    print("--- KEY OBSERVATIONS ---")
    print("1. Each AIMessage carries reasoning_content in additional_kwargs")
    print("2. On subsequent turns, the full history (including reasoning) is sent to the LLM")
    print("3. This lets the model reference its own reasoning from earlier turns")
    print("4. In LangGraph, this is the same pattern — AgentState['messages'] carries the history")
    print()


if __name__ == "__main__":
    demo_single_turn()
    demo_multi_turn()
