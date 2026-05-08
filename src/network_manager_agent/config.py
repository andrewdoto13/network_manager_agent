"""Configuration for the network management agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:
    """Configuration for the LLM backend.

    Defaults to http://localhost:8080/v1 (OpenAI-compatible endpoint).
    Override via LLM_BASE_URL environment variable for local/dev use.
    """
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    api_key: str = os.getenv("LLM_API_KEY", "not-needed")
    model: str = os.getenv("LLM_MODEL", "llm")


SUMMARIZE_THRESHOLD = 14
MESSAGES_TO_ARCHIVE = SUMMARIZE_THRESHOLD // 2
SERVICE_AREA_BUFFER_MILES = 20  # Buffer added to max threshold when filtering service area

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI subclass that captures reasoning_content from responses.

    The base ChatOpenAI class targets the official OpenAI API spec and does not
    extract non-standard response fields like ``reasoning_content`` (used by
    Qwen via llama.cpp).  This subclass overrides ``_create_chat_result`` to
    copy ``reasoning_content`` from the raw message dict into each
    ``AIMessage.additional_kwargs["reasoning_content"]`` so the reasoning trace
    is preserved in the LangGraph message history.
    """

    def _create_chat_result(
        self,
        response: dict[str, Any] | Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        choices = response_dict.get("choices", [])

        for i, gen in enumerate(result.generations):
            if i < len(choices):
                msg_dict = choices[i].get("message", {})
                reasoning_content = msg_dict.get("reasoning_content")
                if reasoning_content:
                    gen.message.additional_kwargs["reasoning_content"] = reasoning_content

        return result


def create_llm(config: LLMConfig | None = None) -> ChatOpenAIWithReasoning:
    """Create and return a ChatOpenAI instance that captures reasoning_content."""

    if config is None:
        config = LLMConfig()

    return ChatOpenAIWithReasoning(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
    )
