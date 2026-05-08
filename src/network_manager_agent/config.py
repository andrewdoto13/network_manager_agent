"""Configuration for the network management agent."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for the LLM backend.

    Defaults to http://10.0.0.228:8080/v1 (OpenAI-compatible endpoint).
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


def create_llm(config: LLMConfig | None = None):
    """Create and return a ChatOpenAI instance."""
    from langchain_openai import ChatOpenAI

    if config is None:
        config = LLMConfig()

    return ChatOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
    )
