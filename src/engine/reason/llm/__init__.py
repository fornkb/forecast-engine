"""Model providers behind one interface. Pick one per client in `reasoning.llm`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from engine.reason.config import LLMConfig
from engine.reason.llm.base import LLMClient, LLMError, LLMResult, api_key, strict_schema

Mode = Literal["live", "record", "replay"]


def _live(config: LLMConfig) -> LLMClient:
    key = api_key(config.key_env)
    options = dict(timeout=config.timeout_seconds, max_retries=config.max_retries)
    if config.provider == "openai":
        from engine.reason.llm.openai_client import OpenAIClient

        return OpenAIClient(key, **options)
    if config.provider == "anthropic":
        from engine.reason.llm.anthropic_client import AnthropicClient

        return AnthropicClient(key, **options)
    if config.provider == "gemini":
        from engine.reason.llm.gemini_client import GeminiClient

        return GeminiClient(key, **options)
    raise LLMError(f"Unknown provider {config.provider!r}")


def get_client(config: LLMConfig, mode: Mode, recordings: Path | None) -> LLMClient:
    """Live client, or a recorder/replayer around it for offline demos and tests."""
    from engine.reason.llm.replay import ReplayClient

    if config.provider == "replay" or mode == "replay":
        if recordings is None:
            raise LLMError("Replay needs reasoning.llm.recordings_dir")
        return ReplayClient(recordings, config.provider)
    if mode == "record":
        if recordings is None:
            raise LLMError("Record needs reasoning.llm.recordings_dir")
        return ReplayClient(recordings, config.provider, inner=_live(config))
    return _live(config)


__all__ = ["LLMClient", "LLMError", "LLMResult", "Mode", "get_client", "strict_schema"]
