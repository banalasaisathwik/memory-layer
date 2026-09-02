"""Lazy client creation for LLM completion providers."""

from __future__ import annotations

from openai import OpenAI

from src.config import Settings, get_config


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_client: OpenAI | None = None
_client_key: tuple[str, str, str | None] | None = None


def _client_arguments(settings: Settings) -> tuple[tuple[str, str, str | None], dict[str, str]]:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY must be configured before creating an LLM client.")

    base_url = settings.llm_base_url
    if settings.llm_provider == "openrouter" and not base_url:
        base_url = OPENROUTER_BASE_URL

    key = (settings.llm_provider, settings.llm_api_key, base_url)
    arguments = {"api_key": settings.llm_api_key}
    if base_url:
        arguments["base_url"] = base_url
    return key, arguments


def get_llm_client() -> OpenAI:
    """Return a cached OpenAI-compatible client without sending a request."""

    global _client, _client_key
    key, arguments = _client_arguments(get_config())
    if _client is None or _client_key != key:
        _client = OpenAI(**arguments)
        _client_key = key
    return _client


def reset_llm_client() -> None:
    """Clear the cached client, primarily for isolated tests or reconfiguration."""

    global _client, _client_key
    _client = None
    _client_key = None
