"""Lazy client creation for embedding providers."""

from __future__ import annotations

from openai import OpenAI

from src.config import Settings, get_config
from src.providers.llm import OPENROUTER_BASE_URL


_client: OpenAI | None = None
_client_key: tuple[str, str, str | None] | None = None


def _client_arguments(settings: Settings) -> tuple[tuple[str, str, str | None], dict[str, str]]:
    if not settings.embedding_api_key:
        raise RuntimeError("EMBEDDING_API_KEY must be configured before creating an embedding client.")

    base_url = settings.embedding_base_url
    if settings.embedding_provider == "openrouter" and not base_url:
        base_url = OPENROUTER_BASE_URL

    key = (settings.embedding_provider, settings.embedding_api_key, base_url)
    arguments = {"api_key": settings.embedding_api_key}
    if base_url:
        arguments["base_url"] = base_url
    return key, arguments


def get_embedding_client() -> OpenAI:
    """Return a cached OpenAI-compatible client without generating embeddings."""

    global _client, _client_key
    key, arguments = _client_arguments(get_config())
    if _client is None or _client_key != key:
        _client = OpenAI(**arguments)
        _client_key = key
    return _client


def reset_embedding_client() -> None:
    """Clear the cached client, primarily for isolated tests or reconfiguration."""

    global _client, _client_key
    _client = None
    _client_key = None
