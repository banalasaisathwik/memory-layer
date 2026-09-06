"""Lazy, OpenAI-compatible clients configured independently by purpose."""

from .embeddings import get_embedding_client, reset_embedding_client
from .llm import get_llm_client, reset_llm_client

__all__ = [
    "get_embedding_client",
    "get_llm_client",
    "reset_embedding_client",
    "reset_llm_client",
]
