"""Durable Memory retrieval and per-conversation Message context retrieval."""

from .errors import (
    EmbeddingError,
    IndexDimensionMismatchError,
    IndexModelMismatchError,
    IndexStateError,
    InvalidEmbeddingError,
    InvalidFilterScopeError,
    InvalidSearchError,
    RetrievalError,
    UserNotFoundError,
)
from .fusion import FusedMemory, RRF_K, reciprocal_rank_fusion
from .schemas import SearchFilters, SearchHit
from .search import search_memories
from .message_vector import (
    conversation_message_index_paths,
    load_conversation_message_index,
    rebuild_conversation_message_index,
    retrieve_semantic_message_context,
    sync_conversation_message_index,
)
from .vector import (
    load_user_memory_index,
    rebuild_user_memory_index,
    sync_user_memory_index,
    user_index_paths,
)

__all__ = [
    "EmbeddingError",
    "conversation_message_index_paths",
    "FusedMemory",
    "IndexDimensionMismatchError",
    "IndexModelMismatchError",
    "IndexStateError",
    "InvalidEmbeddingError",
    "InvalidFilterScopeError",
    "InvalidSearchError",
    "RRF_K",
    "RetrievalError",
    "SearchFilters",
    "SearchHit",
    "UserNotFoundError",
    "load_user_memory_index",
    "load_conversation_message_index",
    "rebuild_conversation_message_index",
    "rebuild_user_memory_index",
    "reciprocal_rank_fusion",
    "search_memories",
    "retrieve_semantic_message_context",
    "sync_conversation_message_index",
    "sync_user_memory_index",
    "user_index_paths",
]
