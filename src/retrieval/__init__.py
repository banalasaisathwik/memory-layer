"""User-scoped hybrid retrieval over durable Memory rows."""

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
from .vector import (
    load_user_memory_index,
    rebuild_user_memory_index,
    sync_user_memory_index,
    user_index_paths,
)

__all__ = [
    "EmbeddingError",
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
    "rebuild_user_memory_index",
    "reciprocal_rank_fusion",
    "search_memories",
    "sync_user_memory_index",
    "user_index_paths",
]
