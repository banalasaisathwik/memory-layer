"""Durable embeddings and exact, per-user FAISS vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import get_config
from src.database.models import Conversation, Memory, User
from src.providers import get_embedding_client

from .errors import (
    EmbeddingError,
    IndexDimensionMismatchError,
    IndexModelMismatchError,
    IndexStateError,
    InvalidEmbeddingError,
    UserNotFoundError,
)
from .schemas import SearchFilters
from .structured import memory_filter_conditions
from .vector_support import (
    embedding_response_vectors,
    load_and_validate_index,
    normalize_embedding,
    persist_index,
    safe_fingerprint,
)


_INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class UserVectorIndex:
    """Loaded derived state for one user and one configured embedding model."""

    index: Any
    memory_ids: list[str]
    embedding_model: str
    dimension: int


def _user_fingerprint(user_external_id: str) -> str:
    """Create a filesystem-safe identity without exposing an external user ID."""

    return safe_fingerprint(user_external_id)


def user_index_paths(
    user_external_id: str,
    *,
    index_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return the FAISS and metadata paths for one user without creating them."""

    directory = index_dir if index_dir is not None else get_config().faiss_index_dir
    stem = _user_fingerprint(user_external_id)
    return directory / f"{stem}.faiss", directory / f"{stem}.json"


def _resolve_user(db: Session, user_external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == user_external_id))
    if user is None:
        raise UserNotFoundError(f"No user exists for external ID {user_external_id!r}.")
    return user


def _user_memories(db: Session, *, user: User) -> list[Memory]:
    """Use stable ordering so persisted positions reproduce across rebuilds."""

    return list(
        db.scalars(
            select(Memory)
            .where(Memory.user_id == user.id)
            .order_by(Memory.created_at.asc(), Memory.id.asc())
        )
    )


def _normalize_embedding(values: object) -> np.ndarray:
    """Validate and normalize one vector for IndexFlatIP cosine search."""

    return normalize_embedding(values)


def _embedding_response_vectors(texts: list[str]) -> list[np.ndarray]:
    """Create one provider request for a batch and validate its response shape."""

    return embedding_response_vectors(
        texts,
        model=get_config().embedding_model,
        client_factory=get_embedding_client,
    )


def _embedding_rows_requiring_sync(memories: list[Memory], *, model: str) -> list[Memory]:
    """Find rows that cannot safely join the configured model's index."""

    return [
        memory
        for memory in memories
        if memory.embedding_model != model or memory.embedding is None
    ]


def _validated_memory_matrix(memories: list[Memory], *, model: str) -> np.ndarray:
    """Turn persisted normalized vectors into one dimension-consistent matrix."""

    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for memory in memories:
        if memory.embedding_model != model:
            raise IndexModelMismatchError(
                "Stored embeddings do not match the configured embedding model; synchronize first."
            )
        try:
            vector = _normalize_embedding(memory.embedding)
        except InvalidEmbeddingError as error:
            raise InvalidEmbeddingError(
                f"Memory {memory.id} has an invalid persisted embedding."
            ) from error
        if dimension is None:
            dimension = vector.size
        elif vector.size != dimension:
            raise InvalidEmbeddingError(
                "Stored embeddings for one model have inconsistent dimensions; re-embed before searching."
            )
        vectors.append(vector)
    if not vectors:
        raise IndexStateError("A FAISS index cannot be built for a user with no memories.")
    return np.vstack(vectors).astype(np.float32)


def _persist_user_index(
    *,
    user_external_id: str,
    index: Any,
    memory_ids: list[str],
    embedding_model: str,
    dimension: int,
) -> Path:
    """Persist index and position mapping with replace-on-complete files."""

    index_path, metadata_path = user_index_paths(user_external_id)
    metadata = {
        "format_version": _INDEX_FORMAT_VERSION,
        "user_fingerprint": _user_fingerprint(user_external_id),
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "memory_ids": memory_ids,
    }
    persist_index(
        index=index,
        index_path=index_path,
        metadata_path=metadata_path,
        metadata=metadata,
        error_message="The per-user FAISS index could not be persisted.",
    )
    return index_path


def load_user_memory_index(
    *,
    user_external_id: str,
    embedding_model: str | None = None,
) -> UserVectorIndex:
    """Load and validate a local derived index before it can serve candidates."""

    expected_model = embedding_model or get_config().embedding_model
    index_path, metadata_path = user_index_paths(user_external_id)

    def _validate_metadata(metadata: dict[str, Any]) -> tuple[list[str], int]:
        if metadata.get("format_version") != _INDEX_FORMAT_VERSION:
            raise IndexStateError("The per-user FAISS metadata format is unsupported.")
        if metadata.get("user_fingerprint") != _user_fingerprint(user_external_id):
            raise IndexStateError("The per-user FAISS metadata belongs to a different user.")
        if metadata.get("embedding_model") != expected_model:
            raise IndexModelMismatchError(
                "The per-user FAISS index uses a different embedding model and must be rebuilt."
            )
        dimension = metadata.get("embedding_dimension")
        memory_ids = metadata.get("memory_ids")
        if not isinstance(dimension, int) or dimension < 1:
            raise IndexStateError("The per-user FAISS metadata has an invalid embedding dimension.")
        if not isinstance(memory_ids, list) or not all(isinstance(item, str) for item in memory_ids):
            raise IndexStateError("The per-user FAISS metadata has an invalid position mapping.")
        return memory_ids, dimension

    index, memory_ids, dimension = load_and_validate_index(
        index_path=index_path,
        metadata_path=metadata_path,
        missing_message="The per-user FAISS index is missing and must be synchronized.",
        corrupt_message="The per-user FAISS index or mapping is corrupt.",
        validate_metadata=_validate_metadata,
    )
    return UserVectorIndex(
        index=index,
        memory_ids=memory_ids,
        embedding_model=expected_model,
        dimension=dimension,
    )


def _build_and_persist_user_index(db: Session, *, user: User) -> Path | None:
    """Build the exact IndexFlatIP state from already durable embeddings."""

    memories = _user_memories(db, user=user)
    if not memories:
        return None
    model = get_config().embedding_model
    matrix = _validated_memory_matrix(memories, model=model)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return _persist_user_index(
        user_external_id=user.external_id,
        index=index,
        memory_ids=[str(memory.id) for memory in memories],
        embedding_model=model,
        dimension=matrix.shape[1],
    )


def _synchronize_embeddings(db: Session, *, user: User) -> bool:
    """Persist only embeddings absent from the current configured model."""

    memories = _user_memories(db, user=user)
    missing = _embedding_rows_requiring_sync(memories, model=get_config().embedding_model)
    if not missing:
        return False
    vectors = _embedding_response_vectors([memory.memory_text for memory in missing])
    expected_dimension: int | None = None
    for memory, vector in zip(missing, vectors, strict=True):
        if expected_dimension is None:
            expected_dimension = vector.size
        elif vector.size != expected_dimension:
            raise InvalidEmbeddingError("The embedding provider returned inconsistent vector dimensions.")
        memory.embedding = vector.astype(float).tolist()
        memory.embedding_model = get_config().embedding_model
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return True


def sync_user_memory_index(db: Session, *, user_external_id: str) -> Path | None:
    """Embed missing rows for the configured model, commit them, and rebuild FAISS."""

    user = _resolve_user(db, user_external_id)
    _synchronize_embeddings(db, user=user)
    return _build_and_persist_user_index(db, user=user)


def rebuild_user_memory_index(db: Session, *, user_external_id: str) -> Path | None:
    """Rebuild derived FAISS state from PostgreSQL's durable embedding columns."""

    return sync_user_memory_index(db, user_external_id=user_external_id)


def _index_matches_database(
    db: Session,
    *,
    user: User,
) -> UserVectorIndex | None:
    """Accept a local index only when its model and full UUID mapping match PostgreSQL."""

    memories = _user_memories(db, user=user)
    if not memories:
        return None
    current_model = get_config().embedding_model
    if _embedding_rows_requiring_sync(memories, model=current_model):
        return None
    matrix = _validated_memory_matrix(memories, model=current_model)
    try:
        loaded = load_user_memory_index(
            user_external_id=user.external_id,
            embedding_model=current_model,
        )
    except IndexStateError:
        return None
    memory_ids = [str(memory.id) for memory in memories]
    if loaded.memory_ids != memory_ids or loaded.dimension != matrix.shape[1]:
        return None
    return loaded


def _ensure_user_memory_index(db: Session, *, user: User) -> UserVectorIndex | None:
    """Recover derived state safely from PostgreSQL when it is missing, stale, or corrupt."""

    loaded = _index_matches_database(db, user=user)
    if loaded is not None:
        return loaded
    sync_user_memory_index(db, user_external_id=user.external_id)
    return _index_matches_database(db, user=user)


def vector_retrieve(
    db: Session,
    query: str,
    *,
    user: User,
    filters: SearchFilters,
    conversation: Conversation | None,
    limit: int,
) -> list[Memory]:
    """Run one query embedding through a user-only exact cosine FAISS index."""

    # A search owns exactly one query embedding. Structured and lexical
    # branches do not make embeddings, and this vector is reused for FAISS.
    query_vector = _embedding_response_vectors([query])[0]
    loaded = _ensure_user_memory_index(db, user=user)
    if loaded is None:
        return []
    if query_vector.size != loaded.dimension:
        raise IndexDimensionMismatchError(
            "The query embedding dimension differs from the persisted index; rebuild with compatible embeddings."
        )
    requested = min(
        loaded.index.ntotal,
        max(limit, limit * get_config().vector_candidate_multiplier),
    )
    if requested == 0:
        return []
    _, positions = loaded.index.search(query_vector.reshape(1, -1), requested)
    candidate_ids = [
        loaded.memory_ids[position]
        for position in positions[0]
        if position >= 0 and position < len(loaded.memory_ids)
    ]
    if not candidate_ids:
        return []
    # PostgreSQL remains authoritative for state and caller filters. This is
    # after FAISS lookup; user isolation was already established by index key.
    rows = list(
        db.scalars(
            select(Memory).where(
                Memory.user_id == user.id,
                Memory.id.in_(candidate_ids),
                *memory_filter_conditions(filters, conversation=conversation),
            )
        )
    )
    by_id = {str(memory.id): memory for memory in rows}
    return [by_id[memory_id] for memory_id in candidate_ids if memory_id in by_id]
