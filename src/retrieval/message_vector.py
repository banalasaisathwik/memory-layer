"""Per-conversation semantic retrieval over durable raw Message embeddings.

This is deliberately separate from long-term Memory retrieval: raw messages
only provide pre-extraction context within the current conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from src.config import get_config
from src.database.models import Conversation, Message, MessageRole, User
from src.providers import get_embedding_client

from .errors import (
    IndexDimensionMismatchError,
    IndexModelMismatchError,
    IndexStateError,
    InvalidEmbeddingError,
    InvalidFilterScopeError,
    InvalidSearchError,
    UserNotFoundError,
)
from .vector_support import (
    embedding_response_vectors,
    normalize_embedding,
    safe_fingerprint,
    temporary_path,
)


_INDEX_FORMAT_VERSION = 1
_MAX_SEMANTIC_CONTEXT_MESSAGES = 100
_CONVERSATIONAL_ROLES = (MessageRole.USER, MessageRole.ASSISTANT)


@dataclass(frozen=True)
class ConversationMessageVectorIndex:
    """Loaded derived state for one owned conversation and embedding model."""

    index: Any
    message_ids: list[str]
    embedding_model: str
    dimension: int


def _user_fingerprint(user_external_id: str) -> str:
    return safe_fingerprint(user_external_id)


def _conversation_fingerprint(user_external_id: str, conversation_external_id: str) -> str:
    """Scope the filename to both identities without exposing either on disk."""

    return safe_fingerprint(f"{user_external_id}\x00{conversation_external_id}")


def conversation_message_index_paths(
    user_external_id: str,
    conversation_external_id: str,
    *,
    index_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return the separate FAISS and metadata paths for one conversation."""

    directory = index_dir if index_dir is not None else get_config().faiss_index_dir / "messages"
    stem = _conversation_fingerprint(user_external_id, conversation_external_id)
    return directory / f"{stem}.faiss", directory / f"{stem}.json"


def _resolve_user(db: Session, user_external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == user_external_id))
    if user is None:
        raise UserNotFoundError(f"No user exists for external ID {user_external_id!r}.")
    return user


def _resolve_conversation(
    db: Session,
    *,
    user: User,
    conversation_external_id: str,
) -> Conversation:
    conversations = list(
        db.scalars(
            select(Conversation).where(
                Conversation.user_id == user.id,
                Conversation.external_id == conversation_external_id,
            )
        )
    )
    if not conversations:
        raise InvalidFilterScopeError("No conversation exists in the supplied user scope.")
    if len(conversations) > 1:
        raise InvalidFilterScopeError("Conversation external ID is ambiguous within the supplied user scope.")
    return conversations[0]


def _conversation_messages(db: Session, *, conversation: Conversation) -> list[Message]:
    """Return deterministic, useful raw conversational text only."""

    return list(
        db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                Message.role.in_(_CONVERSATIONAL_ROLES),
                func.btrim(Message.content) != "",
            )
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    )


def _embedding_response_vectors(texts: list[str]) -> list[np.ndarray]:
    """Generate exactly one provider request for a non-empty text batch."""

    return embedding_response_vectors(
        texts,
        model=get_config().embedding_model,
        client_factory=get_embedding_client,
    )


def _embedding_rows_requiring_sync(messages: list[Message], *, model: str) -> list[Message]:
    """Find absent, invalid, model-mismatched, or dimension-incompatible rows."""

    required: list[Message] = []
    dimensions: set[int] = set()
    for message in messages:
        if message.embedding_model != model or message.embedding is None:
            required.append(message)
            continue
        try:
            dimensions.add(normalize_embedding(message.embedding).size)
        except InvalidEmbeddingError:
            required.append(message)

    # A model should produce one stable vector dimension.  With conflicting
    # durable values there is no safe basis for retaining either group, so
    # synchronize this conversation from the configured provider.
    if len(dimensions) > 1:
        return list(messages)
    return required


def _validated_message_matrix(messages: list[Message], *, model: str) -> np.ndarray:
    """Turn durable normalized Message embeddings into a stable matrix."""

    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for message in messages:
        if message.embedding_model != model:
            raise IndexModelMismatchError(
                "Stored message embeddings do not match the configured embedding model; synchronize first."
            )
        try:
            vector = normalize_embedding(message.embedding)
        except InvalidEmbeddingError as error:
            raise InvalidEmbeddingError(
                f"Message {message.id} has an invalid persisted embedding."
            ) from error
        if dimension is None:
            dimension = vector.size
        elif vector.size != dimension:
            raise InvalidEmbeddingError(
                "Stored message embeddings for one model have inconsistent dimensions; re-embed before searching."
            )
        vectors.append(vector)
    if not vectors:
        raise IndexStateError("A FAISS index cannot be built for a conversation with no eligible messages.")
    return np.vstack(vectors).astype(np.float32)


def _persist_conversation_message_index(
    *,
    user_external_id: str,
    conversation_external_id: str,
    index: Any,
    message_ids: list[str],
    embedding_model: str,
    dimension: int,
) -> Path:
    """Persist derived state with the same replace-on-complete convention as Memory."""

    index_path, metadata_path = conversation_message_index_paths(
        user_external_id,
        conversation_external_id,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": _INDEX_FORMAT_VERSION,
        "index_kind": "conversation_messages",
        "user_fingerprint": _user_fingerprint(user_external_id),
        "conversation_fingerprint": _conversation_fingerprint(
            user_external_id,
            conversation_external_id,
        ),
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "message_ids": message_ids,
    }
    temporary_index = temporary_path(index_path.parent, ".faiss.tmp")
    temporary_metadata = temporary_path(index_path.parent, ".json.tmp")
    try:
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary_index, index_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception as error:
        raise IndexStateError("The per-conversation Message FAISS index could not be persisted.") from error
    finally:
        for path in (temporary_index, temporary_metadata):
            if path.exists():
                path.unlink(missing_ok=True)
    return index_path


def load_conversation_message_index(
    *,
    user_external_id: str,
    conversation_external_id: str,
    embedding_model: str | None = None,
) -> ConversationMessageVectorIndex:
    """Load and validate one conversation-scoped Message index."""

    expected_model = embedding_model or get_config().embedding_model
    index_path, metadata_path = conversation_message_index_paths(
        user_external_id,
        conversation_external_id,
    )
    if not index_path.is_file() or not metadata_path.is_file():
        raise IndexStateError("The per-conversation Message FAISS index is missing and must be synchronized.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != _INDEX_FORMAT_VERSION:
            raise IndexStateError("The per-conversation Message FAISS metadata format is unsupported.")
        if metadata.get("index_kind") != "conversation_messages":
            raise IndexStateError("The FAISS metadata is not a Message context index.")
        if metadata.get("user_fingerprint") != _user_fingerprint(user_external_id):
            raise IndexStateError("The per-conversation Message FAISS metadata belongs to a different user.")
        if metadata.get("conversation_fingerprint") != _conversation_fingerprint(
            user_external_id,
            conversation_external_id,
        ):
            raise IndexStateError("The per-conversation Message FAISS metadata belongs to another conversation.")
        if metadata.get("embedding_model") != expected_model:
            raise IndexModelMismatchError(
                "The per-conversation Message FAISS index uses a different embedding model and must be rebuilt."
            )
        dimension = metadata.get("embedding_dimension")
        message_ids = metadata.get("message_ids")
        if not isinstance(dimension, int) or dimension < 1:
            raise IndexStateError("The per-conversation Message FAISS metadata has an invalid embedding dimension.")
        if not isinstance(message_ids, list) or not all(isinstance(item, str) for item in message_ids):
            raise IndexStateError("The per-conversation Message FAISS metadata has an invalid position mapping.")
        index = faiss.read_index(str(index_path))
        if index.d != dimension or index.ntotal != len(message_ids):
            raise IndexStateError("The Message FAISS index and position mapping do not agree.")
    except IndexStateError:
        raise
    except Exception as error:
        raise IndexStateError("The per-conversation Message FAISS index or mapping is corrupt.") from error
    return ConversationMessageVectorIndex(
        index=index,
        message_ids=message_ids,
        embedding_model=expected_model,
        dimension=dimension,
    )


def _synchronize_embeddings(db: Session, *, conversation: Conversation) -> bool:
    """Persist only embeddings that cannot safely serve the configured model."""

    messages = _conversation_messages(db, conversation=conversation)
    missing = _embedding_rows_requiring_sync(messages, model=get_config().embedding_model)
    if not missing:
        return False
    vectors = _embedding_response_vectors([message.content for message in missing])
    expected_dimension: int | None = None
    for message, vector in zip(missing, vectors, strict=True):
        if expected_dimension is None:
            expected_dimension = vector.size
        elif vector.size != expected_dimension:
            raise InvalidEmbeddingError("The embedding provider returned inconsistent vector dimensions.")
        message.embedding = vector.astype(float).tolist()
        message.embedding_model = get_config().embedding_model
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return True


def _build_and_persist_conversation_message_index(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> Path | None:
    messages = _conversation_messages(db, conversation=conversation)
    if not messages:
        return None
    matrix = _validated_message_matrix(messages, model=get_config().embedding_model)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return _persist_conversation_message_index(
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        index=index,
        message_ids=[str(message.id) for message in messages],
        embedding_model=get_config().embedding_model,
        dimension=matrix.shape[1],
    )


def sync_conversation_message_index(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
) -> Path | None:
    """Persist compatible Message vectors and rebuild this conversation's FAISS state."""

    user = _resolve_user(db, user_external_id)
    conversation = _resolve_conversation(
        db,
        user=user,
        conversation_external_id=conversation_external_id,
    )
    _synchronize_embeddings(db, conversation=conversation)
    return _build_and_persist_conversation_message_index(db, user=user, conversation=conversation)


def rebuild_conversation_message_index(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
) -> Path | None:
    """Rebuild derived Message FAISS state from PostgreSQL's authoritative rows."""

    return sync_conversation_message_index(
        db,
        user_external_id=user_external_id,
        conversation_external_id=conversation_external_id,
    )


def _index_matches_database(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> ConversationMessageVectorIndex | None:
    """Require full ordered UUID mapping and compatible durable embeddings."""

    messages = _conversation_messages(db, conversation=conversation)
    if not messages:
        return None
    model = get_config().embedding_model
    if _embedding_rows_requiring_sync(messages, model=model):
        return None
    matrix = _validated_message_matrix(messages, model=model)
    try:
        loaded = load_conversation_message_index(
            user_external_id=user.external_id,
            conversation_external_id=conversation.external_id,
            embedding_model=model,
        )
    except IndexStateError:
        return None
    message_ids = [str(message.id) for message in messages]
    if loaded.message_ids != message_ids or loaded.dimension != matrix.shape[1]:
        return None
    return loaded


def _ensure_conversation_message_index(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> ConversationMessageVectorIndex | None:
    """Safely recover missing, stale, corrupt, or mismatched Message FAISS state."""

    loaded = _index_matches_database(db, user=user, conversation=conversation)
    if loaded is not None:
        return loaded
    sync_conversation_message_index(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )
    return _index_matches_database(db, user=user, conversation=conversation)


def _before_marker(
    db: Session,
    *,
    conversation: Conversation,
    before_message_id: str | None,
) -> Message | None:
    if before_message_id is None:
        return None
    try:
        marker = db.get(Message, before_message_id)
    except (StatementError, TypeError, ValueError) as error:
        raise InvalidFilterScopeError("The semantic context boundary message does not exist.") from error
    if marker is None or marker.conversation_id != conversation.id:
        raise InvalidFilterScopeError("The semantic context boundary must belong to this conversation.")
    return marker


def _precedes(message: Message, marker: Message | None) -> bool:
    if marker is None:
        return True
    return (message.created_at, str(message.id)) < (marker.created_at, str(marker.id))


def retrieve_semantic_message_context(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
    query_text: str,
    limit: int = 3,
    exclude_message_ids: set[str] | None = None,
    before_message_id: str | None = None,
) -> list[Message]:
    """Return ranked semantic candidates as chronological, DB-validated Messages.

    This function searches only one owned conversation.  FAISS positions remain
    private derived state; returned rows always come from PostgreSQL.
    """

    if not query_text.strip():
        return []
    if limit < 1 or limit > _MAX_SEMANTIC_CONTEXT_MESSAGES:
        raise InvalidSearchError("Semantic message context limit must be between 1 and 100.")

    user = _resolve_user(db, user_external_id)
    conversation = _resolve_conversation(
        db,
        user=user,
        conversation_external_id=conversation_external_id,
    )
    marker = _before_marker(
        db,
        conversation=conversation,
        before_message_id=before_message_id,
    )
    query_vector = _embedding_response_vectors([query_text])[0]
    loaded = _ensure_conversation_message_index(db, user=user, conversation=conversation)
    if loaded is None:
        return []
    if query_vector.size != loaded.dimension:
        raise IndexDimensionMismatchError(
            "The semantic message query dimension differs from the persisted index; rebuild with compatible embeddings."
        )

    requested = min(
        loaded.index.ntotal,
        max(limit, limit * get_config().vector_candidate_multiplier),
    )
    if requested == 0:
        return []
    _, positions = loaded.index.search(query_vector.reshape(1, -1), requested)
    candidate_ids = [
        loaded.message_ids[position]
        for position in positions[0]
        if 0 <= position < len(loaded.message_ids)
    ]
    if not candidate_ids:
        return []

    excluded = {str(message_id) for message_id in (exclude_message_ids or set())}
    rows = list(
        db.scalars(
            select(Message).where(
                Message.conversation_id == conversation.id,
                Message.id.in_(candidate_ids),
                Message.role.in_(_CONVERSATIONAL_ROLES),
                func.btrim(Message.content) != "",
            )
        )
    )
    by_id = {str(message.id): message for message in rows}
    selected = [
        message
        for message_id in candidate_ids
        if (message := by_id.get(message_id)) is not None
        and message_id not in excluded
        and _precedes(message, marker)
    ][:limit]
    selected.sort(key=lambda message: (message.created_at, str(message.id)))
    return selected
