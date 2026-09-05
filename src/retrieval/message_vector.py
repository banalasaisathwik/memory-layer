"""Per-conversation semantic retrieval over durable raw Message embeddings.

This is deliberately separate from long-term Memory retrieval: raw messages
only provide pre-extraction context within the current conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    load_and_validate_index,
    normalize_embedding,
    persist_index,
    safe_fingerprint,
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


@dataclass
class MessageIndexSyncStats:
    """Simple hot-path counters proving append growth no longer forces a full rebuild.

    Diagnostics only -- nothing here changes retrieval behavior. A benchmark
    runner can read these before/after a long ingestion run to see how many
    times each path actually executed.
    """

    full_rebuilds: int = 0
    incremental_appends: int = 0
    unchanged_hits: int = 0


_sync_stats = MessageIndexSyncStats()


def get_message_index_sync_stats() -> MessageIndexSyncStats:
    """Return the process-wide counters accumulated so far."""

    return _sync_stats


def reset_message_index_sync_stats() -> None:
    """Zero the counters; useful at the start of a benchmark run or a test."""

    global _sync_stats
    _sync_stats = MessageIndexSyncStats()


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


def _conversation_message_ids(db: Session, *, conversation: Conversation) -> list[str]:
    """Cheap ordered id-only fetch used to detect append-only growth.

    Deliberately avoids selecting ``content``/``embedding`` columns: this is
    the check run on every retrieval call, so it must stay far cheaper than
    fetching every message's full text and vector across the network.
    """

    return [
        str(message_id)
        for message_id in db.scalars(
            select(Message.id)
            .where(
                Message.conversation_id == conversation.id,
                Message.role.in_(_CONVERSATIONAL_ROLES),
                func.btrim(Message.content) != "",
            )
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    ]


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
    persist_index(
        index=index,
        index_path=index_path,
        metadata_path=metadata_path,
        metadata=metadata,
        error_message="The per-conversation Message FAISS index could not be persisted.",
    )
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

    def _validate_metadata(metadata: dict[str, Any]) -> tuple[list[str], int]:
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
        return message_ids, dimension

    index, message_ids, dimension = load_and_validate_index(
        index_path=index_path,
        metadata_path=metadata_path,
        missing_message="The per-conversation Message FAISS index is missing and must be synchronized.",
        corrupt_message="The per-conversation Message FAISS index or mapping is corrupt.",
        validate_metadata=_validate_metadata,
    )
    return ConversationMessageVectorIndex(
        index=index,
        message_ids=message_ids,
        embedding_model=expected_model,
        dimension=dimension,
    )


def _persist_required_embeddings(db: Session, missing: list[Message], *, model: str) -> None:
    """Embed and durably persist exactly the given rows; the only embedding writer.

    Shared by the full-conversation sync path and the incremental-append path
    so there is exactly one place that turns Message content into a
    persisted, model-tagged embedding column.
    """

    if not missing:
        return
    vectors = _embedding_response_vectors([message.content for message in missing])
    expected_dimension: int | None = None
    for message, vector in zip(missing, vectors, strict=True):
        if expected_dimension is None:
            expected_dimension = vector.size
        elif vector.size != expected_dimension:
            raise InvalidEmbeddingError("The embedding provider returned inconsistent vector dimensions.")
        message.embedding = vector.astype(float).tolist()
        message.embedding_model = model
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def _synchronize_embeddings(db: Session, *, conversation: Conversation) -> bool:
    """Persist only embeddings that cannot safely serve the configured model."""

    messages = _conversation_messages(db, conversation=conversation)
    model = get_config().embedding_model
    missing = _embedding_rows_requiring_sync(messages, model=model)
    if not missing:
        return False
    _persist_required_embeddings(db, missing, model=model)
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
    path = _persist_conversation_message_index(
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        index=index,
        message_ids=[str(message.id) for message in messages],
        embedding_model=get_config().embedding_model,
        dimension=matrix.shape[1],
    )
    _sync_stats.full_rebuilds += 1
    return path


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


def _fast_index_match(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> ConversationMessageVectorIndex | None:
    """Cheap steady-state check: only compares ordered message IDs.

    This is the path a call pays for when nothing changed since the last
    sync -- one id-only query plus a local FAISS/metadata file read, with no
    content or embedding payload transfer and no re-embedding. It returns
    None (never raises) for anything that needs the slower, authoritative
    checks: a missing/corrupt index, a model mismatch, or any ID divergence.
    """

    try:
        loaded = load_conversation_message_index(
            user_external_id=user.external_id,
            conversation_external_id=conversation.external_id,
        )
    except IndexStateError:
        return None
    if loaded.message_ids != _conversation_message_ids(db, conversation=conversation):
        return None
    _sync_stats.unchanged_hits += 1
    return loaded


def _try_incremental_append(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> ConversationMessageVectorIndex | None:
    """Append only newly persisted messages onto a still-valid index.

    Safe only when the persisted index's message IDs are an exact, ordered
    prefix of the database's current message IDs -- i.e. pure append growth
    with no deletion, reordering, or model/dimension change. Returns None
    (making no changes) for anything else, so the caller falls back to the
    full, authoritative validate-or-rebuild path. Existing messages are never
    re-embedded or re-added; only the new suffix is embedded (if not already
    durably embedded) and added to the loaded FAISS index.
    """

    model = get_config().embedding_model
    try:
        loaded = load_conversation_message_index(
            user_external_id=user.external_id,
            conversation_external_id=conversation.external_id,
            embedding_model=model,
        )
    except IndexStateError:
        return None

    old_ids = loaded.message_ids
    current_ids = _conversation_message_ids(db, conversation=conversation)
    if len(current_ids) <= len(old_ids) or current_ids[: len(old_ids)] != old_ids:
        return None

    new_ids = current_ids[len(old_ids) :]
    new_messages = list(
        db.scalars(
            select(Message)
            .where(Message.id.in_(new_ids))
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    )
    if [str(message.id) for message in new_messages] != new_ids:
        # A concurrent delete or an ordering surprise; do not guess.
        return None

    missing = _embedding_rows_requiring_sync(new_messages, model=model)
    _persist_required_embeddings(db, missing, model=model)

    try:
        new_matrix = _validated_message_matrix(new_messages, model=model)
    except (IndexStateError, InvalidEmbeddingError):
        return None
    if new_matrix.shape[1] != loaded.dimension:
        return None

    loaded.index.add(new_matrix)
    updated_ids = old_ids + new_ids
    _persist_conversation_message_index(
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        index=loaded.index,
        message_ids=updated_ids,
        embedding_model=model,
        dimension=loaded.dimension,
    )
    _sync_stats.incremental_appends += 1
    return ConversationMessageVectorIndex(
        index=loaded.index,
        message_ids=updated_ids,
        embedding_model=model,
        dimension=loaded.dimension,
    )


def _ensure_conversation_message_index(
    db: Session,
    *,
    user: User,
    conversation: Conversation,
) -> ConversationMessageVectorIndex | None:
    """Safely recover missing, stale, corrupt, or mismatched Message FAISS state.

    Tries three paths, cheapest and most common first: an unchanged index
    (id comparison only), an append-only incremental update (embed and add
    only the new suffix), then the full, authoritative validate-or-rebuild
    path that already existed. The third path's correctness is unchanged;
    the first two only short-circuit it when they can prove it is safe to.
    """

    fast = _fast_index_match(db, user=user, conversation=conversation)
    if fast is not None:
        return fast
    appended = _try_incremental_append(db, user=user, conversation=conversation)
    if appended is not None:
        return appended
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
    chronological: bool = True,
) -> list[Message]:
    """Return ranked semantic candidates as DB-validated Messages.

    This function searches only one owned conversation.  FAISS positions remain
    private derived state; returned rows always come from PostgreSQL.

    By default the result is sorted chronologically for direct callers.  Pass
    ``chronological=False`` to keep FAISS similarity-rank order instead (best
    match first) -- extraction-context fusion needs the rank order, since
    fusing an already chronologically-sorted branch would discard its
    relevance signal.
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
    if chronological:
        selected.sort(key=lambda message: (message.created_at, str(message.id)))
    return selected
