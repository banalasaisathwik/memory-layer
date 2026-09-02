"""Validated, deterministic PostgreSQL writes for extracted memory candidates."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, MemoryType, Message, User, utcnow

from .fact_keys import build_fact_key, normalize_subject_type, normalize_value_identity
from .predicates import get_predicate_cardinality, resolve_predicate
from .schemas import CandidateMemory


class WriteError(Exception):
    """Raised when a candidate cannot safely be written in its requested scope."""


class WriteAction(str, Enum):
    """The deterministic outcome for one memory candidate."""

    ADD = "ADD"
    NOOP = "NOOP"
    SUPERSEDE = "SUPERSEDE"


@dataclass(frozen=True)
class WriteResult:
    """A compact durable-write outcome suitable for callers and later observability."""

    action: WriteAction
    memory_id: str
    fact_key: str | None
    superseded_memory_id: str | None
    reason: str


def _normalize_memory_text(memory_text: str) -> str:
    """Normalize only exact-text identity for the intentionally conservative open path."""

    normalized = unicodedata.normalize("NFKC", memory_text)
    return " ".join(normalized.split()).casefold()


def _stable_unique_source_ids(source_message_ids: list[str | int]) -> list[str]:
    """Convert validated message IDs to a stable, duplicate-free persistence form."""

    result: list[str] = []
    seen: set[str] = set()
    for source_message_id in source_message_ids:
        normalized = str(source_message_id)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _merge_provenance(existing: Memory, source_message_ids: list[str]) -> None:
    """Append new evidence in deterministic first-seen order without duplicate IDs."""

    existing.source_message_ids = _stable_unique_source_ids(
        [*existing.source_message_ids, *source_message_ids]
    )


def _resolve_user(db: Session, user_external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == user_external_id))
    if user is None:
        raise WriteError(f"No user exists for external ID {user_external_id!r}.")
    return user


def _resolve_conversation(
    db: Session,
    *,
    user: User,
    conversation_external_id: str | None,
) -> Conversation | None:
    if conversation_external_id is None:
        return None

    conversations = list(
        db.scalars(
            select(Conversation).where(Conversation.external_id == conversation_external_id)
        )
    )
    if not conversations:
        raise WriteError(f"No conversation exists for external ID {conversation_external_id!r}.")
    if len(conversations) > 1:
        raise WriteError(f"Conversation external ID {conversation_external_id!r} is ambiguous.")

    conversation = conversations[0]
    if conversation.user_id != user.id:
        raise WriteError("The supplied conversation does not belong to the requested user.")
    return conversation


def _validate_source_messages(
    db: Session,
    *,
    source_message_ids: list[str | int],
    user: User,
    conversation: Conversation | None,
) -> list[str]:
    """Validate application-owned provenance before it can be attached to a memory."""

    normalized_ids = _stable_unique_source_ids(source_message_ids)
    for source_message_id in normalized_ids:
        try:
            message = db.get(Message, source_message_id)
        except (StatementError, ValueError, TypeError) as error:
            raise WriteError(f"Source message {source_message_id!r} does not exist.") from error

        if message is None:
            raise WriteError(f"Source message {source_message_id!r} does not exist.")

        message_conversation = message.conversation
        if message_conversation.user_id != user.id:
            raise WriteError("Source message provenance belongs to a different user.")
        if conversation is not None and message.conversation_id != conversation.id:
            raise WriteError("Source message provenance does not belong to the supplied conversation.")
    return normalized_ids


def _canonical_subject_id(candidate: CandidateMemory, user: User) -> str | None:
    """Supply the only V1 canonical subject identity: the scoped user."""

    return user.external_id if normalize_subject_type(candidate.subject_type) == "user" else None


def _new_memory(
    *,
    candidate: CandidateMemory,
    user: User,
    conversation: Conversation | None,
    source_message_ids: list[str],
    subject_id: str | None,
    predicate: str | None,
    fact_key: str | None,
) -> Memory:
    """Map one candidate to the durable model after all deterministic validation."""

    return Memory(
        user_id=user.id,
        conversation_id=conversation.id if conversation is not None else None,
        memory_type=candidate.memory_type,
        memory_text=candidate.memory_text,
        subject_type="user" if subject_id is not None else candidate.subject_type,
        subject_id=subject_id,
        predicate=predicate,
        value=candidate.value if predicate is not None else None,
        fact_key=fact_key,
        confidence=candidate.confidence,
        importance=candidate.importance,
        source_message_ids=source_message_ids,
        valid_from=utcnow(),
        is_active=True,
    )


def _active_memories_for_fact_key(db: Session, *, user: User, fact_key: str) -> list[Memory]:
    """Find active structured facts inside the mandatory long-term user scope."""

    return list(
        db.scalars(
            select(Memory)
            .where(
                Memory.user_id == user.id,
                Memory.fact_key == fact_key,
                Memory.is_active.is_(True),
            )
            .order_by(Memory.created_at, Memory.id)
        )
    )


def _active_open_semantic_duplicate(
    db: Session,
    *,
    user: User,
    memory_text: str,
) -> Memory | None:
    for memory in db.scalars(
        select(Memory)
        .where(
            Memory.user_id == user.id,
            Memory.memory_type == MemoryType.SEMANTIC,
            Memory.fact_key.is_(None),
            Memory.is_active.is_(True),
        )
        .order_by(Memory.created_at, Memory.id)
    ):
        if _normalize_memory_text(memory.memory_text) == _normalize_memory_text(memory_text):
            return memory
    return None


def _active_episodic_duplicate(
    db: Session,
    *,
    user: User,
    memory_text: str,
    source_message_ids: list[str],
) -> Memory | None:
    wanted_sources = set(source_message_ids)
    for memory in db.scalars(
        select(Memory)
        .where(
            Memory.user_id == user.id,
            Memory.memory_type == MemoryType.EPISODIC,
            Memory.is_active.is_(True),
        )
        .order_by(Memory.created_at, Memory.id)
    ):
        if (
            _normalize_memory_text(memory.memory_text) == _normalize_memory_text(memory_text)
            and set(memory.source_message_ids) == wanted_sources
        ):
            return memory
    return None


def _add_memory(db: Session, memory: Memory, *, reason: str) -> WriteResult:
    db.add(memory)
    db.flush()
    return WriteResult(
        action=WriteAction.ADD,
        memory_id=str(memory.id),
        fact_key=memory.fact_key,
        superseded_memory_id=None,
        reason=reason,
    )


def _write_candidate(
    db: Session,
    *,
    candidate: CandidateMemory,
    user: User,
    conversation: Conversation | None,
) -> WriteResult:
    source_message_ids = _validate_source_messages(
        db,
        source_message_ids=candidate.source_message_ids,
        user=user,
        conversation=conversation,
    )
    subject_id = _canonical_subject_id(candidate, user)
    predicate = resolve_predicate(candidate.predicate)
    fact_key = build_fact_key(candidate, subject_id=subject_id)
    memory = _new_memory(
        candidate=candidate,
        user=user,
        conversation=conversation,
        source_message_ids=source_message_ids,
        subject_id=subject_id,
        predicate=predicate,
        fact_key=fact_key,
    )

    if candidate.memory_type is MemoryType.EPISODIC:
        existing = _active_episodic_duplicate(
            db,
            user=user,
            memory_text=candidate.memory_text,
            source_message_ids=source_message_ids,
        )
        if existing is None:
            return _add_memory(db, memory, reason="new episodic memory")
        _merge_provenance(existing, source_message_ids)
        return WriteResult(
            action=WriteAction.NOOP,
            memory_id=str(existing.id),
            fact_key=None,
            superseded_memory_id=None,
            reason="same episodic memory and source provenance already exist",
        )

    if fact_key is None:
        existing = _active_open_semantic_duplicate(db, user=user, memory_text=candidate.memory_text)
        if existing is None:
            return _add_memory(db, memory, reason="no exact open semantic duplicate exists")
        _merge_provenance(existing, source_message_ids)
        return WriteResult(
            action=WriteAction.NOOP,
            memory_id=str(existing.id),
            fact_key=None,
            superseded_memory_id=None,
            reason="equivalent open semantic memory already exists",
        )

    active_memories = _active_memories_for_fact_key(db, user=user, fact_key=fact_key)
    if not active_memories:
        return _add_memory(db, memory, reason="no active memory exists for this fact key")
    if len(active_memories) > 1:
        raise WriteError("More than one active memory exists for the same scoped fact key.")

    existing = active_memories[0]
    if get_predicate_cardinality(predicate) == "multi" or (
        normalize_value_identity(existing.value) == normalize_value_identity(memory.value)
    ):
        _merge_provenance(existing, source_message_ids)
        return WriteResult(
            action=WriteAction.NOOP,
            memory_id=str(existing.id),
            fact_key=fact_key,
            superseded_memory_id=None,
            reason="equivalent active structured fact already exists",
        )

    boundary = utcnow()
    memory.valid_from = boundary
    db.add(memory)
    db.flush()
    existing.is_active = False
    existing.valid_to = boundary
    existing.superseded_by_id = memory.id
    return WriteResult(
        action=WriteAction.SUPERSEDE,
        memory_id=str(memory.id),
        fact_key=fact_key,
        superseded_memory_id=str(existing.id),
        reason="single-valued fact changed",
    )


def write_memories(
    db: Session,
    candidates: list[CandidateMemory],
    *,
    user_external_id: str,
    conversation_external_id: str | None = None,
) -> list[WriteResult]:
    """Write one extraction batch atomically and own its commit or rollback.

    The caller must pass an existing user and, when supplied, a conversation
    owned by that user. Any validation or database failure rolls back the whole
    batch, including an in-progress supersession.
    """

    try:
        user = _resolve_user(db, user_external_id)
        conversation = _resolve_conversation(
            db,
            user=user,
            conversation_external_id=conversation_external_id,
        )
        results = [
            _write_candidate(db, candidate=candidate, user=user, conversation=conversation)
            for candidate in candidates
        ]
        db.commit()
        return results
    except Exception:
        db.rollback()
        raise
