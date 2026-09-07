"""PostgreSQL retrieval for hard caller filters and optional query-intent evidence."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm import Session

from meminfra.database.models import Conversation, Memory, User

from .schemas import QueryIntent, SearchFilters


def memory_filter_conditions(
    filters: SearchFilters,
    *,
    conversation: Conversation | None,
    include_history: bool | None = None,
) -> Sequence[ColumnElement[bool]]:
    """Return the common state/filter constraints shared by all retrieval branches."""

    conditions: list[ColumnElement[bool]] = []
    effective_include_history = filters.include_history if include_history is None else include_history
    if not effective_include_history:
        conditions.append(Memory.is_active.is_(True))
    if filters.memory_type is not None:
        conditions.append(Memory.memory_type == filters.memory_type)
    if filters.predicate is not None:
        conditions.append(Memory.predicate == filters.predicate)
    if filters.subject_type is not None:
        conditions.append(Memory.subject_type == filters.subject_type)
    if filters.fact_key is not None:
        conditions.append(Memory.fact_key == filters.fact_key)
    if conversation is not None:
        conditions.append(Memory.conversation_id == conversation.id)
    return conditions


def structured_retrieve(
    db: Session,
    *,
    user: User,
    filters: SearchFilters,
    conversation: Conversation | None,
    limit: int,
) -> list[Memory]:
    """Return exact caller-filtered rows, never inferring structure from text."""

    has_structured_filter = any(
        (
            filters.memory_type is not None,
            filters.predicate is not None,
            filters.subject_type is not None,
            filters.fact_key is not None,
            conversation is not None,
        )
    )
    if not has_structured_filter:
        return []

    statement: Select[tuple[Memory]] = (
        select(Memory)
        .where(
            Memory.user_id == user.id,
            *memory_filter_conditions(filters, conversation=conversation),
        )
        .order_by(
            Memory.is_active.desc(),
            Memory.updated_at.desc(),
            Memory.created_at.desc(),
            Memory.id.desc(),
        )
        .limit(limit)
    )
    return list(db.scalars(statement))


def structured_retrieve_from_intent(
    db: Session,
    *,
    user: User,
    intent: QueryIntent,
    filters: SearchFilters,
    conversation: Conversation | None,
    limit: int,
) -> list[Memory]:
    """Return lifecycle-aware structured evidence without changing hard filters.

    Unknown predicates deliberately produce no SQL structure. Query value matching
    uses the writer's normalization so its identity semantics stay aligned.
    """

    # Kept local because importing the memory package during retrieval module
    # initialization would create a package-import cycle.
    from meminfra.memory.fact_keys import normalize_value_identity
    from meminfra.memory.predicates import resolve_predicate

    predicate = resolve_predicate(intent.predicate)
    if predicate is None:
        return []

    explicit_history_exclusion = "include_history" in filters.model_fields_set and not filters.include_history
    if intent.temporal_scope == "historical" and explicit_history_exclusion:
        return []

    if intent.temporal_scope == "current" or explicit_history_exclusion:
        is_active: bool | None = True
    elif intent.temporal_scope == "historical":
        is_active = False
    else:
        is_active = None

    conditions: list[ColumnElement[bool]] = [
        Memory.user_id == user.id,
        Memory.predicate == predicate,
        *memory_filter_conditions(filters, conversation=conversation, include_history=True),
    ]
    if is_active is not None:
        conditions.append(Memory.is_active.is_(is_active))

    statement: Select[tuple[Memory]] = select(Memory).where(*conditions).order_by(
        Memory.is_active.desc(),
        Memory.updated_at.desc(),
        Memory.created_at.desc(),
        Memory.id.desc(),
    )
    candidates = list(db.scalars(statement))
    if intent.value is not None:
        target_value = normalize_value_identity(intent.value)
        candidates = [memory for memory in candidates if normalize_value_identity(memory.value) == target_value]
    return candidates[:limit]
