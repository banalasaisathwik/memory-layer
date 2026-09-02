"""Deterministic PostgreSQL retrieval for caller-supplied memory structure."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, User

from .schemas import SearchFilters


def memory_filter_conditions(
    filters: SearchFilters,
    *,
    conversation: Conversation | None,
) -> Sequence[ColumnElement[bool]]:
    """Return the common state/filter constraints shared by all retrieval branches."""

    conditions: list[ColumnElement[bool]] = []
    if not filters.include_history:
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
