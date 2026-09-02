"""PostgreSQL native full-text retrieval over durable memory text."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, User

from .schemas import SearchFilters
from .structured import memory_filter_conditions


def lexical_retrieve(
    db: Session,
    query: str,
    *,
    user: User,
    filters: SearchFilters,
    conversation: Conversation | None,
    limit: int,
) -> list[Memory]:
    """Use PostgreSQL FTS, then a narrow literal fallback for punctuation-heavy IDs."""

    vector = func.to_tsvector("simple", Memory.memory_text)
    tsquery = func.websearch_to_tsquery("simple", query)
    rank = func.ts_rank_cd(vector, tsquery)
    conditions = [
        Memory.user_id == user.id,
        *memory_filter_conditions(filters, conversation=conversation),
    ]
    matches = list(
        db.scalars(
            select(Memory)
            .where(*conditions, vector.op("@@")(tsquery))
            .order_by(rank.desc(), Memory.created_at.desc(), Memory.id.desc())
            .limit(limit)
        )
    )
    if matches:
        return matches

    # `websearch_to_tsquery` safely handles normal queries, but a literal
    # fallback keeps identifiers such as KAN-4 useful when tokenization loses
    # their punctuation. It remains fully user- and state-scoped.
    literal_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return list(
        db.scalars(
            select(Memory)
            .where(*conditions, Memory.memory_text.ilike(f"%{literal_query}%", escape="\\"))
            .order_by(Memory.created_at.desc(), Memory.id.desc())
            .limit(limit)
        )
    )
