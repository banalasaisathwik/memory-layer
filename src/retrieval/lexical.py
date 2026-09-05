"""Lexical retrieval over durable memory text: PostgreSQL FTS, and genuine BM25.

Both backends share the same user/conversation/state scope conditions from
``memory_filter_conditions``. ``lexical_retrieve`` (PostgreSQL
``websearch_to_tsquery``/``ts_rank_cd``) is retained for ablation/comparison
against ``bm25_retrieve``; production retrieval uses BM25 (see
``src.retrieval.search``).
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, User

from .bm25 import bm25_rank
from .schemas import SearchFilters
from .structured import memory_filter_conditions


def bm25_retrieve(
    db: Session,
    query: str,
    *,
    user: User,
    filters: SearchFilters,
    conversation: Conversation | None,
    limit: int,
) -> list[Memory]:
    """Score every scoped memory with Okapi BM25; never AND-filtered by FTS first.

    The corpus is scoped to this user (and any caller filters/conversation)
    *before* scoring, exactly like the structured and FTS branches --
    cross-user lexical leakage is prevented by the SQL scope, not by a
    post-hoc filter over BM25 output.
    """

    conditions = [
        Memory.user_id == user.id,
        *memory_filter_conditions(filters, conversation=conversation),
    ]
    candidates = list(
        db.scalars(
            select(Memory).where(*conditions).order_by(Memory.created_at.asc(), Memory.id.asc())
        )
    )
    hits = bm25_rank(query, candidates, text_of=lambda memory: memory.memory_text)
    return [hit.item for hit in hits[:limit]]


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
