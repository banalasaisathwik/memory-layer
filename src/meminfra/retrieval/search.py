"""Public user-scoped hybrid memory search API."""

from __future__ import annotations

from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from meminfra.database.models import Conversation, User

from .errors import InvalidFilterScopeError, InvalidSearchError, UserNotFoundError
from .fusion import DEFAULT_AGREEMENT_DISCOUNT, FusionStrategy, fuse_rankings
from .lexical import bm25_retrieve, lexical_retrieve
from .schemas import SearchFilters, SearchHit
from .structured import structured_retrieve
from .vector import vector_retrieve


_MAX_SEARCH_LIMIT = 100
_BRANCH_CANDIDATE_MULTIPLIER = 5

LexicalBackend = Literal["bm25", "postgres_fts"]


def _resolve_user(db: Session, user_external_id: str) -> User:
    user = db.scalar(select(User).where(User.external_id == user_external_id))
    if user is None:
        raise UserNotFoundError(f"No user exists for external ID {user_external_id!r}.")
    return user


def _resolve_filter_conversation(
    db: Session,
    *,
    user: User,
    filters: SearchFilters,
) -> Conversation | None:
    if filters.conversation_external_id is None:
        return None
    conversations = list(
        db.scalars(
            select(Conversation).where(
                Conversation.external_id == filters.conversation_external_id,
                Conversation.user_id == user.id,
            )
        )
    )
    if not conversations:
        raise InvalidFilterScopeError("The supplied conversation does not belong to the requested user.")
    if len(conversations) > 1:
        raise InvalidFilterScopeError("The supplied conversation external ID is ambiguous for the requested user.")
    return conversations[0]


def search_memories(
    db: Session,
    query: str,
    *,
    user_external_id: str,
    limit: int = 10,
    filters: SearchFilters | None = None,
    lexical_backend: LexicalBackend = "bm25",
    fusion_strategy: FusionStrategy = "discounted_agreement",
) -> list[SearchHit]:
    """Search one user's active memories with structured, lexical, and FAISS branches.

    ``lexical_backend`` selects the lexical branch: ``"bm25"`` (the default
    production behavior, genuine Okapi BM25 scoring) or ``"postgres_fts"``
    (the original ``websearch_to_tsquery``/``ts_rank_cd`` branch, retained
    for side-by-side ablation against BM25).

    ``fusion_strategy`` selects how the structured, lexical, and vector
    branches are combined into one final ranking. The default,
    ``"discounted_agreement"``, is the project's current validated default
    (a conv-30 LoCoMo ablation showed +3 net question-level gain over equal
    RRF at ``agreement_discount=0.10``): it scores each candidate as its
    strongest branch's reciprocal rank plus a discounted bonus for any
    additional branches that also matched, so one excellent single-branch
    match is no longer routinely outranked by two only-mediocre branch
    matches. ``"rrf"`` reproduces the previous equal-weight
    ``reciprocal_rank_fusion`` behavior exactly, and remains available for
    backward compatibility and ablation.

    A missing, stale, or corrupt local FAISS index is rebuilt from PostgreSQL
    during the vector branch. That rebuild may call the configured embedding
    provider only for rows missing the configured embedding model.
    """

    if not isinstance(query, str) or not query.strip():
        raise InvalidSearchError("query must not be empty or whitespace-only.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_SEARCH_LIMIT:
        raise InvalidSearchError(f"limit must be an integer from 1 to {_MAX_SEARCH_LIMIT}.")
    active_filters = filters or SearchFilters()
    user = _resolve_user(db, user_external_id)
    conversation = _resolve_filter_conversation(db, user=user, filters=active_filters)
    branch_limit = min(_MAX_SEARCH_LIMIT, limit * _BRANCH_CANDIDATE_MULTIPLIER)

    structured = structured_retrieve(
        db,
        user=user,
        filters=active_filters,
        conversation=conversation,
        limit=branch_limit,
    )
    lexical_retrieve_fn = bm25_retrieve if lexical_backend == "bm25" else lexical_retrieve
    lexical = lexical_retrieve_fn(
        db,
        query,
        user=user,
        filters=active_filters,
        conversation=conversation,
        limit=branch_limit,
    )
    vector = vector_retrieve(
        db,
        query,
        user=user,
        filters=active_filters,
        conversation=conversation,
        limit=limit,
    )
    fused = fuse_rankings(
        structured=structured,
        lexical=lexical,
        vector=vector,
        strategy=fusion_strategy,
        lambda_=DEFAULT_AGREEMENT_DISCOUNT,
    )
    return [
        SearchHit(
            memory_id=str(hit.memory.id),
            memory_text=hit.memory.memory_text,
            memory_type=hit.memory.memory_type.value,
            fact_key=hit.memory.fact_key,
            predicate=hit.memory.predicate,
            value=hit.memory.value,
            is_active=hit.memory.is_active,
            confidence=hit.memory.confidence,
            importance=hit.memory.importance,
            created_at=hit.memory.created_at,
            valid_from=hit.memory.valid_from,
            valid_to=hit.memory.valid_to,
            score=hit.score,
            structured_rank=hit.structured_rank,
            lexical_rank=hit.lexical_rank,
            vector_rank=hit.vector_rank,
        )
        for hit in fused[:limit]
    ]
