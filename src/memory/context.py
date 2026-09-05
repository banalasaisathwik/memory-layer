"""Bounded, persisted conversation context for interpreting one target interaction."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from src.config import get_config
from src.database.models import Conversation, ConversationSummary, Message, User
from src.retrieval import (
    RetrievalError,
    bm25_rank,
    fuse_message_rankings,
    retrieve_semantic_message_context,
)


class ContextError(Exception):
    """Raised when a requested conversation context cannot be safely resolved."""


_LEXICAL_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "can",
        "could",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "with",
        "would",
        "you",
        "your",
    }
)
_MAX_TARGET_LEXICAL_TERMS = 12


class ChatMessage(BaseModel):
    """A normal user or assistant message used only as LLM input data."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject empty message content without changing meaningful wording."""

        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only.")
        return value


class ConversationContext(BaseModel):
    """Persisted summary plus bounded context that predates the target interaction."""

    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    recent_messages: list[ChatMessage] = Field(default_factory=list)
    older_lexical_messages: list[ChatMessage] = Field(default_factory=list)
    older_semantic_messages: list[ChatMessage] = Field(default_factory=list)
    # This is the prompt-ready lexical/semantic union.  Branch fields remain
    # inspectable for callers while the extractor receives one deduplicated set.
    older_relevant_messages: list[ChatMessage] = Field(default_factory=list)
    semantic_retrieval_error: str | None = None

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        """Treat an absent summary as None while rejecting malformed blank text."""

        if value is not None and not value.strip():
            raise ValueError("summary must not be empty or whitespace-only.")
        return value

    def effective_older_messages(self) -> list[ChatMessage]:
        """Return the prompt-ready older-context union.

        build_extraction_context() always populates older_relevant_messages
        with its UUID-deduplicated, chronologically bounded merge. A caller
        that constructs ConversationContext directly with only the lexical and
        semantic branch fields (for example, a test or a lightweight caller
        that skips the full pipeline) gets an equivalent best-effort union
        here, deduplicated by (role, content) since raw ChatMessage carries no
        message ID or timestamp to merge on.
        """

        if self.older_relevant_messages:
            return self.older_relevant_messages
        merged: list[ChatMessage] = []
        seen: set[tuple[str, str]] = set()
        for message in [*self.older_lexical_messages, *self.older_semantic_messages]:
            identity = (message.role, message.content)
            if identity not in seen:
                seen.add(identity)
                merged.append(message)
        return merged


def _resolve_conversation(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
) -> Conversation:
    """Resolve one unambiguous conversation inside the supplied user scope."""

    conversations = list(
        db.scalars(
            select(Conversation).join(User).where(
                User.external_id == user_external_id,
                Conversation.external_id == conversation_external_id,
            )
        )
    )
    if not conversations:
        raise ContextError("No conversation exists in the supplied user scope.")
    if len(conversations) > 1:
        raise ContextError("Conversation external ID is ambiguous within the supplied user scope.")
    return conversations[0]


def _target_messages(
    db: Session,
    *,
    conversation: Conversation,
    target_message_ids: list[str],
) -> list[Message]:
    """Validate that every application-selected target belongs to this conversation."""

    if not target_message_ids:
        raise ContextError("At least one target message ID is required.")

    targets: list[Message] = []
    seen: set[str] = set()
    for target_message_id in target_message_ids:
        normalized_id = str(target_message_id)
        if normalized_id in seen:
            continue
        seen.add(normalized_id)
        try:
            message = db.get(Message, normalized_id)
        except (StatementError, TypeError, ValueError) as error:
            raise ContextError(f"Target message {normalized_id!r} does not exist.") from error
        if message is None:
            raise ContextError(f"Target message {normalized_id!r} does not exist.")
        if message.conversation_id != conversation.id:
            raise ContextError("Target messages must belong to the supplied conversation.")
        targets.append(message)
    return targets


def _target_lexical_query(targets: list[Message]) -> str:
    """Build a small OR query from target terms without semantic rewriting."""

    terms: list[str] = []
    seen: set[str] = set()
    for target in targets:
        for raw_term in re.findall(r"[A-Za-z0-9]+", target.content.casefold()):
            if len(raw_term) < 2 or raw_term in _LEXICAL_QUERY_STOP_WORDS or raw_term in seen:
                continue
            seen.add(raw_term)
            terms.append(raw_term)
            if len(terms) == _MAX_TARGET_LEXICAL_TERMS:
                return " | ".join(terms)
    return " | ".join(terms)


def _target_semantic_query(targets: list[Message]) -> str:
    """Use one deterministic target-interaction query without rewriting it."""

    user_text = [target.content for target in targets if target.role.value == "user"]
    return "\n".join(user_text or [target.content for target in targets])


def _select_older_messages(
    lexical_messages_by_rank: list[Message],
    semantic_messages_by_rank: list[Message],
    *,
    limit: int,
) -> list[Message]:
    """Fuse relevance-ranked branches with RRF, cap, then order for readability.

    Both arguments must be in each branch's own relevance-rank order (best
    match first) so ``fuse_message_rankings`` can score them; the
    chronological order used for the extraction prompt is applied only after
    the top-K selection below, never before it.
    """

    fused = fuse_message_rankings(lexical=lexical_messages_by_rank, semantic=semantic_messages_by_rank)
    selected = [item.message for item in fused[:limit]]
    selected.sort(key=lambda message: (message.created_at, str(message.id)))
    return selected


def _eligible_older_messages(
    db: Session,
    *,
    conversation: Conversation,
    before_target: ColumnElement[bool],
    excluded_ids: list[object],
) -> list[Message]:
    """Fetch every same-conversation message BM25 is allowed to score.

    Scoped to this conversation and strictly before the target interaction,
    excluding the target and recent-window rows -- the same temporal/context
    boundary the old FTS query enforced. Deliberately has no text filter of
    its own: BM25 must generate its own lexical ranking rather than reranking
    an already AND-filtered candidate set.
    """

    return list(
        db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                before_target,
                ~Message.id.in_(excluded_ids),
            )
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    )


def _bm25_older_messages(
    db: Session,
    *,
    conversation: Conversation,
    before_target: ColumnElement[bool],
    excluded_ids: list[object],
    query_text: str,
    limit: int,
) -> list[Message]:
    """Rank eligible older messages with genuine BM25; this is the default lexical branch."""

    candidates = _eligible_older_messages(
        db, conversation=conversation, before_target=before_target, excluded_ids=excluded_ids
    )
    hits = bm25_rank(query_text, candidates, text_of=lambda message: message.content)
    return [hit.item for hit in hits[:limit]]


def _postgres_fts_older_messages(
    db: Session,
    *,
    conversation: Conversation,
    before_target: ColumnElement[bool],
    excluded_ids: list[object],
    query_text: str,
    explicit_query: bool,
    limit: int,
) -> list[Message]:
    """The original ``to_tsquery``/``ts_rank_cd`` branch, retained for FTS-vs-BM25 ablation.

    ``explicit_query`` matches the old behavior: a caller-supplied
    ``older_lexical_query`` is parsed with ``plainto_tsquery`` (natural
    language), while the default target-derived, already ``|``-joined query
    uses ``to_tsquery`` (raw operator syntax). Not used by
    ``build_extraction_context`` by default; kept importable so the lexical
    ablation comparison can run both backends over the same fixtures without
    duplicating the query-scoping logic.
    """

    lexical_query = (
        func.plainto_tsquery("simple", query_text)
        if explicit_query
        else func.to_tsquery("simple", query_text)
    )
    lexical_vector = func.to_tsvector("simple", Message.content)
    return list(
        db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                before_target,
                ~Message.id.in_(excluded_ids),
                lexical_vector.op("@@")(lexical_query),
            )
            .order_by(
                func.ts_rank_cd(lexical_vector, lexical_query).desc(),
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )
    )


def build_extraction_context(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
    target_message_ids: list[str],
    recent_message_limit: int | None = None,
    older_lexical_query: str | None = None,
    older_lexical_limit: int | None = None,
    older_semantic_limit: int | None = None,
    older_context_limit: int | None = None,
) -> ConversationContext:
    """Return only context that predates an application-selected target interaction.

    The returned summary and messages help an extractor resolve references, but
    callers must keep the target interaction separate as the only new-memory
    evidence.
    """

    recent_limit = (
        get_config().extraction_recent_messages
        if recent_message_limit is None
        else recent_message_limit
    )
    lexical_limit = (
        get_config().extraction_lexical_messages
        if older_lexical_limit is None
        else older_lexical_limit
    )
    semantic_limit = (
        get_config().extraction_semantic_messages
        if older_semantic_limit is None
        else older_semantic_limit
    )
    old_context_limit = (
        get_config().extraction_old_messages
        if older_context_limit is None
        else older_context_limit
    )
    if recent_limit < 1:
        raise ContextError("recent_message_limit must be greater than zero.")
    if lexical_limit < 1:
        raise ContextError("older_lexical_limit must be greater than zero.")
    if semantic_limit < 1:
        raise ContextError("older_semantic_limit must be greater than zero.")
    if old_context_limit < 1:
        raise ContextError("older_context_limit must be greater than zero.")
    if older_lexical_query is not None and not older_lexical_query.strip():
        raise ContextError("older_lexical_query must not be empty or whitespace-only.")

    conversation = _resolve_conversation(
        db,
        user_external_id=user_external_id,
        conversation_external_id=conversation_external_id,
    )
    targets = _target_messages(
        db,
        conversation=conversation,
        target_message_ids=target_message_ids,
    )
    first_target = min(targets, key=lambda message: (message.created_at, str(message.id)))
    before_target = or_(
        Message.created_at < first_target.created_at,
        and_(
            Message.created_at == first_target.created_at,
            Message.id < first_target.id,
        ),
    )
    recent_messages = list(
        db.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation.id,
                ~Message.id.in_([message.id for message in targets]),
                before_target,
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(recent_limit)
        )
    )
    recent_messages.reverse()
    lexical_messages_by_rank: list[Message] = []
    lexical_query_text = (
        older_lexical_query if older_lexical_query is not None else _target_lexical_query(targets)
    )
    if lexical_query_text.strip():
        excluded_ids = [
            *(message.id for message in targets),
            *(message.id for message in recent_messages),
        ]
        # Kept in relevance-rank order (best match first) for RRF fusion below.
        # older_lexical_messages, the inspectable/prompt debug field, gets its
        # own chronologically-sorted copy further down. BM25 scores every
        # eligible older message itself; it is never a rerank of an
        # already-AND-filtered FTS candidate set.
        lexical_messages_by_rank = _bm25_older_messages(
            db,
            conversation=conversation,
            before_target=before_target,
            excluded_ids=excluded_ids,
            query_text=lexical_query_text,
            limit=lexical_limit,
        )
    older_lexical_messages = sorted(
        lexical_messages_by_rank, key=lambda message: (message.created_at, str(message.id))
    )

    # Semantic message context is an optional enhancement.  It remains scoped
    # to this conversation, excludes target/recent rows, and can fail without
    # making the safe summary/lexical/recent context unavailable.
    semantic_messages_by_rank: list[Message] = []
    semantic_retrieval_error: str | None = None
    try:
        # chronological=False keeps FAISS similarity-rank order (best match
        # first) for RRF fusion below; older_semantic_messages, the
        # inspectable/prompt debug field, gets its own chronological copy.
        semantic_messages_by_rank = retrieve_semantic_message_context(
            db,
            user_external_id=user_external_id,
            conversation_external_id=conversation_external_id,
            query_text=_target_semantic_query(targets),
            limit=semantic_limit,
            exclude_message_ids={
                *(str(message.id) for message in targets),
                *(str(message.id) for message in recent_messages),
            },
            before_message_id=str(first_target.id),
            chronological=False,
        )
    except RetrievalError as error:
        # RetrievalError messages intentionally contain only safe categories,
        # so callers can observe graceful degradation without leaking provider
        # responses, credentials, or filesystem details.
        semantic_retrieval_error = str(error)
    older_semantic_messages = sorted(
        semantic_messages_by_rank, key=lambda message: (message.created_at, str(message.id))
    )

    older_relevant_messages = _select_older_messages(
        lexical_messages_by_rank,
        semantic_messages_by_rank,
        limit=old_context_limit,
    )
    persisted_summary = db.scalar(
        select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id)
    )

    return ConversationContext(
        summary=persisted_summary.summary_text if persisted_summary is not None else None,
        recent_messages=[
            ChatMessage(role=message.role.value, content=message.content)
            for message in recent_messages
        ],
        older_lexical_messages=[
            ChatMessage(role=message.role.value, content=message.content)
            for message in older_lexical_messages
        ],
        older_semantic_messages=[
            ChatMessage(role=message.role.value, content=message.content)
            for message in older_semantic_messages
        ],
        older_relevant_messages=[
            ChatMessage(role=message.role.value, content=message.content)
            for message in older_relevant_messages
        ],
        semantic_retrieval_error=semantic_retrieval_error,
    )
