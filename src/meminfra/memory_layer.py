"""Public developer-facing facade over the memory-layer pipeline.

An application should not need to understand or manually call extraction,
context building, deterministic writes, summary updates, or hybrid retrieval
one step at a time. :class:`MemoryLayer` orchestrates those existing
production functions behind three methods: :meth:`MemoryLayer.add`,
:meth:`MemoryLayer.search`, and :meth:`MemoryLayer.answer`. It introduces no
new memory-quality behavior: ranking, deduplication, and write semantics are
unchanged from the underlying ``meminfra.memory`` / ``meminfra.retrieval`` modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from meminfra.config import get_config
from meminfra.database.models import Conversation, Message, MessageRole, User, utcnow
from meminfra.memory import (
    SummaryError,
    WriteResult,
    build_extraction_context,
    extract_memories,
    update_conversation_summary,
    write_memories,
)
from meminfra.memory.context import ChatMessage
from meminfra.memory.prompts import ANSWER_READER_SYSTEM_PROMPT
from meminfra.providers import get_llm_client
from meminfra.retrieval import SearchFilters, SearchHit, search_memories


class MemoryLayerError(Exception):
    """Raised for facade-level input problems not already covered by a lower-level error."""


class AnswerError(Exception):
    """Raised when a retrieved-memory answer cannot be safely generated."""


_MessagesInput = Annotated[list[ChatMessage], Field(min_length=1)]
_messages_adapter = TypeAdapter(_MessagesInput)

_ABSTENTION_TOKEN = "UNKNOWN"
_ABSTENTION_ANSWER = "The retrieved memories do not contain enough information to answer this question."


@dataclass(frozen=True)
class AddResult:
    """The outcome of one :meth:`MemoryLayer.add` call."""

    message_ids: list[str]
    write_results: list[WriteResult]
    summary_updated: bool
    warnings: list[str] = field(default_factory=list)
    extracted_candidate_count: int = 0


@dataclass(frozen=True)
class AnswerResult:
    """The outcome of one :meth:`MemoryLayer.answer` call."""

    answer: str
    supporting_memory_ids: list[str]
    retrieved_memories: list[SearchHit]
    abstained: bool


def _validate_scope_id(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryLayerError(f"{name} must not be empty or whitespace-only.")
    return value


def _group_interactions(messages: list[Message]) -> list[list[Message]]:
    """Split persisted messages into extractor-compatible interaction units.

    A user message immediately followed by an assistant message forms one
    two-message interaction. Any other message -- a standalone user message,
    a trailing user message with no assistant reply yet, or a lone assistant
    message -- forms its own one-message interaction. This mirrors the
    extractor's existing "at most one current user/assistant interaction"
    contract without parsing conversation structure any further.
    """

    interactions: list[list[Message]] = []
    index = 0
    while index < len(messages):
        current = messages[index]
        has_reply = (
            index + 1 < len(messages)
            and current.role is MessageRole.USER
            and messages[index + 1].role is MessageRole.ASSISTANT
        )
        if has_reply:
            interactions.append([current, messages[index + 1]])
            index += 2
        else:
            interactions.append([current])
            index += 1
    return interactions


def _response_content(response: Any) -> str | None:
    """Read normal OpenAI-compatible chat-completion content defensively."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _reader_request_content(query: str, hits: list[SearchHit]) -> str:
    """Serialize the query and a bounded, locally-referenced memory list as data."""

    memories = [
        {
            "ref": f"M{index}",
            "text": hit.memory_text,
            "status": "active" if hit.is_active else "historical",
        }
        for index, hit in enumerate(hits)
    ]
    return json.dumps({"query": query, "memories": memories}, ensure_ascii=False)


class MemoryLayer:
    """A minimal public facade over ingestion, retrieval, and grounded answering."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # -- ingestion -----------------------------------------------------

    def add(
        self,
        *,
        user_id: str,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> AddResult:
        """Persist chat messages, extract and write memories, and maybe update the summary.

        Flow: persist ``messages`` as real ``Message`` rows -> split them into
        bounded user/assistant interaction units -> build extraction context
        and extract candidates per interaction -> write validated candidates
        -> update the conversation's rolling summary if enough new history has
        accumulated. A missing user or conversation is created automatically,
        scoped so that two different users may safely reuse the same
        ``conversation_id``.

        A summary-update failure never rolls back memory already written: it
        is reported through ``AddResult.warnings`` instead of raising.
        """

        user_id = _validate_scope_id(user_id, name="user_id")
        conversation_id = _validate_scope_id(conversation_id, name="conversation_id")
        validated_messages = _messages_adapter.validate_python(messages)

        user = self._get_or_create_user(user_id)
        conversation = self._get_or_create_conversation(user, conversation_id)
        persisted = self._persist_messages(conversation, validated_messages)

        write_results: list[WriteResult] = []
        extracted_candidate_count = 0
        for interaction in _group_interactions(persisted):
            target_ids = [str(message.id) for message in interaction]
            context = build_extraction_context(
                self.db,
                user_external_id=user_id,
                conversation_external_id=conversation_id,
                target_message_ids=target_ids,
            )
            candidates = extract_memories(
                [{"role": message.role.value, "content": message.content} for message in interaction],
                source_message_ids=target_ids,
                context=context,
            )
            extracted_candidate_count += len(candidates)
            if candidates:
                write_results.extend(
                    write_memories(
                        self.db,
                        candidates,
                        user_external_id=user_id,
                        conversation_external_id=conversation_id,
                    )
                )

        warnings: list[str] = []
        try:
            updated_summary = update_conversation_summary(
                self.db,
                user_external_id=user_id,
                conversation_external_id=conversation_id,
            )
            summary_updated = updated_summary is not None
        except SummaryError:
            summary_updated = False
            warnings.append("summary_update_failed")

        return AddResult(
            message_ids=[str(message.id) for message in persisted],
            write_results=write_results,
            summary_updated=summary_updated,
            warnings=warnings,
            extracted_candidate_count=extracted_candidate_count,
        )

    def _get_or_create_user(self, user_id: str) -> User:
        user = self.db.scalars(select(User).where(User.external_id == user_id)).first()
        if user is None:
            user = User(external_id=user_id)
            self.db.add(user)
            self.db.commit()
        return user

    def _get_or_create_conversation(self, user: User, conversation_id: str) -> Conversation:
        # Scoped by user_id, so the same conversation_id used by another user
        # never resolves here; it is always a distinct row per user.
        conversation = self.db.scalars(
            select(Conversation).where(
                Conversation.external_id == conversation_id,
                Conversation.user_id == user.id,
            )
        ).first()
        if conversation is None:
            conversation = Conversation(external_id=conversation_id, user_id=user.id)
            self.db.add(conversation)
            self.db.commit()
        return conversation

    def _persist_messages(self, conversation: Conversation, messages: list[ChatMessage]) -> list[Message]:
        base_time = utcnow()
        rows = [
            Message(
                conversation_id=conversation.id,
                role=MessageRole(message.role),
                content=message.content,
                created_at=base_time + timedelta(microseconds=index),
            )
            for index, message in enumerate(messages)
        ]
        self.db.add_all(rows)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return rows

    # -- retrieval -------------------------------------------------------

    def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Return raw memory search hits. A thin wrapper: ranking is unchanged."""

        user_id = _validate_scope_id(user_id, name="user_id")
        return search_memories(self.db, query, user_external_id=user_id, limit=limit, filters=filters)

    # -- grounded answering ------------------------------------------------

    def answer(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        filters: SearchFilters | None = None,
    ) -> AnswerResult:
        """Answer a question using only memories retrieved by :meth:`search`.

        Abstains without an LLM call when no memories are retrieved. Otherwise
        an LLM reader is grounded strictly in the retrieved memory text and
        told to distinguish active from historical (superseded) memories, and
        to answer with the literal token ``UNKNOWN`` when the retrieved
        memories do not support an answer. ``supporting_memory_ids`` is the
        full retrieved set used as context, not an LLM-chosen subset -- the
        model is never asked to invent or select memory IDs.
        """

        user_id = _validate_scope_id(user_id, name="user_id")
        hits = self.search(user_id=user_id, query=query, limit=limit, filters=filters)
        if not hits:
            return AnswerResult(
                answer=_ABSTENTION_ANSWER,
                supporting_memory_ids=[],
                retrieved_memories=[],
                abstained=True,
            )

        settings = get_config()
        if not settings.llm_model:
            raise AnswerError("LLM_MODEL must be configured before generating an answer.")

        try:
            response = get_llm_client().chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": ANSWER_READER_SYSTEM_PROMPT},
                    {"role": "user", "content": _reader_request_content(query, hits)},
                ],
                temperature=0,
            )
        except Exception as error:
            raise AnswerError("LLM answer request failed.") from error

        content = _response_content(response)
        if content is None or not content.strip():
            raise AnswerError("LLM answer returned an empty response.")

        text = content.strip()
        if text == _ABSTENTION_TOKEN:
            return AnswerResult(
                answer=_ABSTENTION_ANSWER,
                supporting_memory_ids=[],
                retrieved_memories=hits,
                abstained=True,
            )

        return AnswerResult(
            answer=text,
            supporting_memory_ids=[hit.memory_id for hit in hits],
            retrieved_memories=hits,
            abstained=False,
        )
