"""Incremental rolling summaries for persisted conversation history."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from src.config import get_config
from src.database.models import Conversation, ConversationSummary, Message
from src.providers import get_llm_client

from .context import ContextError, _resolve_conversation
from .prompts import SUMMARY_SYSTEM_PROMPT


class SummaryError(Exception):
    """Raised when a rolling summary cannot be generated or persisted safely."""


def _response_content(response: Any) -> str | None:
    """Read normal OpenAI-compatible chat-completion content defensively."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _summary_request_content(
    previous_summary: str | None,
    messages: list[Message],
) -> str:
    """Serialize only the previous summary and newly eligible history as data."""

    return json.dumps(
        {
            "previous_summary": previous_summary,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
        },
        ensure_ascii=False,
    )


def _messages_after_coverage(
    db: Session,
    *,
    conversation: Conversation,
    summary: ConversationSummary | None,
) -> list[Message]:
    """Load only history not yet represented by the current rolling summary."""

    statement = select(Message).where(Message.conversation_id == conversation.id)
    if summary is not None and summary.covered_through_message_id is not None:
        try:
            covered_message = db.get(Message, summary.covered_through_message_id)
        except (StatementError, TypeError, ValueError) as error:
            raise SummaryError("The summary coverage marker is invalid.") from error
        if covered_message is None or covered_message.conversation_id != conversation.id:
            raise SummaryError("The summary coverage marker is outside its conversation.")
        statement = statement.where(
            or_(
                Message.created_at > covered_message.created_at,
                and_(
                    Message.created_at == covered_message.created_at,
                    Message.id > covered_message.id,
                ),
            )
        )
    return list(db.scalars(statement.order_by(Message.created_at, Message.id)))


def update_conversation_summary(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
) -> ConversationSummary | None:
    """Update one summary when enough unsummarized history has accumulated.

    The current keep window remains outside the summary, and an existing summary
    is sent with only newly eligible messages rather than the full transcript.
    This function owns its commit or rollback when it changes persistence.
    """

    try:
        conversation = _resolve_conversation(
            db,
            user_external_id=user_external_id,
            conversation_external_id=conversation_external_id,
        )
        existing_summary = db.scalar(
            select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id)
        )
        unsummarized_messages = _messages_after_coverage(
            db,
            conversation=conversation,
            summary=existing_summary,
        )
        settings = get_config()
        eligible_messages = unsummarized_messages[: -settings.summary_recent_keep]
        if len(eligible_messages) < settings.summary_trigger_messages:
            return None

        if not settings.llm_model:
            raise SummaryError("LLM_MODEL must be configured before generating a summary.")
        try:
            response = get_llm_client().chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _summary_request_content(
                            existing_summary.summary_text if existing_summary is not None else None,
                            eligible_messages,
                        ),
                    },
                ],
                temperature=0,
            )
        except Exception as error:
            raise SummaryError("LLM summary request failed.") from error

        content = _response_content(response)
        if content is None or not content.strip():
            raise SummaryError("LLM summary returned an empty response.")

        if existing_summary is None:
            persisted_summary = ConversationSummary(
                conversation_id=conversation.id,
                summary_text=content.strip(),
                covered_through_message_id=eligible_messages[-1].id,
            )
            db.add(persisted_summary)
        else:
            existing_summary.summary_text = content.strip()
            existing_summary.covered_through_message_id = eligible_messages[-1].id
            persisted_summary = existing_summary
        db.commit()
        return persisted_summary
    except ContextError as error:
        db.rollback()
        raise SummaryError("The conversation for summary generation could not be resolved.") from error
    except SummaryError:
        db.rollback()
        raise
    except Exception as error:
        db.rollback()
        raise SummaryError("Conversation summary persistence failed.") from error
