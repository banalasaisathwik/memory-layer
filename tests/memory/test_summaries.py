"""PostgreSQL integration tests for local, mocked rolling summaries."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from meminfra.config import configure, reset_config
from meminfra.database import (
    Conversation,
    ConversationSummary,
    Message,
    MessageRole,
    SessionLocal,
    User,
    create_tables,
    reset_engine,
)
from meminfra.memory import SummaryError, update_conversation_summary


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; summary integration tests never use DATABASE_URL.",
    ),
]


class FakeCompletions:
    """Record summary requests while returning a fully local fake completion."""

    def __init__(self, *, content: str | None = "Updated summary.", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL)
    create_tables()
    yield
    reset_engine()
    reset_config()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
        session.rollback()


@pytest.fixture
def fake_completions(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions()
    monkeypatch.setattr("meminfra.memory.summaries.get_llm_client", lambda: FakeClient(completions))
    return completions


def _conversation(
    db,
    *,
    suffix: str | None = None,
    conversation_external_id: str | None = None,
) -> Conversation:
    suffix = suffix or uuid4().hex
    user = User(external_id=f"summary-user-{suffix}")
    conversation = Conversation(
        external_id=conversation_external_id or f"summary-conversation-{suffix}",
        user=user,
    )
    db.add_all([user, conversation])
    db.commit()
    return conversation


def _messages(db, conversation: Conversation, *, start: int, stop: int) -> list[Message]:
    messages = [
        Message(
            conversation_id=conversation.id,
            role=MessageRole.USER if number % 2 else MessageRole.ASSISTANT,
            content=f"message {number}",
            created_at=datetime(2026, 9, 2, tzinfo=timezone.utc) + timedelta(minutes=number),
        )
        for number in range(start, stop + 1)
    ]
    db.add_all(messages)
    db.commit()
    return messages


def _configure_summary_settings() -> None:
    configure(
        llm_model="test-summary-model",
        summary_trigger_messages=6,
        summary_recent_keep=2,
    )


def _summary_count(db, conversation: Conversation) -> int:
    return db.scalar(
        select(func.count()).select_from(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation.id
        )
    )


def test_below_threshold_skips_the_provider_and_persistence(db, fake_completions: FakeCompletions) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    _messages(db, conversation, start=1, stop=7)

    result = update_conversation_summary(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
    )

    assert result is None
    assert fake_completions.calls == []
    assert _summary_count(db, conversation) == 0


def test_initial_summary_leaves_the_recent_window_uncovered(db, fake_completions: FakeCompletions) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    messages = _messages(db, conversation, start=1, stop=8)

    summary = update_conversation_summary(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
    )

    assert summary is not None
    assert _summary_count(db, conversation) == 1
    assert summary.summary_text == "Updated summary."
    assert summary.covered_through_message_id == messages[5].id
    request = json.loads(fake_completions.calls[0]["messages"][1]["content"])
    assert request["previous_summary"] is None
    assert [message["content"] for message in request["messages"]] == [
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
        "message 6",
    ]


def test_summary_input_never_crosses_to_another_conversation_or_user(
    db,
    fake_completions: FakeCompletions,
) -> None:
    _configure_summary_settings()
    selected = _conversation(db)
    foreign = _conversation(db)
    _messages(db, selected, start=1, stop=8)
    _messages(db, foreign, start=1, stop=10)

    summary = update_conversation_summary(
        db,
        user_external_id=selected.user.external_id,
        conversation_external_id=selected.external_id,
    )

    assert summary is not None
    assert _summary_count(db, foreign) == 0
    request = json.loads(fake_completions.calls[0]["messages"][1]["content"])
    assert [message["content"] for message in request["messages"]] == [
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
        "message 6",
    ]


def test_incremental_summary_uses_only_newly_eligible_messages_and_updates_one_row(
    db,
    fake_completions: FakeCompletions,
) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    first_messages = _messages(db, conversation, start=1, stop=8)
    first_summary = update_conversation_summary(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
    )
    assert first_summary is not None
    fake_completions.content = "Second updated summary."
    later_messages = _messages(db, conversation, start=9, stop=14)

    second_summary = update_conversation_summary(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
    )

    assert second_summary is not None
    assert second_summary.id == first_summary.id
    assert _summary_count(db, conversation) == 1
    assert second_summary.summary_text == "Second updated summary."
    assert second_summary.covered_through_message_id == later_messages[3].id
    assert db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    ) == 14
    request = json.loads(fake_completions.calls[1]["messages"][1]["content"])
    assert request["previous_summary"] == "Updated summary."
    assert [message["content"] for message in request["messages"]] == [
        "message 7",
        "message 8",
        "message 9",
        "message 10",
        "message 11",
        "message 12",
    ]
    assert first_messages[5].id != second_summary.covered_through_message_id


def test_provider_and_empty_response_fail_without_a_summary_row(db, fake_completions: FakeCompletions) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    _messages(db, conversation, start=1, stop=8)
    fake_completions.error = RuntimeError("provider unavailable")

    with pytest.raises(SummaryError, match="request failed") as provider_error:
        update_conversation_summary(
            db,
            user_external_id=conversation.user.external_id,
            conversation_external_id=conversation.external_id,
        )

    assert isinstance(provider_error.value.__cause__, RuntimeError)
    assert _summary_count(db, conversation) == 0

    fake_completions.error = None
    fake_completions.content = "   "
    with pytest.raises(SummaryError, match="empty response"):
        update_conversation_summary(
            db,
            user_external_id=conversation.user.external_id,
            conversation_external_id=conversation.external_id,
        )

    assert _summary_count(db, conversation) == 0


def test_summary_resolution_stays_in_the_requested_user_scope(
    db,
    fake_completions: FakeCompletions,
) -> None:
    _configure_summary_settings()
    shared_conversation_id = f"shared-summary-{uuid4().hex}"
    selected = _conversation(
        db,
        suffix=f"selected-{uuid4().hex}",
        conversation_external_id=shared_conversation_id,
    )
    foreign = _conversation(
        db,
        suffix=f"foreign-{uuid4().hex}",
        conversation_external_id=shared_conversation_id,
    )
    _messages(db, selected, start=1, stop=8)
    _messages(db, foreign, start=1, stop=8)

    summary = update_conversation_summary(
        db,
        user_external_id=selected.user.external_id,
        conversation_external_id=shared_conversation_id,
    )

    assert summary is not None
    assert summary.conversation_id == selected.id
    assert _summary_count(db, selected) == 1
    assert _summary_count(db, foreign) == 0


def test_persistence_failure_rolls_back_the_new_summary(
    db,
    fake_completions: FakeCompletions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    _messages(db, conversation, start=1, stop=8)

    def fail_commit() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(SummaryError, match="persistence failed"):
        update_conversation_summary(
            db,
            user_external_id=conversation.user.external_id,
            conversation_external_id=conversation.external_id,
        )

    assert _summary_count(db, conversation) == 0


def test_persistence_failure_keeps_an_existing_summary_and_coverage_marker(
    db,
    fake_completions: FakeCompletions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_summary_settings()
    conversation = _conversation(db)
    _messages(db, conversation, start=1, stop=8)
    first_summary = update_conversation_summary(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
    )
    assert first_summary is not None
    original_id = first_summary.id
    original_text = first_summary.summary_text
    original_coverage = first_summary.covered_through_message_id
    _messages(db, conversation, start=9, stop=14)
    fake_completions.content = "This update must roll back."

    def fail_commit() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(SummaryError, match="persistence failed"):
        update_conversation_summary(
            db,
            user_external_id=conversation.user.external_id,
            conversation_external_id=conversation.external_id,
        )

    persisted_summary = db.scalar(
        select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id)
    )
    assert persisted_summary is not None
    assert persisted_summary.id == original_id
    assert persisted_summary.summary_text == original_text
    assert persisted_summary.covered_through_message_id == original_coverage
