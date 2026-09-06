"""PostgreSQL integration tests for bounded extraction context selection."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

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
from meminfra.memory import ContextError, build_extraction_context


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; context integration tests never use DATABASE_URL.",
    ),
]


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


def _conversation(
    db,
    *,
    suffix: str | None = None,
    conversation_external_id: str | None = None,
) -> Conversation:
    identifier = suffix or uuid4().hex
    user = User(external_id=f"context-user-{identifier}")
    conversation = Conversation(
        external_id=conversation_external_id or f"context-conversation-{identifier}",
        user=user,
    )
    db.add_all([user, conversation])
    db.commit()
    return conversation


def _message(db, conversation: Conversation, *, number: int, content: str) -> Message:
    message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER if number % 2 else MessageRole.ASSISTANT,
        content=content,
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc) + timedelta(minutes=number),
    )
    db.add(message)
    db.commit()
    return message


def test_context_loads_its_summary_and_a_chronological_bounded_window(db) -> None:
    conversation = _conversation(db)
    messages = [
        _message(db, conversation, number=number, content=f"message {number}")
        for number in range(1, 9)
    ]
    db.add(
        ConversationSummary(
            conversation_id=conversation.id,
            summary_text="Earlier decision context.",
            covered_through_message_id=messages[1].id,
        )
    )
    db.commit()

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(messages[-1].id)],
        recent_message_limit=3,
    )

    assert context.summary == "Earlier decision context."
    assert [message.content for message in context.recent_messages] == [
        "message 5",
        "message 6",
        "message 7",
    ]


def test_context_excludes_targets_and_never_reads_another_conversation_or_user(db) -> None:
    selected = _conversation(db)
    foreign = _conversation(db)
    selected_messages = [
        _message(db, selected, number=number, content=f"selected {number}")
        for number in range(1, 4)
    ]
    _message(db, foreign, number=1, content="foreign user message")
    _message(db, foreign, number=2, content="foreign assistant message")

    context = build_extraction_context(
        db,
        user_external_id=selected.user.external_id,
        conversation_external_id=selected.external_id,
        target_message_ids=[str(selected_messages[-1].id)],
        recent_message_limit=6,
    )

    assert [message.content for message in context.recent_messages] == ["selected 1", "selected 2"]
    assert all("foreign" not in message.content for message in context.recent_messages)

    with pytest.raises(ContextError, match="must belong"):
        build_extraction_context(
            db,
            user_external_id=selected.user.external_id,
            conversation_external_id=selected.external_id,
            target_message_ids=[str(selected_messages[-1].id), str(_message(db, foreign, number=3, content="foreign target").id)],
        )


def test_context_without_a_summary_returns_none_cleanly(db) -> None:
    conversation = _conversation(db)
    prior = _message(db, conversation, number=1, content="earlier context")
    target = _message(db, conversation, number=2, content="current target")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
    )

    assert context.summary is None
    assert [message.content for message in context.recent_messages] == [prior.content]


def test_context_optionally_adds_bounded_older_lexical_matches_after_recent_messages(db) -> None:
    conversation = _conversation(db)
    foreign = _conversation(db)
    older_match = _message(db, conversation, number=1, content="The original database was PostgreSQL.")
    _message(db, foreign, number=1, content="Foreign PostgreSQL reference.")
    _message(db, conversation, number=2, content="recent message 2")
    _message(db, conversation, number=3, content="recent message 3")
    _message(db, conversation, number=4, content="recent message 4")
    _message(db, conversation, number=5, content="recent message 5")
    target = _message(db, conversation, number=6, content="Yes, use that database.")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=2,
        older_lexical_query="PostgreSQL",
        older_lexical_limit=1,
    )

    assert [message.content for message in context.recent_messages] == [
        "recent message 4",
        "recent message 5",
    ]
    assert [message.content for message in context.older_lexical_messages] == [older_match.content]


def test_default_lexical_query_uses_target_text(db) -> None:
    conversation = _conversation(db)
    older_match = _message(db, conversation, number=1, content="Atlas is Project Alpha's component.")
    _message(db, conversation, number=2, content="recent context")
    target = _message(db, conversation, number=3, content="Atlas is failing again.")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_limit=1,
    )

    assert [message.content for message in context.older_lexical_messages] == [older_match.content]


@pytest.mark.parametrize(
    ("query", "content"),
    [
        ("Atlas", "Atlas is the deployment component."),
        ("Project Alpha", "Project Alpha owns the rollout."),
        ("migration error", "The migration error needs a schema fix."),
        ("deployment", "The deployment completed yesterday."),
    ],
)
def test_lexical_context_finds_distinctive_raw_terms(
    db,
    query: str,
    content: str,
) -> None:
    conversation = _conversation(db)
    expected = _message(db, conversation, number=1, content=content)
    _message(db, conversation, number=2, content="unrelated recent message")
    target = _message(db, conversation, number=3, content="Please investigate the earlier issue.")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_query=query,
        older_lexical_limit=1,
    )

    assert [message.content for message in context.older_lexical_messages] == [expected.content]


def test_lexical_matches_are_ranked_then_presented_chronologically(db) -> None:
    conversation = _conversation(db)
    first_match = _message(db, conversation, number=1, content="Atlas migration was prepared.")
    later_match = _message(db, conversation, number=2, content="Atlas deployment then failed.")
    _message(db, conversation, number=3, content="most recent raw context")
    target = _message(db, conversation, number=4, content="Atlas is failing again.")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_query="Atlas",
        older_lexical_limit=2,
    )

    assert [message.content for message in context.older_lexical_messages] == [
        first_match.content,
        later_match.content,
    ]


def test_bm25_lexical_context_does_not_require_every_query_term(db) -> None:
    """Regression for Problem B: the old ``plainto_tsquery`` explicit-query
    path treated every word as required (AND semantics), so a natural,
    multi-word query would find nothing unless the message happened to
    contain every word. BM25 must still find a message sharing only one
    term."""

    conversation = _conversation(db)
    relevant = _message(db, conversation, number=1, content="The Atlas deployment finally succeeded.")
    _message(db, conversation, number=2, content="unrelated recent message")
    target = _message(db, conversation, number=3, content="Please check on it.")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_query="Atlas rollout status update",
        older_lexical_limit=1,
    )

    assert [message.content for message in context.older_lexical_messages] == [relevant.content]


def test_bm25_lexical_context_never_reads_another_conversation(db) -> None:
    conversation = _conversation(db)
    foreign = _conversation(db)
    _message(db, foreign, number=1, content="Atlas belongs to a foreign conversation.")
    _message(db, conversation, number=2, content="unrelated recent message")
    target = _message(db, conversation, number=3, content="Atlas status?")

    context = build_extraction_context(
        db,
        user_external_id=conversation.user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_query="Atlas",
        older_lexical_limit=5,
    )

    assert context.older_lexical_messages == []


def test_context_is_scoped_to_the_requested_user_when_conversation_ids_match(db) -> None:
    shared_conversation_id = f"shared-context-{uuid4().hex}"
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
    selected_older = _message(db, selected, number=1, content="Atlas belongs to selected user.")
    _message(db, selected, number=2, content="selected recent context")
    selected_target = _message(db, selected, number=3, content="Atlas failed again.")
    _message(db, foreign, number=1, content="Atlas belongs to foreign user.")
    _message(db, foreign, number=2, content="foreign target")

    context = build_extraction_context(
        db,
        user_external_id=selected.user.external_id,
        conversation_external_id=shared_conversation_id,
        target_message_ids=[str(selected_target.id)],
        recent_message_limit=1,
        older_lexical_query="Atlas",
        older_lexical_limit=2,
    )

    assert [message.content for message in context.older_lexical_messages] == [selected_older.content]
    assert all("foreign" not in message.content for message in context.recent_messages)
    assert all("foreign" not in message.content for message in context.older_lexical_messages)

    with pytest.raises(ContextError, match="must belong"):
        build_extraction_context(
            db,
            user_external_id=foreign.user.external_id,
            conversation_external_id=shared_conversation_id,
            target_message_ids=[str(selected_target.id)],
        )
