"""PostgreSQL integration tests guarded by TEST_DATABASE_URL."""

from __future__ import annotations

import os
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from src.config import configure, reset_config
from src.database import (
    Conversation,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    SessionLocal,
    User,
    create_tables,
    get_engine,
    reset_engine,
)
from src.database.models import utcnow


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; database integration tests never use DATABASE_URL.",
)


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    # This explicit value is the only URL database tests pass into the application.
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL)
    create_tables()
    yield
    reset_engine()
    reset_config()


def test_engine_connects_and_creates_current_tables() -> None:
    engine = get_engine()

    with engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1

    table_names = set(inspect(engine).get_table_names())
    assert {"users", "conversations", "messages", "memories"}.issubset(table_names)

    checked_out_before = engine.pool.checkedout()
    with SessionLocal() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1
        assert engine.pool.checkedout() == checked_out_before + 1
    assert engine.pool.checkedout() == checked_out_before


def test_schema_constraints_and_indexes() -> None:
    inspector = inspect(get_engine())

    user_indexes = inspector.get_indexes("users")
    assert any(
        index["column_names"] == ["external_id"] and index["unique"]
        for index in user_indexes
    )

    assert {foreign_key["referred_table"] for foreign_key in inspector.get_foreign_keys("conversations")} == {
        "users"
    }
    assert {foreign_key["referred_table"] for foreign_key in inspector.get_foreign_keys("messages")} == {
        "conversations"
    }
    assert {foreign_key["referred_table"] for foreign_key in inspector.get_foreign_keys("memories")} == {
        "users",
        "conversations",
        "memories",
    }

    memory_columns = {column["name"]: column for column in inspector.get_columns("memories")}
    assert memory_columns["conversation_id"]["nullable"] is True
    assert memory_columns["fact_key"]["nullable"] is True
    assert {
        frozenset({"user_id"}),
        frozenset({"conversation_id"}),
        frozenset({"fact_key"}),
        frozenset({"is_active"}),
    } <= {
        frozenset(index["column_names"]) for index in inspector.get_indexes("memories")
    }


def test_user_owns_conversations_and_messages() -> None:
    external_suffix = uuid4().hex

    with SessionLocal() as session:
        user = User(external_id=f"user-{external_suffix}")
        first_conversation = Conversation(external_id=f"conversation-a-{external_suffix}", user=user)
        second_conversation = Conversation(external_id=f"conversation-b-{external_suffix}", user=user)
        message = Message(
            conversation=first_conversation,
            role=MessageRole.USER,
            content="Please remember that I prefer concise explanations.",
        )
        session.add_all([user, first_conversation, second_conversation, message])
        session.commit()

        assert user.id is not None
        assert {conversation.id for conversation in user.conversations} == {
            first_conversation.id,
            second_conversation.id,
        }
        assert message.conversation_id == first_conversation.id
        assert message.role is MessageRole.USER


def test_user_external_id_must_be_unique() -> None:
    external_id = f"user-{uuid4().hex}"

    with SessionLocal() as session:
        session.add(User(external_id=external_id))
        session.commit()

        session.add(User(external_id=external_id))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_structured_and_unstructured_memories_persist() -> None:
    external_suffix = uuid4().hex
    valid_from = utcnow()
    valid_to = valid_from + timedelta(days=30)

    with SessionLocal() as session:
        user = User(external_id=f"user-{external_suffix}")
        conversation = Conversation(external_id=f"conversation-{external_suffix}", user=user)
        source_message = Message(
            conversation=conversation,
            role=MessageRole.ASSISTANT,
            content="I will use implementation-first technical explanations.",
        )
        session.add_all([user, conversation, source_message])
        session.flush()

        historical_structured = Memory(
            user=user,
            conversation=conversation,
            memory_type=MemoryType.SEMANTIC,
            memory_text="User lives in Bangalore",
            subject_type="user",
            subject_id=user.external_id,
            predicate="location",
            value="Bangalore",
            fact_key=f"user:{user.external_id}:location",
            confidence=0.95,
            importance=8,
            source_message_ids=[str(source_message.id)],
            valid_from=valid_from,
            valid_to=valid_to,
            is_active=False,
        )
        current_structured = Memory(
            user=user,
            conversation=conversation,
            memory_type=MemoryType.SEMANTIC,
            memory_text="User lives in Bengaluru",
            subject_type="user",
            subject_id=user.external_id,
            predicate="location",
            value="Bengaluru",
            fact_key=f"user:{user.external_id}:location",
            source_message_ids=[str(source_message.id)],
        )
        historical_structured.superseded_by = current_structured
        unstructured = Memory(
            user=user,
            memory_type=MemoryType.SEMANTIC,
            memory_text="User prefers implementation-first technical explanations.",
            fact_key=None,
            source_message_ids=[str(source_message.id)],
        )
        session.add_all([historical_structured, current_structured, unstructured])
        session.commit()

        persisted_structured = session.get(Memory, historical_structured.id)
        persisted_current = session.get(Memory, current_structured.id)
        persisted_unstructured = session.get(Memory, unstructured.id)

        assert persisted_structured is not None
        assert persisted_structured.conversation_id == conversation.id
        assert persisted_structured.fact_key == f"user:{user.external_id}:location"
        assert persisted_structured.source_message_ids == [str(source_message.id)]
        assert persisted_structured.valid_from == valid_from
        assert persisted_structured.valid_to == valid_to
        assert persisted_structured.is_active is False
        assert persisted_structured.superseded_by_id == current_structured.id

        assert persisted_current is not None
        assert persisted_current.fact_key == f"user:{user.external_id}:location"
        assert persisted_current.valid_to is None
        assert persisted_current.superseded_by_id is None
        assert persisted_current.is_active is True

        assert persisted_unstructured is not None
        assert persisted_unstructured.conversation_id is None
        assert persisted_unstructured.fact_key is None
        assert persisted_unstructured.predicate is None
        assert persisted_unstructured.value is None
