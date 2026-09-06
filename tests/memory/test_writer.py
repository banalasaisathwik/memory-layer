"""PostgreSQL integration tests for deterministic memory writes."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from meminfra.config import configure, reset_config
from meminfra.database import (
    Conversation,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    SessionLocal,
    User,
    create_tables,
    reset_engine,
)
from meminfra.memory import CandidateMemory, WriteAction, WriteError, write_memories


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; writer integration tests never use DATABASE_URL.",
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


def _user(db, *, external_id: str | None = None) -> User:
    user = User(external_id=external_id or f"user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _conversation(db, user: User) -> Conversation:
    conversation = Conversation(external_id=f"conversation-{uuid4().hex}", user_id=user.id)
    db.add(conversation)
    db.commit()
    return conversation


def _message(db, conversation: Conversation) -> Message:
    message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content="Please remember this detail.",
    )
    db.add(message)
    db.commit()
    return message


def _candidate(**overrides: object) -> CandidateMemory:
    data: dict[str, object] = {
        "memory_text": "User lives in Bangalore",
        "subject_type": "user",
        "predicate": "location",
        "value": "Bangalore",
        "confidence": 0.95,
        "importance": 0.8,
    }
    data.update(overrides)
    return CandidateMemory(**data)


def _memory_count(db, user: User) -> int:
    return db.scalar(select(func.count()).select_from(Memory).where(Memory.user_id == user.id))


def test_missing_user_fails_without_writing(db) -> None:
    with pytest.raises(WriteError, match="No user exists"):
        write_memories(db, [_candidate()], user_external_id="missing-user")


def test_missing_conversation_fails_without_writing(db) -> None:
    user = _user(db)

    with pytest.raises(WriteError, match="No conversation exists"):
        write_memories(
            db,
            [_candidate()],
            user_external_id=user.external_id,
            conversation_external_id="missing-conversation",
        )


def test_structured_add_persists_canonical_fields_without_conversation(db) -> None:
    user = _user(db)

    results = write_memories(db, [_candidate()], user_external_id=user.external_id)

    result = results[0]
    persisted = db.get(Memory, result.memory_id)
    assert result.action is WriteAction.ADD
    assert result.fact_key == f"user:{user.external_id}:location"
    assert result.superseded_memory_id is None
    assert persisted is not None
    assert persisted.conversation_id is None
    assert persisted.subject_id == user.external_id
    assert persisted.predicate == "location"
    assert persisted.value == "Bangalore"
    assert persisted.fact_key == result.fact_key
    assert persisted.confidence == 0.95
    assert persisted.importance == 0.8


def test_same_user_conversation_and_provenance_are_attached(db) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    message = _message(db, conversation)

    result = write_memories(
        db,
        [_candidate(source_message_ids=[str(message.id)])],
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )[0]

    persisted = db.get(Memory, result.memory_id)
    assert persisted is not None
    assert persisted.conversation_id == conversation.id
    assert persisted.source_message_ids == [str(message.id)]


def test_cross_user_conversation_and_provenance_are_rejected(db) -> None:
    owner = _user(db)
    foreign_user = _user(db)
    foreign_conversation = _conversation(db, foreign_user)
    foreign_message = _message(db, foreign_conversation)

    with pytest.raises(WriteError, match="No conversation exists"):
        write_memories(
            db,
            [_candidate()],
            user_external_id=owner.external_id,
            conversation_external_id=foreign_conversation.external_id,
        )

    with pytest.raises(WriteError, match="different user"):
        write_memories(
            db,
            [_candidate(source_message_ids=[str(foreign_message.id)])],
            user_external_id=owner.external_id,
        )


def test_two_users_may_independently_use_the_same_conversation_external_id(db) -> None:
    user_a = _user(db)
    user_b = _user(db)
    conversation_a = Conversation(external_id="main", user_id=user_a.id)
    conversation_b = Conversation(external_id="main", user_id=user_b.id)
    db.add_all([conversation_a, conversation_b])
    db.commit()

    result_a = write_memories(
        db,
        [_candidate(memory_text="User A prefers PostgreSQL", value="PostgreSQL")],
        user_external_id=user_a.external_id,
        conversation_external_id="main",
    )[0]
    result_b = write_memories(
        db,
        [_candidate(memory_text="User B prefers MongoDB", value="MongoDB")],
        user_external_id=user_b.external_id,
        conversation_external_id="main",
    )[0]

    memory_a = db.get(Memory, result_a.memory_id)
    memory_b = db.get(Memory, result_b.memory_id)
    assert memory_a.user_id == user_a.id
    assert memory_a.conversation_id == conversation_a.id
    assert memory_b.user_id == user_b.id
    assert memory_b.conversation_id == conversation_b.id


def test_conversation_external_id_belonging_only_to_another_user_is_rejected(db) -> None:
    owner = _user(db)
    foreign_user = _user(db)
    foreign_conversation = Conversation(external_id="main", user_id=foreign_user.id)
    db.add(foreign_conversation)
    db.commit()

    with pytest.raises(WriteError, match="No conversation exists"):
        write_memories(
            db,
            [_candidate()],
            user_external_id=owner.external_id,
            conversation_external_id="main",
        )


def test_single_user_conversation_write_behavior_is_unchanged(db) -> None:
    user = _user(db)
    conversation = _conversation(db, user)

    result = write_memories(
        db,
        [_candidate()],
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )[0]

    persisted = db.get(Memory, result.memory_id)
    assert result.action is WriteAction.ADD
    assert persisted.conversation_id == conversation.id


def test_provenance_must_match_the_supplied_conversation(db) -> None:
    user = _user(db)
    selected_conversation = _conversation(db, user)
    other_conversation = _conversation(db, user)
    other_message = _message(db, other_conversation)

    with pytest.raises(WriteError, match="does not belong to the supplied conversation"):
        write_memories(
            db,
            [_candidate(source_message_ids=[str(other_message.id)])],
            user_external_id=user.external_id,
            conversation_external_id=selected_conversation.external_id,
        )


def test_equivalent_structured_value_is_a_noop_and_merges_provenance(db) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first_message = _message(db, conversation)
    second_message = _message(db, conversation)

    first = write_memories(
        db,
        [_candidate(source_message_ids=[str(first_message.id)])],
        user_external_id=user.external_id,
    )[0]
    repeated = write_memories(
        db,
        [_candidate(value="  bangalore  ", source_message_ids=[str(second_message.id)])],
        user_external_id=user.external_id,
    )[0]

    persisted = db.get(Memory, first.memory_id)
    assert repeated.action is WriteAction.NOOP
    assert repeated.memory_id == first.memory_id
    assert _memory_count(db, user) == 1
    assert persisted is not None
    assert persisted.source_message_ids == [str(first_message.id), str(second_message.id)]


def test_changed_single_value_supersedes_without_deleting_history(db) -> None:
    user = _user(db)
    first = write_memories(db, [_candidate(value="Hyderabad")], user_external_id=user.external_id)[0]

    result = write_memories(db, [_candidate(value="Bangalore")], user_external_id=user.external_id)[0]

    old_memory = db.get(Memory, first.memory_id)
    new_memory = db.get(Memory, result.memory_id)
    assert result.action is WriteAction.SUPERSEDE
    assert result.superseded_memory_id == first.memory_id
    assert old_memory is not None and new_memory is not None
    assert old_memory.is_active is False
    assert old_memory.valid_to == new_memory.valid_from
    assert old_memory.superseded_by_id == new_memory.id
    assert new_memory.is_active is True
    assert _memory_count(db, user) == 2


def test_multi_valued_facts_coexist_and_repeat_is_a_noop(db) -> None:
    user = _user(db)
    python = _candidate(
        memory_text="User knows Python",
        predicate="programming_language",
        value="Python",
    )
    go = _candidate(
        memory_text="User knows Go",
        predicate="programming_language",
        value="Go",
    )

    first, second = write_memories(db, [python, go], user_external_id=user.external_id)
    repeated = write_memories(db, [python], user_external_id=user.external_id)[0]

    active = list(
        db.scalars(select(Memory).where(Memory.user_id == user.id, Memory.is_active.is_(True)))
    )
    assert first.action is WriteAction.ADD
    assert second.action is WriteAction.ADD
    assert repeated.action is WriteAction.NOOP
    assert {memory.fact_key for memory in active} == {
        f"user:{user.external_id}:programming_language:python",
        f"user:{user.external_id}:programming_language:go",
    }


def test_open_semantic_unknown_predicates_dedupe_only_exact_text(db) -> None:
    user = _user(db)
    open_memory = _candidate(
        memory_text="User prefers implementation-first explanations",
        predicate="explanation_style_preference",
        value="implementation-first",
    )
    different_open_memory = _candidate(
        memory_text="User prefers diagrams for architecture discussions",
        predicate="explanation_style_preference",
        value="diagrams",
    )

    first = write_memories(db, [open_memory], user_external_id=user.external_id)[0]
    repeated = write_memories(
        db,
        [open_memory.model_copy(update={"memory_text": " user prefers implementation-first explanations "})],
        user_external_id=user.external_id,
    )[0]
    different = write_memories(db, [different_open_memory], user_external_id=user.external_id)[0]

    persisted = db.get(Memory, first.memory_id)
    assert first.action is WriteAction.ADD
    assert repeated.action is WriteAction.NOOP
    assert different.action is WriteAction.ADD
    assert persisted is not None
    assert persisted.predicate is None
    assert persisted.value is None
    assert persisted.fact_key is None


def test_non_user_subject_persists_without_an_invented_canonical_id(db) -> None:
    user = _user(db)
    project_memory = _candidate(
        memory_text="Project Alpha is active",
        subject_type="project",
        predicate="project_status",
        value="active",
    )

    result = write_memories(db, [project_memory], user_external_id=user.external_id)[0]

    persisted = db.get(Memory, result.memory_id)
    assert result.action is WriteAction.ADD
    assert persisted is not None
    assert persisted.subject_id is None
    assert persisted.predicate == "project_status"
    assert persisted.fact_key is None


def test_episodic_duplicate_requires_the_same_provenance(db) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first_message = _message(db, conversation)
    second_message = _message(db, conversation)
    episode = _candidate(
        memory_type=MemoryType.EPISODIC,
        memory_text="User is debugging authentication today",
        predicate=None,
        value=None,
        source_message_ids=[str(first_message.id)],
    )

    first = write_memories(db, [episode], user_external_id=user.external_id)[0]
    repeated = write_memories(db, [episode], user_external_id=user.external_id)[0]
    later_episode = write_memories(
        db,
        [episode.model_copy(update={"source_message_ids": [str(second_message.id)]})],
        user_external_id=user.external_id,
    )[0]

    assert first.action is WriteAction.ADD
    assert repeated.action is WriteAction.NOOP
    assert later_episode.action is WriteAction.ADD
    assert _memory_count(db, user) == 2
    assert db.get(Memory, first.memory_id).fact_key is None


def test_batch_failure_rolls_back_adds_and_in_progress_supersession(db) -> None:
    user = _user(db)
    initial = write_memories(db, [_candidate(value="Hyderabad")], user_external_id=user.external_id)[0]
    changed = _candidate(value="Bangalore")
    invalid_provenance = _candidate(
        memory_text="User prefers PostgreSQL",
        predicate="database_preference",
        value="PostgreSQL",
        source_message_ids=["not-a-message-id"],
    )

    with pytest.raises(WriteError, match="does not exist"):
        write_memories(db, [changed, invalid_provenance], user_external_id=user.external_id)

    old_memory = db.get(Memory, initial.memory_id)
    assert old_memory is not None
    assert old_memory.is_active is True
    assert old_memory.superseded_by_id is None
    assert _memory_count(db, user) == 1


def test_same_logical_fact_for_two_users_is_isolated(db) -> None:
    first_user = _user(db)
    second_user = _user(db)

    first = write_memories(db, [_candidate()], user_external_id=first_user.external_id)[0]
    second = write_memories(db, [_candidate()], user_external_id=second_user.external_id)[0]

    assert first.action is WriteAction.ADD
    assert second.action is WriteAction.ADD
    assert first.memory_id != second.memory_id
    assert db.get(Memory, first.memory_id).user_id == first_user.id
    assert db.get(Memory, second.memory_id).user_id == second_user.id
