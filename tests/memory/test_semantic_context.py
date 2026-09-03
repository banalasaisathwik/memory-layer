"""PostgreSQL integration tests for conversation-scoped semantic raw-message context."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import inspect

from src.config import configure, reset_config
from src.database import Conversation, Message, MessageRole, SessionLocal, User, create_tables, reset_engine
from src.memory import build_extraction_context, retrieve_semantic_message_context
from src.retrieval import (
    IndexDimensionMismatchError,
    IndexModelMismatchError,
    conversation_message_index_paths,
    load_conversation_message_index,
    sync_conversation_message_index,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; semantic context integration tests never use DATABASE_URL.",
    ),
]


class FakeEmbeddingClient:
    """Small deterministic semantic space; it never makes an external request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=self._vector_for(text))
                for index, text in enumerate(input)
            ]
        )

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        normalized = text.casefold()
        if "dimension query" in normalized:
            return [1.0, 0.0, 0.0]
        if "lexical-only" in normalized:
            return [0.2, 0.98]
        if "irrelevant" in normalized or "recent" in normalized:
            return [0.0, 1.0]
        if "later historical" in normalized:
            return [1.0, 0.0]
        if "first historical" in normalized:
            return [0.8, 0.2]
        if any(
            term in normalized
            for term in (
                "atlas",
                "component",
                "shared-anchor",
                "semantic-only",
                "semantic query",
                "reference meaning",
            )
        ):
            return [1.0, 0.0]
        return [0.6, 0.8]


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL, embedding_model="fake-message-model")
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
def fake_embeddings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeEmbeddingClient:
    client = FakeEmbeddingClient()
    configure(
        embedding_model="fake-message-model",
        faiss_index_dir=tmp_path,
        vector_candidate_multiplier=5,
        extraction_semantic_messages=3,
        extraction_old_messages=6,
    )
    monkeypatch.setattr("src.retrieval.message_vector.get_embedding_client", lambda: client)
    return client


def _user(db, *, external_id: str | None = None) -> User:
    user = User(external_id=external_id or f"semantic-user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _conversation(db, user: User, *, external_id: str | None = None) -> Conversation:
    conversation = Conversation(external_id=external_id or f"semantic-conversation-{uuid4().hex}", user_id=user.id)
    db.add(conversation)
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


def test_message_embeddings_persist_reuse_and_reembed_on_model_change(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    message = _message(db, conversation, number=1, content="Project Alpha uses Atlas for deployments.")

    first_path = sync_conversation_message_index(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )
    calls_after_first_sync = len(fake_embeddings.calls)
    sync_conversation_message_index(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )
    configure(embedding_model="replacement-message-model")
    sync_conversation_message_index(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
    )

    persisted = db.get(Message, message.id)
    assert first_path is not None and first_path.is_file()
    assert persisted is not None
    assert persisted.embedding == pytest.approx([1.0, 0.0])
    assert persisted.embedding_model == "replacement-message-model"
    assert len(fake_embeddings.calls) == calls_after_first_sync + 1
    assert fake_embeddings.calls[-1][0] == "replacement-message-model"


def test_message_indexes_are_hashed_per_conversation_and_reload_with_uuid_mapping(db, fake_embeddings) -> None:
    user = _user(db)
    first = _conversation(db, user)
    second = _conversation(db, user)
    first_message = _message(db, first, number=1, content="Atlas is the first component.")
    second_message = _message(db, second, number=1, content="Atlas is the second component.")

    first_path = sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=first.external_id
    )
    second_path = sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=second.external_id
    )
    loaded = load_conversation_message_index(
        user_external_id=user.external_id,
        conversation_external_id=first.external_id,
    )
    expected_paths = conversation_message_index_paths(user.external_id, first.external_id)

    assert first_path == expected_paths[0]
    assert first_path is not None and second_path is not None and first_path != second_path
    assert first_path.parent.name == "messages"
    assert user.external_id not in first_path.name and first.external_id not in first_path.name
    assert loaded.message_ids == [str(first_message.id)]
    assert str(second_message.id) not in loaded.message_ids


def test_semantic_retrieval_finds_a_paraphrase_without_lexical_overlap_and_is_chronological(
    db, fake_embeddings
) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    old = _message(db, conversation, number=1, content="Project Alpha uses Atlas for deployments.")
    _message(db, conversation, number=2, content="recent unrelated message")
    target = _message(db, conversation, number=3, content="That component we discussed is failing again.")

    context = build_extraction_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_limit=2,
        older_semantic_limit=2,
    )

    assert context.older_lexical_messages == []
    assert [message.content for message in context.older_semantic_messages] == [old.content]
    assert [message.content for message in context.older_relevant_messages] == [old.content]
    assert [call for call in fake_embeddings.calls if call[1] == [target.content]] == [
        ("fake-message-model", [target.content])
    ]


def test_semantic_context_excludes_recent_and_target_rows_and_returns_selected_rows_chronologically(
    db, fake_embeddings
) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first = _message(db, conversation, number=1, content="first historical reference")
    later = _message(db, conversation, number=2, content="later historical reference")
    recent = _message(db, conversation, number=3, content="recent component reference")
    target = _message(db, conversation, number=4, content="component reference meaning")

    results = retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text=target.content,
        limit=2,
        exclude_message_ids={str(recent.id), str(target.id)},
        before_message_id=str(target.id),
    )

    assert [message.id for message in results] == [first.id, later.id]
    assert recent.id not in {message.id for message in results}
    assert target.id not in {message.id for message in results}


def test_semantic_context_isolates_users_and_conversations_even_when_external_ids_match(db, fake_embeddings) -> None:
    first_user = _user(db)
    second_user = _user(db)
    shared_external_id = f"shared-{uuid4().hex}"
    first = _conversation(db, first_user, external_id=shared_external_id)
    second = _conversation(db, second_user, external_id=shared_external_id)
    first_message = _message(db, first, number=1, content="Atlas belongs to first user.")
    foreign_message = _message(db, second, number=1, content="Atlas belongs to second user.")

    results = retrieve_semantic_message_context(
        db,
        user_external_id=first_user.external_id,
        conversation_external_id=shared_external_id,
        query_text="component reference meaning",
    )

    assert [message.id for message in results] == [first_message.id]
    assert foreign_message.id not in {message.id for message in results}


def test_stale_corrupt_and_orphan_message_indexes_rebuild_from_postgresql(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first = _message(db, conversation, number=1, content="Atlas first context.")
    sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    index_path, metadata_path = conversation_message_index_paths(user.external_id, conversation.external_id)

    index_path.unlink()
    missing = retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text="component reference meaning",
    )
    assert [message.id for message in missing] == [first.id]

    index_path.write_bytes(b"not a FAISS index")
    corrupt = retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text="component reference meaning",
    )
    assert [message.id for message in corrupt] == [first.id]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["message_ids"] = [str(uuid4())]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    orphan = retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text="component reference meaning",
    )
    assert [message.id for message in orphan] == [first.id]

    second = _message(db, conversation, number=2, content="Atlas second context.")
    stale = retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text="component reference meaning",
    )
    assert {message.id for message in stale} == {first.id, second.id}


def test_message_index_model_and_query_dimension_mismatches_are_explicit(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1, content="Atlas component context.")
    sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )

    configure(embedding_model="other-message-model")
    with pytest.raises(IndexModelMismatchError, match="different embedding model"):
        load_conversation_message_index(
            user_external_id=user.external_id,
            conversation_external_id=conversation.external_id,
        )

    configure(embedding_model="fake-message-model")
    with pytest.raises(IndexDimensionMismatchError, match="dimension"):
        retrieve_semantic_message_context(
            db,
            user_external_id=user.external_id,
            conversation_external_id=conversation.external_id,
            query_text="dimension query",
        )


def test_context_merges_lexical_and_semantic_old_messages_once_and_degrades_explicitly(
    db, fake_embeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    shared = _message(db, conversation, number=1, content="shared-anchor lexicalanchor history")
    lexical_only = _message(db, conversation, number=2, content="lexical-only lexicalanchor history")
    semantic_only = _message(db, conversation, number=3, content="semantic-only history")
    _message(db, conversation, number=4, content="recent unrelated message")
    target = _message(db, conversation, number=5, content="component reference meaning")

    context = build_extraction_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(target.id)],
        recent_message_limit=1,
        older_lexical_query="lexicalanchor",
        older_lexical_limit=2,
        older_semantic_limit=2,
        older_context_limit=3,
    )

    assert [message.content for message in context.older_relevant_messages] == [
        shared.content,
        lexical_only.content,
        semantic_only.content,
    ]

    class FailingEmbeddingClient:
        embeddings = None

        def __init__(self) -> None:
            self.embeddings = self

        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            raise RuntimeError("provider unavailable")

    another_target = _message(db, conversation, number=6, content="component reference meaning again")
    monkeypatch.setattr(
        "src.retrieval.message_vector.get_embedding_client", lambda: FailingEmbeddingClient()
    )
    failed_context = build_extraction_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        target_message_ids=[str(another_target.id)],
        recent_message_limit=1,
        older_lexical_query="lexicalanchor",
        older_lexical_limit=1,
    )

    assert failed_context.semantic_retrieval_error == (
        "The configured embedding provider could not generate vectors."
    )
    assert len(failed_context.older_lexical_messages) == 1
    assert "lexicalanchor" in failed_context.older_lexical_messages[0].content
    assert failed_context.recent_messages


def test_message_embedding_columns_exist_after_the_milestone_61_migration() -> None:
    with SessionLocal() as session:
        columns = {column["name"] for column in inspect(session.bind).get_columns("messages")}

    assert {"embedding", "embedding_model"} <= columns
