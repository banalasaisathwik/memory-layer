"""PostgreSQL integration tests for incremental per-conversation Message FAISS sync.

These exercise ``_ensure_conversation_message_index()``'s three-path design
directly through the public ``retrieve_semantic_message_context()`` /
``sync_conversation_message_index()`` entry points, using the sync-stats
counters (``full_rebuilds`` / ``incremental_appends`` / ``unchanged_hits``)
to prove which path actually ran. No live provider calls: the embedding
client is a small deterministic fake, like tests/memory/test_semantic_context.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from meminfra.config import configure, reset_config
from meminfra.database import Conversation, Message, MessageRole, SessionLocal, User, create_tables, reset_engine
from meminfra.retrieval import (
    conversation_message_index_paths,
    get_message_index_sync_stats,
    load_conversation_message_index,
    reset_message_index_sync_stats,
    retrieve_semantic_message_context,
    sync_conversation_message_index,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; message-index sync integration tests never use DATABASE_URL.",
    ),
]


class FakeEmbeddingClient:
    """Fixed 2D vectors: ranking is irrelevant here, only sync-path behavior is tested."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)]
        )


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL, embedding_model="fake-sync-model")
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
        embedding_model="fake-sync-model",
        faiss_index_dir=tmp_path,
        vector_candidate_multiplier=5,
    )
    monkeypatch.setattr("meminfra.retrieval.message_vector.get_embedding_client", lambda: client)
    reset_message_index_sync_stats()
    return client


def _user(db) -> User:
    user = User(external_id=f"sync-user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _conversation(db, user: User) -> Conversation:
    conversation = Conversation(external_id=f"sync-conversation-{uuid4().hex}", user_id=user.id)
    db.add(conversation)
    db.commit()
    return conversation


def _message(db, conversation: Conversation, *, number: int, content: str | None = None) -> Message:
    message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER if number % 2 else MessageRole.ASSISTANT,
        content=content or f"message number {number} unique content",
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc) + timedelta(minutes=number),
    )
    db.add(message)
    db.commit()
    return message


def _search(db, user: User, conversation: Conversation, *, query: str = "component reference meaning"):
    return retrieve_semantic_message_context(
        db,
        user_external_id=user.external_id,
        conversation_external_id=conversation.external_id,
        query_text=query,
    )


def test_empty_index_builds_fully_from_scratch(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _message(db, conversation, number=2)

    results = _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert len(results) == 2
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 0
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.index.ntotal == 2


def test_single_append_only_embeds_and_adds_the_new_message(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _message(db, conversation, number=2)
    _search(db, user, conversation)  # initial full build
    calls_after_build = len(fake_embeddings.calls)

    third = _message(db, conversation, number=3)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 1
    # Only the new message's content was sent for embedding (plus the query
    # text itself); M1/M2 content is never re-submitted for embedding.
    new_calls = fake_embeddings.calls[calls_after_build:]
    message_content_calls = [call for call in new_calls if call[1] == [third.content]]
    assert len(message_content_calls) == 1
    assert all(call[1] != ["message number 1 unique content", "message number 2 unique content"] for call in new_calls)
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.index.ntotal == 3
    assert loaded.message_ids[-1] == str(third.id)


def test_repeated_appends_each_stay_incremental(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _message(db, conversation, number=2)
    _search(db, user, conversation)

    for number in (3, 4, 5):
        _message(db, conversation, number=number)
        _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 3
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.index.ntotal == 5


def test_repeated_search_with_no_new_messages_does_not_rebuild_or_append(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _search(db, user, conversation)

    _search(db, user, conversation)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 0
    assert stats.unchanged_hits == 2


def test_model_mismatch_falls_back_to_full_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _search(db, user, conversation)

    configure(embedding_model="other-sync-model")
    _message(db, conversation, number=2)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.embedding_model == "other-sync-model"
    assert loaded.index.ntotal == 2


def test_dimension_divergence_falls_back_to_full_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _search(db, user, conversation)
    index_path, metadata_path = conversation_message_index_paths(user.external_id, conversation.external_id)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_dimension"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    _message(db, conversation, number=2)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.dimension == 2
    assert loaded.index.ntotal == 2


def test_metadata_id_divergence_does_not_blindly_append(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first = _message(db, conversation, number=1)
    _search(db, user, conversation)
    index_path, metadata_path = conversation_message_index_paths(user.external_id, conversation.external_id)

    # Simulate a diverged prefix: the persisted mapping's second entry does
    # not correspond to any message that will actually be appended next.
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["message_ids"] = [str(first.id), str(uuid4())]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    second = _message(db, conversation, number=2)
    third = _message(db, conversation, number=3)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.incremental_appends == 0
    assert stats.full_rebuilds == 2
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.message_ids == [str(first.id), str(second.id), str(third.id)]
    assert loaded.index.ntotal == 3


def test_message_removed_from_eligible_set_forces_safe_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    first = _message(db, conversation, number=1)
    second = _message(db, conversation, number=2)
    _search(db, user, conversation)

    # Blank content makes the message ineligible (mirrors the same btrim()
    # filter used everywhere messages are read), simulating a message that
    # disappeared from the indexable set without touching FAISS directly.
    second.content = "   "
    db.commit()

    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.incremental_appends == 0
    assert stats.full_rebuilds == 2
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.message_ids == [str(first.id)]
    assert loaded.index.ntotal == 1


def test_corrupt_index_file_recovers_via_full_rebuild_not_incremental(db, fake_embeddings) -> None:
    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _search(db, user, conversation)
    index_path, _ = conversation_message_index_paths(user.external_id, conversation.external_id)

    index_path.write_bytes(b"not a real faiss index")
    _message(db, conversation, number=2)
    _search(db, user, conversation)

    stats = get_message_index_sync_stats()
    assert stats.incremental_appends == 0
    assert stats.full_rebuilds == 2
    loaded = load_conversation_message_index(
        user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    assert loaded.index.ntotal == 2


def test_sync_conversation_message_index_still_performs_a_full_rebuild(db, fake_embeddings) -> None:
    """Direct callers of the explicit rebuild entry point always get a full rebuild."""

    user = _user(db)
    conversation = _conversation(db, user)
    _message(db, conversation, number=1)
    _message(db, conversation, number=2)

    sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )
    sync_conversation_message_index(
        db, user_external_id=user.external_id, conversation_external_id=conversation.external_id
    )

    stats = get_message_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
