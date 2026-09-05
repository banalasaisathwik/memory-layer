"""PostgreSQL integration tests for incremental durable Memory FAISS sync.

Mirrors ``tests/retrieval/test_message_vector_sync.py``: exercises the
three-path ``_ensure_user_memory_index()`` design (fast unchanged match,
append-only incremental update, full validate-or-rebuild) through the public
``vector_retrieve()`` / ``sync_user_memory_index()`` entry points, using the
sync-stats counters to prove which path actually ran. No live provider
calls: the embedding client is a small deterministic fake.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.config import configure, reset_config
from src.database import Memory, MemoryType, SessionLocal, User, create_tables, reset_engine
from src.retrieval import SearchFilters
from src.retrieval.vector import (
    get_memory_index_sync_stats,
    load_user_memory_index,
    reset_memory_index_sync_stats,
    sync_user_memory_index,
    user_index_paths,
    vector_retrieve,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; memory-index sync integration tests never use DATABASE_URL.",
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
    configure(database_url=TEST_DATABASE_URL, embedding_model="fake-vector-sync-model")
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
        embedding_model="fake-vector-sync-model",
        faiss_index_dir=tmp_path,
        vector_candidate_multiplier=5,
    )
    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: client)
    reset_memory_index_sync_stats()
    return client


def _user(db) -> User:
    user = User(external_id=f"vector-sync-user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _memory(db, user: User, *, number: int, text: str | None = None, is_active: bool = True) -> Memory:
    memory = Memory(
        user_id=user.id,
        memory_type=MemoryType.SEMANTIC,
        memory_text=text or f"memory number {number} unique fact",
        is_active=is_active,
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc) + timedelta(minutes=number),
    )
    db.add(memory)
    db.commit()
    return memory


def _search(db, user: User, *, query: str = "unique fact reference"):
    return vector_retrieve(
        db,
        query,
        user=user,
        filters=SearchFilters(include_history=True),
        conversation=None,
        limit=10,
    )


def test_empty_index_builds_fully_from_scratch(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _memory(db, user, number=2)

    results = _search(db, user)

    stats = get_memory_index_sync_stats()
    assert len(results) == 2
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 0
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.index.ntotal == 2


def test_single_append_only_embeds_and_adds_the_new_memory(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _memory(db, user, number=2)
    _search(db, user)  # initial full build
    calls_after_build = len(fake_embeddings.calls)

    third = _memory(db, user, number=3)
    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 1
    new_calls = fake_embeddings.calls[calls_after_build:]
    memory_text_calls = [call for call in new_calls if call[1] == [third.memory_text]]
    assert len(memory_text_calls) == 1
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.index.ntotal == 3
    assert loaded.memory_ids[-1] == str(third.id)


def test_repeated_search_with_no_new_memories_does_not_rebuild_or_append(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _search(db, user)

    _search(db, user)
    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.incremental_appends == 0
    assert stats.unchanged_hits == 2


def test_supersession_state_change_alone_stays_on_fast_path(db, fake_embeddings) -> None:
    """Flipping is_active must not force a rebuild: filtering happens post-FAISS."""

    user = _user(db)
    first = _memory(db, user, number=1)
    _search(db, user)

    first.is_active = False
    db.commit()

    active_only = vector_retrieve(
        db,
        "unique fact reference",
        user=user,
        filters=SearchFilters(include_history=False),
        conversation=None,
        limit=10,
    )

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 1
    assert stats.unchanged_hits == 1
    assert stats.incremental_appends == 0
    # The FAISS index still contains the row; live DB filtering removes it.
    assert all(memory.id != first.id for memory in active_only)


def test_model_mismatch_falls_back_to_full_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _search(db, user)

    configure(embedding_model="other-vector-sync-model")
    _memory(db, user, number=2)
    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.embedding_model == "other-vector-sync-model"
    assert loaded.index.ntotal == 2


def test_dimension_divergence_falls_back_to_full_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _search(db, user)
    _, metadata_path = user_index_paths(user.external_id)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_dimension"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    _memory(db, user, number=2)
    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.dimension == 2
    assert loaded.index.ntotal == 2


def test_corrupt_index_file_recovers_via_full_rebuild_not_incremental(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _search(db, user)
    index_path, _ = user_index_paths(user.external_id)

    index_path.write_bytes(b"not a real faiss index")
    _memory(db, user, number=2)
    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.incremental_appends == 0
    assert stats.full_rebuilds == 2
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.index.ntotal == 2


def test_missing_index_forces_rebuild(db, fake_embeddings) -> None:
    user = _user(db)
    _memory(db, user, number=1)
    _search(db, user)
    index_path, metadata_path = user_index_paths(user.external_id)
    index_path.unlink()

    _search(db, user)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 2
    loaded = load_user_memory_index(user_external_id=user.external_id)
    assert loaded.index.ntotal == 1


def test_user_isolation_between_indexes(db, fake_embeddings) -> None:
    user_a = _user(db)
    user_b = _user(db)
    _memory(db, user_a, number=1, text="alpha only fact")
    _memory(db, user_b, number=1, text="beta only fact")

    results_a = _search(db, user_a, query="alpha only fact")
    results_b = _search(db, user_b, query="beta only fact")

    assert all(memory.user_id == user_a.id for memory in results_a)
    assert all(memory.user_id == user_b.id for memory in results_b)


def test_sync_user_memory_index_still_performs_a_full_rebuild(db, fake_embeddings) -> None:
    """Direct callers of the explicit rebuild entry point always get a full rebuild."""

    user = _user(db)
    _memory(db, user, number=1)
    _memory(db, user, number=2)

    sync_user_memory_index(db, user_external_id=user.external_id)
    sync_user_memory_index(db, user_external_id=user.external_id)

    stats = get_memory_index_sync_stats()
    assert stats.full_rebuilds == 2
    assert stats.incremental_appends == 0
