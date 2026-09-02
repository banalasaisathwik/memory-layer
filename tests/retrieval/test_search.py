"""PostgreSQL integration tests for user-scoped hybrid memory retrieval."""

from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import inspect

from src.config import configure, reset_config
from src.database import Conversation, Memory, MemoryType, SessionLocal, User, create_tables, reset_engine
from src.retrieval import (
    EmbeddingError,
    IndexDimensionMismatchError,
    InvalidFilterScopeError,
    InvalidSearchError,
    SearchFilters,
    UserNotFoundError,
    load_user_memory_index,
    rebuild_user_memory_index,
    search_memories,
    sync_user_memory_index,
    user_index_paths,
)


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; retrieval integration tests never use DATABASE_URL.",
    ),
]


class FakeEmbeddingClient:
    """Deterministic vectors that make semantic ranking and provider calls inspectable."""

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
        if "postgresql" in normalized or "which database" in normalized or "database does" in normalized:
            return [1.0, 0.0]
        if "mysql" in normalized:
            return [0.0, 1.0]
        if "project alpha" in normalized or "kan-4" in normalized:
            return [0.7, 0.7]
        if "bangalore" in normalized:
            return [0.8, 0.2]
        if "hyderabad" in normalized:
            return [0.2, 0.8]
        return [0.6, 0.8]


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL, embedding_model="fake-embedding-model")
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
        embedding_model="fake-embedding-model",
        faiss_index_dir=tmp_path,
        vector_candidate_multiplier=5,
    )
    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: client)
    return client


def _user(db, *, external_id: str | None = None) -> User:
    user = User(external_id=external_id or f"user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _conversation(db, user: User, *, external_id: str | None = None) -> Conversation:
    conversation = Conversation(external_id=external_id or f"conversation-{uuid4().hex}", user_id=user.id)
    db.add(conversation)
    db.commit()
    return conversation


def _memory(
    db,
    user: User,
    text: str,
    *,
    conversation: Conversation | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    predicate: str | None = None,
    subject_type: str | None = None,
    fact_key: str | None = None,
    value: str | None = None,
    is_active: bool = True,
    importance: float = 0.5,
) -> Memory:
    memory = Memory(
        user_id=user.id,
        conversation_id=conversation.id if conversation is not None else None,
        memory_type=memory_type,
        memory_text=text,
        predicate=predicate,
        subject_type=subject_type,
        fact_key=fact_key,
        value=value,
        importance=importance,
        source_message_ids=[],
        is_active=is_active,
    )
    db.add(memory)
    db.commit()
    return memory


def test_structured_filters_are_exact_active_by_default_and_can_include_history(db, fake_embeddings) -> None:
    user = _user(db, external_id=f"user-{uuid4().hex}")
    old = _memory(
        db,
        user,
        "User lived in Hyderabad",
        predicate="location",
        subject_type="user",
        fact_key=f"user:{user.external_id}:location",
        value="Hyderabad",
        is_active=False,
    )
    current = _memory(
        db,
        user,
        "User lives in Bangalore",
        predicate="location",
        subject_type="user",
        fact_key=f"user:{user.external_id}:location",
        value="Bangalore",
    )

    current_only = search_memories(
        db,
        "location",
        user_external_id=user.external_id,
        filters=SearchFilters(fact_key=current.fact_key),
    )
    with_history = search_memories(
        db,
        "location",
        user_external_id=user.external_id,
        filters=SearchFilters(fact_key=current.fact_key, include_history=True),
    )
    by_predicate = search_memories(
        db,
        "location",
        user_external_id=user.external_id,
        filters=SearchFilters(predicate="location"),
    )
    by_subject_type = search_memories(
        db,
        "location",
        user_external_id=user.external_id,
        filters=SearchFilters(subject_type="user"),
    )
    by_memory_type = search_memories(
        db,
        "location",
        user_external_id=user.external_id,
        filters=SearchFilters(memory_type=MemoryType.SEMANTIC),
    )

    assert [hit.memory_id for hit in current_only] == [str(current.id)]
    assert {hit.memory_id for hit in with_history} == {str(old.id), str(current.id)}
    assert current_only[0].structured_rank == 1
    assert [hit.memory_id for hit in by_predicate] == [str(current.id)]
    assert [hit.memory_id for hit in by_subject_type] == [str(current.id)]
    assert [hit.memory_id for hit in by_memory_type] == [str(current.id)]
    assert all(hit.vector_rank is not None for hit in with_history)


def test_conversation_filter_is_owned_explicit_and_cross_conversation_search_is_user_scoped(db, fake_embeddings) -> None:
    user = _user(db)
    first = _conversation(db, user)
    second = _conversation(db, user)
    user_level = _memory(db, user, "User prefers PostgreSQL")
    first_memory = _memory(db, user, "Project Alpha is in conversation one", conversation=first)
    second_memory = _memory(db, user, "Project Alpha is in conversation two", conversation=second)
    other_user = _user(db)
    foreign = _conversation(db, other_user, external_id=f"foreign-{uuid4().hex}")

    unrestricted = search_memories(db, "PostgreSQL", user_external_id=user.external_id)
    narrowed = search_memories(
        db,
        "Project Alpha",
        user_external_id=user.external_id,
        filters=SearchFilters(conversation_external_id=first.external_id),
    )

    assert str(user_level.id) in {hit.memory_id for hit in unrestricted}
    assert [hit.memory_id for hit in narrowed] == [str(first_memory.id)]
    assert set(load_user_memory_index(user_external_id=user.external_id).memory_ids) == {
        str(user_level.id),
        str(first_memory.id),
        str(second_memory.id),
    }
    with pytest.raises(InvalidFilterScopeError, match="does not belong"):
        search_memories(
            db,
            "Project Alpha",
            user_external_id=user.external_id,
            filters=SearchFilters(conversation_external_id=foreign.external_id),
        )


def test_postgresql_lexical_search_handles_technical_terms_identifiers_and_user_isolation(db, fake_embeddings) -> None:
    user = _user(db)
    second_user = _user(db)
    preferred = _memory(db, user, "Project Alpha uses PostgreSQL for KAN-4.", importance=0.9)
    _memory(db, user, "Project Alpha draft notes.", importance=0.1)
    foreign = _memory(db, second_user, "Project Alpha uses PostgreSQL for KAN-4.")

    alpha_hits = search_memories(db, "Project Alpha", user_external_id=user.external_id)
    ranked_hits = search_memories(db, "Project Alpha PostgreSQL", user_external_id=user.external_id)
    identifier_hits = search_memories(db, "KAN-4", user_external_id=user.external_id)
    foreign_hits = search_memories(db, "Project Alpha", user_external_id=second_user.external_id)

    assert str(preferred.id) in {hit.memory_id for hit in alpha_hits}
    assert ranked_hits[0].memory_id == str(preferred.id)
    assert identifier_hits[0].memory_id == str(preferred.id)
    assert all(hit.memory_id != str(foreign.id) for hit in alpha_hits)
    assert [hit.memory_id for hit in foreign_hits] == [str(foreign.id)]


def test_sync_persists_normalized_embeddings_per_user_and_skips_noop_reembedding(db, fake_embeddings) -> None:
    first_user = _user(db)
    second_user = _user(db)
    memory = _memory(db, first_user, "User prefers PostgreSQL")
    _memory(db, second_user, "User prefers MySQL")

    first_path = sync_user_memory_index(db, user_external_id=first_user.external_id)
    second_path = sync_user_memory_index(db, user_external_id=second_user.external_id)
    persisted = db.get(Memory, memory.id)
    loaded = load_user_memory_index(user_external_id=first_user.external_id)
    expected_paths = user_index_paths(first_user.external_id)
    calls_after_first_sync = len(fake_embeddings.calls)
    sync_user_memory_index(db, user_external_id=first_user.external_id)

    assert persisted is not None
    assert persisted.embedding_model == "fake-embedding-model"
    assert persisted.embedding == pytest.approx([1.0, 0.0])
    assert first_path == expected_paths[0]
    assert first_path is not None and first_path.is_file()
    assert expected_paths[1].is_file()
    assert second_path is not None and second_path != first_path
    assert first_user.external_id not in first_path.name
    assert loaded.index.ntotal == 1
    assert loaded.memory_ids == [str(memory.id)]
    assert len(fake_embeddings.calls) == calls_after_first_sync


def test_vector_semantic_search_is_exact_cosine_and_filters_historical_candidates(db, fake_embeddings) -> None:
    user = _user(db)
    historical = _memory(db, user, "User prefers PostgreSQL", is_active=False)
    current = _memory(db, user, "User prefers MySQL", is_active=True)

    hits = search_memories(db, "Which database does the user like?", user_external_id=user.external_id)

    assert [hit.memory_id for hit in hits] == [str(current.id)]
    assert all(hit.memory_id != str(historical.id) for hit in hits)
    assert hits[0].vector_rank == 1
    assert [call for call in fake_embeddings.calls if call[1] == ["Which database does the user like?"]] == [
        ("fake-embedding-model", ["Which database does the user like?"])
    ]


def test_model_change_reembeds_and_rebuilds_the_user_index(db, fake_embeddings) -> None:
    user = _user(db)
    memory = _memory(db, user, "User prefers PostgreSQL")
    sync_user_memory_index(db, user_external_id=user.external_id)

    configure(embedding_model="replacement-fake-model")
    rebuilt = rebuild_user_memory_index(db, user_external_id=user.external_id)
    loaded = load_user_memory_index(user_external_id=user.external_id)

    assert rebuilt is not None
    assert db.get(Memory, memory.id).embedding_model == "replacement-fake-model"
    assert loaded.embedding_model == "replacement-fake-model"
    assert fake_embeddings.calls[-1][0] == "replacement-fake-model"


def test_search_recovers_missing_corrupt_and_stale_user_indexes(db, fake_embeddings) -> None:
    user = _user(db)
    first = _memory(db, user, "User prefers PostgreSQL")
    sync_user_memory_index(db, user_external_id=user.external_id)
    index_path, metadata_path = user_index_paths(user.external_id)

    index_path.unlink()
    fake_embeddings.calls.clear()
    missing_hits = search_memories(db, "Which database does the user prefer?", user_external_id=user.external_id)

    assert [hit.memory_id for hit in missing_hits] == [str(first.id)]
    assert index_path.is_file() and metadata_path.is_file()
    assert fake_embeddings.calls == [("fake-embedding-model", ["Which database does the user prefer?"])]

    index_path.write_bytes(b"not a FAISS index")
    fake_embeddings.calls.clear()
    corrupt_hits = search_memories(db, "Which database does the user prefer?", user_external_id=user.external_id)

    assert [hit.memory_id for hit in corrupt_hits] == [str(first.id)]
    assert fake_embeddings.calls == [("fake-embedding-model", ["Which database does the user prefer?"])]

    second = _memory(db, user, "The user also maintains Project Alpha")
    fake_embeddings.calls.clear()
    stale_hits = search_memories(db, "Which database does the user prefer?", user_external_id=user.external_id)

    assert {hit.memory_id for hit in stale_hits} == {str(first.id), str(second.id)}
    assert any(call[1] == [second.memory_text] for call in fake_embeddings.calls)
    assert load_user_memory_index(user_external_id=user.external_id).memory_ids == [
        str(first.id),
        str(second.id),
    ]


def test_vector_post_filter_oversamples_active_candidates_and_honors_structure(db, fake_embeddings) -> None:
    user = _user(db)
    for number in range(4):
        _memory(
            db,
            user,
            f"User previously preferred PostgreSQL version {number}",
            predicate="database",
            is_active=False,
        )
    current = _memory(
        db,
        user,
        "User currently prefers PostgreSQL",
        predicate="database",
        is_active=True,
    )
    _memory(
        db,
        user,
        "User currently prefers MySQL",
        predicate="language",
        is_active=True,
    )

    hits = search_memories(
        db,
        "Which database does the user prefer?",
        user_external_id=user.external_id,
        limit=1,
        filters=SearchFilters(predicate="database"),
    )

    assert [hit.memory_id for hit in hits] == [str(current.id)]
    assert hits[0].vector_rank == 1
    assert hits[0].structured_rank == 1


def test_dimension_mismatch_and_embedding_provider_failure_are_explicit(db, fake_embeddings, monkeypatch) -> None:
    user = _user(db)
    _memory(db, user, "User prefers PostgreSQL")
    sync_user_memory_index(db, user_external_id=user.external_id)

    class DimensionMismatchClient:
        embeddings = None

        def __init__(self) -> None:
            self.embeddings = self

        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])])

    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: DimensionMismatchClient())
    with pytest.raises(IndexDimensionMismatchError, match="dimension"):
        search_memories(db, "Which database does the user prefer?", user_external_id=user.external_id)

    class FailingEmbeddingClient:
        embeddings = None

        def __init__(self) -> None:
            self.embeddings = self

        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: FailingEmbeddingClient())
    with pytest.raises(EmbeddingError, match="could not generate"):
        search_memories(db, "Which database does the user prefer?", user_external_id=user.external_id)


def test_search_rejects_empty_query_limit_missing_user_and_cross_user_leakage(db, fake_embeddings) -> None:
    user = _user(db)
    other_user = _user(db)
    _memory(db, user, "User prefers PostgreSQL")

    with pytest.raises(InvalidSearchError, match="query"):
        search_memories(db, "  ", user_external_id=user.external_id)
    with pytest.raises(InvalidSearchError, match="limit"):
        search_memories(db, "PostgreSQL", user_external_id=user.external_id, limit=101)
    with pytest.raises(UserNotFoundError, match="No user exists"):
        search_memories(db, "PostgreSQL", user_external_id="missing-user")

    assert search_memories(db, "PostgreSQL", user_external_id=other_user.external_id) == []


def test_database_exposes_embedding_columns_and_functional_fts_index() -> None:
    with SessionLocal() as session:
        inspector = inspect(session.bind)
        columns = {column["name"] for column in inspector.get_columns("memories")}
        indexes = {index["name"] for index in inspector.get_indexes("memories")}

    assert {"embedding", "embedding_model"} <= columns
    assert "ix_memories_memory_text_fts" in indexes
