"""One integration test for the eval runner's plumbing against a real test database.

This uses TEST_DATABASE_URL like the rest of the project's database integration
tests, but monkeypatches the LLM and embedding clients so it never makes a live
provider request. It exercises the actual ingestion -> extraction -> write ->
search -> metrics path in evals/runner.py, not extraction quality.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import configure, reset_config
from src.database import SessionLocal, create_tables, reset_engine
from evals.runner import run_case
from evals.schemas import EvalCase, EvalMessage, ExpectedMemory


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; eval runner integration test never uses DATABASE_URL.",
    ),
]


class FakeCompletions:
    """Always propose one fixed PostgreSQL-preference memory, mirroring test_extractor.py."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = json.dumps(
            {
                "memories": [
                    {
                        "memory_type": "semantic",
                        "memory_text": "User prefers PostgreSQL",
                        "subject_type": "user",
                        "subject_name": None,
                        "predicate": "database_preference",
                        "value": "PostgreSQL",
                        "confidence": 0.9,
                        "importance": 0.7,
                    }
                ]
            }
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLLMClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    """Deterministic vectors, mirroring tests/retrieval/test_search.py."""

    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=self._vector_for(text)) for index, text in enumerate(input)]
        )

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        normalized = text.casefold()
        if "postgresql" in normalized or "database" in normalized:
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(
        database_url=TEST_DATABASE_URL,
        llm_model="test-model",
        embedding_model="fake-embedding-model",
    )
    create_tables()
    yield
    reset_engine()
    reset_config()


@pytest.fixture
def fake_providers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeCompletions:
    completions = FakeCompletions()
    monkeypatch.setattr("src.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path)
    return completions


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
        session.rollback()


def test_run_case_ingests_extracts_writes_and_scores_retrieval(db, fake_providers: FakeCompletions) -> None:
    case = EvalCase(
        id="runner-integration-001",
        category="single_fact",
        messages=[EvalMessage(role="user", content="I prefer PostgreSQL over MongoDB.")],
        query="Which database does the user prefer?",
        expected_memories=[ExpectedMemory(required_terms=["postgresql"])],
    )

    result = run_case(db, case, top_k=5)

    assert fake_providers.calls, "extract_memories should have called the LLM client"
    assert len(result.retrieved) >= 1
    assert "postgresql" in result.retrieved[0].memory_text.casefold()
    assert result.metrics.hit_at_1 == 1
    assert result.metrics.rank == 1
    assert result.isolation_failures == 0
    assert result.superseded_returned == 0


def test_run_case_isolation_case_never_returns_the_other_users_memory(db, fake_providers: FakeCompletions) -> None:
    case = EvalCase(
        id="runner-integration-isolation-001",
        category="user_isolation",
        messages=[EvalMessage(role="user", content="I prefer PostgreSQL over MongoDB.")],
        isolation_messages=[EvalMessage(role="user", content="I prefer PostgreSQL over MongoDB too.")],
        query="Which database does the user prefer?",
        expected_memories=[ExpectedMemory(required_terms=["postgresql"])],
    )

    result = run_case(db, case, top_k=5)

    assert result.isolation_failures == 0
