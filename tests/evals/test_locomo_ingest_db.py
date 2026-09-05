"""DB integration test for LoCoMo ingestion's dia_id -> Message UUID mapping.

Like the rest of the project's database integration tests, this uses
TEST_DATABASE_URL and is skipped without it; it never falls back to
DATABASE_URL. Every LLM and embedding call is mocked so it never makes a
live provider request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from src.config import configure, reset_config
from src.database import Message, SessionLocal, create_tables, reset_engine
from evals.locomo.dataset import load_locomo_dataset
from evals.locomo.ingest import ingest_sample

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_sample.json"
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo ingestion integration test never uses DATABASE_URL.",
    ),
]


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        # No candidate memories: this test verifies provenance mapping and
        # ingestion plumbing, not extraction quality.
        content = json.dumps({"memories": []})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLLMClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)])


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL, llm_model="test-model", embedding_model="fake-model")
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
def fake_providers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeCompletions:
    completions = FakeCompletions()
    monkeypatch.setattr("src.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr("src.retrieval.message_vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path, embedding_model="fake-model")
    return completions


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def test_ingest_sample_maps_every_dia_id_to_a_persisted_message_in_order(db, fake_providers) -> None:
    sample = load_locomo_dataset(FIXTURE_PATH)[0]
    user_id = _unique("locomo-test")

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")

    all_dia_ids = [turn.dia_id for turn in sample.turns]
    assert set(outcome.dia_id_to_message_id) == set(all_dia_ids)
    assert outcome.sessions_ingested == 2
    assert outcome.messages_persisted == len(all_dia_ids)

    for dia_id, message_id in outcome.dia_id_to_message_id.items():
        message = db.get(Message, message_id)
        assert message is not None
        turn = next(t for t in sample.turns if t.dia_id == dia_id)
        assert turn.speaker in message.content
        assert turn.text in message.content

    # The reverse map is a true inverse: no two dia_ids collapse to one message.
    assert len(outcome.message_id_to_dia_id) == len(outcome.dia_id_to_message_id)


def test_ingest_sample_preserves_speaker_roles_across_sessions(db, fake_providers) -> None:
    sample = load_locomo_dataset(FIXTURE_PATH)[0]
    user_id = _unique("locomo-test")

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")

    for turn in sample.turns:
        message_id = outcome.dia_id_to_message_id[turn.dia_id]
        message = db.get(Message, message_id)
        expected_role = sample.role_for_speaker(turn.speaker)
        assert message.role.value == expected_role


def test_ingest_sample_prefixes_only_the_first_turn_of_each_session_with_its_timestamp(db, fake_providers) -> None:
    sample = load_locomo_dataset(FIXTURE_PATH)[0]
    user_id = _unique("locomo-test")

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")

    first_of_session_1 = db.get(Message, outcome.dia_id_to_message_id["D1:1"])
    second_of_session_1 = db.get(Message, outcome.dia_id_to_message_id["D1:2"])
    first_of_session_2 = db.get(Message, outcome.dia_id_to_message_id["D2:1"])

    assert first_of_session_1.content.startswith("[Session date:")
    assert not second_of_session_1.content.startswith("[Session date:")
    assert first_of_session_2.content.startswith("[Session date:")
