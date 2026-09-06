"""DB integration tests for LoCoMo ingestion's checkpoint/resume safety.

Mirrors tests/evals/test_locomo_ingest_db.py: TEST_DATABASE_URL-gated, never
falls back to DATABASE_URL, and mocks every LLM/embedding call. These cover
evals/locomo/ingest.py's resume_from_session / on_session_complete contract:
skipping already-completed sessions, recovering their dia_id mapping from
already-persisted rows, and refusing to guess when persisted state doesn't
exactly match what a checkpoint claims is complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from meminfra.config import configure, reset_config
from meminfra.database import Message, SessionLocal, create_tables, reset_engine
from evals.locomo.dataset import load_locomo_dataset
from evals.locomo.ingest import LocomoResumeAmbiguousError, ingest_sample
from evals.locomo.schemas import LocomoSample

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_sample.json"
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo resume integration tests never use DATABASE_URL.",
    ),
]


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
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
    monkeypatch.setattr("meminfra.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("meminfra.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr("meminfra.retrieval.message_vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path, embedding_model="fake-model")
    return completions


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _sample() -> LocomoSample:
    return load_locomo_dataset(FIXTURE_PATH)[0]


def _only_session(sample: LocomoSample, session_number: int) -> LocomoSample:
    return LocomoSample(
        sample_id=sample.sample_id,
        speaker_a=sample.speaker_a,
        speaker_b=sample.speaker_b,
        turns=[turn for turn in sample.turns if turn.session_number == session_number],
        qa=[],
    )


def test_resume_skips_completed_session_and_recovers_its_mapping(db, fake_providers) -> None:
    sample = _sample()
    user_id = _unique("locomo-resume")
    session_1_turns = [turn for turn in sample.turns if turn.session_number == 1]
    session_2_turns = [turn for turn in sample.turns if turn.session_number == 2]

    # Phase 1: only session 1 ever completes (models a process that later dies).
    first_outcome = ingest_sample(
        db, _only_session(sample, 1), user_id=user_id, conversation_id="conversation"
    )
    assert first_outcome.sessions_ingested == 1
    session_1_message_ids = dict(first_outcome.dia_id_to_message_id)

    # Phase 2: restart with the full sample, resuming from the checkpointed session 1.
    completed_sessions: list[int] = []
    resumed_outcome = ingest_sample(
        db,
        sample,
        user_id=user_id,
        conversation_id="conversation",
        resume_from_session=1,
        on_session_complete=lambda session_number, result: completed_sessions.append(session_number),
    )

    assert completed_sessions == [2]
    assert resumed_outcome.sessions_ingested == 1
    # Session 1's dia_ids resolve to the exact same Message rows as before -- not re-persisted.
    for dia_id, message_id in session_1_message_ids.items():
        assert resumed_outcome.dia_id_to_message_id[dia_id] == message_id
    assert set(resumed_outcome.dia_id_to_message_id) == {turn.dia_id for turn in sample.turns}
    assert resumed_outcome.messages_persisted == len(session_1_turns) + len(session_2_turns)

    for turn in session_2_turns:
        message = db.get(Message, resumed_outcome.dia_id_to_message_id[turn.dia_id])
        assert turn.text in message.content


def test_resume_never_calls_add_for_the_recovered_session(db, fake_providers) -> None:
    sample = _sample()
    user_id = _unique("locomo-resume")
    ingest_sample(db, _only_session(sample, 1), user_id=user_id, conversation_id="conversation")
    calls_before_resume = len(fake_providers.calls)

    ingest_sample(db, sample, user_id=user_id, conversation_id="conversation", resume_from_session=1)

    # Only session 2's interactions triggered new extraction calls.
    session_2_turn_count = len([turn for turn in sample.turns if turn.session_number == 2])
    assert len(fake_providers.calls) - calls_before_resume <= session_2_turn_count


def test_resume_refuses_when_persisted_state_exceeds_the_checkpoint(db, fake_providers) -> None:
    sample = _sample()
    user_id = _unique("locomo-resume")
    # Both sessions actually completed, but the checkpoint being resumed from
    # only claims session 1 is done -- an ambiguous, unexpected mismatch.
    ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")

    with pytest.raises(LocomoResumeAmbiguousError, match="expected exactly"):
        ingest_sample(db, sample, user_id=user_id, conversation_id="conversation", resume_from_session=1)


def test_resume_refuses_when_no_conversation_exists_yet(db, fake_providers) -> None:
    sample = _sample()
    user_id = _unique("locomo-resume-never-started")

    with pytest.raises(LocomoResumeAmbiguousError, match="no persisted conversation exists"):
        ingest_sample(db, sample, user_id=user_id, conversation_id="conversation", resume_from_session=1)


def test_on_session_complete_is_not_called_for_a_skipped_session(db, fake_providers) -> None:
    sample = _sample()
    user_id = _unique("locomo-resume")
    ingest_sample(db, _only_session(sample, 1), user_id=user_id, conversation_id="conversation")

    seen: list[int] = []
    ingest_sample(
        db,
        sample,
        user_id=user_id,
        conversation_id="conversation",
        resume_from_session=1,
        on_session_complete=lambda session_number, result: seen.append(session_number),
    )

    assert 1 not in seen
    assert seen == [2]
