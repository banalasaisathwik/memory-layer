"""DB integration tests for evals/locomo/recover.py.

Mirrors the conventions in tests/evals/test_locomo_ingest_db.py and
test_locomo_ingest_resume.py: TEST_DATABASE_URL-gated, never falls back to
DATABASE_URL, and mocks every LLM/embedding call so no test ever reaches a
live provider.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx2
import openai
import pytest
from sqlalchemy import select

from meminfra.config import configure, reset_config
from meminfra.database import Memory, Message, SessionLocal, create_tables, reset_engine
from evals.locomo.ingest import ingest_sample
from evals.locomo.recover import (
    _classify_extraction_error,
    plan_session_recovery,
    recover_sessions,
)
from evals.locomo.schemas import LocomoSample, LocomoTurn
from meminfra.memory import ExtractionError

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo recovery integration tests never use DATABASE_URL.",
    ),
]


def _synthetic_sample(sample_id: str, *, interaction_count: int = 6) -> LocomoSample:
    """One session with `interaction_count` user/assistant interactions."""

    turns: list[LocomoTurn] = []
    dia = 1
    for i in range(interaction_count):
        turns.append(
            LocomoTurn(
                dia_id=f"D1:{dia}",
                speaker="Alice",
                text=f"user turn {i}",
                session_number=1,
                session_date_time="10:00 am on 1 January, 2024",
            )
        )
        dia += 1
        turns.append(
            LocomoTurn(
                dia_id=f"D1:{dia}",
                speaker="Bob",
                text=f"assistant turn {i}",
                session_number=1,
                session_date_time="10:00 am on 1 January, 2024",
            )
        )
        dia += 1
    return LocomoSample(sample_id=sample_id, speaker_a="Alice", speaker_b="Bob", turns=turns, qa=[])


class ScriptedCompletions:
    """Returns one scripted response per call, by call index; extras repeat the last."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        index = len(self.calls)
        self.calls.append(kwargs)
        response = self.responses[index] if index < len(self.responses) else self.responses[-1]
        if isinstance(response, Exception):
            raise response
        content = json.dumps(response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLLMClient:
    def __init__(self, completions: ScriptedCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)])


def _memory_response(text: str) -> dict:
    return {"memories": [{"memory_text": text, "memory_type": "semantic"}]}


_EMPTY_RESPONSE = {"memories": []}


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


def _patch_providers(monkeypatch: pytest.MonkeyPatch, completions: ScriptedCompletions, *, tmp_path) -> None:
    monkeypatch.setattr("meminfra.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("meminfra.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr("meminfra.retrieval.message_vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path, embedding_model="fake-model")


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _ingest_with_failure_after(db, sample, *, user_id, monkeypatch, tmp_path, succeed_count: int):
    """Simulate MemoryLayer.add() persisting the session then failing partway through extraction."""

    responses: list[Any] = [_memory_response(f"fact {i}") for i in range(succeed_count)]
    responses.append(ExtractionError("LLM extraction request failed."))
    completions = ScriptedCompletions(responses)
    _patch_providers(monkeypatch, completions, tmp_path=tmp_path)

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")
    assert outcome.sessions_failed == 1
    return outcome, completions


def test_boundary_detection_and_no_replay_of_completed_interactions(db, monkeypatch, tmp_path) -> None:
    """B: chronological interaction reconstruction. C: no duplicate replay of completed work."""

    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=6)
    user_id = _unique("locomo-resume")

    # Interactions 0, 1, 2 succeed (each writes one memory); interaction 3 fails.
    _ingest_with_failure_after(db, sample, user_id=user_id, monkeypatch=monkeypatch, tmp_path=tmp_path, succeed_count=3)

    plan = plan_session_recovery(db, sample, 1, user_id=user_id, conversation_id="conversation")
    assert len(plan.interactions) == 6
    assert [interaction.covered for interaction in plan.interactions] == [True, True, True, False, False, False]
    assert plan.boundary_index == 3
    assert [interaction.index for interaction in plan.recovery_candidates] == [3, 4, 5]

    # Recovery must only ever call extract_memories() for the 3 uncompleted interactions.
    recovery_completions = ScriptedCompletions([_memory_response("recovered fact") for _ in range(3)])
    _patch_providers(monkeypatch, recovery_completions, tmp_path=tmp_path)

    report = recover_sessions(
        db, sample, [1], user_id=user_id, conversation_id="conversation", dry_run=False
    )
    assert len(recovery_completions.calls) == 3
    outcome = report.sessions[0]
    assert [result.interaction_index for result in outcome.results] == [3, 4, 5]
    assert all(result.status == "recovered" for result in outcome.results)


def test_recovery_uses_original_message_ids_and_no_new_messages(db, monkeypatch, tmp_path) -> None:
    """A: no new Message rows. D: original message IDs preserved as provenance."""

    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=4)
    user_id = _unique("locomo-resume")

    _ingest_with_failure_after(db, sample, user_id=user_id, monkeypatch=monkeypatch, tmp_path=tmp_path, succeed_count=2)

    all_messages_before = list(db.scalars(select(Message)))
    ids_before = {str(message.id) for message in all_messages_before}

    plan = plan_session_recovery(db, sample, 1, user_id=user_id, conversation_id="conversation")
    expected_ids = {message_id for interaction in plan.recovery_candidates for message_id in interaction.message_ids}

    recovery_completions = ScriptedCompletions([_memory_response("recovered fact") for _ in range(2)])
    _patch_providers(monkeypatch, recovery_completions, tmp_path=tmp_path)
    report = recover_sessions(db, sample, [1], user_id=user_id, conversation_id="conversation", dry_run=False)

    all_messages_after = list(db.scalars(select(Message)))
    ids_after = {str(message.id) for message in all_messages_after}
    assert ids_after == ids_before, "recovery must never insert a new Message row"

    written_memory_ids = [
        write_result.memory_id
        for outcome in report.sessions
        for result in outcome.results
        for write_result in result.write_results
    ]
    assert written_memory_ids, "expected at least one recovered memory"
    for memory_id in written_memory_ids:
        memory = db.get(Memory, memory_id)
        assert set(memory.source_message_ids).issubset(expected_ids)
        assert set(memory.source_message_ids).issubset(ids_before)


def test_provider_schema_failure_is_recorded_without_writing_a_fake_memory(db, monkeypatch, tmp_path) -> None:
    """E: a schema-invalid provider response is recorded and writes nothing."""

    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=2)
    user_id = _unique("locomo-resume")

    _ingest_with_failure_after(db, sample, user_id=user_id, monkeypatch=monkeypatch, tmp_path=tmp_path, succeed_count=0)

    # A boolean where memory_text (a string) is required -- the real observed failure mode.
    invalid_response = {"memories": [{"memory_text": True}]}
    recovery_completions = ScriptedCompletions([invalid_response, invalid_response])
    _patch_providers(monkeypatch, recovery_completions, tmp_path=tmp_path)

    memories_before = list(db.scalars(select(Memory)))

    report = recover_sessions(db, sample, [1], user_id=user_id, conversation_id="conversation", dry_run=False)

    memories_after = list(db.scalars(select(Memory)))
    assert len(memories_after) == len(memories_before), "a schema-invalid response must never write a memory"

    outcome = report.sessions[0]
    assert all(result.status == "failed" for result in outcome.results)
    assert all(result.error is not None and result.error.error_category == "schema_validation" for result in outcome.results)


def test_rate_limit_error_is_classified_as_rate_limited() -> None:
    """F: a mocked 429 is reported as rate-limited, not guessed at."""

    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx2.Response(429, request=request)
    cause = openai.RateLimitError("rate limited", response=response, body=None)
    error = ExtractionError("LLM extraction request failed.")
    error.__cause__ = cause

    assert _classify_extraction_error(error) == "rate_limit"


def test_timeout_error_is_classified_as_timeout_not_connection() -> None:
    """APITimeoutError subclasses APIConnectionError; timeout must win the classification."""

    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    cause = openai.APITimeoutError(request=request)
    error = ExtractionError("LLM extraction request failed.")
    error.__cause__ = cause

    assert _classify_extraction_error(error) == "timeout"


def test_connection_error_is_classified_as_connection() -> None:
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    cause = openai.APIConnectionError(request=request)
    error = ExtractionError("LLM extraction request failed.")
    error.__cause__ = cause

    assert _classify_extraction_error(error) == "connection"


def test_invalid_json_error_is_classified_as_invalid_json(db, monkeypatch, tmp_path) -> None:
    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=1)
    user_id = _unique("locomo-resume")
    _ingest_with_failure_after(db, sample, user_id=user_id, monkeypatch=monkeypatch, tmp_path=tmp_path, succeed_count=0)

    class BadJsonCompletions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])

    completions = BadJsonCompletions()
    _patch_providers(monkeypatch, completions, tmp_path=tmp_path)  # type: ignore[arg-type]

    report = recover_sessions(db, sample, [1], user_id=user_id, conversation_id="conversation", dry_run=False)
    outcome = report.sessions[0]
    assert len(outcome.results) == 1
    assert outcome.results[0].status == "failed"
    assert outcome.results[0].error is not None
    assert outcome.results[0].error.error_category == "invalid_json"


def test_no_recovery_needed_when_a_later_interaction_is_covered(db, monkeypatch, tmp_path) -> None:
    """A session with no gap at all (last interaction covered) reports zero recovery candidates."""

    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=2)
    user_id = _unique("locomo-resume")
    completions = ScriptedCompletions([_EMPTY_RESPONSE, _memory_response("last interaction fact")])
    _patch_providers(monkeypatch, completions, tmp_path=tmp_path)

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")
    assert outcome.sessions_ingested == 1

    plan = plan_session_recovery(db, sample, 1, user_id=user_id, conversation_id="conversation")
    assert plan.boundary_index == len(plan.interactions)
    assert plan.recovery_candidates == []


def test_boundary_is_conservative_when_a_session_legitimately_wrote_nothing(db, monkeypatch, tmp_path) -> None:
    """Known limitation: a fully-successful, zero-output session cannot be told apart from a
    session that never ran at all, since both leave zero covering Memory rows. Recovery
    conservatively treats every interaction as a candidate rather than silently skipping a
    session that might genuinely have failed on interaction 0 -- see recover.py's boundary
    detection docstring. This is intentional, documented behavior, not a bug: a
    write_memories() NOOP for a duplicate is harmless, so over-recovering here costs at most
    a few redundant provider calls, never a duplicate memory.
    """

    sample = _synthetic_sample(_unique("conv-recover"), interaction_count=2)
    user_id = _unique("locomo-resume")
    completions = ScriptedCompletions([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    _patch_providers(monkeypatch, completions, tmp_path=tmp_path)

    outcome = ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")
    assert outcome.sessions_ingested == 1

    plan = plan_session_recovery(db, sample, 1, user_id=user_id, conversation_id="conversation")
    assert plan.boundary_index == 0
    assert len(plan.recovery_candidates) == len(plan.interactions)
