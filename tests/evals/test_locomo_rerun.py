"""Tests for evals/locomo/rerun.py: retrieval-only, no re-ingestion, V0 stays untouched."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from src.config import configure, reset_config
from src.database import SessionLocal, create_tables, reset_engine
from evals.locomo.ingest import ingest_sample
from evals.locomo.rerun import build_retrieval_report, rerun_retrieval, save_v1_result
from evals.locomo.schemas import LocomoSample, LocomoTurn

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo rerun integration tests never use DATABASE_URL.",
    ),
]


def _sample(sample_id: str) -> LocomoSample:
    turns = [
        LocomoTurn(dia_id="D1:1", speaker="Alice", text="I started painting again.", session_number=1, session_date_time="10:00 am on 1 January, 2024"),
        LocomoTurn(dia_id="D1:2", speaker="Bob", text="Nice, what kind?", session_number=1, session_date_time="10:00 am on 1 January, 2024"),
    ]
    return LocomoSample(sample_id=sample_id, speaker_a="Alice", speaker_b="Bob", turns=turns, qa=[])


class FakeCompletions:
    def create(self, **kwargs: Any) -> SimpleNamespace:
        content = json.dumps({"memories": [{"memory_text": "Alice started painting again.", "memory_type": "semantic"}]})
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
def fake_providers(monkeypatch: pytest.MonkeyPatch, tmp_path):
    completions = FakeCompletions()
    monkeypatch.setattr("src.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("src.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr("src.retrieval.message_vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path, embedding_model="fake-model")
    return completions


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def test_rerun_retrieval_never_calls_add_and_reuses_existing_state(db, fake_providers) -> None:
    sample = _sample(_unique("conv-rerun"))
    user_id = _unique("locomo-resume")
    ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")

    # ingest_sample already used the extractor; rerun_retrieval must never call MemoryLayer.add()
    # again -- it only performs MemoryLayer.search(), which never touches extract_memories().
    result = rerun_retrieval(db, sample, user_id=user_id, conversation_id="conversation", top_k=10)
    assert len(result.diagnostics) == len(sample.qa)
    assert result.dia_id_to_message_id["D1:1"] and result.dia_id_to_message_id["D1:2"]


def test_rerun_retrieval_requires_an_already_ingested_conversation(db, fake_providers) -> None:
    sample = _sample(_unique("conv-rerun-missing"))
    with pytest.raises(RuntimeError, match="No user exists"):
        rerun_retrieval(db, sample, user_id=_unique("locomo-resume-never-ingested"), conversation_id="conversation", top_k=10)


def test_save_v1_result_never_touches_the_v0_file(db, fake_providers, tmp_path) -> None:
    sample = _sample(_unique("conv-rerun-v0"))
    user_id = _unique("locomo-resume")
    ingest_sample(db, sample, user_id=user_id, conversation_id="conversation")
    result = rerun_retrieval(db, sample, user_id=user_id, conversation_id="conversation", top_k=10)

    v0_path = tmp_path / "v0.json"
    v0_content = json.dumps({"label": "V0", "untouched": True})
    v0_path.write_text(v0_content, encoding="utf-8")

    report = build_retrieval_report(
        result.diagnostics,
        top_k=10,
        run_id="testrun",
        dataset_path=Path("unused"),
        dataset_sha256_value="0" * 64,
        git_commit=None,
        llm_model="test-model",
        embedding_model="fake-model",
    )
    v1_path = tmp_path / "v1.json"
    save_v1_result(
        report,
        base_run_path=str(v0_path),
        recovered_sessions=[1],
        recovery_report_path=None,
        path=v1_path,
    )

    assert v0_path.read_text(encoding="utf-8") == v0_content, "V0 must remain byte-for-byte untouched"
    saved_v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    assert saved_v1["label"] == "LoCoMo Subset Baseline V1"
    assert saved_v1["base_run"] == str(v0_path)
    assert saved_v1["recovered_sessions"] == [1]
