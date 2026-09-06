"""DB integration tests for the LoCoMo runner's --resume checkpoint wiring.

Mirrors tests/evals/test_locomo_runner.py's fixtures and fakes. These cover
run_benchmark(resume=True): a deterministic per-sample user_id, a checkpoint
written after each completed session, incompatible-checkpoint refusal, and
an idempotent second call that re-ingests nothing once every session is
already checkpointed.
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
from meminfra.database import create_tables, reset_engine

from evals.locomo.checkpoint import load_checkpoint, new_checkpoint, save_checkpoint
from evals.locomo.runner import LocomoEnvironmentError, run_benchmark
from evals.locomo.schemas import LocomoQA, LocomoSample, LocomoTurn

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
FIXTURE_DATASET_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_sample.json"
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo runner resume integration test never uses DATABASE_URL.",
    ),
]


class FakeExtractionCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = json.dumps({"memories": []})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLLMClient:
    def __init__(self, completions: Any) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)])


def _two_session_sample(sample_id: str) -> LocomoSample:
    turns = [
        LocomoTurn(
            dia_id="D1:1", speaker="Alice", text="I adopted a cat named Whiskers.",
            session_number=1, session_date_time="9:00 am on 1 January, 2024",
        ),
        LocomoTurn(
            dia_id="D1:2", speaker="Bob", text="That's great!",
            session_number=1, session_date_time="9:00 am on 1 January, 2024",
        ),
        LocomoTurn(
            dia_id="D2:1", speaker="Alice", text="Whiskers learned a new trick.",
            session_number=2, session_date_time="9:00 am on 2 January, 2024",
        ),
        LocomoTurn(
            dia_id="D2:2", speaker="Bob", text="Nice!",
            session_number=2, session_date_time="9:00 am on 2 January, 2024",
        ),
    ]
    qa = [LocomoQA(question="What pet did Alice adopt?", answer="a cat", category_id=4, evidence=["D1:1"])]
    return LocomoSample(sample_id=sample_id, speaker_a="Alice", speaker_b="Bob", turns=turns, qa=qa)


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
def fake_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeExtractionCompletions:
    completions = FakeExtractionCompletions()
    monkeypatch.setattr("meminfra.memory.extractor.get_llm_client", lambda: FakeLLMClient(completions))
    monkeypatch.setattr("meminfra.retrieval.vector.get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr("meminfra.retrieval.message_vector.get_embedding_client", lambda: FakeEmbeddingClient())
    configure(faiss_index_dir=tmp_path, embedding_model="fake-model")
    return completions


def test_resume_writes_a_checkpoint_after_each_completed_session(fake_extraction, tmp_path) -> None:
    sample = _two_session_sample(f"resume-conv-{uuid4().hex}")
    checkpoint_dir = tmp_path / "checkpoints"

    run_benchmark(
        [sample],
        top_k=5,
        mode="retrieval",
        max_questions=None,
        run_id=uuid4().hex[:8],
        resume=True,
        checkpoint_dir=checkpoint_dir,
        dataset_path=FIXTURE_DATASET_PATH,
        show_progress=False,
    )

    checkpoint = load_checkpoint(sample.sample_id, checkpoint_dir=checkpoint_dir)
    assert checkpoint is not None
    assert checkpoint.last_completed_session == 2
    assert checkpoint.user_id == f"locomo-resume-{sample.sample_id}"


def test_resume_second_call_reingests_nothing_once_fully_checkpointed(fake_extraction, tmp_path) -> None:
    sample = _two_session_sample(f"resume-conv-{uuid4().hex}")
    checkpoint_dir = tmp_path / "checkpoints"

    run_benchmark(
        [sample], top_k=5, mode="retrieval", max_questions=None, run_id=uuid4().hex[:8],
        resume=True, checkpoint_dir=checkpoint_dir, dataset_path=FIXTURE_DATASET_PATH, show_progress=False,
    )
    calls_after_first_run = len(fake_extraction.calls)

    diagnostics = run_benchmark(
        [sample], top_k=5, mode="retrieval", max_questions=None, run_id=uuid4().hex[:8],
        resume=True, checkpoint_dir=checkpoint_dir, dataset_path=FIXTURE_DATASET_PATH, show_progress=False,
    )

    assert len(fake_extraction.calls) == calls_after_first_run
    assert len(diagnostics) == 1
    assert diagnostics[0].retrieval_metrics is not None


def test_resume_refuses_an_incompatible_checkpoint(fake_extraction, tmp_path) -> None:
    sample = _two_session_sample(f"resume-conv-{uuid4().hex}")
    checkpoint_dir = tmp_path / "checkpoints"
    user_id = f"locomo-resume-{sample.sample_id}"
    save_checkpoint(
        new_checkpoint(
            sample_id=sample.sample_id,
            dataset_sha256="not-the-real-dataset-hash",
            llm_model="test-model",
            embedding_model="fake-model",
            user_id=user_id,
            conversation_id="conversation",
            last_completed_session=1,
            git_commit=None,
        ),
        checkpoint_dir=checkpoint_dir,
    )

    with pytest.raises(LocomoEnvironmentError, match="incompatible"):
        run_benchmark(
            [sample], top_k=5, mode="retrieval", max_questions=None, run_id=uuid4().hex[:8],
            resume=True, checkpoint_dir=checkpoint_dir, dataset_path=FIXTURE_DATASET_PATH, show_progress=False,
        )


def test_non_resume_runs_never_read_or_write_a_checkpoint(fake_extraction, tmp_path) -> None:
    sample = _two_session_sample(f"resume-conv-{uuid4().hex}")
    checkpoint_dir = tmp_path / "checkpoints"

    run_benchmark(
        [sample], top_k=5, mode="retrieval", max_questions=None, run_id=uuid4().hex[:8],
        resume=False, checkpoint_dir=checkpoint_dir, dataset_path=FIXTURE_DATASET_PATH, show_progress=False,
    )

    assert load_checkpoint(sample.sample_id, checkpoint_dir=checkpoint_dir) is None
