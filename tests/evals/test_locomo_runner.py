"""DB integration test for the LoCoMo runner's ingest -> search -> answer plumbing.

Mirrors tests/evals/test_runner.py and tests/test_memory_layer.py: uses
TEST_DATABASE_URL (skipped without it, never falls back to DATABASE_URL) and
mocks every LLM/embedding call so it never makes a live provider request.
This exercises evals/locomo/runner.py's actual run_benchmark() path, not
extraction or ranking quality.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from meminfra.config import configure, reset_config
from meminfra.database import SessionLocal, create_tables, reset_engine
from evals.locomo.runner import run_benchmark
from evals.locomo.schemas import LocomoQA, LocomoSample, LocomoTurn

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; LoCoMo runner integration test never uses DATABASE_URL.",
    ),
]


def _target_interaction_content(call: dict[str, Any]) -> str:
    """Pull the raw JSON line out of extractor._request_content()'s TARGET INTERACTION section."""

    content = call["messages"][1]["content"]
    _, _, remainder = content.partition("TARGET INTERACTION\n")
    _, _, json_line = remainder.partition("\n")
    return json_line


class FakeExtractionCompletions:
    """Echo the target interaction's own message content back as one memory.

    This keeps retrieval realistic (each ingested interaction produces a
    distinct, lexically matching memory) without needing real extraction.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        target = json.loads(_target_interaction_content(kwargs))
        memory_text = " ".join(message["content"] for message in target["messages"])
        content = json.dumps(
            {
                "memories": [
                    {
                        "memory_type": "semantic",
                        "memory_text": memory_text,
                        "subject_type": "user",
                        "subject_name": None,
                        "predicate": None,
                        "value": None,
                        "confidence": 0.9,
                        "importance": 0.8,
                    }
                ]
            }
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeAnswerCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeLLMClient:
    def __init__(self, completions: Any) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)])


def _sample(sample_id: str, speaker_a: str, speaker_b: str, topic_sentence: str, evidence_text: str) -> LocomoSample:
    turns = [
        LocomoTurn(
            dia_id="D1:1",
            speaker=speaker_a,
            text=evidence_text,
            session_number=1,
            session_date_time="9:00 am on 1 January, 2024",
        ),
        LocomoTurn(
            dia_id="D1:2",
            speaker=speaker_b,
            text=topic_sentence,
            session_number=1,
            session_date_time="9:00 am on 1 January, 2024",
        ),
    ]
    qa = [
        LocomoQA(
            question=f"What did {speaker_a} say?",
            answer=evidence_text,
            category_id=4,
            evidence=["D1:1"],
        )
    ]
    return LocomoSample(sample_id=sample_id, speaker_a=speaker_a, speaker_b=speaker_b, turns=turns, qa=qa)


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


def test_run_benchmark_retrieval_finds_evidence_and_never_leaks_across_samples(fake_extraction) -> None:
    run_id = uuid4().hex[:8]
    sample_a = _sample("conv-a", "Alice", "Bob", "That sounds fun.", "I went hiking in the mountains yesterday.")
    sample_b = _sample("conv-b", "Xavier", "Yolanda", "Nice work!", "I finished painting the fence this weekend.")

    diagnostics = run_benchmark([sample_a, sample_b], top_k=5, mode="retrieval", max_questions=None, run_id=run_id)

    assert len(diagnostics) == 2
    for diagnostic in diagnostics:
        assert diagnostic.retrieval_metrics is not None
        assert diagnostic.retrieval_metrics.hit_at_1 == 1
        assert diagnostic.isolation_failures == 0
        # The top hit's provenance must resolve back to this sample's own evidence dia_id.
        assert "D1:1" in diagnostic.retrieved_provenance_dia_ids[0]


def test_run_benchmark_respects_max_questions_budget(fake_extraction) -> None:
    run_id = uuid4().hex[:8]
    sample_a = _sample("conv-c", "Alice", "Bob", "Cool.", "I adopted a cat named Whiskers.")
    sample_b = _sample("conv-d", "Sam", "Evan", "Great!", "I started a new job downtown.")

    diagnostics = run_benchmark([sample_a, sample_b], top_k=5, mode="retrieval", max_questions=1, run_id=run_id)

    assert len(diagnostics) == 1


def test_run_benchmark_qa_mode_scores_a_normal_question_with_f1(monkeypatch: pytest.MonkeyPatch, fake_extraction) -> None:
    answer_completions = FakeAnswerCompletions("I went hiking in the mountains yesterday.")
    monkeypatch.setattr("meminfra.memory_layer.get_llm_client", lambda: FakeLLMClient(answer_completions))

    run_id = uuid4().hex[:8]
    sample = _sample("conv-e", "Alice", "Bob", "Nice.", "I went hiking in the mountains yesterday.")

    diagnostics = run_benchmark([sample], top_k=5, mode="qa", max_questions=None, run_id=run_id)

    assert len(diagnostics) == 1
    assert diagnostics[0].qa_score == pytest.approx(1.0)
    assert diagnostics[0].abstained is False


def test_run_benchmark_qa_mode_scores_adversarial_abstention(monkeypatch: pytest.MonkeyPatch, fake_extraction) -> None:
    answer_completions = FakeAnswerCompletions("UNKNOWN")
    monkeypatch.setattr("meminfra.memory_layer.get_llm_client", lambda: FakeLLMClient(answer_completions))

    run_id = uuid4().hex[:8]
    sample = _sample("conv-f", "Alice", "Bob", "Hm.", "I might have visited Paris once.")
    sample.qa[0] = LocomoQA(
        question="What did Alice realize after her trip?",
        adversarial_answer="she loves croissants",
        category_id=5,
        evidence=["D1:1"],
    )

    diagnostics = run_benchmark([sample], top_k=5, mode="qa", max_questions=None, run_id=run_id)

    assert len(diagnostics) == 1
    assert diagnostics[0].abstained is True
    assert diagnostics[0].qa_score == 1.0
    assert diagnostics[0].is_correct_abstention is True
