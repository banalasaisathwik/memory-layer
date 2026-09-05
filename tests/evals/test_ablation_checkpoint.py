"""Unit tests for ablation checkpoint/resume (Part F, no database)."""

from __future__ import annotations

import pytest

from evals.locomo.ablation_checkpoint import (
    AblationCheckpoint,
    AblationCheckpointError,
    append_question_result,
    load_ablation_checkpoint,
    load_question_results,
    save_ablation_checkpoint,
)


def _checkpoint(**overrides) -> AblationCheckpoint:
    defaults = dict(
        sample_id="conv-30",
        dataset_sha256="dataset-abc",
        embedding_model="text-embedding-3-small",
        top_k=10,
        lexical_backends=["postgres_fts", "bm25"],
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
        completed_question_indices=[],
    )
    defaults.update(overrides)
    return AblationCheckpoint(**defaults)


def test_no_checkpoint_returns_none(tmp_path) -> None:
    assert load_ablation_checkpoint("conv-30", checkpoint_dir=tmp_path) is None


def test_save_then_load_round_trips(tmp_path) -> None:
    state = _checkpoint(completed_question_indices=[0, 1, 2])
    save_ablation_checkpoint(state, checkpoint_dir=tmp_path)

    loaded = load_ablation_checkpoint("conv-30", checkpoint_dir=tmp_path)
    assert loaded is not None
    assert loaded.completed_question_indices == [0, 1, 2]
    assert loaded.dataset_sha256 == "dataset-abc"


def test_questions_1_to_3_complete_then_restart_skips_them(tmp_path) -> None:
    state = _checkpoint()
    for index in range(3):
        append_question_result("conv-30", {"question_index": index, "question": f"q{index}"}, checkpoint_dir=tmp_path)
        state.completed_question_indices.append(index)
    save_ablation_checkpoint(state, checkpoint_dir=tmp_path)

    # Simulate a restart: fresh load.
    resumed = load_ablation_checkpoint("conv-30", checkpoint_dir=tmp_path)
    assert resumed is not None
    assert resumed.completed_question_indices == [0, 1, 2]

    results = load_question_results("conv-30", checkpoint_dir=tmp_path)
    assert set(results) == {0, 1, 2}

    # The next question to run is index 3.
    remaining = [i for i in range(6) if i not in resumed.completed_question_indices]
    assert remaining == [3, 4, 5]


def test_incompatible_fingerprint_is_detected() -> None:
    checkpoint = _checkpoint(top_k=10, dataset_sha256="dataset-abc")

    problems = checkpoint.incompatibilities(
        sample_id="conv-30",
        dataset_sha256="dataset-DIFFERENT",
        embedding_model="text-embedding-3-small",
        top_k=5,
        lexical_backends=["postgres_fts", "bm25"],
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
    )

    assert any("dataset_sha256" in p for p in problems)
    assert any("top_k" in p for p in problems)


def test_compatible_fingerprint_has_no_problems() -> None:
    checkpoint = _checkpoint()
    problems = checkpoint.incompatibilities(
        sample_id="conv-30",
        dataset_sha256="dataset-abc",
        embedding_model="text-embedding-3-small",
        top_k=10,
        lexical_backends=["postgres_fts", "bm25"],
        rrf_k=60,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    assert problems == []


def test_corrupt_checkpoint_raises_rather_than_silently_restarting(tmp_path) -> None:
    path = tmp_path / "ablation-conv-30.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(AblationCheckpointError):
        load_ablation_checkpoint("conv-30", checkpoint_dir=tmp_path)


def test_results_jsonl_skips_corrupt_trailing_line(tmp_path) -> None:
    append_question_result("conv-30", {"question_index": 0, "question": "q0"}, checkpoint_dir=tmp_path)
    from evals.locomo.ablation_checkpoint import ablation_results_path

    path = ablation_results_path("conv-30", checkpoint_dir=tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated garbage\n")

    results = load_question_results("conv-30", checkpoint_dir=tmp_path)
    assert set(results) == {0}
