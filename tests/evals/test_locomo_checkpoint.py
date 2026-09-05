"""Unit tests for LoCoMo checkpoint/resume state -- no database, no provider calls.

evals/locomo/checkpoint.py is pure file I/O plus compatibility comparison, so
these run unconditionally (unlike the ingestion/runner integration tests,
which need TEST_DATABASE_URL).
"""

from __future__ import annotations

import json

import pytest

from evals.locomo.checkpoint import (
    CheckpointError,
    checkpoint_path,
    delete_checkpoint,
    load_checkpoint,
    new_checkpoint,
    save_checkpoint,
)


def _checkpoint(**overrides):
    defaults = dict(
        sample_id="conv-1",
        dataset_sha256="abc123",
        llm_model="gpt-test",
        embedding_model="embed-test",
        user_id="locomo-resume-conv-1",
        conversation_id="conversation",
        last_completed_session=3,
        git_commit="deadbeef",
    )
    defaults.update(overrides)
    return new_checkpoint(**defaults)


def test_save_and_load_roundtrip(tmp_path) -> None:
    state = _checkpoint()

    path = save_checkpoint(state, checkpoint_dir=tmp_path)
    loaded = load_checkpoint("conv-1", checkpoint_dir=tmp_path)

    assert path == checkpoint_path("conv-1", checkpoint_dir=tmp_path)
    assert path.is_file()
    assert loaded == state


def test_load_missing_checkpoint_returns_none(tmp_path) -> None:
    assert load_checkpoint("never-ingested", checkpoint_dir=tmp_path) is None


def test_load_corrupt_checkpoint_raises_instead_of_guessing(tmp_path) -> None:
    path = checkpoint_path("conv-2", checkpoint_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{", encoding="utf-8")

    with pytest.raises(CheckpointError, match="could not be parsed"):
        load_checkpoint("conv-2", checkpoint_dir=tmp_path)


def test_save_overwrites_atomically_leaving_no_temp_file(tmp_path) -> None:
    save_checkpoint(_checkpoint(last_completed_session=1), checkpoint_dir=tmp_path)
    save_checkpoint(_checkpoint(last_completed_session=2), checkpoint_dir=tmp_path)

    loaded = load_checkpoint("conv-1", checkpoint_dir=tmp_path)
    assert loaded.last_completed_session == 2
    leftover_temp_files = list(tmp_path.glob("*.tmp"))
    assert leftover_temp_files == []


def test_delete_checkpoint_is_safe_when_absent(tmp_path) -> None:
    delete_checkpoint("nothing-here", checkpoint_dir=tmp_path)  # must not raise

    save_checkpoint(_checkpoint(), checkpoint_dir=tmp_path)
    delete_checkpoint("conv-1", checkpoint_dir=tmp_path)
    assert load_checkpoint("conv-1", checkpoint_dir=tmp_path) is None


def test_compatible_checkpoint_has_no_incompatibilities() -> None:
    state = _checkpoint()

    problems = state.incompatibilities(
        sample_id="conv-1",
        dataset_sha256="abc123",
        llm_model="gpt-test",
        embedding_model="embed-test",
    )

    assert problems == []


@pytest.mark.parametrize(
    "overrides,expected_snippet",
    [
        ({"dataset_sha256": "different-hash"}, "dataset_sha256"),
        ({"llm_model": "different-model"}, "llm_model"),
        ({"embedding_model": "different-model"}, "embedding_model"),
        ({"sample_id": "conv-2"}, "sample_id"),
    ],
)
def test_incompatible_checkpoint_reports_every_mismatch(overrides, expected_snippet) -> None:
    state = _checkpoint()
    current = dict(
        sample_id="conv-1",
        dataset_sha256="abc123",
        llm_model="gpt-test",
        embedding_model="embed-test",
    )
    current.update(overrides)

    problems = state.incompatibilities(**current)

    assert problems
    assert any(expected_snippet in problem for problem in problems)


def test_multiple_mismatches_are_all_reported_at_once() -> None:
    state = _checkpoint()

    problems = state.incompatibilities(
        sample_id="conv-1",
        dataset_sha256="different-hash",
        llm_model="different-model",
        embedding_model="embed-test",
    )

    assert len(problems) == 2


def test_checkpoint_path_is_stable_and_filesystem_safe(tmp_path) -> None:
    first = checkpoint_path("conv-26", checkpoint_dir=tmp_path)
    second = checkpoint_path("conv-26", checkpoint_dir=tmp_path)

    assert first == second
    assert first.suffix == ".json"
    assert first.parent == tmp_path


def test_saved_checkpoint_json_is_human_readable(tmp_path) -> None:
    save_checkpoint(_checkpoint(), checkpoint_dir=tmp_path)

    raw = json.loads(checkpoint_path("conv-1", checkpoint_dir=tmp_path).read_text(encoding="utf-8"))

    assert raw["sample_id"] == "conv-1"
    assert raw["last_completed_session"] == 3
    assert raw["dataset_sha256"] == "abc123"
