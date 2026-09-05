"""Checkpoint/resume for the retrieval-only lexical ablation runner (Part F).

Mirrors ``evals/locomo/checkpoint.py``'s pattern for the main LoCoMo ingestion
runner: a small JSON fingerprint file plus replace-on-complete durability, and
a fingerprint compatibility check that must pass before any resume is
trusted. This module is a distinct file (not a reuse of
``LocomoCheckpoint``) because the ablation's fingerprint is different in
kind -- it tracks retrieval-only configuration (top_k, lexical backends, RRF
constant, BM25 params, embedding model, dataset hash), never an ingestion
session count, since the ablation never ingests anything.

Per-question results are appended to a sibling JSON-lines file as each
question completes, so a resumed run can skip every already-completed
question index without recomputing it. A checkpoint whose fingerprint does
not match the current run's configuration is never resumed -- the caller
must start fresh (Part F's explicit requirement: do not silently reuse an
incompatible checkpoint).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ABLATION_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


class AblationCheckpointError(RuntimeError):
    """Raised when a persisted ablation checkpoint cannot be trusted."""


@dataclass
class AblationCheckpoint:
    """Fingerprint plus progress for one retrieval-ablation run."""

    sample_id: str
    dataset_sha256: str
    embedding_model: str | None
    top_k: int
    lexical_backends: list[str]
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    completed_question_indices: list[int] = field(default_factory=list)
    updated_at: str = ""

    def incompatibilities(
        self,
        *,
        sample_id: str,
        dataset_sha256: str,
        embedding_model: str | None,
        top_k: int,
        lexical_backends: list[str],
        rrf_k: int,
        bm25_k1: float,
        bm25_b: float,
    ) -> list[str]:
        """Every reason this checkpoint cannot safely be resumed under the current run.

        An empty list means it is safe to resume. Any mismatch here means the
        checkpoint's completed-question results were computed under different
        retrieval semantics, so the caller must discard them and start fresh
        rather than mixing incompatible results into one report.
        """

        problems: list[str] = []
        if self.sample_id != sample_id:
            problems.append(f"sample_id: checkpoint={self.sample_id!r}, requested={sample_id!r}")
        if self.dataset_sha256 != dataset_sha256:
            problems.append(
                f"dataset_sha256: checkpoint={self.dataset_sha256!r}, current={dataset_sha256!r} "
                "(the vendored dataset file changed since this checkpoint was written)"
            )
        if self.embedding_model != embedding_model:
            problems.append(f"embedding_model: checkpoint={self.embedding_model!r}, current={embedding_model!r}")
        if self.top_k != top_k:
            problems.append(f"top_k: checkpoint={self.top_k!r}, current={top_k!r}")
        if list(self.lexical_backends) != list(lexical_backends):
            problems.append(f"lexical_backends: checkpoint={self.lexical_backends!r}, current={lexical_backends!r}")
        if self.rrf_k != rrf_k:
            problems.append(f"rrf_k: checkpoint={self.rrf_k!r}, current={rrf_k!r}")
        if self.bm25_k1 != bm25_k1:
            problems.append(f"bm25_k1: checkpoint={self.bm25_k1!r}, current={bm25_k1!r}")
        if self.bm25_b != bm25_b:
            problems.append(f"bm25_b: checkpoint={self.bm25_b!r}, current={bm25_b!r}")
        return problems


def _safe_filename(sample_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in sample_id)


def ablation_checkpoint_path(sample_id: str, *, checkpoint_dir: Path | None = None) -> Path:
    directory = checkpoint_dir if checkpoint_dir is not None else DEFAULT_ABLATION_CHECKPOINT_DIR
    return directory / f"ablation-{_safe_filename(sample_id)}.json"


def ablation_results_path(sample_id: str, *, checkpoint_dir: Path | None = None) -> Path:
    directory = checkpoint_dir if checkpoint_dir is not None else DEFAULT_ABLATION_CHECKPOINT_DIR
    return directory / f"ablation-{_safe_filename(sample_id)}-results.jsonl"


def load_ablation_checkpoint(sample_id: str, *, checkpoint_dir: Path | None = None) -> AblationCheckpoint | None:
    """Return the persisted checkpoint for one sample, or None if it does not exist.

    Raises ``AblationCheckpointError`` (never silently ignored) if the file
    exists but cannot be parsed -- a corrupt checkpoint must never be treated
    as "no checkpoint" (silent restart) or guessed at.
    """

    path = ablation_checkpoint_path(sample_id, checkpoint_dir=checkpoint_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AblationCheckpoint(**raw)
    except Exception as error:
        raise AblationCheckpointError(
            f"Ablation checkpoint file {path} exists but could not be parsed; refusing to guess its state: {error}"
        ) from error


def save_ablation_checkpoint(state: AblationCheckpoint, *, checkpoint_dir: Path | None = None) -> Path:
    """Persist one checkpoint with the project's replace-on-complete convention."""

    state.updated_at = datetime.now(timezone.utc).isoformat()
    path = ablation_checkpoint_path(state.sample_id, checkpoint_dir=checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json.tmp", delete=False, mode="w", encoding="utf-8"
    )
    temporary_path = Path(handle.name)
    try:
        json.dump(asdict(state), handle, indent=2)
        handle.close()
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def delete_ablation_checkpoint(sample_id: str, *, checkpoint_dir: Path | None = None) -> None:
    """Remove a sample's ablation checkpoint and results, if any."""

    ablation_checkpoint_path(sample_id, checkpoint_dir=checkpoint_dir).unlink(missing_ok=True)
    ablation_results_path(sample_id, checkpoint_dir=checkpoint_dir).unlink(missing_ok=True)


def append_question_result(
    sample_id: str,
    result: dict,
    *,
    checkpoint_dir: Path | None = None,
) -> None:
    """Append one completed question's result as one JSON-line record."""

    path = ablation_results_path(sample_id, checkpoint_dir=checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result))
        handle.write("\n")


def load_question_results(sample_id: str, *, checkpoint_dir: Path | None = None) -> dict[int, dict]:
    """Return every already-persisted question result, keyed by question_index.

    A truncated/corrupt trailing line (e.g. a crash mid-write) is skipped
    rather than raised, since JSON-lines appends are not individually
    atomic; the checkpoint's ``completed_question_indices`` remains the
    authoritative record of what is safe to skip, not the mere presence of
    a results line.
    """

    path = ablation_results_path(sample_id, checkpoint_dir=checkpoint_dir)
    if not path.is_file():
        return {}
    results: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        index = record.get("question_index")
        if isinstance(index, int):
            results[index] = record
    return results
