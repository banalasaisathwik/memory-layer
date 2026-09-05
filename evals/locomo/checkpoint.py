"""Session-level checkpoint/resume for long LoCoMo ingestion runs.

This is deliberately benchmark-only: it never touches production
``MemoryLayer``, its schema, or its write semantics. A checkpoint file
records that one LoCoMo sample's ingestion has durably completed sessions
``1..last_completed_session`` -- nothing more. It is written (see
``evals/locomo/ingest.py``'s ``on_session_complete`` hook) only after
``MemoryLayer.add()`` has returned successfully for that session, so a
session that raised partway through never gets checkpointed as done.

Resuming a checkpointed run reuses the same deterministic ``user_id`` /
``conversation_id`` so the already-ingested sessions' Message rows and
derived memory state are still there; ``ingest_sample()`` skips sessions the
checkpoint already covers and recovers their dia_id<->Message.id mapping by
re-reading the already-persisted rows (never inserting anything by hand).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


class CheckpointError(RuntimeError):
    """Raised when a persisted checkpoint cannot be trusted or safely resumed."""


@dataclass
class LocomoCheckpoint:
    """Durable record of how far one LoCoMo sample's ingestion has safely progressed."""

    sample_id: str
    dataset_sha256: str
    llm_model: str | None
    embedding_model: str | None
    user_id: str
    conversation_id: str
    last_completed_session: int
    updated_at: str
    git_commit: str | None = None

    def incompatibilities(
        self,
        *,
        sample_id: str,
        dataset_sha256: str,
        llm_model: str | None,
        embedding_model: str | None,
    ) -> list[str]:
        """Return every reason this checkpoint cannot safely be resumed under the current run.

        An empty list means the checkpoint is compatible. Callers must check
        this before resuming -- silently continuing with stale benchmark
        state (a different dataset copy, or a different LLM/embedding model
        than produced the existing memory state) would make the resulting
        numbers meaningless.
        """

        problems: list[str] = []
        if self.sample_id != sample_id:
            problems.append(f"sample_id: checkpoint={self.sample_id!r}, requested={sample_id!r}")
        if self.dataset_sha256 != dataset_sha256:
            problems.append(
                f"dataset_sha256: checkpoint={self.dataset_sha256!r}, current={dataset_sha256!r} "
                "(the vendored dataset file changed since this checkpoint was written)"
            )
        if self.llm_model != llm_model:
            problems.append(f"llm_model: checkpoint={self.llm_model!r}, current={llm_model!r}")
        if self.embedding_model != embedding_model:
            problems.append(f"embedding_model: checkpoint={self.embedding_model!r}, current={embedding_model!r}")
        return problems


def _safe_filename(sample_id: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in sample_id)


def checkpoint_path(sample_id: str, *, checkpoint_dir: Path | None = None) -> Path:
    """Return the deterministic checkpoint file path for one sample; never creates it."""

    directory = checkpoint_dir if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    return directory / f"{_safe_filename(sample_id)}.json"


def load_checkpoint(sample_id: str, *, checkpoint_dir: Path | None = None) -> LocomoCheckpoint | None:
    """Return the persisted checkpoint for one sample, or None if it does not exist.

    Raises CheckpointError (never silently ignored) if the file exists but is
    unreadable or malformed -- a corrupt checkpoint must never be treated as
    "no checkpoint" (which would silently restart, doubling work) or guessed
    at (which could duplicate messages).
    """

    path = checkpoint_path(sample_id, checkpoint_dir=checkpoint_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LocomoCheckpoint(**raw)
    except Exception as error:
        raise CheckpointError(
            f"Checkpoint file {path} exists but could not be parsed; refusing to guess its state: {error}"
        ) from error


def save_checkpoint(state: LocomoCheckpoint, *, checkpoint_dir: Path | None = None) -> Path:
    """Persist one checkpoint with the project's replace-on-complete convention."""

    path = checkpoint_path(state.sample_id, checkpoint_dir=checkpoint_dir)
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


def delete_checkpoint(sample_id: str, *, checkpoint_dir: Path | None = None) -> None:
    """Remove a sample's checkpoint, if any; used once a sample is fully evaluated."""

    checkpoint_path(sample_id, checkpoint_dir=checkpoint_dir).unlink(missing_ok=True)


def new_checkpoint(
    *,
    sample_id: str,
    dataset_sha256: str,
    llm_model: str | None,
    embedding_model: str | None,
    user_id: str,
    conversation_id: str,
    last_completed_session: int,
    git_commit: str | None,
) -> LocomoCheckpoint:
    return LocomoCheckpoint(
        sample_id=sample_id,
        dataset_sha256=dataset_sha256,
        llm_model=llm_model,
        embedding_model=embedding_model,
        user_id=user_id,
        conversation_id=conversation_id,
        last_completed_session=last_completed_session,
        updated_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
    )
