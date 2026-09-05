"""Ingest LoCoMo conversations through MemoryLayer.add() and score retrieval/QA.

Flow for one selected LoCoMo sample:

    sessions (chronological) -> MemoryLayer.add() once per session
                              -> dia_id <-> persisted Message.id mapping
    each QA question         -> MemoryLayer.search()  -> evidence-provenance retrieval metrics
                              -> MemoryLayer.answer()  -> LoCoMo-compatible QA score

Retrieval and QA are always scored and reported separately; see
evals/README.md for the full methodology, including the coarse per-interaction
provenance caveat that this milestone deliberately does not fix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import configure, get_config, reset_config
from src.database import Memory, SessionLocal, User, create_tables, reset_engine
from src.memory_layer import AddResult, AnswerError, MemoryLayer
from src.retrieval import MessageIndexSyncStats, get_message_index_sync_stats, reset_message_index_sync_stats

from .checkpoint import load_checkpoint, new_checkpoint, save_checkpoint
from .dataset import DATASET_SOURCE_URL, DEFAULT_DATASET_PATH, dataset_sha256, load_locomo_dataset
from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
from .ingest import IngestOutcome, ingest_sample
from .metrics import aggregate_retrieval_metrics, evaluate_question_retrieval, group_retrieval_by_category
from .qa_metrics import aggregate_qa_metrics, group_qa_by_category
from .report import render_report, save_report
from .schemas import CATEGORY_NAMES, LocomoRunMetadata, LocomoRunReport, LocomoSample, QuestionDiagnostic
from .scoring import ScoringError, score_qa


class LocomoEnvironmentError(RuntimeError):
    """Raised when required database or provider configuration is missing."""


def _check_provider_config() -> None:
    settings = get_config()
    missing = []
    if not settings.llm_model:
        missing.append("LLM_MODEL")
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY")
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if missing:
        raise LocomoEnvironmentError(
            "Missing required provider configuration for the LoCoMo benchmark: " + ", ".join(missing) + "."
        )


def _git_commit_sha() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _select_samples(samples: list[LocomoSample], *, conversation_index: int | None) -> list[LocomoSample]:
    if conversation_index is None:
        return samples
    if not (0 <= conversation_index < len(samples)):
        raise LocomoEnvironmentError(
            f"--conversation {conversation_index} is out of range; the dataset has {len(samples)} "
            f"conversations (valid indices: 0..{len(samples) - 1})."
        )
    return [samples[conversation_index]]


def _evaluate_question(
    db: Session,
    memory: MemoryLayer,
    *,
    sample: LocomoSample,
    question_index: int,
    user_id: str,
    primary_user_db_id: object,
    top_k: int,
    mode: str,
    message_id_to_dia_id: dict[str, str],
) -> QuestionDiagnostic:
    qa = sample.qa[question_index]
    diagnostic = QuestionDiagnostic(
        sample_id=sample.sample_id,
        question_index=question_index,
        question=qa.question,
        category_id=qa.category_id,
        category_name=CATEGORY_NAMES[qa.category_id],
        gold_answer=qa.answer,
        adversarial_trap_answer=qa.adversarial_answer,
        gold_evidence=qa.evidence,
        unresolved_evidence=qa.unresolved_evidence,
    )

    if mode in ("retrieval", "both"):
        hits = memory.search(user_id=user_id, query=qa.question, limit=top_k)
        provenance_by_rank: list[set[str]] = []
        isolation_failures = 0
        for hit in hits:
            diagnostic.retrieved_memory_ids.append(hit.memory_id)
            diagnostic.retrieved_memory_texts.append(hit.memory_text)
            memory_row = db.get(Memory, hit.memory_id)
            if memory_row is None or memory_row.user_id != primary_user_db_id:
                isolation_failures += 1
                provenance_by_rank.append(set())
                diagnostic.retrieved_provenance_dia_ids.append([])
                continue
            provenance = {
                message_id_to_dia_id[message_id]
                for message_id in memory_row.source_message_ids
                if message_id in message_id_to_dia_id
            }
            provenance_by_rank.append(provenance)
            diagnostic.retrieved_provenance_dia_ids.append(sorted(provenance))
        diagnostic.isolation_failures = isolation_failures
        diagnostic.retrieval_metrics = evaluate_question_retrieval(provenance_by_rank, qa.evidence)

    if mode in ("qa", "both"):
        try:
            result = memory.answer(user_id=user_id, query=qa.question, limit=top_k)
        except AnswerError as error:
            diagnostic.error = f"answer() failed: {error}"
        else:
            diagnostic.predicted_answer = result.answer
            diagnostic.abstained = result.abstained
            try:
                score, is_correct_abstention = score_qa(
                    category_id=qa.category_id,
                    predicted_answer=result.answer,
                    gold_answer=qa.answer,
                    abstained=result.abstained,
                )
                diagnostic.qa_score = score
                diagnostic.is_correct_abstention = is_correct_abstention
            except ScoringError as error:
                diagnostic.error = f"scoring failed: {error}"

    return diagnostic


def _session_turn_counts(sample: LocomoSample) -> dict[int, int]:
    counts: dict[int, int] = {}
    for turn in sample.turns:
        counts[turn.session_number] = counts.get(turn.session_number, 0) + 1
    return counts


def _make_session_progress_hook(
    *,
    sample: LocomoSample,
    user_id: str,
    conversation_id: str,
    resume_from_session: int,
    resume: bool,
    dataset_sha256_value: str | None,
    git_commit: str | None,
    checkpoint_dir: Path | None,
    show_progress: bool,
) -> tuple[Callable[[int, AddResult], None], Callable[[int, str], None]]:
    """Build the per-session callbacks: print progress and, if resuming, checkpoint.

    A checkpoint is only ever written from the success callback -- never for
    a session that raised. See evals/locomo/checkpoint.py and ingest.py's
    on_session_complete/on_session_failed contract. Both callbacks advance
    the turns-processed counter (a failed session's messages are still
    durably persisted; only its extraction is incomplete), so a failure
    shows up live instead of only in the final returned warnings list.
    """

    turn_counts = _session_turn_counts(sample)
    total_sessions = len(turn_counts)
    total_turns = len(sample.turns)
    turns_processed = sum(count for session, count in turn_counts.items() if session <= resume_from_session)
    memories_written = 0
    started_at = time.monotonic()

    def _on_session_complete(session_number: int, result: AddResult) -> None:
        nonlocal turns_processed, memories_written
        if resume:
            save_checkpoint(
                new_checkpoint(
                    sample_id=sample.sample_id,
                    dataset_sha256=dataset_sha256_value,
                    llm_model=get_config().llm_model,
                    embedding_model=get_config().embedding_model,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    last_completed_session=session_number,
                    git_commit=git_commit,
                ),
                checkpoint_dir=checkpoint_dir,
            )
        turns_processed += turn_counts[session_number]
        memories_written += len(result.write_results)
        if show_progress:
            elapsed = time.monotonic() - started_at
            print(
                f"LoCoMo sample: {sample.sample_id}  Session: {session_number}/{total_sessions}  "
                f"Turns in session: {turn_counts[session_number]}  "
                f"Total turns processed: {turns_processed}/{total_turns}  "
                f"Memories written so far: {memories_written}  "
                f"Warnings: {len(result.warnings)}  Elapsed: {elapsed:.0f}s",
                flush=True,
            )

    def _on_session_failed(session_number: int, warning: str) -> None:
        nonlocal turns_processed
        # No checkpoint write: an unsuccessful session must never be marked complete.
        turns_processed += turn_counts[session_number]
        if show_progress:
            elapsed = time.monotonic() - started_at
            print(
                f"LoCoMo sample: {sample.sample_id}  Session: {session_number}/{total_sessions}  FAILED  "
                f"Total turns processed: {turns_processed}/{total_turns}  Elapsed: {elapsed:.0f}s\n"
                f"  warning: {warning}",
                flush=True,
            )

    return _on_session_complete, _on_session_failed


def run_benchmark(
    samples: list[LocomoSample],
    *,
    top_k: int,
    mode: str,
    max_questions: int | None,
    run_id: str,
    resume: bool = False,
    checkpoint_dir: Path | None = None,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    show_progress: bool = True,
    on_sample_ingested: Callable[[LocomoSample, IngestOutcome], None] | None = None,
) -> list[QuestionDiagnostic]:
    """Ingest each sample once, then evaluate its questions against that one memory state.

    ``resume=True`` uses a deterministic, dataset/sample-scoped ``user_id``
    (instead of one tagged with the fresh, random ``run_id``) so a later
    invocation can find and safely continue the same conversation, guarded
    by a session-level checkpoint (see evals/locomo/checkpoint.py). Without
    it, every run is isolated exactly as before -- a fresh user per sample,
    full re-ingestion, no checkpoint read or written.

    ``on_sample_ingested``, if given, is called once per sample with its
    full ``IngestOutcome`` right after ``ingest_sample()`` returns -- this is
    how a caller (see ``main()``) recovers extraction warnings/failures for
    the saved report without changing this function's own return type.
    """

    diagnostics: list[QuestionDiagnostic] = []
    questions_used = 0
    dataset_sha256_value = dataset_sha256(dataset_path) if resume else None
    git_commit = _git_commit_sha() if resume else None

    with SessionLocal() as db:
        for sample in samples:
            if max_questions is not None and questions_used >= max_questions:
                break

            conversation_id = "conversation"
            resume_from_session = 0
            if resume:
                user_id = f"locomo-resume-{sample.sample_id}"
                checkpoint = load_checkpoint(sample.sample_id, checkpoint_dir=checkpoint_dir)
                if checkpoint is not None:
                    problems = checkpoint.incompatibilities(
                        sample_id=sample.sample_id,
                        dataset_sha256=dataset_sha256_value,
                        llm_model=get_config().llm_model,
                        embedding_model=get_config().embedding_model,
                    )
                    if checkpoint.user_id != user_id or checkpoint.conversation_id != conversation_id:
                        problems.append(
                            "user_id/conversation_id: checkpoint="
                            f"{checkpoint.user_id!r}/{checkpoint.conversation_id!r}, current={user_id!r}/{conversation_id!r}"
                        )
                    if problems:
                        raise LocomoEnvironmentError(
                            f"Checkpoint for {sample.sample_id} is incompatible with this run and will not be "
                            "reused (refusing to silently continue with stale benchmark state): "
                            + "; ".join(problems)
                            + ". Delete the checkpoint file under evals/checkpoints/ to force a fresh run."
                        )
                    resume_from_session = checkpoint.last_completed_session
            else:
                user_id = f"locomo-{sample.sample_id}-{run_id}"

            memory = MemoryLayer(db)
            on_session_complete, on_session_failed = _make_session_progress_hook(
                sample=sample,
                user_id=user_id,
                conversation_id=conversation_id,
                resume_from_session=resume_from_session,
                resume=resume,
                dataset_sha256_value=dataset_sha256_value,
                git_commit=git_commit,
                checkpoint_dir=checkpoint_dir,
                show_progress=show_progress,
            )
            outcome = ingest_sample(
                db,
                sample,
                user_id=user_id,
                conversation_id=conversation_id,
                resume_from_session=resume_from_session,
                on_session_complete=on_session_complete,
                on_session_failed=on_session_failed,
            )
            if on_sample_ingested is not None:
                on_sample_ingested(sample, outcome)
            primary_user = db.scalar(select(User).where(User.external_id == user_id))
            if primary_user is None:
                raise RuntimeError(f"{sample.sample_id}: ingestion did not create user {user_id!r}.")

            for question_index in range(len(sample.qa)):
                if max_questions is not None and questions_used >= max_questions:
                    break
                diagnostics.append(
                    _evaluate_question(
                        db,
                        memory,
                        sample=sample,
                        question_index=question_index,
                        user_id=user_id,
                        primary_user_db_id=primary_user.id,
                        top_k=top_k,
                        mode=mode,
                        message_id_to_dia_id=outcome.message_id_to_dia_id,
                    )
                )
                questions_used += 1

    return diagnostics


def build_report(
    diagnostics: list[QuestionDiagnostic],
    *,
    top_k: int,
    mode: str,
    run_id: str,
    dataset_path: Path,
    resumed: bool = False,
    duration_seconds: float | None = None,
    sync_stats: MessageIndexSyncStats | None = None,
    sessions_ingested: int | None = None,
    sessions_failed: int | None = None,
    ingestion_warnings: list[str] | None = None,
) -> LocomoRunReport:
    settings = get_config()
    # Not len(samples): --max-questions can stop the run before every selected
    # sample is actually ingested, so this counts samples that produced at
    # least one evaluated question instead of samples merely selected.
    conversations_ingested = len({diagnostic.sample_id for diagnostic in diagnostics})
    metadata = LocomoRunMetadata(
        dataset_source=DATASET_SOURCE_URL,
        dataset_sha256=dataset_sha256(dataset_path),
        git_commit=_git_commit_sha(),
        run_at=datetime.now(timezone.utc),
        run_id=run_id,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        top_k=top_k,
        mode=mode,
        conversations_ingested=conversations_ingested,
        questions_evaluated=len(diagnostics),
        resumed=resumed,
        ingestion_and_eval_duration_seconds=duration_seconds,
        message_index_full_rebuilds=sync_stats.full_rebuilds if sync_stats is not None else None,
        message_index_incremental_appends=sync_stats.incremental_appends if sync_stats is not None else None,
        message_index_unchanged_hits=sync_stats.unchanged_hits if sync_stats is not None else None,
        sessions_ingested=sessions_ingested,
        sessions_failed=sessions_failed,
        ingestion_warnings=ingestion_warnings or [],
    )

    retrieval = aggregate_retrieval_metrics(diagnostics) if mode in ("retrieval", "both") else None
    retrieval_by_category = group_retrieval_by_category(diagnostics) if mode in ("retrieval", "both") else {}
    qa = aggregate_qa_metrics(diagnostics) if mode in ("qa", "both") else None
    qa_by_category = group_qa_by_category(diagnostics) if mode in ("qa", "both") else {}

    return LocomoRunReport(
        metadata=metadata,
        retrieval=retrieval,
        retrieval_by_category=retrieval_by_category,
        qa=qa,
        qa_by_category=qa_by_category,
        questions=diagnostics,
    )


def default_results_path(run_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "results" / f"locomo-{timestamp}-{run_id}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LoCoMo long-term memory benchmark.")
    parser.add_argument("--conversation", type=int, default=None, help="Run only this 0-indexed sample.")
    parser.add_argument(
        "--max-questions", type=int, default=None, help="Cap the total number of questions evaluated this run."
    )
    parser.add_argument("--mode", choices=["retrieval", "qa", "both"], default="retrieval")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", default=None, help="Path for the JSON run report.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Use a deterministic, checkpointed user_id per sample so a run interrupted mid-ingestion "
            "(e.g. a dropped database connection) can be safely continued by running the same command "
            "again, instead of re-ingesting from scratch under a fresh, isolated user_id. "
            "--max-questions only bounds QA evaluation cost, never ingestion cost."
        ),
    )
    args = parser.parse_args(argv)

    try:
        eval_database_url = get_eval_database_url()
        reset_config()
        reset_engine()
        _check_provider_config()
        samples = _select_samples(load_locomo_dataset(args.dataset_path), conversation_index=args.conversation)
    except (LocomoEnvironmentError, EvalDatabaseConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print_eval_database_banner(eval_database_url)
    configure(database_url=eval_database_url)
    create_tables()
    run_id = uuid4().hex[:8]
    reset_message_index_sync_stats()
    ingestion_started = time.monotonic()
    ingestion_warnings: list[str] = []
    sessions_ingested_total = 0
    sessions_failed_total = 0

    def _collect_ingestion_outcome(sample: LocomoSample, outcome) -> None:
        nonlocal sessions_ingested_total, sessions_failed_total
        ingestion_warnings.extend(f"{sample.sample_id}: {warning}" for warning in outcome.warnings)
        sessions_ingested_total += outcome.sessions_ingested
        sessions_failed_total += outcome.sessions_failed

    try:
        diagnostics = run_benchmark(
            samples,
            top_k=args.top_k,
            mode=args.mode,
            max_questions=args.max_questions,
            run_id=run_id,
            resume=args.resume,
            dataset_path=args.dataset_path,
            on_sample_ingested=_collect_ingestion_outcome,
        )
        ingestion_and_eval_duration = time.monotonic() - ingestion_started
        sync_stats = get_message_index_sync_stats()
        report = build_report(
            diagnostics,
            top_k=args.top_k,
            mode=args.mode,
            run_id=run_id,
            dataset_path=args.dataset_path,
            resumed=args.resume,
            duration_seconds=ingestion_and_eval_duration,
            sync_stats=sync_stats,
            sessions_ingested=sessions_ingested_total,
            sessions_failed=sessions_failed_total,
            ingestion_warnings=ingestion_warnings,
        )
    finally:
        reset_engine()
        reset_config()

    print(render_report(report))
    print(
        f"\nRun duration (ingestion + evaluation): {ingestion_and_eval_duration:.1f}s"
        f"\nMessage-index sync: {sync_stats.full_rebuilds} full rebuild(s), "
        f"{sync_stats.incremental_appends} incremental append(s), {sync_stats.unchanged_hits} unchanged hit(s)"
        f"\nSessions ingested: {sessions_ingested_total}   Sessions failed: {sessions_failed_total}"
    )
    if ingestion_warnings:
        print("Ingestion warnings:")
        for warning in ingestion_warnings:
            print(f"  - {warning}")
    output_path = Path(args.output) if args.output else default_results_path(run_id)
    save_report(report, output_path)
    print(f"\nSaved JSON report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
