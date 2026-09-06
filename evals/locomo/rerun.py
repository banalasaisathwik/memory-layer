"""Rerun LoCoMo retrieval-only evaluation against an already-ingested memory state.

This never calls ``ingest_sample()`` or ``MemoryLayer.add()``. It reuses the
already-persisted conversation (same ``user_id``/``conversation_id`` as the
baseline run being repeated) and only exercises ``MemoryLayer.search()``,
exactly mirroring ``evals/locomo/runner.py``'s retrieval-only evaluation path
so a rerun after recovery is comparable question-for-question with a prior
baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from meminfra.database.models import Memory, User
from meminfra.memory_layer import MemoryLayer

from .metrics import aggregate_retrieval_metrics, evaluate_question_retrieval, group_retrieval_by_category
from .recover import build_dia_id_mapping, session_message_slices
from .schemas import CATEGORY_NAMES, LocomoRunMetadata, LocomoRunReport, LocomoSample, QuestionDiagnostic


@dataclass(frozen=True)
class RetrievalRerunResult:
    diagnostics: list[QuestionDiagnostic]
    dia_id_to_message_id: dict[str, str]
    message_id_to_dia_id: dict[str, str]


def _evaluate_question_retrieval_only(
    db: Session,
    memory: MemoryLayer,
    *,
    sample: LocomoSample,
    question_index: int,
    user_id: str,
    primary_user_db_id: object,
    top_k: int,
    message_id_to_dia_id: dict[str, str],
) -> QuestionDiagnostic:
    """Mirror evals/locomo/runner.py's `_evaluate_question` retrieval branch exactly."""

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
    return diagnostic


def rerun_retrieval(
    db: Session,
    sample: LocomoSample,
    *,
    user_id: str,
    conversation_id: str,
    top_k: int,
) -> RetrievalRerunResult:
    """Evaluate every QA question's retrieval against the current memory state.

    Requires the conversation to already exist (ingested by a prior run,
    possibly since repaired by ``evals.locomo.recover``); raises if it does
    not, rather than silently ingesting anything.
    """

    from meminfra.database.models import Conversation

    user = db.scalar(select(User).where(User.external_id == user_id))
    if user is None:
        raise RuntimeError(f"No user exists for external ID {user_id!r}; nothing to rerun retrieval against.")
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.external_id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise RuntimeError(f"No conversation exists for external ID {conversation_id!r} under user {user_id!r}.")

    slices = session_message_slices(db, sample, conversation=conversation)
    dia_id_to_message_id, message_id_to_dia_id = build_dia_id_mapping(sample, slices)

    memory = MemoryLayer(db)
    diagnostics = [
        _evaluate_question_retrieval_only(
            db,
            memory,
            sample=sample,
            question_index=question_index,
            user_id=user_id,
            primary_user_db_id=user.id,
            top_k=top_k,
            message_id_to_dia_id=message_id_to_dia_id,
        )
        for question_index in range(len(sample.qa))
    ]
    return RetrievalRerunResult(
        diagnostics=diagnostics,
        dia_id_to_message_id=dia_id_to_message_id,
        message_id_to_dia_id=message_id_to_dia_id,
    )


def build_retrieval_report(
    diagnostics: list[QuestionDiagnostic],
    *,
    top_k: int,
    run_id: str,
    dataset_path: Path,
    dataset_sha256_value: str,
    git_commit: str | None,
    llm_model: str | None,
    embedding_model: str | None,
) -> LocomoRunReport:
    metadata = LocomoRunMetadata(
        dataset_source="https://github.com/snap-research/locomo (data/locomo10.json)",
        dataset_sha256=dataset_sha256_value,
        git_commit=git_commit,
        run_at=datetime.now(timezone.utc),
        run_id=run_id,
        llm_model=llm_model,
        embedding_model=embedding_model,
        top_k=top_k,
        mode="retrieval",
        conversations_ingested=len({d.sample_id for d in diagnostics}),
        questions_evaluated=len(diagnostics),
        resumed=True,
    )
    return LocomoRunReport(
        metadata=metadata,
        retrieval=aggregate_retrieval_metrics(diagnostics),
        retrieval_by_category=group_retrieval_by_category(diagnostics),
        qa=None,
        qa_by_category={},
        questions=diagnostics,
    )


def default_v1_result_path(sample_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "results" / f"locomo-v1-{sample_id}-{timestamp}.json"


def save_v1_result(
    report: LocomoRunReport,
    *,
    base_run_path: str,
    recovered_sessions: list[int],
    recovery_report_path: str | None,
    path: Path,
) -> None:
    """Save V1 as an explicitly-labeled, immutable JSON file distinct from V0.

    The wrapping fields (base_run, recovered_sessions, recovery_report_path,
    label) are additive metadata about *how this run relates to V0*; they are
    written alongside the normal LocomoRunReport payload rather than folded
    into LocomoRunMetadata, which stays the unmodified production schema.
    """

    payload = {
        "label": "LoCoMo Subset Baseline V1",
        "base_run": base_run_path,
        "recovered_sessions": recovered_sessions,
        "recovery_report_path": recovery_report_path,
        "report": report.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess
    from uuid import uuid4

    from meminfra.config import configure, get_config, reset_config
    from meminfra.database import SessionLocal, create_tables, reset_engine

    from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
    from .dataset import DEFAULT_DATASET_PATH, dataset_sha256, load_locomo_dataset

    parser = argparse.ArgumentParser(description="Rerun LoCoMo retrieval-only evaluation (no re-ingestion).")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--conversation-id", default="conversation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-run", required=True, help="Path to the V0 result JSON this rerun compares against.")
    parser.add_argument("--recovered-sessions", type=int, nargs="+", default=[])
    parser.add_argument("--recovery-report", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    try:
        eval_database_url = get_eval_database_url()
    except EvalDatabaseConfigError as error:
        print(f"error: {error}")
        return 1

    print_eval_database_banner(eval_database_url)
    reset_config()
    reset_engine()
    configure(database_url=eval_database_url)
    create_tables()

    samples = load_locomo_dataset(args.dataset_path)
    sample = next((candidate for candidate in samples if candidate.sample_id == args.sample), None)
    if sample is None:
        print(f"error: sample {args.sample!r} not found.")
        return 1

    user_id = args.user_id or f"locomo-resume-{args.sample}"
    settings = get_config()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = None

    try:
        with SessionLocal() as db:
            result = rerun_retrieval(
                db,
                sample,
                user_id=user_id,
                conversation_id=args.conversation_id,
                top_k=args.top_k,
            )
    finally:
        reset_engine()
        reset_config()

    report = build_retrieval_report(
        result.diagnostics,
        top_k=args.top_k,
        run_id=uuid4().hex[:8],
        dataset_path=args.dataset_path,
        dataset_sha256_value=dataset_sha256(args.dataset_path),
        git_commit=git_commit,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )

    output_path = Path(args.output) if args.output else default_v1_result_path(args.sample)
    save_v1_result(
        report,
        base_run_path=args.base_run,
        recovered_sessions=sorted(args.recovered_sessions),
        recovery_report_path=args.recovery_report,
        path=output_path,
    )
    print(f"Saved V1 result to {output_path}")
    if report.retrieval is not None:
        r = report.retrieval
        print(
            f"Hit@1={r.hit_at_1:.2f} Hit@3={r.hit_at_3:.2f} Hit@5={r.hit_at_5:.2f} Hit@10={r.hit_at_10:.2f}  "
            f"Recall@1={r.recall_at_1:.2f} Recall@3={r.recall_at_3:.2f} Recall@5={r.recall_at_5:.2f} "
            f"Recall@10={r.recall_at_10:.2f}  MRR={r.mrr:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
