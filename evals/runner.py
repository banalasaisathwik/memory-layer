"""Run eval cases through the production memory pipeline and score retrieval.

Flow for one EvalCase:

    messages -> Message rows (real ingestion)
             -> build_extraction_context() per user turn
             -> extract_memories()
             -> write_memories()
    query    -> search_memories()
             -> gold matching (evals.metrics)
             -> CaseResult

This module never reimplements extraction, writing, or retrieval; it only
calls the existing src.memory / src.retrieval APIs and scores what comes
back.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from src.config import configure, get_config, reset_config
from src.database import Conversation, Memory, Message, MessageRole, SessionLocal, User, create_tables, reset_engine
from src.memory import build_extraction_context, extract_memories, write_memories
from src.retrieval import search_memories

from .datasets import DATASETS
from .db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
from .metrics import aggregate_case_results, evaluate_retrieval, group_by_category
from .report import render_table, save_report
from .schemas import CaseResult, EvalCase, EvalMessage, RetrievedMemory, RunReport


class EvalEnvironmentError(RuntimeError):
    """Raised when required database or provider configuration is missing."""


def _check_provider_config() -> None:
    """Fail with one clear message naming every missing provider setting."""

    settings = get_config()
    missing = []
    if not settings.llm_model:
        missing.append("LLM_MODEL")
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY")
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if missing:
        raise EvalEnvironmentError(
            "Missing required provider configuration for the eval harness: " + ", ".join(missing) + "."
        )


def _ingest_conversation(
    db: Session,
    *,
    user_external_id: str,
    conversation_external_id: str,
    messages: list[EvalMessage],
) -> tuple[User, Conversation]:
    """Persist messages and run extraction+write for each user turn.

    Each user message is treated as its own one-message target interaction,
    with everything before it in the conversation available as extraction
    context. Assistant messages are persisted for that context but are never
    extraction targets themselves, since they cannot introduce new user facts.
    """

    user = User(external_id=user_external_id)
    conversation = Conversation(external_id=conversation_external_id, user=user)
    db.add_all([user, conversation])
    db.commit()

    base_time = datetime.now(timezone.utc)
    for index, message in enumerate(messages):
        row = Message(
            conversation_id=conversation.id,
            role=MessageRole(message.role),
            content=message.content,
            created_at=base_time + timedelta(seconds=index),
        )
        db.add(row)
        db.commit()

        if message.role != "user":
            continue

        context = build_extraction_context(
            db,
            user_external_id=user_external_id,
            conversation_external_id=conversation_external_id,
            target_message_ids=[str(row.id)],
        )
        candidates = extract_memories(
            [{"role": "user", "content": message.content}],
            source_message_ids=[str(row.id)],
            context=context,
        )
        write_memories(
            db,
            candidates,
            user_external_id=user_external_id,
            conversation_external_id=conversation_external_id,
        )

    return user, conversation


def run_case(db: Session, case: EvalCase, *, top_k: int) -> CaseResult:
    """Run one EvalCase end to end and score its retrieved memories."""

    run_suffix = uuid4().hex[:8]
    primary_user_id = f"eval-{case.id}-{run_suffix}"
    # Conversation external IDs only need to be unique per user (write_memories()
    # scopes that lookup by user_id), so every case can reuse the same "main"
    # conversation name for its primary user and, when present, its isolation
    # user without colliding.
    primary_user, _conversation = _ingest_conversation(
        db,
        user_external_id=primary_user_id,
        conversation_external_id="main",
        messages=case.messages,
    )

    if case.isolation_messages:
        _ingest_conversation(
            db,
            user_external_id=f"eval-{case.id}-isolation-{run_suffix}",
            conversation_external_id="main",
            messages=case.isolation_messages,
        )

    hits = search_memories(db, case.query, user_external_id=primary_user.external_id, limit=top_k)

    retrieved = [
        RetrievedMemory(memory_id=hit.memory_id, memory_text=hit.memory_text, is_active=hit.is_active, score=hit.score)
        for hit in hits
    ]
    metrics = evaluate_retrieval([hit.memory_text for hit in hits], case.expected_memories)
    # Prove the real invariant directly: every returned memory's durable row
    # must belong to the user that was passed to search_memories(), not just
    # to a set of IDs this harness happens to have written.
    isolation_failures = 0
    for hit in hits:
        memory = db.get(Memory, hit.memory_id)
        if memory is None or memory.user_id != primary_user.id:
            isolation_failures += 1
    superseded_returned = sum(1 for hit in hits if not hit.is_active)

    return CaseResult(
        case_id=case.id,
        category=case.category,
        query=case.query,
        retrieved=retrieved,
        metrics=metrics,
        isolation_failures=isolation_failures,
        superseded_returned=superseded_returned,
    )


def run_dataset(cases: list[EvalCase], *, top_k: int = 5) -> list[CaseResult]:
    """Run every case in one dataset against a fresh session per case."""

    results = []
    with SessionLocal() as db:
        for case in cases:
            results.append(run_case(db, case, top_k=top_k))
    return results


def build_report(results: list[CaseResult], *, dataset: str, top_k: int) -> RunReport:
    settings = get_config()
    return RunReport(
        run_at=datetime.now(timezone.utc),
        dataset=dataset,
        top_k=top_k,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        aggregate=aggregate_case_results(results),
        by_category=group_by_category(results),
        cases=results,
    )


def default_results_path(dataset: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parent / "results" / f"{dataset}-{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a memory-layer retrieval evaluation dataset.")
    parser.add_argument("--dataset", default="smoke", choices=sorted(DATASETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Path for the JSON run report.")
    args = parser.parse_args(argv)

    try:
        eval_database_url = get_eval_database_url()
        reset_config()
        reset_engine()
        _check_provider_config()
    except (EvalEnvironmentError, EvalDatabaseConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print_eval_database_banner(eval_database_url)
    configure(database_url=eval_database_url)
    create_tables()
    try:
        results = run_dataset(DATASETS[args.dataset], top_k=args.top_k)
        report = build_report(results, dataset=args.dataset, top_k=args.top_k)
    finally:
        reset_engine()
        reset_config()

    print(render_table(report))
    output_path = Path(args.output) if args.output else default_results_path(args.dataset)
    save_report(report, output_path)
    print(f"\nSaved JSON report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
