"""One-off eval script: WRITE-path A/B ablation (OLD_WRITE vs CURRENT_WRITE).

Not part of the maintained evals CLI. Composes only existing, unmodified
evals/locomo and meminfra public functions to answer one question: did the
recent write-side predicate/canonicalization change (controlled predicate
vocabulary + aliases in src/meminfra/memory/predicates.py and prompts.py)
cause the LoCoMo retrieval regression?

This script performs ONE ingestion phase per invocation (--phase old or
--phase current) into a uniquely-named user, sharing a --run-id across both
invocations so their outputs pair up. The orchestrating shell is responsible
for materializing OLD_WRITE / CURRENT_WRITE into predicates.py/prompts.py
before invoking this script and restoring CURRENT_WRITE afterward -- this
script does not touch those files itself.

Retrieval is evaluated with infer_query_intent=False for both phases
(QueryIntent is proven to make zero difference on this dataset and is kept
identical/off here so the only variable under test is the write path).

Usage:
    python -m evals.locomo.run_write_ablation --phase old --run-id abcd1234
    python -m evals.locomo.run_write_ablation --phase current --run-id abcd1234
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from meminfra.config import configure, get_config, reset_config
from meminfra.database import Memory, SessionLocal, User, create_tables, reset_engine
from meminfra.database.models import MemoryType
from meminfra.memory_layer import MemoryLayer

from .dataset import DEFAULT_DATASET_PATH, load_locomo_dataset
from .ingest import ingest_sample
from .metrics import aggregate_retrieval_metrics, evaluate_question_retrieval
from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner

CONV_INDEX = 1  # conv-30: 19 sessions, 369 turns, 105 QA
TOP_K = 5
EXPECTED_SESSIONS = 19
EXPECTED_TURNS = 369
EXPECTED_QA = 105


def _check_provider_config() -> list[str]:
    settings = get_config()
    missing = []
    if not settings.llm_model:
        missing.append("LLM_MODEL")
    if not settings.llm_api_key:
        missing.append("LLM_API_KEY")
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["old", "current"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    phase = args.phase
    run_id = args.run_id

    # ---- 1. resolve source ----
    import meminfra

    print(f"[{phase}] meminfra source path: {Path(meminfra.__file__).resolve().parent}")

    # ---- 2. verify dataset ----
    samples = load_locomo_dataset(DEFAULT_DATASET_PATH)
    if not (0 <= CONV_INDEX < len(samples)):
        print(f"INVALID RUN: conversation index {CONV_INDEX} out of range (dataset has {len(samples)} samples)")
        return 1
    sample = samples[CONV_INDEX]
    session_numbers = {turn.session_number for turn in sample.turns}
    n_sessions, n_turns, n_qa = len(session_numbers), len(sample.turns), len(sample.qa)
    print(f"[{phase}] Dataset check: sample_id={sample.sample_id} sessions={n_sessions} turns={n_turns} qa={n_qa}")
    if (n_sessions, n_turns, n_qa) != (EXPECTED_SESSIONS, EXPECTED_TURNS, EXPECTED_QA):
        print(
            "INVALID RUN: dataset counts do not match expected "
            f"(sessions={EXPECTED_SESSIONS}, turns={EXPECTED_TURNS}, qa={EXPECTED_QA}); stopping."
        )
        return 1

    # ---- 3. eval database safety ----
    try:
        eval_database_url = get_eval_database_url()
    except EvalDatabaseConfigError as error:
        print(f"INVALID RUN: {error}")
        return 1
    reset_config()
    reset_engine()
    missing = _check_provider_config()
    if missing:
        print(f"INVALID RUN: missing provider configuration: {', '.join(missing)}")
        return 1
    print_eval_database_banner(eval_database_url)
    configure(database_url=eval_database_url)
    create_tables()

    user_id = f"locomo-write-ablation-{phase}-{sample.sample_id}-{run_id}"
    conversation_id = "conversation"

    # ---- 4/5/6. fresh ingestion (fresh FAISS follows automatically: new user => new index on first search) ----
    write_actions: Counter[str] = Counter()

    def _on_session_complete(session_number: int, result) -> None:
        for wr in result.write_results:
            write_actions[wr.action.value] += 1
        print(f"  [{phase}] session {session_number}/{n_sessions} ingested, {len(result.write_results)} write result(s)")

    def _on_session_failed(session_number: int, warning: str) -> None:
        print(f"  [{phase}] session {session_number}/{n_sessions} FAILED: {warning}")

    print(f"\n[{phase}] Ingesting {sample.sample_id} as user {user_id!r} ...")
    ingest_started = time.monotonic()
    with SessionLocal() as db:
        outcome = ingest_sample(
            db,
            sample,
            user_id=user_id,
            conversation_id=conversation_id,
            on_session_complete=_on_session_complete,
            on_session_failed=_on_session_failed,
        )
    ingest_elapsed = time.monotonic() - ingest_started

    print(f"\n[{phase}] Ingestion stats (elapsed {ingest_elapsed:.0f}s):")
    print(f"  turns processed: {n_turns}")
    print(f"  messages persisted: {outcome.messages_persisted}")
    print(f"  sessions ingested: {outcome.sessions_ingested}   sessions failed: {outcome.sessions_failed}")
    print(f"  memories written (total write results): {sum(write_actions.values())}")
    print(f"    ADD: {write_actions.get('ADD', 0)}  NOOP: {write_actions.get('NOOP', 0)}  SUPERSEDE: {write_actions.get('SUPERSEDE', 0)}")
    if outcome.warnings:
        print(f"  extraction/write warnings ({len(outcome.warnings)}):")
        for w in outcome.warnings:
            print(f"    - {w}")
    else:
        print("  extraction/write warnings: none")

    # ---- 7. structured memory sanity check + full memory dump ----
    with SessionLocal() as db:
        user_row = db.scalar(select(User).where(User.external_id == user_id))
        assert user_row is not None
        total_memories = db.scalar(select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id)) or 0
        non_null_predicate = db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id, Memory.predicate.is_not(None))
        ) or 0
        non_null_fact_key = db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id, Memory.fact_key.is_not(None))
        ) or 0
        semantic_count = db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id, Memory.memory_type == MemoryType.SEMANTIC)
        ) or 0
        episodic_count = db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id, Memory.memory_type == MemoryType.EPISODIC)
        ) or 0
        open_semantic = db.scalar(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.user_id == user_row.id,
                Memory.memory_type == MemoryType.SEMANTIC,
                Memory.fact_key.is_(None),
                Memory.is_active.is_(True),
            )
        ) or 0
        predicate_sample = list(
            db.scalars(
                select(Memory.predicate)
                .where(Memory.user_id == user_row.id, Memory.predicate.is_not(None))
                .distinct()
                .limit(50)
            )
        )
        inactive_count = db.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user_row.id, Memory.is_active.is_(False))
        ) or 0

        all_rows = list(db.scalars(select(Memory).where(Memory.user_id == user_row.id)))
        memory_dump = []
        for row in all_rows:
            prov_dia_ids = sorted(
                {outcome.message_id_to_dia_id[mid] for mid in row.source_message_ids if mid in outcome.message_id_to_dia_id}
            )
            memory_dump.append(
                {
                    "id": str(row.id),
                    "memory_type": row.memory_type.value if hasattr(row.memory_type, "value") else str(row.memory_type),
                    "memory_text": row.memory_text,
                    "subject_type": row.subject_type,
                    "predicate": row.predicate,
                    "value": row.value,
                    "fact_key": row.fact_key,
                    "is_active": row.is_active,
                    "superseded_by_id": str(row.superseded_by_id) if row.superseded_by_id else None,
                    "source_message_ids": list(row.source_message_ids),
                    "source_dia_ids": prov_dia_ids,
                }
            )

    print(f"\n[{phase}] Structured memory sanity check:")
    print(f"  total memories: {total_memories}")
    print(f"  non-null predicate: {non_null_predicate}")
    print(f"  non-null fact_key: {non_null_fact_key}")
    print(f"  semantic: {semantic_count}   episodic: {episodic_count}")
    print(f"  open semantic (no fact_key, active): {open_semantic}")
    print(f"  inactive (superseded/expired) count: {inactive_count}")
    print(f"  distinct predicates observed (sample): {sorted(predicate_sample)}")

    # ---- 8. retrieval evaluation, infer_query_intent=False, k=5, no .answer() ----
    diagnostics = []
    per_question = []
    with SessionLocal() as db:
        memory = MemoryLayer(db)
        primary_user = db.scalar(select(User).where(User.external_id == user_id))
        assert primary_user is not None

        print(f"\n[{phase}] Evaluating {n_qa} QA at k={TOP_K}, infer_query_intent=False ...")
        for qi, qa in enumerate(sample.qa):
            hits = memory.search(user_id=user_id, query=qa.question, limit=TOP_K, infer_query_intent=False)
            provenance_by_rank = []
            retrieved_ids = []
            retrieved_texts = []
            retrieved_provenance = []
            for hit in hits:
                row = db.get(Memory, hit.memory_id)
                retrieved_ids.append(str(hit.memory_id))
                if row is None or row.user_id != primary_user.id:
                    provenance_by_rank.append(set())
                    retrieved_texts.append(None)
                    retrieved_provenance.append([])
                    continue
                prov = {outcome.message_id_to_dia_id[mid] for mid in row.source_message_ids if mid in outcome.message_id_to_dia_id}
                provenance_by_rank.append(prov)
                retrieved_texts.append(row.memory_text)
                retrieved_provenance.append(sorted(prov))
            metrics = evaluate_question_retrieval(provenance_by_rank, qa.evidence)
            diagnostics.append(metrics)
            per_question.append(
                {
                    "question_index": qi,
                    "question": qa.question,
                    "category_id": qa.category_id,
                    "gold_evidence": list(qa.evidence),
                    "retrieved_memory_ids": retrieved_ids,
                    "retrieved_memory_texts": retrieved_texts,
                    "retrieved_provenance_dia_ids": retrieved_provenance,
                    "rank": metrics.rank,
                    "hit_at_5": metrics.hit_at_5,
                    "recall_at_5": metrics.recall_at_5,
                    "reciprocal_rank": metrics.reciprocal_rank,
                    "has_evidence": metrics.has_evidence,
                }
            )

    class _Diag:
        def __init__(self, m):
            self.retrieval_metrics = m
            self.isolation_failures = 0

    agg = aggregate_retrieval_metrics([_Diag(m) for m in diagnostics])

    print("\n" + "=" * 70)
    print(f"WRITE_{phase.upper()} (QueryIntent OFF)")
    print("=" * 70)
    print(f"  questions evaluated: {agg.questions_evaluated}  excluded (no evidence): {agg.questions_excluded_no_evidence}")
    print(f"  Hit@5    = {agg.hit_at_5:.3f}")
    print(f"  Recall@5 = {agg.recall_at_5:.3f}")
    print(f"  MRR      = {agg.mrr:.3f}")

    # ---- save raw output ----
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"locomo-write-ablation-{phase}-{run_id}.json"
    payload = {
        "run_id": run_id,
        "phase": phase,
        "sample_id": sample.sample_id,
        "user_id": user_id,
        "dataset_counts": {"sessions": n_sessions, "turns": n_turns, "qa": n_qa},
        "ingestion": {
            "messages_persisted": outcome.messages_persisted,
            "sessions_ingested": outcome.sessions_ingested,
            "sessions_failed": outcome.sessions_failed,
            "write_actions": dict(write_actions),
            "warnings": outcome.warnings,
            "elapsed_seconds": ingest_elapsed,
        },
        "structured_sanity": {
            "total_memories": total_memories,
            "non_null_predicate": non_null_predicate,
            "non_null_fact_key": non_null_fact_key,
            "semantic": semantic_count,
            "episodic": episodic_count,
            "open_semantic": open_semantic,
            "inactive": inactive_count,
            "predicates_sample": sorted(predicate_sample),
        },
        "retrieval_aggregate": {
            "hit_at_5": agg.hit_at_5,
            "recall_at_5": agg.recall_at_5,
            "mrr": agg.mrr,
            "questions_evaluated": agg.questions_evaluated,
            "questions_excluded_no_evidence": agg.questions_excluded_no_evidence,
        },
        "memory_dump": memory_dump,
        "per_question": per_question,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[{phase}] Saved JSON report to {out_path}")

    reset_engine()
    reset_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
