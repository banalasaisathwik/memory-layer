"""One-off eval script: QueryIntent A/B ablation over a single fresh ingestion.

Not part of the maintained evals CLI. Composes only existing, unmodified
evals/locomo and meminfra public functions to answer one question: given the
CURRENT write-path extraction (controlled-predicate registry), how much does
enabling QueryIntent-driven structured retrieval (search_memories(...,
infer_query_intent=True)) change Hit@5 / Recall@5 / MRR versus leaving it
disabled -- against the *same* freshly ingested LoCoMo conv-30 memory state
and the *same* FAISS index for both runs.

Usage:
    python -m evals.locomo.run_query_intent_ablation
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select

from meminfra.config import configure, get_config, reset_config
from meminfra.database import Memory, SessionLocal, User, create_tables, reset_engine
from meminfra.database.models import MemoryType
from meminfra.memory_layer import MemoryLayer

import meminfra.retrieval.search as search_mod
from meminfra.retrieval.query_intent import QueryIntentError

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
    # ---- 1. resolve source ----
    import meminfra

    print(f"meminfra source path: {Path(meminfra.__file__).resolve().parent}")
    try:
        import importlib.metadata as importlib_metadata

        version = importlib_metadata.version("meminfra")
        print(f"meminfra installed version metadata: {version}")
    except Exception:
        print("meminfra installed version metadata: not found (editable/local source, pyproject.toml declares 0.2.0)")

    # ---- 2. verify dataset ----
    samples = load_locomo_dataset(DEFAULT_DATASET_PATH)
    if not (0 <= CONV_INDEX < len(samples)):
        print(f"INVALID RUN: conversation index {CONV_INDEX} out of range (dataset has {len(samples)} samples)")
        return 1
    sample = samples[CONV_INDEX]
    session_numbers = {turn.session_number for turn in sample.turns}
    n_sessions, n_turns, n_qa = len(session_numbers), len(sample.turns), len(sample.qa)
    print(f"Dataset check: sample_id={sample.sample_id} sessions={n_sessions} turns={n_turns} qa={n_qa}")
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

    run_id = uuid4().hex[:8]
    user_id = f"locomo-qi-ablation-{sample.sample_id}-{run_id}"
    conversation_id = "conversation"

    # ---- 4/5/6. fresh ingestion (fresh FAISS follows automatically: new user => new index on first search) ----
    write_actions: Counter[str] = Counter()

    def _on_session_complete(session_number: int, result) -> None:
        for wr in result.write_results:
            write_actions[wr.action.value] += 1
        print(f"  session {session_number}/{n_sessions} ingested, {len(result.write_results)} write result(s)")

    def _on_session_failed(session_number: int, warning: str) -> None:
        print(f"  session {session_number}/{n_sessions} FAILED: {warning}")

    print(f"\nIngesting {sample.sample_id} as user {user_id!r} ...")
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

    print(f"\nIngestion stats (elapsed {ingest_elapsed:.0f}s):")
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

    # ---- 7. structured memory sanity check ----
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
                .limit(20)
            )
        )

    print(f"\nStructured memory sanity check:")
    print(f"  total memories: {total_memories}")
    print(f"  non-null predicate: {non_null_predicate}")
    print(f"  non-null fact_key: {non_null_fact_key}")
    print(f"  semantic: {semantic_count}   episodic: {episodic_count}")
    print(f"  open semantic (no fact_key, active): {open_semantic}")
    print(f"  distinct predicates observed (sample): {sorted(predicate_sample)}")

    # ---- instrumentation for QueryIntent activation metrics (Run B only) ----
    intent_log: list[dict] = []
    structured_candidate_counts: list[int] = []

    _original_extract = search_mod.extract_query_intent
    _original_structured_from_intent = search_mod.structured_retrieve_from_intent

    def _instrumented_extract(query: str):
        try:
            intent = _original_extract(query)
            intent_log.append(
                {
                    "query": query,
                    "ok": True,
                    "predicate": intent.predicate,
                    "value": intent.value,
                    "temporal_scope": intent.temporal_scope,
                    "error_category": None,
                }
            )
            return intent
        except QueryIntentError as error:
            intent_log.append(
                {
                    "query": query,
                    "ok": False,
                    "predicate": None,
                    "value": None,
                    "temporal_scope": None,
                    "error_category": error.category,
                }
            )
            raise

    def _instrumented_structured_from_intent(*args, **kwargs):
        result = _original_structured_from_intent(*args, **kwargs)
        structured_candidate_counts.append(len(result))
        return result

    # ---- 8/9. Run A (intent disabled) then Run B (intent enabled), same memory state ----
    diagnostics_a = []
    diagnostics_b = []
    b_activation_rows = []

    with SessionLocal() as db:
        memory = MemoryLayer(db)
        primary_user = db.scalar(select(User).where(User.external_id == user_id))
        assert primary_user is not None

        print(f"\nRun A (QueryIntent disabled): evaluating {n_qa} QA at k={TOP_K} ...")
        for qi, qa in enumerate(sample.qa):
            hits = memory.search(user_id=user_id, query=qa.question, limit=TOP_K, infer_query_intent=False)
            provenance_by_rank = []
            for hit in hits:
                row = db.get(Memory, hit.memory_id)
                if row is None or row.user_id != primary_user.id:
                    provenance_by_rank.append(set())
                    continue
                provenance_by_rank.append(
                    {outcome.message_id_to_dia_id[mid] for mid in row.source_message_ids if mid in outcome.message_id_to_dia_id}
                )
            diagnostics_a.append(evaluate_question_retrieval(provenance_by_rank, qa.evidence))

        print(f"Run B (QueryIntent enabled): evaluating {n_qa} QA at k={TOP_K} ...")
        search_mod.extract_query_intent = _instrumented_extract
        search_mod.structured_retrieve_from_intent = _instrumented_structured_from_intent
        try:
            for qi, qa in enumerate(sample.qa):
                pre_intent_len = len(intent_log)
                pre_struct_len = len(structured_candidate_counts)
                hits = memory.search(user_id=user_id, query=qa.question, limit=TOP_K, infer_query_intent=True)
                provenance_by_rank = []
                structured_hit_matches_gold = False
                gold = set(qa.evidence)
                for hit in hits:
                    row = db.get(Memory, hit.memory_id)
                    if row is None or row.user_id != primary_user.id:
                        provenance_by_rank.append(set())
                        continue
                    prov = {outcome.message_id_to_dia_id[mid] for mid in row.source_message_ids if mid in outcome.message_id_to_dia_id}
                    provenance_by_rank.append(prov)
                    if hit.structured_rank is not None and (prov & gold):
                        structured_hit_matches_gold = True
                diagnostics_b.append(evaluate_question_retrieval(provenance_by_rank, qa.evidence))

                new_intent_entries = intent_log[pre_intent_len:]
                new_struct_counts = structured_candidate_counts[pre_struct_len:]
                b_activation_rows.append(
                    {
                        "intent_attempted": len(new_intent_entries) > 0,
                        "intent_ok": bool(new_intent_entries) and new_intent_entries[-1]["ok"],
                        "predicate": new_intent_entries[-1]["predicate"] if new_intent_entries else None,
                        "value": new_intent_entries[-1]["value"] if new_intent_entries else None,
                        "temporal_scope": new_intent_entries[-1]["temporal_scope"] if new_intent_entries else None,
                        "error_category": new_intent_entries[-1]["error_category"] if new_intent_entries else None,
                        "structured_candidates": new_struct_counts[-1] if new_struct_counts else 0,
                        "structured_hit_matches_gold": structured_hit_matches_gold,
                    }
                )
        finally:
            search_mod.extract_query_intent = _original_extract
            search_mod.structured_retrieve_from_intent = _original_structured_from_intent

    # ---- results ----
    class _Diag:
        def __init__(self, m):
            self.retrieval_metrics = m
            self.isolation_failures = 0

    agg_a = aggregate_retrieval_metrics([_Diag(m) for m in diagnostics_a])
    agg_b = aggregate_retrieval_metrics([_Diag(m) for m in diagnostics_b])

    print("\n" + "=" * 70)
    print("CURRENT_WRITE_INTENT_OFF (Run A)")
    print("=" * 70)
    print(f"  questions evaluated: {agg_a.questions_evaluated}  excluded (no evidence): {agg_a.questions_excluded_no_evidence}")
    print(f"  Hit@5    = {agg_a.hit_at_5:.3f}")
    print(f"  Recall@5 = {agg_a.recall_at_5:.3f}")
    print(f"  MRR      = {agg_a.mrr:.3f}")

    print("\n" + "=" * 70)
    print("CURRENT_WRITE_INTENT_ON (Run B)")
    print("=" * 70)
    print(f"  questions evaluated: {agg_b.questions_evaluated}  excluded (no evidence): {agg_b.questions_excluded_no_evidence}")
    print(f"  Hit@5    = {agg_b.hit_at_5:.3f}")
    print(f"  Recall@5 = {agg_b.recall_at_5:.3f}")
    print(f"  MRR      = {agg_b.mrr:.3f}")

    # ---- 11. activation metrics ----
    attempted = sum(1 for r in b_activation_rows if r["intent_attempted"])
    successful = sum(1 for r in b_activation_rows if r["intent_ok"])
    predicate_non_null = sum(1 for r in b_activation_rows if r["predicate"])
    value_non_null = sum(1 for r in b_activation_rows if r["value"])
    scope_current = sum(1 for r in b_activation_rows if r["temporal_scope"] == "current")
    scope_historical = sum(1 for r in b_activation_rows if r["temporal_scope"] == "historical")
    scope_any = sum(1 for r in b_activation_rows if r["temporal_scope"] == "any")
    structured_ge1 = sum(1 for r in b_activation_rows if r["structured_candidates"] >= 1)
    structured_top5_correct = sum(1 for r in b_activation_rows if r["structured_hit_matches_gold"])
    fallback_count = sum(1 for r in b_activation_rows if r["intent_attempted"] and not r["intent_ok"])
    error_categories = Counter(r["error_category"] for r in b_activation_rows if r["error_category"])

    print("\n" + "=" * 70)
    print("QueryIntent activation metrics (Run B, total QA = %d)" % n_qa)
    print("=" * 70)
    print(f"  intent extraction attempted: {attempted}")
    print(f"  intent extraction successful: {successful}")
    print(f"  predicate non-null: {predicate_non_null}")
    print(f"  value non-null: {value_non_null}")
    print(f"  temporal_scope=current: {scope_current}")
    print(f"  temporal_scope=historical: {scope_historical}")
    print(f"  temporal_scope=any: {scope_any}")
    print(f"  structured branch returned >=1 candidate: {structured_ge1}")
    print(f"  structured branch returned the eventual top-5 relevant memory: {structured_top5_correct}")
    print(f"  structured_activation_rate = {structured_ge1}/{n_qa} = {structured_ge1 / n_qa:.3f}")

    print("\n" + "=" * 70)
    print("QueryIntent failures (Run B)")
    print("=" * 70)
    print(f"  fallback count (attempted but not ok -> continued without structured-intent branch): {fallback_count}")
    print(f"  failure categories: {dict(error_categories)}")
    print("  note: internal per-attempt repair-retries inside extract_query_intent (up to 3 per call)")
    print("        are not separately observable without instrumenting inside meminfra source; only the")
    print("        final per-question success/failure outcome is reported above.")

    # ---- save raw output ----
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"locomo-qi-ablation-{run_id}.json"
    payload = {
        "run_id": run_id,
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
            "predicates_sample": sorted(predicate_sample),
        },
        "run_a_intent_off": {
            "hit_at_5": agg_a.hit_at_5,
            "recall_at_5": agg_a.recall_at_5,
            "mrr": agg_a.mrr,
            "questions_evaluated": agg_a.questions_evaluated,
            "questions_excluded_no_evidence": agg_a.questions_excluded_no_evidence,
        },
        "run_b_intent_on": {
            "hit_at_5": agg_b.hit_at_5,
            "recall_at_5": agg_b.recall_at_5,
            "mrr": agg_b.mrr,
            "questions_evaluated": agg_b.questions_evaluated,
            "questions_excluded_no_evidence": agg_b.questions_excluded_no_evidence,
        },
        "activation_metrics": {
            "total_qa": n_qa,
            "intent_attempted": attempted,
            "intent_successful": successful,
            "predicate_non_null": predicate_non_null,
            "value_non_null": value_non_null,
            "temporal_scope_current": scope_current,
            "temporal_scope_historical": scope_historical,
            "temporal_scope_any": scope_any,
            "structured_branch_ge1_candidate": structured_ge1,
            "structured_branch_top5_correct": structured_top5_correct,
            "structured_activation_rate": structured_ge1 / n_qa,
            "fallback_count": fallback_count,
            "error_categories": dict(error_categories),
        },
        "per_question_activation": b_activation_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved JSON report to {out_path}")

    reset_engine()
    reset_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
