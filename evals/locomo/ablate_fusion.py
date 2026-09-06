"""Fusion-policy ablation over an already-ingested, frozen LoCoMo sample.

This is a retrieval-only ablation, exactly like ``ablate_lexical.py``: it
never ingests anything, never calls ``MemoryLayer.add()``, never inserts a
``User``/``Conversation``/``Message``/``Memory`` row, and only reads existing
rows. The one thing it varies is the *fusion* step -- how the structured,
lexical (BM25), and vector branch rankings are combined into one final
ordering -- never candidate generation, candidate limits, BM25 parameters,
the embedding model, or top-K.

For every question, the structured/lexical/vector candidate branches are
computed exactly once (shared-branch pattern from ``ablate_lexical.py`` /
``benchmark_state.py``), then every fusion variant below is evaluated purely
in memory against those same three candidate lists:

    A. BM25 only            -- rank order of the BM25 branch alone
    B. Vector only           -- rank order of the vector branch alone
    C. Current equal RRF     -- production ``reciprocal_rank_fusion`` (unchanged)
    D/E/F. Weighted RRF      -- BM25 weight 0.25 / 0.50 / 0.75, vector anchored at 1.0
    G/H/I. Discounted agreement -- lambda 0.10 / 0.25 / 0.50

Weighted RRF at bm25_weight=1.0 is mathematically identical to Current equal
RRF (structured is a no-op branch for this benchmark, weighted at 1.0), so it
is intentionally not included as a separate variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import Conversation, ConversationSummary, Memory, Message, User
from src.retrieval.fusion import RRF_K, discounted_agreement_fusion, reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from src.retrieval.search import _BRANCH_CANDIDATE_MULTIPLIER, _MAX_SEARCH_LIMIT
from src.retrieval.structured import structured_retrieve
from src.retrieval.schemas import SearchFilters

from .ablate_lexical import (
    _diagnostic_from_fused,
    _noop,
    _rank_of_gold,
    _resolve_embedding_dimension,
    _vector_candidates_from_embedding,
    embed_question_batch,
    snapshot_db_counts,
)
from .benchmark_state import BenchmarkState, load_benchmark_state
from .diagnose import classify_stage
from .embedding_cache import EmbeddingCacheContext, QueryEmbeddingCache
from .instrumentation import DbNetworkCounters, Timings
from .metrics import aggregate_retrieval_metrics, group_retrieval_by_category
from .schemas import LocomoRunMetadata, LocomoRunReport, LocomoSample, QuestionDiagnostic

STAGE_KEYS = ("SUCCESS", "NO_GOLD_MEMORY", "RETRIEVAL_MISS", "RANKING_MISS")

VariantKind = Literal["single_branch", "fusion"]


@dataclass(frozen=True)
class FusionVariant:
    """One fusion configuration under ablation, self-describing for the report."""

    label: str
    kind: VariantKind
    strategy: str | None = None
    branch: str | None = None
    bm25_weight: float | None = None
    lambda_: float | None = None

    def describe(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "strategy": self.strategy,
            "branch": self.branch,
            "bm25_weight": self.bm25_weight,
            "lambda": self.lambda_,
        }


VARIANTS: tuple[FusionVariant, ...] = (
    FusionVariant(label="bm25_only", kind="single_branch", branch="bm25"),
    FusionVariant(label="vector_only", kind="single_branch", branch="vector"),
    FusionVariant(label="current_rrf", kind="fusion", strategy="rrf"),
    FusionVariant(label="weighted_rrf_bm25_0.25", kind="fusion", strategy="weighted_rrf", bm25_weight=0.25),
    FusionVariant(label="weighted_rrf_bm25_0.50", kind="fusion", strategy="weighted_rrf", bm25_weight=0.50),
    FusionVariant(label="weighted_rrf_bm25_0.75", kind="fusion", strategy="weighted_rrf", bm25_weight=0.75),
    FusionVariant(label="discounted_agreement_lambda_0.10", kind="fusion", strategy="discounted_agreement", lambda_=0.10),
    FusionVariant(label="discounted_agreement_lambda_0.25", kind="fusion", strategy="discounted_agreement", lambda_=0.25),
    FusionVariant(label="discounted_agreement_lambda_0.50", kind="fusion", strategy="discounted_agreement", lambda_=0.50),
)

BASELINE_LABEL = "current_rrf"
KNOWN_FAILURE_STAGES = ("RETRIEVAL_MISS", "RANKING_MISS")


# --------------------------------------------------------------------------
# Full-database invariant snapshot (Part 22): users, conversations, messages,
# memories, conversation_summaries -- must be byte-identical before/after.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FullDbInvariants:
    users: int
    conversations: int
    messages: int
    memories: int
    conversation_summaries: int

    def to_dict(self) -> dict:
        return {
            "users": self.users,
            "conversations": self.conversations,
            "messages": self.messages,
            "memories": self.memories,
            "conversation_summaries": self.conversation_summaries,
        }


def snapshot_full_db_invariants(db: Session) -> FullDbInvariants:
    return FullDbInvariants(
        users=int(db.scalar(select(func.count()).select_from(User)) or 0),
        conversations=int(db.scalar(select(func.count()).select_from(Conversation)) or 0),
        messages=int(db.scalar(select(func.count()).select_from(Message)) or 0),
        memories=int(db.scalar(select(func.count()).select_from(Memory)) or 0),
        conversation_summaries=int(db.scalar(select(func.count()).select_from(ConversationSummary)) or 0),
    )


# --------------------------------------------------------------------------
# Shared-branch retrieval, fused once per variant, purely in memory.
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Every candidate branch and every variant's fused outcome for one question."""

    question_index: int
    question: str
    structured_candidate_ids: list[str]
    vector_candidate_ids: list[str]
    bm25_candidate_ids: list[str]
    variant_fused_ids: dict[str, list[str]]
    variant_diagnostics: dict[str, QuestionDiagnostic]


def _fuse_variant(
    variant: FusionVariant,
    *,
    structured_candidates: list[Memory],
    bm25_candidates: list[Memory],
    vector_candidates: list[Memory],
    top_k: int,
) -> list[Memory]:
    if variant.kind == "single_branch":
        source = bm25_candidates if variant.branch == "bm25" else vector_candidates
        return source[:top_k]
    if variant.strategy == "rrf":
        fused = reciprocal_rank_fusion(structured=structured_candidates, lexical=bm25_candidates, vector=vector_candidates, k=RRF_K)
    elif variant.strategy == "weighted_rrf":
        fused = weighted_reciprocal_rank_fusion(
            structured=structured_candidates,
            lexical=bm25_candidates,
            vector=vector_candidates,
            k=RRF_K,
            bm25_weight=variant.bm25_weight,
        )
    elif variant.strategy == "discounted_agreement":
        fused = discounted_agreement_fusion(
            structured=structured_candidates,
            lexical=bm25_candidates,
            vector=vector_candidates,
            k=RRF_K,
            lambda_=variant.lambda_,
        )
    else:
        raise ValueError(f"Unknown variant strategy: {variant.strategy!r}")
    return [hit.memory for hit in fused][:top_k]


def evaluate_question_all_variants(
    db: Session,
    *,
    sample: LocomoSample,
    question_index: int,
    state: BenchmarkState,
    top_k: int,
    query_vector: np.ndarray,
    timings: Timings | None = None,
    counters: DbNetworkCounters | None = None,
) -> QuestionResult:
    qa = sample.qa[question_index]
    filters = SearchFilters()
    branch_limit = min(_MAX_SEARCH_LIMIT, top_k * _BRANCH_CANDIDATE_MULTIPLIER)

    with (timings.measure("structured_branch") if timings else _noop()):
        structured_candidates = structured_retrieve(
            db, user=state.user, filters=filters, conversation=None, limit=branch_limit
        )

    with (timings.measure("vector_branch") if timings else _noop()):
        vector_candidates = _vector_candidates_from_embedding(db, query_vector, user=state.user, limit=top_k)
    if counters is not None:
        counters.vector_searches += 1

    with (timings.measure("bm25_branch") if timings else _noop()):
        bm25_candidates = [hit.item for hit in state.bm25_corpus.score(qa.question)[:branch_limit]]

    variant_fused_ids: dict[str, list[str]] = {}
    variant_diagnostics: dict[str, QuestionDiagnostic] = {}
    with (timings.measure("fusion_all_variants") if timings else _noop()):
        for variant in VARIANTS:
            fused_memories = _fuse_variant(
                variant,
                structured_candidates=structured_candidates,
                bm25_candidates=bm25_candidates,
                vector_candidates=vector_candidates,
                top_k=top_k,
            )
            if counters is not None:
                counters.fusion_computations += 1
            variant_fused_ids[variant.label] = [str(m.id) for m in fused_memories]
            variant_diagnostics[variant.label] = _diagnostic_from_fused(
                sample=sample,
                question_index=question_index,
                fused_memories=fused_memories,
                primary_user_db_id=state.user.id,
                message_id_to_dia_id=state.message_id_to_dia_id,
            )

    return QuestionResult(
        question_index=question_index,
        question=qa.question,
        structured_candidate_ids=[str(m.id) for m in structured_candidates],
        vector_candidate_ids=[str(m.id) for m in vector_candidates],
        bm25_candidate_ids=[str(m.id) for m in bm25_candidates],
        variant_fused_ids=variant_fused_ids,
        variant_diagnostics=variant_diagnostics,
    )


def run_fusion_ablation(
    db: Session,
    sample: LocomoSample,
    *,
    state: BenchmarkState,
    top_k: int,
    embedding_cache: QueryEmbeddingCache,
    timings: Timings | None = None,
    counters: DbNetworkCounters | None = None,
) -> list[QuestionResult]:
    questions = [qa.question for qa in sample.qa]

    with (timings.measure("query_embedding_cache_load") if timings else _noop()):
        vectors_by_question = embedding_cache.ensure_batch(questions, embed_many=embed_question_batch)
    if counters is not None:
        counters.query_embedding_provider_calls += embedding_cache.stats.provider_batch_calls
        counters.query_embedding_cache_hits += embedding_cache.stats.hits

    results: list[QuestionResult] = []
    for question_index in range(len(sample.qa)):
        query_vector = vectors_by_question[sample.qa[question_index].question]
        result = evaluate_question_all_variants(
            db,
            sample=sample,
            question_index=question_index,
            state=state,
            top_k=top_k,
            query_vector=query_vector,
            timings=timings,
            counters=counters,
        )
        results.append(result)
    return results


# --------------------------------------------------------------------------
# Aggregation, stage counts, net-impact / regression analysis
# --------------------------------------------------------------------------


def build_report(
    diagnostics: list[QuestionDiagnostic],
    *,
    top_k: int,
    variant_label: str,
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
        run_id=f"fusion-ablation-{variant_label}",
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


def stage_by_question(diagnostics: list[QuestionDiagnostic], *, covered_dia_ids: set[str], k: int = 5) -> dict[int, str | None]:
    return {d.question_index: classify_stage(d, covered_dia_ids=covered_dia_ids, k=k) for d in diagnostics}


def stage_counts(stages: dict[int, str | None]) -> dict[str, int]:
    counts = {key: 0 for key in STAGE_KEYS}
    for outcome in stages.values():
        if outcome is not None:
            counts[outcome] += 1
    return counts


@dataclass(frozen=True)
class NetImpact:
    failures_fixed: int
    successes_lost: int
    net_success_gain: int
    rank_improved: int
    rank_worsened: int
    rank_unchanged: int

    def to_dict(self) -> dict:
        return {
            "failures_fixed": self.failures_fixed,
            "successes_lost": self.successes_lost,
            "net_success_gain": self.net_success_gain,
            "rank_improved": self.rank_improved,
            "rank_worsened": self.rank_worsened,
            "rank_unchanged": self.rank_unchanged,
        }


def compute_net_impact(
    *,
    baseline_stages: dict[int, str | None],
    variant_stages: dict[int, str | None],
    baseline_diagnostics: dict[int, QuestionDiagnostic],
    variant_diagnostics: dict[int, QuestionDiagnostic],
) -> NetImpact:
    failures_fixed = 0
    successes_lost = 0
    rank_improved = 0
    rank_worsened = 0
    rank_unchanged = 0
    for index, before in baseline_stages.items():
        after = variant_stages.get(index)
        if before is None or after is None:
            continue
        if before in KNOWN_FAILURE_STAGES and after == "SUCCESS":
            failures_fixed += 1
        if before == "SUCCESS" and after != "SUCCESS":
            successes_lost += 1

        before_rank = baseline_diagnostics[index].retrieval_metrics.rank if baseline_diagnostics[index].retrieval_metrics else None
        after_rank = variant_diagnostics[index].retrieval_metrics.rank if variant_diagnostics[index].retrieval_metrics else None
        if before_rank is None and after_rank is None:
            rank_unchanged += 1
        elif after_rank is None:
            rank_worsened += 1
        elif before_rank is None:
            rank_improved += 1
        elif after_rank < before_rank:
            rank_improved += 1
        elif after_rank > before_rank:
            rank_worsened += 1
        else:
            rank_unchanged += 1
    return NetImpact(
        failures_fixed=failures_fixed,
        successes_lost=successes_lost,
        net_success_gain=failures_fixed - successes_lost,
        rank_improved=rank_improved,
        rank_worsened=rank_worsened,
        rank_unchanged=rank_unchanged,
    )


@dataclass(frozen=True)
class SingleBranchGoldCase:
    question_index: int
    kind: Literal["vector_only", "bm25_only"]
    bm25_rank: int | None
    vector_rank: int | None

    def to_dict(self) -> dict:
        return {"question_index": self.question_index, "kind": self.kind, "bm25_rank": self.bm25_rank, "vector_rank": self.vector_rank}


def find_single_branch_gold_cases(
    results: list[QuestionResult],
    *,
    sample: LocomoSample,
    gold_lookup,
) -> list[SingleBranchGoldCase]:
    """Questions whose gold-linked memory appears in exactly one branch's candidates."""

    cases: list[SingleBranchGoldCase] = []
    for result in results:
        qa = sample.qa[result.question_index]
        gold = set(qa.evidence)
        if not gold:
            continue
        gold_memory_ids = {mid for mid, dia_ids in gold_lookup.dia_ids_by_memory_id.items() if dia_ids & gold}
        if not gold_memory_ids:
            continue
        bm25_rank = _rank_of_gold(result.bm25_candidate_ids, gold_memory_ids)
        vector_rank = _rank_of_gold(result.vector_candidate_ids, gold_memory_ids)
        if vector_rank is not None and bm25_rank is None:
            cases.append(SingleBranchGoldCase(result.question_index, "vector_only", bm25_rank, vector_rank))
        elif bm25_rank is not None and vector_rank is None:
            cases.append(SingleBranchGoldCase(result.question_index, "bm25_only", bm25_rank, vector_rank))
    return cases


def single_branch_gold_behavior(
    cases: list[SingleBranchGoldCase],
    *,
    kind: Literal["vector_only", "bm25_only"],
    baseline_diagnostics: dict[int, QuestionDiagnostic],
    variant_diagnostics: dict[int, QuestionDiagnostic],
) -> dict:
    subset = [c for c in cases if c.kind == kind]
    retained_top5 = 0
    promoted = 0
    demoted = 0
    unchanged = 0
    for case in subset:
        baseline_rank = baseline_diagnostics[case.question_index].retrieval_metrics.rank if baseline_diagnostics[case.question_index].retrieval_metrics else None
        variant_rank = variant_diagnostics[case.question_index].retrieval_metrics.rank if variant_diagnostics[case.question_index].retrieval_metrics else None
        if variant_rank is not None and variant_rank <= 5:
            retained_top5 += 1
        if baseline_rank is None and variant_rank is not None:
            promoted += 1
        elif baseline_rank is not None and variant_rank is not None and variant_rank < baseline_rank:
            promoted += 1
        elif baseline_rank is not None and (variant_rank is None or variant_rank > baseline_rank):
            demoted += 1
        else:
            unchanged += 1
    return {
        "total_cases": len(subset),
        "retained_top5": retained_top5,
        "promoted_vs_current_rrf": promoted,
        "demoted_vs_current_rrf": demoted,
        "unchanged_vs_current_rrf": unchanged,
    }


# --------------------------------------------------------------------------
# Result artifact
# --------------------------------------------------------------------------


def default_result_path(sample_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parents[1] / "results"
    return results_dir / f"locomo-conv30-v2-fusion-ablation-{timestamp}.json"


def _metric_row(report: LocomoRunReport) -> dict:
    r = report.retrieval
    return {
        "hit_at_1": r.hit_at_1,
        "hit_at_3": r.hit_at_3,
        "hit_at_5": r.hit_at_5,
        "hit_at_10": r.hit_at_10,
        "recall_at_1": r.recall_at_1,
        "recall_at_3": r.recall_at_3,
        "recall_at_5": r.recall_at_5,
        "recall_at_10": r.recall_at_10,
        "mrr": r.mrr,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess

    from src.config import configure, get_config, reset_config
    from src.database import SessionLocal, create_tables, reset_engine

    from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
    from .dataset import DEFAULT_DATASET_PATH, dataset_sha256, load_locomo_dataset

    parser = argparse.ArgumentParser(description="Fusion-policy ablation over an already-ingested LoCoMo sample.")
    parser.add_argument("--sample", default="conv-30")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--conversation-id", default="conversation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
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
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None

    dataset_sha256_value = dataset_sha256(args.dataset_path)
    timings = Timings()
    counters = DbNetworkCounters()
    start_time = datetime.now(timezone.utc)

    try:
        with SessionLocal() as db:
            db_invariants_before = snapshot_full_db_invariants(db)

            with timings.measure("load_benchmark_state"):
                state = load_benchmark_state(db, sample, user_id=user_id, conversation_external_id=args.conversation_id)
            counters.full_memory_corpus_loads += 1
            counters.bm25_corpus_builds += 1

            db_counts_before = snapshot_db_counts(db, user=state.user, conversation=state.conversation)
            print("Before ablation:")
            print(f"  Messages: {db_counts_before.message_count}")
            print(f"  Memories: {db_counts_before.memory_count}")

            embedding_context = EmbeddingCacheContext(
                dataset_sha256=dataset_sha256_value,
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                dimension=_resolve_embedding_dimension(state),
            )
            embedding_cache = QueryEmbeddingCache(context=embedding_context)

            print(f"\nRunning fusion ablation ({len(VARIANTS)} variants) over {len(sample.qa)} questions ...")
            results = run_fusion_ablation(
                db, sample, state=state, top_k=args.top_k, embedding_cache=embedding_cache, timings=timings, counters=counters
            )

            reports: dict[str, LocomoRunReport] = {}
            for variant in VARIANTS:
                diagnostics = [r.variant_diagnostics[variant.label] for r in results]
                reports[variant.label] = build_report(
                    diagnostics,
                    top_k=args.top_k,
                    variant_label=variant.label,
                    dataset_sha256_value=dataset_sha256_value,
                    git_commit=git_commit,
                    llm_model=settings.llm_model,
                    embedding_model=settings.embedding_model,
                )

            for variant in VARIANTS:
                r = reports[variant.label].retrieval
                print(
                    f"  [{variant.label:<34}] Hit@1={r.hit_at_1:.3f} Hit@3={r.hit_at_3:.3f} Hit@5={r.hit_at_5:.3f} "
                    f"Hit@10={r.hit_at_10:.3f} Recall@5={r.recall_at_5:.3f} MRR={r.mrr:.3f}"
                )

            stages_by_variant: dict[str, dict[int, str | None]] = {}
            counts_by_variant: dict[str, dict[str, int]] = {}
            diagnostics_by_variant_index: dict[str, dict[int, QuestionDiagnostic]] = {}
            for variant in VARIANTS:
                diagnostics = [r.variant_diagnostics[variant.label] for r in results]
                stages = stage_by_question(diagnostics, covered_dia_ids=state.covered_dia_ids, k=5)
                stages_by_variant[variant.label] = stages
                counts_by_variant[variant.label] = stage_counts(stages)
                diagnostics_by_variant_index[variant.label] = {d.question_index: d for d in diagnostics}

            no_gold_counts = {counts_by_variant[v.label]["NO_GOLD_MEMORY"] for v in VARIANTS}
            if len(no_gold_counts) != 1 or 18 not in no_gold_counts:
                print(f"\nSTOP: NO_GOLD_MEMORY changed across variants or is not 18: {no_gold_counts}")
                return 1

            baseline_stages = stages_by_variant[BASELINE_LABEL]
            baseline_diags = diagnostics_by_variant_index[BASELINE_LABEL]
            net_impact_by_variant: dict[str, NetImpact] = {}
            failures_fixed_by_variant: dict[str, list[int]] = {}
            for variant in VARIANTS:
                if variant.label == BASELINE_LABEL:
                    continue
                impact = compute_net_impact(
                    baseline_stages=baseline_stages,
                    variant_stages=stages_by_variant[variant.label],
                    baseline_diagnostics=baseline_diags,
                    variant_diagnostics=diagnostics_by_variant_index[variant.label],
                )
                net_impact_by_variant[variant.label] = impact
                failures_fixed_by_variant[variant.label] = [
                    index
                    for index, before in baseline_stages.items()
                    if before in KNOWN_FAILURE_STAGES and stages_by_variant[variant.label].get(index) == "SUCCESS"
                ]

            single_branch_cases = find_single_branch_gold_cases(results, sample=sample, gold_lookup=state.gold_lookup)
            vector_only_cases = [c for c in single_branch_cases if c.kind == "vector_only"]
            bm25_only_cases = [c for c in single_branch_cases if c.kind == "bm25_only"]
            single_branch_report: dict[str, dict] = {}
            for variant in VARIANTS:
                single_branch_report[variant.label] = {
                    "vector_only_gold": single_branch_gold_behavior(
                        single_branch_cases,
                        kind="vector_only",
                        baseline_diagnostics=baseline_diags,
                        variant_diagnostics=diagnostics_by_variant_index[variant.label],
                    ),
                    "bm25_only_gold": single_branch_gold_behavior(
                        single_branch_cases,
                        kind="bm25_only",
                        baseline_diagnostics=baseline_diags,
                        variant_diagnostics=diagnostics_by_variant_index[variant.label],
                    ),
                }

            # Section 14: detailed per-question analysis of the existing 25
            # retrieval-side failures (18 RETRIEVAL_MISS + 7 RANKING_MISS under
            # current_rrf), reusing only already-computed candidate ids/ranks.
            results_by_index = {r.question_index: r for r in results}
            known_failure_indices = sorted(i for i, s in baseline_stages.items() if s in KNOWN_FAILURE_STAGES)
            failure_25_analysis: dict[str, list[dict]] = {}
            for variant in VARIANTS:
                rows = []
                for index in known_failure_indices:
                    result = results_by_index[index]
                    qa = sample.qa[index]
                    gold_memory_ids = {
                        mid for mid, dia_ids in state.gold_lookup.dia_ids_by_memory_id.items() if dia_ids & set(qa.evidence)
                    }
                    old_rank = baseline_diags[index].retrieval_metrics.rank if baseline_diags[index].retrieval_metrics else None
                    new_diag = diagnostics_by_variant_index[variant.label][index]
                    new_rank = new_diag.retrieval_metrics.rank if new_diag.retrieval_metrics else None
                    if new_rank is not None and new_rank <= 5:
                        classification = "fixed" if variant.label != BASELINE_LABEL else "success"
                    elif old_rank is not None and new_rank is not None and new_rank < old_rank:
                        classification = "improved_not_fixed"
                    elif old_rank is not None and new_rank is not None and new_rank > old_rank:
                        classification = "worsened"
                    else:
                        classification = "still_failing"
                    rows.append(
                        {
                            "question_index": index,
                            "old_final_rank": old_rank,
                            "new_final_rank": new_rank,
                            "gold_bm25_rank": _rank_of_gold(result.bm25_candidate_ids, gold_memory_ids),
                            "gold_vector_rank": _rank_of_gold(result.vector_candidate_ids, gold_memory_ids),
                            "classification": classification,
                        }
                    )
                failure_25_analysis[variant.label] = rows

            # Section 16: representative fusion-math worked examples for a
            # handful of questions where discounted agreement changes the
            # outcome relative to current_rrf.
            representative_examples = []
            for index in known_failure_indices:
                result = results_by_index[index]
                qa = sample.qa[index]
                gold_memory_ids = {
                    mid for mid, dia_ids in state.gold_lookup.dia_ids_by_memory_id.items() if dia_ids & set(qa.evidence)
                }
                gold_bm25_rank = _rank_of_gold(result.bm25_candidate_ids, gold_memory_ids)
                gold_vector_rank = _rank_of_gold(result.vector_candidate_ids, gold_memory_ids)
                if gold_vector_rank is None and gold_bm25_rank is None:
                    continue
                if gold_bm25_rank is not None and gold_vector_rank is not None:
                    continue  # only single-branch gold cases make the intended contrast
                gold_rr = 1.0 / (RRF_K + (gold_bm25_rank or gold_vector_rank))
                # Find the top current_rrf competitor at this question that out-scored gold.
                old_rank = baseline_diags[index].retrieval_metrics.rank if baseline_diags[index].retrieval_metrics else None
                new_rank_010 = (
                    diagnostics_by_variant_index["discounted_agreement_lambda_0.10"][index].retrieval_metrics.rank
                    if diagnostics_by_variant_index["discounted_agreement_lambda_0.10"][index].retrieval_metrics
                    else None
                )
                if old_rank is not None and new_rank_010 is not None and new_rank_010 >= old_rank:
                    continue
                representative_examples.append(
                    {
                        "question_index": index,
                        "question": qa.question,
                        "gold_bm25_rank": gold_bm25_rank,
                        "gold_vector_rank": gold_vector_rank,
                        "gold_current_rrf_score": gold_rr,
                        "old_final_rank_current_rrf": old_rank,
                        "new_final_rank_discounted_0.10": new_rank_010,
                    }
                )
                if len(representative_examples) >= 5:
                    break

            db_counts_after = snapshot_db_counts(db, user=state.user, conversation=state.conversation)
            db_invariants_after = snapshot_full_db_invariants(db)
    finally:
        reset_engine()
        reset_config()

    unchanged = (
        db_counts_before.message_count == db_counts_after.message_count
        and db_counts_before.memory_count == db_counts_after.memory_count
        and db_counts_before.memory_checksum == db_counts_after.memory_checksum
        and db_invariants_before == db_invariants_after
    )
    print(f"\nAfter ablation: messages={db_counts_after.message_count} memories={db_counts_after.memory_count}")
    print(f"DB state unchanged: {unchanged}")
    if not unchanged:
        print("WARNING: database state changed during a retrieval-only ablation.")

    print("\nK=5 stage counts:")
    print(f"  {'VARIANT':<36}{'SUCCESS':>8}{'NO_GOLD':>8}{'RETR_MISS':>10}{'RANK_MISS':>10}")
    for variant in VARIANTS:
        c = counts_by_variant[variant.label]
        print(f"  {variant.label:<36}{c['SUCCESS']:>8}{c['NO_GOLD_MEMORY']:>8}{c['RETRIEVAL_MISS']:>10}{c['RANKING_MISS']:>10}")

    print("\nNet question impact vs current_rrf:")
    for variant in VARIANTS:
        if variant.label == BASELINE_LABEL:
            continue
        impact = net_impact_by_variant[variant.label]
        print(
            f"  {variant.label:<36} fixed={impact.failures_fixed:>3} lost={impact.successes_lost:>3} "
            f"net={impact.net_success_gain:>+3} rank_improved={impact.rank_improved:>3} rank_worsened={impact.rank_worsened:>3}"
        )

    # Selection: recall@5 desc, hit@5 desc, mrr desc, positive net gain, fewer RANKING_MISS.
    def _selection_key(label: str) -> tuple:
        r = reports[label].retrieval
        impact = net_impact_by_variant.get(label)
        net_gain = impact.net_success_gain if impact else 0
        return (
            -r.recall_at_5,
            -r.hit_at_5,
            -r.mrr,
            -(1 if net_gain > 0 else 0),
            counts_by_variant[label]["RANKING_MISS"],
        )

    non_baseline_labels = [v.label for v in VARIANTS if v.label not in ("bm25_only", "vector_only")]
    ranked = sorted(non_baseline_labels, key=_selection_key)
    best_label = ranked[0]
    baseline_recall5 = reports[BASELINE_LABEL].retrieval.recall_at_5
    best_recall5 = reports[best_label].retrieval.recall_at_5
    MEANINGFUL_IMPROVEMENT = 0.02  # absolute Recall@5 points; below this, keep current fusion.
    if best_label == BASELINE_LABEL or (best_recall5 - baseline_recall5) < MEANINGFUL_IMPROVEMENT:
        recommendation = "KEEP_CURRENT_FUSION"
        recommended_params = None
    elif reports[best_label].retrieval is not None and best_label.startswith("weighted_rrf"):
        recommendation = "USE_WEIGHTED_RRF"
        recommended_params = {"bm25_weight": next(v.bm25_weight for v in VARIANTS if v.label == best_label)}
    else:
        recommendation = "USE_DISCOUNTED_AGREEMENT"
        recommended_params = {"lambda": next(v.lambda_ for v in VARIANTS if v.label == best_label)}

    print(f"\nRecommendation: {recommendation} (best variant by selection criteria: {best_label})")

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    print()
    print(timings.render())
    print()
    print(counters.render(question_count=len(sample.qa)))
    print(f"\nTotal wall time: {elapsed:.2f}s")

    output_path = Path(args.output) if args.output else default_result_path(args.sample)
    payload = {
        "label": "LoCoMo Fusion-Policy Ablation (frozen V2-recovered conv-30)",
        "note": (
            "Fusion-only ablation: candidate generation, candidate limits, top-K, BM25 params, "
            "and the embedding model are all unchanged. Only how structured/lexical/vector branch "
            "rankings are combined into a final score varies between variants."
        ),
        "sample_id": args.sample,
        "top_k": args.top_k,
        "rrf_k": RRF_K,
        "variants": [v.describe() for v in VARIANTS],
        "metrics": {v.label: _metric_row(reports[v.label]) for v in VARIANTS},
        "k5_stage_counts": counts_by_variant,
        "net_impact_vs_current_rrf": {label: impact.to_dict() for label, impact in net_impact_by_variant.items()},
        "known_failures_fixed_by_variant": failures_fixed_by_variant,
        "known_25_failure_analysis_by_variant": failure_25_analysis,
        "representative_fusion_math_examples": representative_examples,
        "single_branch_gold_cases": {
            "vector_only_gold_count": len(vector_only_cases),
            "bm25_only_gold_count": len(bm25_only_cases),
            "vector_only_gold_cases": [c.to_dict() for c in vector_only_cases],
            "bm25_only_gold_cases": [c.to_dict() for c in bm25_only_cases],
            "behavior_by_variant": single_branch_report,
        },
        "db_invariants": {
            "before": db_invariants_before.to_dict(),
            "after": db_invariants_after.to_dict(),
            "unchanged": unchanged,
            "messages_expected": 369,
            "memories_expected": 250,
        },
        "performance_counters": {
            "total_wall_time_seconds": elapsed,
            "query_embedding_provider_calls": counters.query_embedding_provider_calls,
            "query_embedding_cache_hits": counters.query_embedding_cache_hits,
            "vector_searches": counters.vector_searches,
            "bm25_corpus_builds": counters.bm25_corpus_builds,
            "fusion_computations": counters.fusion_computations,
            "diagnostic_reruns": counters.diagnostic_reruns,
            "full_embedding_corpus_db_reloads": counters.full_memory_corpus_loads,
        },
        "recommendation": recommendation,
        "recommended_params": recommended_params,
        "full_reports": {v.label: reports[v.label].model_dump(mode="json") for v in VARIANTS},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved fusion ablation artifact to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
