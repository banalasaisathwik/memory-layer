"""Retrieval-only lexical-backend ablation over an already-ingested LoCoMo sample.

Compares two hybrid-search configurations against the exact same persisted
conv-30 database state:

    A. structured + PostgreSQL FTS lexical + vector, fused by the existing
       equal-weight ``reciprocal_rank_fusion`` (``lexical_backend="postgres_fts"``)
    B. structured + BM25 lexical + vector, fused by the same RRF
       (``lexical_backend="bm25"``)

This module never calls ``MemoryLayer.add()``, never ingests a LoCoMo sample,
never calls ``extract_memories()``/``write_memories()``, and never inserts a
``User``/``Conversation``/``Message``/``Memory`` row. It only reads existing
rows and calls the same lower-level retrieval branch functions that
``meminfra.retrieval.search.search_memories()`` composes internally (structured,
lexical/BM25, vector, RRF), never a second, diverging implementation of
production retrieval semantics.

Architecture (shared-branch ablation, per question)::

    question
       |
       +--> structured_candidates  (structured_retrieve, shared)
       +--> query_embedding        (cache-or-provider, shared, computed once)
       +--> vector_candidates      (vector branch, shared, computed once)
       |
       +--> fts_candidates   (lexical_retrieve)  --\\
       |                                            +--> RRF --> fts_fused
       +--> bm25_candidates  (PreparedBM25Corpus)  --\\
                                                       +--> RRF --> bm25_fused

Both fused outputs reuse the exact same ``reciprocal_rank_fusion`` and branch
candidate lists ``search_memories()`` would produce for the same query and
default filters -- see ``tests/retrieval/test_lexical_ablation.py`` for the
parity tests proving this. The only intentional divergence from calling
``search_memories()`` twice is: (1) the structured and vector branches are
computed once and reused for both backends instead of twice, and (2) the BM25
branch scores a ``PreparedBM25Corpus`` built once per run instead of
re-tokenizing the whole corpus on every question (see ``bm25_corpus.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from meminfra.config import get_config
from meminfra.database.models import Conversation, Memory, Message, User
from meminfra.providers import get_embedding_client
from meminfra.retrieval.errors import IndexDimensionMismatchError
from meminfra.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from meminfra.retrieval.lexical import lexical_retrieve
from meminfra.retrieval.schemas import SearchFilters
from meminfra.retrieval.search import _BRANCH_CANDIDATE_MULTIPLIER, _MAX_SEARCH_LIMIT, LexicalBackend, search_memories
from meminfra.retrieval.structured import memory_filter_conditions, structured_retrieve
from meminfra.retrieval.vector import _ensure_user_memory_index
from meminfra.retrieval.vector_support import embedding_response_vectors

from .ablation_checkpoint import (
    AblationCheckpoint,
    append_question_result,
    delete_ablation_checkpoint,
    load_ablation_checkpoint,
    load_question_results,
    save_ablation_checkpoint,
)
from .benchmark_state import BenchmarkState, GoldMemoryLookup, load_benchmark_state
from .bm25_corpus import BM25_B, BM25_K1
from .diagnose import active_memory_covered_dia_ids, classify_stage
from .embedding_cache import EmbeddingCacheContext, QueryEmbeddingCache
from .instrumentation import DbNetworkCounters, Timings
from .metrics import aggregate_retrieval_metrics, evaluate_question_retrieval, group_retrieval_by_category
from .recover import build_dia_id_mapping, session_message_slices
from .schemas import (
    CATEGORY_NAMES,
    LocomoRunMetadata,
    LocomoRunReport,
    LocomoSample,
    QuestionDiagnostic,
)

BACKENDS: tuple[LexicalBackend, ...] = ("postgres_fts", "bm25")
BACKEND_LABEL = {"postgres_fts": "Postgres FTS", "bm25": "BM25"}
STAGE_KEYS = ("SUCCESS", "NO_GOLD_MEMORY", "RETRIEVAL_MISS", "RANKING_MISS")

# _BRANCH_CANDIDATE_MULTIPLIER / _MAX_SEARCH_LIMIT (imported above) mirror
# meminfra.retrieval.search's private branch-sizing constants exactly, so the
# shared candidate lists this module builds have the identical shape
# search_memories() would produce -- imported by name, never re-declared,
# so the two can never silently drift.


# --------------------------------------------------------------------------
# Read-only DB state verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DbCounts:
    conversation_count: int
    message_count: int
    memory_count: int
    memory_checksum: str

    def to_dict(self) -> dict:
        return {
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "memory_count": self.memory_count,
            "memory_checksum": self.memory_checksum,
        }


def snapshot_db_counts(db: Session, *, user: User, conversation: Conversation) -> DbCounts:
    """A read-only fingerprint of the exact rows this ablation must not change."""

    conversation_count = db.scalar(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user.id)
    )
    message_count = db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    )
    memory_rows = list(
        db.scalars(select(Memory).where(Memory.user_id == user.id).order_by(Memory.id.asc()))
    )
    digest = hashlib.sha256()
    for memory in memory_rows:
        digest.update(
            "|".join(
                [
                    str(memory.id),
                    str(memory.is_active),
                    str(memory.updated_at),
                    str(sorted(memory.source_message_ids)),
                    memory.memory_text,
                ]
            ).encode("utf-8")
        )
    return DbCounts(
        conversation_count=int(conversation_count or 0),
        message_count=int(message_count or 0),
        memory_count=len(memory_rows),
        memory_checksum=digest.hexdigest(),
    )


# --------------------------------------------------------------------------
# Shared-branch retrieval (Part B) + prepared BM25 (Part C)
# --------------------------------------------------------------------------


def _vector_candidates_from_embedding(
    db: Session,
    query_vector: np.ndarray,
    *,
    user: User,
    limit: int,
) -> list[Memory]:
    """Reproduce ``vector_retrieve``'s post-embedding FAISS + DB-filter logic exactly.

    ``meminfra.retrieval.vector.vector_retrieve`` always computes its own query
    embedding internally, so it cannot be handed a cached vector directly.
    This function duplicates only its logic *after* the embedding step
    (calling the same private ``_ensure_user_memory_index`` production sync
    path, the same requested-candidate-count formula, and the same
    ``memory_filter_conditions`` scoping) so a cached embedding can be reused
    across resumed/rerun ablation questions. It is intentionally eval-only;
    production ``vector_retrieve``/``search_memories()`` is untouched.
    Numeric parity with ``vector_retrieve`` is verified by
    ``tests/retrieval/test_lexical_ablation.py``.
    """

    loaded = _ensure_user_memory_index(db, user=user)
    if loaded is None:
        return []
    if query_vector.size != loaded.dimension:
        raise IndexDimensionMismatchError(
            "The query embedding dimension differs from the persisted index; rebuild with compatible embeddings."
        )
    requested = min(loaded.index.ntotal, max(limit, limit * get_config().vector_candidate_multiplier))
    if requested == 0:
        return []
    _, positions = loaded.index.search(query_vector.reshape(1, -1), requested)
    candidate_ids = [
        loaded.memory_ids[position]
        for position in positions[0]
        if position >= 0 and position < len(loaded.memory_ids)
    ]
    if not candidate_ids:
        return []
    filters = SearchFilters()
    rows = list(
        db.scalars(
            select(Memory).where(
                Memory.user_id == user.id,
                Memory.id.in_(candidate_ids),
                *memory_filter_conditions(filters, conversation=None),
            )
        )
    )
    by_id = {str(memory.id): memory for memory in rows}
    return [by_id[memory_id] for memory_id in candidate_ids if memory_id in by_id]


@dataclass
class QuestionResult:
    """Everything computed once for one question, reused by both backends and diagnostics.

    Storing this (Part D) means the stage-diagnostics pass below never needs
    to recall ``vector_retrieve``, ``lexical_retrieve``, ``bm25_retrieve``, a
    prepared-BM25 score, or ``search_memories`` a second time for a question
    already evaluated here -- it reads straight from this dataclass.
    """

    question_index: int
    question: str
    structured_candidate_ids: list[str]
    vector_candidate_ids: list[str]
    fts_candidate_ids: list[str]
    bm25_candidate_ids: list[str]
    fts_fused_ids: list[str]
    bm25_fused_ids: list[str]
    fts_diagnostic: QuestionDiagnostic
    bm25_diagnostic: QuestionDiagnostic

    def to_checkpoint_dict(self) -> dict:
        return {
            "question_index": self.question_index,
            "question": self.question,
            "structured_candidate_ids": self.structured_candidate_ids,
            "vector_candidate_ids": self.vector_candidate_ids,
            "fts_candidate_ids": self.fts_candidate_ids,
            "bm25_candidate_ids": self.bm25_candidate_ids,
            "fts_fused_ids": self.fts_fused_ids,
            "bm25_fused_ids": self.bm25_fused_ids,
            "fts_diagnostic": self.fts_diagnostic.model_dump(mode="json"),
            "bm25_diagnostic": self.bm25_diagnostic.model_dump(mode="json"),
        }

    @staticmethod
    def from_checkpoint_dict(raw: dict) -> "QuestionResult":
        return QuestionResult(
            question_index=raw["question_index"],
            question=raw["question"],
            structured_candidate_ids=raw["structured_candidate_ids"],
            vector_candidate_ids=raw["vector_candidate_ids"],
            fts_candidate_ids=raw["fts_candidate_ids"],
            bm25_candidate_ids=raw["bm25_candidate_ids"],
            fts_fused_ids=raw["fts_fused_ids"],
            bm25_fused_ids=raw["bm25_fused_ids"],
            fts_diagnostic=QuestionDiagnostic.model_validate(raw["fts_diagnostic"]),
            bm25_diagnostic=QuestionDiagnostic.model_validate(raw["bm25_diagnostic"]),
        )


def _diagnostic_from_fused(
    *,
    sample: LocomoSample,
    question_index: int,
    fused_memories: list[Memory],
    primary_user_db_id: object,
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
    provenance_by_rank: list[set[str]] = []
    isolation_failures = 0
    for memory_row in fused_memories:
        diagnostic.retrieved_memory_ids.append(str(memory_row.id))
        diagnostic.retrieved_memory_texts.append(memory_row.memory_text)
        if memory_row.user_id != primary_user_db_id:
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


def evaluate_question_shared(
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
    """Evaluate one question's FTS and BM25 hybrid results from shared branches.

    Computes ``structured_candidates``, ``vector_candidates``, ``fts_candidates``,
    and ``bm25_candidates`` exactly once, then fuses each lexical branch with
    the *same* structured/vector candidates via the same
    ``reciprocal_rank_fusion`` production code path search_memories() uses --
    never a second, hand-rolled fusion.
    """

    qa = sample.qa[question_index]
    filters = SearchFilters()
    branch_limit = min(_MAX_SEARCH_LIMIT, top_k * _BRANCH_CANDIDATE_MULTIPLIER)

    with (timings.measure("structured_branch") if timings else _noop()):
        structured_candidates = structured_retrieve(
            db, user=state.user, filters=filters, conversation=None, limit=branch_limit
        )

    with (timings.measure("vector_branch") if timings else _noop()):
        vector_candidates = _vector_candidates_from_embedding(
            db, query_vector, user=state.user, limit=top_k
        )
    if counters is not None:
        counters.vector_searches += 1

    with (timings.measure("fts_branch") if timings else _noop()):
        fts_candidates = lexical_retrieve(
            db, qa.question, user=state.user, filters=filters, conversation=None, limit=branch_limit
        )
    if counters is not None:
        counters.fts_queries += 1

    with (timings.measure("bm25_branch") if timings else _noop()):
        bm25_candidates = [hit.item for hit in state.bm25_corpus.score(qa.question)[:branch_limit]]

    with (timings.measure("rrf") if timings else _noop()):
        fts_fused = [
            hit.memory
            for hit in reciprocal_rank_fusion(
                structured=structured_candidates, lexical=fts_candidates, vector=vector_candidates, k=RRF_K
            )
        ][:top_k]
        bm25_fused = [
            hit.memory
            for hit in reciprocal_rank_fusion(
                structured=structured_candidates, lexical=bm25_candidates, vector=vector_candidates, k=RRF_K
            )
        ][:top_k]

    with (timings.measure("metrics") if timings else _noop()):
        fts_diagnostic = _diagnostic_from_fused(
            sample=sample,
            question_index=question_index,
            fused_memories=fts_fused,
            primary_user_db_id=state.user.id,
            message_id_to_dia_id=state.message_id_to_dia_id,
        )
        bm25_diagnostic = _diagnostic_from_fused(
            sample=sample,
            question_index=question_index,
            fused_memories=bm25_fused,
            primary_user_db_id=state.user.id,
            message_id_to_dia_id=state.message_id_to_dia_id,
        )

    return QuestionResult(
        question_index=question_index,
        question=qa.question,
        structured_candidate_ids=[str(m.id) for m in structured_candidates],
        vector_candidate_ids=[str(m.id) for m in vector_candidates],
        fts_candidate_ids=[str(m.id) for m in fts_candidates],
        bm25_candidate_ids=[str(m.id) for m in bm25_candidates],
        fts_fused_ids=[str(m.id) for m in fts_fused],
        bm25_fused_ids=[str(m.id) for m in bm25_fused],
        fts_diagnostic=fts_diagnostic,
        bm25_diagnostic=bm25_diagnostic,
    )


class _noop:
    """A trivial context manager used when no ``Timings`` instance was passed."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


def embed_question_batch(texts: list[str]) -> list[np.ndarray]:
    """Batch-embed benchmark questions via the same provider abstraction production uses.

    Delegates to ``embedding_response_vectors`` (``src/retrieval/vector_support.py``),
    which already supports a ``list[str]`` batch call -- no new provider
    abstraction is introduced here (Part E3).
    """

    return embedding_response_vectors(
        texts, model=get_config().embedding_model, client_factory=get_embedding_client
    )


def run_shared_ablation(
    db: Session,
    sample: LocomoSample,
    *,
    state: BenchmarkState,
    top_k: int,
    embedding_cache: QueryEmbeddingCache,
    already_completed: dict[int, QuestionResult] | None = None,
    timings: Timings | None = None,
    counters: DbNetworkCounters | None = None,
    on_question_complete=None,
) -> list[QuestionResult]:
    """Evaluate every question in ``sample.qa``, skipping any already completed.

    ``already_completed`` (from a resumed checkpoint) is trusted as-is and
    never recomputed (Part F). ``on_question_complete`` is called with each
    newly computed ``QuestionResult`` immediately after it is produced, so a
    caller can persist per-question checkpoint progress.
    """

    already_completed = already_completed or {}
    questions = [qa.question for qa in sample.qa]

    with (timings.measure("query_embedding_cache_load") if timings else _noop()):
        vectors_by_question = embedding_cache.ensure_batch(
            [q for index, q in enumerate(questions) if index not in already_completed],
            embed_many=embed_question_batch,
        )
    if counters is not None:
        counters.query_embedding_provider_calls += embedding_cache.stats.provider_batch_calls

    results: list[QuestionResult] = []
    for question_index in range(len(sample.qa)):
        if question_index in already_completed:
            results.append(already_completed[question_index])
            continue
        query_vector = vectors_by_question[sample.qa[question_index].question]
        result = evaluate_question_shared(
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
        if on_question_complete is not None:
            on_question_complete(result)
    return results


# --------------------------------------------------------------------------
# Legacy per-backend evaluation (kept for direct search_memories() comparison /
# parity tests -- not used in the optimized run_shared_ablation() path above)
# --------------------------------------------------------------------------


def _evaluate_question(
    db: Session,
    *,
    sample: LocomoSample,
    question_index: int,
    user_id: str,
    primary_user_db_id: object,
    top_k: int,
    message_id_to_dia_id: dict[str, str],
    lexical_backend: LexicalBackend,
) -> QuestionDiagnostic:
    qa = sample.qa[question_index]
    hits = search_memories(
        db,
        qa.question,
        user_external_id=user_id,
        limit=top_k,
        lexical_backend=lexical_backend,
    )
    fused_memories = [db.get(Memory, hit.memory_id) for hit in hits]
    fused_memories = [memory for memory in fused_memories if memory is not None]
    return _diagnostic_from_fused(
        sample=sample,
        question_index=question_index,
        fused_memories=fused_memories,
        primary_user_db_id=primary_user_db_id,
        message_id_to_dia_id=message_id_to_dia_id,
    )


def run_backend(
    db: Session,
    sample: LocomoSample,
    *,
    user_id: str,
    primary_user_db_id: object,
    top_k: int,
    message_id_to_dia_id: dict[str, str],
    lexical_backend: LexicalBackend,
) -> list[QuestionDiagnostic]:
    """Reference implementation calling ``search_memories()`` directly, per backend.

    Retained only for parity testing against ``run_shared_ablation`` /
    ``evaluate_question_shared`` -- the optimized ablation path never calls
    this for a real run since it would recompute the vector branch twice.
    """

    return [
        _evaluate_question(
            db,
            sample=sample,
            question_index=question_index,
            user_id=user_id,
            primary_user_db_id=primary_user_db_id,
            top_k=top_k,
            message_id_to_dia_id=message_id_to_dia_id,
            lexical_backend=lexical_backend,
        )
        for question_index in range(len(sample.qa))
    ]


def build_report(
    diagnostics: list[QuestionDiagnostic],
    *,
    top_k: int,
    run_id: str,
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


# --------------------------------------------------------------------------
# Stage classification + transition analysis
# --------------------------------------------------------------------------


def stage_by_question(
    diagnostics: list[QuestionDiagnostic],
    *,
    covered_dia_ids: set[str],
    k: int = 5,
) -> dict[int, str | None]:
    return {d.question_index: classify_stage(d, covered_dia_ids=covered_dia_ids, k=k) for d in diagnostics}


def stage_counts(stages: dict[int, str | None]) -> dict[str, int]:
    counts = {key: 0 for key in STAGE_KEYS}
    for outcome in stages.values():
        if outcome is not None:
            counts[outcome] += 1
    return counts


def transition_counts(fts_stages: dict[int, str | None], bm25_stages: dict[int, str | None]) -> dict[str, int]:
    transitions: dict[str, int] = {}
    for index in sorted(fts_stages):
        before = fts_stages[index]
        after = bm25_stages.get(index)
        if before is None or after is None:
            continue
        label = "UNCHANGED" if before == after else f"{before} -> {after}"
        transitions[label] = transitions.get(label, 0) + 1
    return transitions


# --------------------------------------------------------------------------
# Branch-level diagnostics for previously-failing / changed questions
#
# Part D: this reuses the candidate id lists already computed by
# evaluate_question_shared()/run_shared_ablation() above -- it never calls
# lexical_retrieve/bm25_retrieve/vector_retrieve/search_memories again for a
# question that already has a QuestionResult.
# --------------------------------------------------------------------------


def _rank_of_gold(memory_ids_in_order: list[str], gold_memory_ids: set[str]) -> int | None:
    for rank, memory_id in enumerate(memory_ids_in_order, start=1):
        if memory_id in gold_memory_ids:
            return rank
    return None


@dataclass(frozen=True)
class BranchDiagnostic:
    question_index: int
    question: str
    gold_evidence: list[str]
    gold_memory_ids: list[str]
    fts_lexical_rank: int | None
    bm25_lexical_rank: int | None
    vector_rank: int | None
    fts_final_rank: int | None
    bm25_final_rank: int | None
    fts_lexical_top: list[str]
    bm25_lexical_top: list[str]
    vector_top: list[str]
    classification: str

    def to_dict(self) -> dict:
        return {
            "question_index": self.question_index,
            "question": self.question,
            "gold_evidence": self.gold_evidence,
            "gold_memory_ids": self.gold_memory_ids,
            "fts_lexical_rank": self.fts_lexical_rank,
            "bm25_lexical_rank": self.bm25_lexical_rank,
            "vector_rank": self.vector_rank,
            "fts_final_rank": self.fts_final_rank,
            "bm25_final_rank": self.bm25_final_rank,
            "fts_lexical_top_ids": self.fts_lexical_top,
            "bm25_lexical_top_ids": self.bm25_lexical_top,
            "vector_top_ids": self.vector_top,
            "classification": self.classification,
        }


def classify_bm25_effect(
    *,
    fts_lexical_rank: int | None,
    bm25_lexical_rank: int | None,
    vector_rank: int | None,
    fts_final_rank: int | None,
    bm25_final_rank: int | None,
) -> str:
    """CANDIDATE_RECOVERY / RRF_BOOST / BOTH / UNCLEAR, per the task's definitions."""

    bm25_found_lexically = bm25_lexical_rank is not None
    fts_found_lexically = fts_lexical_rank is not None
    found_by_vector = vector_rank is not None
    improved_final = (bm25_final_rank is not None) and (
        fts_final_rank is None or bm25_final_rank < fts_final_rank
    )

    if not improved_final:
        return "UNCLEAR"

    recovered_candidate = bm25_found_lexically and not fts_found_lexically and not found_by_vector
    boosted_existing = bm25_found_lexically and (fts_found_lexically or found_by_vector)

    if recovered_candidate and not boosted_existing:
        return "CANDIDATE_RECOVERY"
    if boosted_existing and not recovered_candidate:
        return "RRF_BOOST"
    if recovered_candidate and boosted_existing:
        return "BOTH"
    return "UNCLEAR"


def branch_diagnostic_from_result(
    result: QuestionResult,
    *,
    gold_lookup: GoldMemoryLookup,
    gold_evidence: list[str],
) -> BranchDiagnostic:
    """Build a BranchDiagnostic entirely from an already-computed QuestionResult.

    No DB access and no retrieval call happens here -- everything needed was
    already computed once by ``evaluate_question_shared``.
    """

    gold = set(gold_evidence)
    gold_memory_ids = {
        memory_id for memory_id, dia_ids in gold_lookup.dia_ids_by_memory_id.items() if dia_ids & gold
    }

    fts_lexical_rank = _rank_of_gold(result.fts_candidate_ids, gold_memory_ids)
    bm25_lexical_rank = _rank_of_gold(result.bm25_candidate_ids, gold_memory_ids)
    vector_rank = _rank_of_gold(result.vector_candidate_ids, gold_memory_ids)
    fts_final_rank = result.fts_diagnostic.retrieval_metrics.rank if result.fts_diagnostic.retrieval_metrics else None
    bm25_final_rank = (
        result.bm25_diagnostic.retrieval_metrics.rank if result.bm25_diagnostic.retrieval_metrics else None
    )

    classification = classify_bm25_effect(
        fts_lexical_rank=fts_lexical_rank,
        bm25_lexical_rank=bm25_lexical_rank,
        vector_rank=vector_rank,
        fts_final_rank=fts_final_rank,
        bm25_final_rank=bm25_final_rank,
    )

    return BranchDiagnostic(
        question_index=result.question_index,
        question=result.question,
        gold_evidence=gold_evidence,
        gold_memory_ids=sorted(gold_memory_ids),
        fts_lexical_rank=fts_lexical_rank,
        bm25_lexical_rank=bm25_lexical_rank,
        vector_rank=vector_rank,
        fts_final_rank=fts_final_rank,
        bm25_final_rank=bm25_final_rank,
        fts_lexical_top=result.fts_candidate_ids[:10],
        bm25_lexical_top=result.bm25_candidate_ids[:10],
        vector_top=result.vector_candidate_ids[:10],
        classification=classification,
    )


# --------------------------------------------------------------------------
# Result artifacts
# --------------------------------------------------------------------------


def default_result_paths(sample_id: str) -> tuple[Path, Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parents[1] / "results"
    return (
        results_dir / f"locomo-v1-retrieval-postgres-fts-{sample_id}-{timestamp}.json",
        results_dir / f"locomo-v1-retrieval-bm25-{sample_id}-{timestamp}.json",
        results_dir / f"locomo-v1-fts-vs-bm25-{sample_id}-{timestamp}.json",
    )


def save_backend_result(
    report: LocomoRunReport,
    *,
    backend: LexicalBackend,
    base_v1_path: str,
    db_counts_before: DbCounts,
    db_counts_after: DbCounts,
    path: Path,
) -> None:
    payload = {
        "label": f"LoCoMo V1 Retrieval Ablation — {BACKEND_LABEL[backend]}",
        "ablation": "retrieval-only lexical backend, same V1 database state, no ingestion",
        "lexical_backend": backend,
        "base_v1_result": base_v1_path,
        "db_counts_before": db_counts_before.to_dict(),
        "db_counts_after": db_counts_after.to_dict(),
        "report": report.model_dump(mode="json"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_comparison(
    *,
    fts_report: LocomoRunReport,
    bm25_report: LocomoRunReport,
    fts_stage_counts: dict[str, int],
    bm25_stage_counts: dict[str, int],
    transitions: dict[str, int],
    retrieval_miss_24_analysis: list[dict],
    branch_examples: list[dict],
    path: Path,
) -> None:
    def metric_row(field_name: str) -> dict:
        fts_value = getattr(fts_report.retrieval, field_name)
        bm25_value = getattr(bm25_report.retrieval, field_name)
        return {"postgres_fts": fts_value, "bm25": bm25_value, "delta": bm25_value - fts_value}

    metrics = {
        field_name: metric_row(field_name)
        for field_name in [
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
        ]
    }
    stage_comparison = {
        key: {
            "postgres_fts": fts_stage_counts[key],
            "bm25": bm25_stage_counts[key],
            "delta": bm25_stage_counts[key] - fts_stage_counts[key],
        }
        for key in STAGE_KEYS
    }
    payload = {
        "label": "LoCoMo V1 Retrieval Ablation — Postgres FTS vs BM25 comparison",
        "note": (
            "Retrieval-only ablation on the already-ingested V1 memory state. "
            "Does not measure extraction-context RRF; only the durable-memory "
            "lexical backend varies between the two runs."
        ),
        "metrics": metrics,
        "k5_stage_counts": stage_comparison,
        "k5_transitions": transitions,
        "previous_24_retrieval_miss_analysis": retrieval_miss_24_analysis,
        "branch_diagnostic_examples": branch_examples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess
    from uuid import uuid4

    from meminfra.config import configure, get_config, reset_config
    from meminfra.database import SessionLocal, create_tables, reset_engine

    from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
    from .dataset import DEFAULT_DATASET_PATH, dataset_sha256, load_locomo_dataset

    parser = argparse.ArgumentParser(
        description="Retrieval-only ablation: Postgres FTS vs BM25 lexical backend, same V1 DB state."
    )
    parser.add_argument("--sample", default="conv-30")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--conversation-id", default="conversation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--v1-result", default="evals/results/locomo-v1-conv-30.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing ablation checkpoint.")
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
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = None

    v1_raw = json.loads(Path(args.v1_result).read_text(encoding="utf-8"))
    v1_report = LocomoRunReport.model_validate(v1_raw["report"] if "report" in v1_raw else v1_raw)

    dataset_sha256_value = dataset_sha256(args.dataset_path)
    timings = Timings()
    counters = DbNetworkCounters()

    # Part F: check for a resumable checkpoint before doing any work.
    fingerprint_kwargs = dict(
        sample_id=args.sample,
        dataset_sha256=dataset_sha256_value,
        embedding_model=settings.embedding_model,
        top_k=args.top_k,
        lexical_backends=list(BACKENDS),
        rrf_k=RRF_K,
        bm25_k1=BM25_K1,
        bm25_b=BM25_B,
    )
    already_completed: dict[int, QuestionResult] = {}
    if not args.no_resume:
        existing = load_ablation_checkpoint(args.sample)
        if existing is not None:
            problems = existing.incompatibilities(**fingerprint_kwargs)
            if problems:
                print("Existing ablation checkpoint is incompatible with this run; starting fresh:")
                for problem in problems:
                    print(f"  - {problem}")
                delete_ablation_checkpoint(args.sample)
            else:
                raw_results = load_question_results(args.sample)
                for index in existing.completed_question_indices:
                    if index in raw_results:
                        already_completed[index] = QuestionResult.from_checkpoint_dict(raw_results[index])
                print(f"Resuming ablation checkpoint: {len(already_completed)} question(s) already completed.")

    checkpoint = AblationCheckpoint(
        completed_question_indices=sorted(already_completed),
        **fingerprint_kwargs,
    )

    def _on_question_complete(result: QuestionResult) -> None:
        append_question_result(args.sample, result.to_checkpoint_dict())
        checkpoint.completed_question_indices.append(result.question_index)
        checkpoint.completed_question_indices.sort()
        save_ablation_checkpoint(checkpoint)

    try:
        with SessionLocal() as db:
            with timings.measure("load_benchmark_state"):
                state = load_benchmark_state(
                    db, sample, user_id=user_id, conversation_external_id=args.conversation_id
                )
            counters.full_memory_corpus_loads += 1
            counters.bm25_corpus_builds += 1

            db_counts_before = snapshot_db_counts(db, user=state.user, conversation=state.conversation)
            print("Before ablation:")
            print(f"  Conversations: {db_counts_before.conversation_count}")
            print(f"  Messages:      {db_counts_before.message_count}")
            print(f"  Memories:      {db_counts_before.memory_count}")

            embedding_context = EmbeddingCacheContext(
                dataset_sha256=dataset_sha256_value,
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                dimension=_resolve_embedding_dimension(state),
            )
            embedding_cache = QueryEmbeddingCache(context=embedding_context)

            print(f"\nRunning shared-branch retrieval ablation over {len(sample.qa)} questions ...")
            results = run_shared_ablation(
                db,
                sample,
                state=state,
                top_k=args.top_k,
                embedding_cache=embedding_cache,
                already_completed=already_completed,
                timings=timings,
                counters=counters,
                on_question_complete=_on_question_complete,
            )

            fts_diagnostics = [r.fts_diagnostic for r in results]
            bm25_diagnostics = [r.bm25_diagnostic for r in results]

            reports: dict[LexicalBackend, LocomoRunReport] = {
                "postgres_fts": build_report(
                    fts_diagnostics,
                    top_k=args.top_k,
                    run_id=uuid4().hex[:8],
                    dataset_sha256_value=dataset_sha256_value,
                    git_commit=git_commit,
                    llm_model=settings.llm_model,
                    embedding_model=settings.embedding_model,
                ),
                "bm25": build_report(
                    bm25_diagnostics,
                    top_k=args.top_k,
                    run_id=uuid4().hex[:8],
                    dataset_sha256_value=dataset_sha256_value,
                    git_commit=git_commit,
                    llm_model=settings.llm_model,
                    embedding_model=settings.embedding_model,
                ),
            }
            for backend, report in reports.items():
                r = report.retrieval
                print(
                    f"  [{BACKEND_LABEL[backend]}] Hit@1={r.hit_at_1:.3f} Hit@3={r.hit_at_3:.3f} "
                    f"Hit@5={r.hit_at_5:.3f} Hit@10={r.hit_at_10:.3f}  Recall@1={r.recall_at_1:.3f} "
                    f"Recall@3={r.recall_at_3:.3f} Recall@5={r.recall_at_5:.3f} Recall@10={r.recall_at_10:.3f}  "
                    f"MRR={r.mrr:.3f}"
                )

            fts_stages = stage_by_question(fts_diagnostics, covered_dia_ids=state.covered_dia_ids, k=5)
            bm25_stages = stage_by_question(bm25_diagnostics, covered_dia_ids=state.covered_dia_ids, k=5)
            fts_counts = stage_counts(fts_stages)
            bm25_counts = stage_counts(bm25_stages)
            transitions = transition_counts(fts_stages, bm25_stages)

            v1_stages = stage_by_question(v1_report.questions, covered_dia_ids=state.covered_dia_ids, k=5)
            previous_retrieval_miss_indices = sorted(i for i, s in v1_stages.items() if s == "RETRIEVAL_MISS")

            interesting_indices = sorted(
                set(previous_retrieval_miss_indices)
                | {i for i in fts_stages if fts_stages[i] != bm25_stages.get(i)}
            )
            # Part D: diagnostics reuse each QuestionResult's already-computed
            # candidates -- no retrieval call happens in this loop.
            with timings.measure("diagnostics"):
                results_by_index = {r.question_index: r for r in results}
                branch_diagnostics: dict[int, BranchDiagnostic] = {
                    index: branch_diagnostic_from_result(
                        results_by_index[index],
                        gold_lookup=state.gold_lookup,
                        gold_evidence=sample.qa[index].evidence,
                    )
                    for index in interesting_indices
                    if index in results_by_index
                }

            db_counts_after = snapshot_db_counts(db, user=state.user, conversation=state.conversation)
    finally:
        reset_engine()
        reset_config()

    print("\nAfter ablation:")
    print(f"  Conversations: {db_counts_after.conversation_count}")
    print(f"  Messages:      {db_counts_after.message_count}")
    print(f"  Memories:      {db_counts_after.memory_count}")
    unchanged = (
        db_counts_before.conversation_count == db_counts_after.conversation_count
        and db_counts_before.message_count == db_counts_after.message_count
        and db_counts_before.memory_count == db_counts_after.memory_count
        and db_counts_before.memory_checksum == db_counts_after.memory_checksum
    )
    print(f"  DB state unchanged: {unchanged}")
    if not unchanged:
        print("  WARNING: database state changed during a retrieval-only ablation.")

    print("\nK=5 stage diagnosis:")
    print(f"  {'STAGE':<16}{'FTS':>6}{'BM25':>6}{'DELTA':>8}")
    for key in STAGE_KEYS:
        print(f"  {key:<16}{fts_counts[key]:>6}{bm25_counts[key]:>6}{bm25_counts[key] - fts_counts[key]:>8}")

    print(f"\nPrevious V1 RETRIEVAL_MISS count: {len(previous_retrieval_miss_indices)}")
    fixed = sum(1 for i in previous_retrieval_miss_indices if bm25_stages.get(i) == "SUCCESS")
    improved_not_fixed = sum(
        1 for i in previous_retrieval_miss_indices if bm25_stages.get(i) == "RANKING_MISS"
    )
    unchanged_miss = sum(1 for i in previous_retrieval_miss_indices if bm25_stages.get(i) == "RETRIEVAL_MISS")
    regressed = sum(
        1
        for i in previous_retrieval_miss_indices
        if bm25_stages.get(i) not in ("RETRIEVAL_MISS", "RANKING_MISS", "SUCCESS")
    )
    print(f"  fixed (-> SUCCESS):            {fixed}")
    print(f"  improved but still below K=5:  {improved_not_fixed}")
    print(f"  unchanged (still miss):        {unchanged_miss}")
    print(f"  other/regressed:               {regressed}")

    retrieval_miss_24_analysis = []
    for index in previous_retrieval_miss_indices:
        fts_diag = fts_diagnostics[index]
        bm25_diag = bm25_diagnostics[index]
        branch = branch_diagnostics.get(index)
        retrieval_miss_24_analysis.append(
            {
                "question_index": index,
                "question": fts_diag.question,
                "fts_status": fts_stages.get(index),
                "bm25_status": bm25_stages.get(index),
                "fts_final_rank": fts_diag.retrieval_metrics.rank if fts_diag.retrieval_metrics else None,
                "bm25_final_rank": bm25_diag.retrieval_metrics.rank if bm25_diag.retrieval_metrics else None,
                "branch_diagnostic": branch.to_dict() if branch is not None else None,
            }
        )

    branch_examples = [
        branch_diagnostics[i].to_dict()
        for i in sorted(branch_diagnostics)
        if fts_stages.get(i) != bm25_stages.get(i)
    ][:5]

    output_dir = Path(args.output_dir) if args.output_dir else None
    fts_path, bm25_path, comparison_path = default_result_paths(args.sample)
    if output_dir is not None:
        fts_path = output_dir / fts_path.name
        bm25_path = output_dir / bm25_path.name
        comparison_path = output_dir / comparison_path.name

    save_backend_result(
        reports["postgres_fts"],
        backend="postgres_fts",
        base_v1_path=args.v1_result,
        db_counts_before=db_counts_before,
        db_counts_after=db_counts_after,
        path=fts_path,
    )
    save_backend_result(
        reports["bm25"],
        backend="bm25",
        base_v1_path=args.v1_result,
        db_counts_before=db_counts_before,
        db_counts_after=db_counts_after,
        path=bm25_path,
    )
    save_comparison(
        fts_report=reports["postgres_fts"],
        bm25_report=reports["bm25"],
        fts_stage_counts=fts_counts,
        bm25_stage_counts=bm25_counts,
        transitions=transitions,
        retrieval_miss_24_analysis=retrieval_miss_24_analysis,
        branch_examples=branch_examples,
        path=comparison_path,
    )

    print(f"\nSaved Postgres FTS result to {fts_path}")
    print(f"Saved BM25 result to {bm25_path}")
    print(f"Saved comparison to {comparison_path}")

    print()
    print(timings.render())
    print()
    print(counters.render(question_count=len(sample.qa)))

    # A fully completed run's checkpoint is no longer needed for resume.
    delete_ablation_checkpoint(args.sample)

    return 0


def _resolve_embedding_dimension(state: BenchmarkState) -> int:
    """Best-effort embedding dimension for the cache key, from any loaded memory."""

    for memory in state.active_memories:
        if memory.embedding is not None:
            return len(memory.embedding)
    # No active memory to infer from; fall back to the configured provider's
    # dimension being validated on first real embedding call instead.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
