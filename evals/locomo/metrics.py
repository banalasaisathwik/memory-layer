"""Evidence-provenance retrieval metrics: Hit@K, Evidence Recall@K, MRR.

A retrieved Memory's provenance is the set of gold LoCoMo dia_ids its
Memory.source_message_ids resolve back to (via the dia_id <-> Message.id
mapping built at ingestion time). Metrics compare that provenance, rank by
rank, against one question's gold evidence dia_ids.

Questions with no gold evidence (``evidence == []`` in the released dataset)
are excluded from the aggregate Hit@K/Recall@K/MRR denominators -- a recall
score against zero gold items is not meaningful -- but every exclusion is
counted so the aggregate always reports how many questions it left out.
"""

from __future__ import annotations

from .schemas import QuestionDiagnostic, QuestionRetrievalMetrics, RetrievalAggregate

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def evaluate_question_retrieval(
    provenance_by_rank: list[set[str]],
    gold_evidence: list[str],
) -> QuestionRetrievalMetrics:
    """Score one question's ranked provenance sets against its gold evidence dia_ids."""

    gold = set(gold_evidence)
    has_evidence = bool(gold)

    rank: int | None = None
    for index, provenance in enumerate(provenance_by_rank, start=1):
        if provenance & gold:
            rank = index
            break

    def recall_at(k: int) -> float:
        if not has_evidence:
            return 0.0
        covered: set[str] = set()
        for provenance in provenance_by_rank[:k]:
            covered |= provenance & gold
        return len(covered) / len(gold)

    def hit_at(k: int) -> int:
        return 1 if rank is not None and rank <= k else 0

    return QuestionRetrievalMetrics(
        rank=rank,
        hit_at_1=hit_at(1),
        hit_at_3=hit_at(3),
        hit_at_5=hit_at(5),
        hit_at_10=hit_at(10),
        recall_at_1=recall_at(1),
        recall_at_3=recall_at(3),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        reciprocal_rank=(1.0 / rank if rank is not None else 0.0),
        has_evidence=has_evidence,
        gold_evidence_count=len(gold),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval_metrics(diagnostics: list[QuestionDiagnostic]) -> RetrievalAggregate:
    """Average retrieval metrics over questions with gold evidence and retrieval_metrics set."""

    scored = [d for d in diagnostics if d.retrieval_metrics is not None]
    with_evidence = [d for d in scored if d.retrieval_metrics.has_evidence]
    excluded = len(scored) - len(with_evidence)

    return RetrievalAggregate(
        questions_evaluated=len(with_evidence),
        questions_excluded_no_evidence=excluded,
        hit_at_1=_mean([d.retrieval_metrics.hit_at_1 for d in with_evidence]),
        hit_at_3=_mean([d.retrieval_metrics.hit_at_3 for d in with_evidence]),
        hit_at_5=_mean([d.retrieval_metrics.hit_at_5 for d in with_evidence]),
        hit_at_10=_mean([d.retrieval_metrics.hit_at_10 for d in with_evidence]),
        recall_at_1=_mean([d.retrieval_metrics.recall_at_1 for d in with_evidence]),
        recall_at_3=_mean([d.retrieval_metrics.recall_at_3 for d in with_evidence]),
        recall_at_5=_mean([d.retrieval_metrics.recall_at_5 for d in with_evidence]),
        recall_at_10=_mean([d.retrieval_metrics.recall_at_10 for d in with_evidence]),
        mrr=_mean([d.retrieval_metrics.reciprocal_rank for d in with_evidence]),
        isolation_failures=sum(d.isolation_failures for d in scored),
    )


def group_retrieval_by_category(diagnostics: list[QuestionDiagnostic]) -> dict[str, RetrievalAggregate]:
    """Aggregate retrieval metrics per LoCoMo category, in first-seen order."""

    by_category: dict[str, list[QuestionDiagnostic]] = {}
    for diagnostic in diagnostics:
        if diagnostic.retrieval_metrics is None:
            continue
        by_category.setdefault(diagnostic.category_name, []).append(diagnostic)
    return {category: aggregate_retrieval_metrics(items) for category, items in by_category.items()}
