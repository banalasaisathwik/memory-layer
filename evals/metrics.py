"""Deterministic, LLM-free retrieval metrics for search_memories() output."""

from __future__ import annotations

from .schemas import AggregateMetrics, CaseMetrics, CaseResult, ExpectedMemory


K_VALUES: tuple[int, ...] = (1, 3, 5)


def normalize_text(text: str) -> str:
    """Normalize only case and whitespace, matching the gold matcher's guarantee."""

    return " ".join(text.casefold().split())


def matches_expected(memory_text: str, expected: ExpectedMemory) -> bool:
    """A memory matches when every required term appears as a substring."""

    normalized = normalize_text(memory_text)
    return all(normalize_text(term) in normalized for term in expected.required_terms)


def rank_of_first_relevant(
    memory_texts: list[str],
    expected_memories: list[ExpectedMemory],
) -> int | None:
    """Return the 1-indexed rank of the first hit that matches any gold memory."""

    for rank, memory_text in enumerate(memory_texts, start=1):
        if any(matches_expected(memory_text, expected) for expected in expected_memories):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> int:
    """1 if a relevant memory was found at or before position k, else 0."""

    return 1 if rank is not None and rank <= k else 0


def recall_at_k(
    memory_texts: list[str],
    expected_memories: list[ExpectedMemory],
    k: int,
) -> float:
    """Fraction of gold memories matched by at least one hit within the top k."""

    top = memory_texts[:k]
    matched = sum(
        1
        for expected in expected_memories
        if any(matches_expected(text, expected) for text in top)
    )
    return matched / len(expected_memories)


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0


def evaluate_retrieval(
    memory_texts: list[str],
    expected_memories: list[ExpectedMemory],
) -> CaseMetrics:
    """Compute Hit@K, Recall@K, and RR for one case's ranked results."""

    rank = rank_of_first_relevant(memory_texts, expected_memories)
    return CaseMetrics(
        rank=rank,
        hit_at_1=hit_at_k(rank, 1),
        hit_at_3=hit_at_k(rank, 3),
        hit_at_5=hit_at_k(rank, 5),
        recall_at_1=recall_at_k(memory_texts, expected_memories, 1),
        recall_at_3=recall_at_k(memory_texts, expected_memories, 3),
        recall_at_5=recall_at_k(memory_texts, expected_memories, 5),
        reciprocal_rank=reciprocal_rank(rank),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_case_results(results: list[CaseResult]) -> AggregateMetrics:
    """Average metrics across cases; isolation and supersession failures are summed."""

    if not results:
        return AggregateMetrics(
            cases=0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            hit_at_5=0.0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mrr=0.0,
            isolation_failures=0,
            superseded_returned=0,
        )

    return AggregateMetrics(
        cases=len(results),
        hit_at_1=_mean([r.metrics.hit_at_1 for r in results]),
        hit_at_3=_mean([r.metrics.hit_at_3 for r in results]),
        hit_at_5=_mean([r.metrics.hit_at_5 for r in results]),
        recall_at_1=_mean([r.metrics.recall_at_1 for r in results]),
        recall_at_3=_mean([r.metrics.recall_at_3 for r in results]),
        recall_at_5=_mean([r.metrics.recall_at_5 for r in results]),
        mrr=_mean([r.metrics.reciprocal_rank for r in results]),
        isolation_failures=sum(r.isolation_failures for r in results),
        superseded_returned=sum(r.superseded_returned for r in results),
    )


def group_by_category(results: list[CaseResult]) -> dict[str, AggregateMetrics]:
    """Aggregate metrics per case category, in first-seen category order."""

    by_category: dict[str, list[CaseResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)
    return {category: aggregate_case_results(cases) for category, cases in by_category.items()}
