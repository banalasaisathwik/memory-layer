"""Unit tests for deterministic, LLM-free eval metrics."""

from __future__ import annotations

from evals.metrics import (
    aggregate_case_results,
    evaluate_retrieval,
    group_by_category,
    hit_at_k,
    matches_expected,
    normalize_text,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)
from evals.schemas import CaseMetrics, CaseResult, ExpectedMemory, RetrievedMemory


def test_normalize_text_folds_case_and_collapses_whitespace() -> None:
    assert normalize_text("  User   Prefers\nPostgreSQL ") == "user prefers postgresql"


def test_matches_expected_requires_every_term_as_a_substring() -> None:
    expected = ExpectedMemory(required_terms=["memorylayer", "postgresql"])

    assert matches_expected("User uses PostgreSQL for MemoryLayer", expected)
    assert not matches_expected("User uses PostgreSQL", expected)


def test_matches_expected_is_case_and_whitespace_insensitive() -> None:
    expected = ExpectedMemory(required_terms=["post gre sql".replace(" ", "")])

    assert matches_expected("  POSTGRESQL  is preferred", expected)


def test_rank_of_first_relevant_finds_first_matching_position() -> None:
    expected = [ExpectedMemory(required_terms=["postgresql"])]

    assert rank_of_first_relevant(["User likes chess", "User prefers PostgreSQL"], expected) == 2
    assert rank_of_first_relevant(["User prefers PostgreSQL"], expected) == 1
    assert rank_of_first_relevant(["User likes chess"], expected) is None
    assert rank_of_first_relevant([], expected) is None


def test_hit_at_k_true_only_at_or_before_rank() -> None:
    assert hit_at_k(2, 1) == 0
    assert hit_at_k(2, 3) == 1
    assert hit_at_k(None, 5) == 0


def test_recall_at_k_counts_distinct_gold_memories_matched_within_top_k() -> None:
    expected = [
        ExpectedMemory(required_terms=["python"]),
        ExpectedMemory(required_terms=["typescript"]),
    ]
    texts = ["User knows Python", "User knows TypeScript", "irrelevant"]

    assert recall_at_k(texts, expected, 1) == 0.5
    assert recall_at_k(texts, expected, 2) == 1.0
    assert recall_at_k(["irrelevant"], expected, 5) == 0.0


def test_reciprocal_rank_is_the_inverse_rank_or_zero() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_evaluate_retrieval_computes_all_fields_for_one_case() -> None:
    expected = [ExpectedMemory(required_terms=["postgresql"])]
    metrics = evaluate_retrieval(["irrelevant", "User prefers PostgreSQL"], expected)

    assert metrics.rank == 2
    assert (metrics.hit_at_1, metrics.hit_at_3, metrics.hit_at_5) == (0, 1, 1)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.reciprocal_rank == 0.5


def _case_result(case_id: str, category: str, *, rank: int | None) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        category=category,
        query="q",
        retrieved=[RetrievedMemory(memory_id="m1", memory_text="text", is_active=True, score=1.0)],
        metrics=CaseMetrics(
            rank=rank,
            hit_at_1=hit_at_k(rank, 1),
            hit_at_3=hit_at_k(rank, 3),
            hit_at_5=hit_at_k(rank, 5),
            recall_at_1=1.0 if rank == 1 else 0.0,
            recall_at_3=1.0 if rank is not None else 0.0,
            recall_at_5=1.0 if rank is not None else 0.0,
            reciprocal_rank=reciprocal_rank(rank),
        ),
        isolation_failures=0,
        superseded_returned=0,
    )


def test_aggregate_case_results_averages_across_cases() -> None:
    results = [_case_result("a", "single_fact", rank=1), _case_result("b", "single_fact", rank=None)]

    aggregate = aggregate_case_results(results)

    assert aggregate.cases == 2
    assert aggregate.hit_at_1 == 0.5
    assert aggregate.mrr == 0.5


def test_aggregate_case_results_sums_isolation_and_supersession_failures() -> None:
    leaking = _case_result("a", "user_isolation", rank=1).model_copy(
        update={"isolation_failures": 2, "superseded_returned": 1}
    )
    clean = _case_result("b", "user_isolation", rank=1)

    aggregate = aggregate_case_results([leaking, clean])

    assert aggregate.isolation_failures == 2
    assert aggregate.superseded_returned == 1


def test_aggregate_case_results_on_empty_input_is_zeroed_not_a_crash() -> None:
    aggregate = aggregate_case_results([])

    assert aggregate.cases == 0
    assert aggregate.mrr == 0.0


def test_group_by_category_partitions_and_aggregates_per_category() -> None:
    results = [
        _case_result("a", "single_fact", rank=1),
        _case_result("b", "single_fact", rank=None),
        _case_result("c", "update", rank=1),
    ]

    by_category = group_by_category(results)

    assert set(by_category) == {"single_fact", "update"}
    assert by_category["single_fact"].cases == 2
    assert by_category["single_fact"].hit_at_1 == 0.5
    assert by_category["update"].cases == 1
    assert by_category["update"].hit_at_1 == 1.0
