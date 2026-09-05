"""Pure unit tests for evidence-provenance retrieval metrics. No network, no DB."""

from __future__ import annotations

import pytest

from evals.locomo.metrics import aggregate_retrieval_metrics, evaluate_question_retrieval, group_retrieval_by_category
from evals.locomo.schemas import QuestionDiagnostic


def test_hit_and_recall_when_evidence_is_covered_at_rank_two() -> None:
    provenance_by_rank = [{"D1:9"}, {"D1:3", "D1:5"}, {"D1:7"}]
    metrics = evaluate_question_retrieval(provenance_by_rank, ["D1:3"])

    assert metrics.rank == 2
    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_3 == 1
    assert metrics.hit_at_10 == 1
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0


def test_recall_is_partial_when_only_some_gold_evidence_is_covered() -> None:
    provenance_by_rank = [{"D1:1"}, {"D1:2"}, set()]
    metrics = evaluate_question_retrieval(provenance_by_rank, ["D1:1", "D1:2", "D1:3"])

    assert metrics.recall_at_1 == pytest.approx(1 / 3)
    assert metrics.recall_at_3 == pytest.approx(2 / 3)
    assert metrics.hit_at_1 == 1


def test_no_gold_evidence_is_not_scored_as_zero_recall() -> None:
    metrics = evaluate_question_retrieval([{"D1:1"}], [])

    assert metrics.has_evidence is False
    assert metrics.recall_at_10 == 0.0
    assert metrics.hit_at_10 == 0
    assert metrics.rank is None


def _diagnostic(*, has_evidence: bool, hit: int, category: str = "single_hop") -> QuestionDiagnostic:
    provenance = [{"D1:1"}] if hit else [set()]
    metrics = evaluate_question_retrieval(provenance, ["D1:1"] if has_evidence else [])
    return QuestionDiagnostic(
        sample_id="s",
        question_index=0,
        question="q",
        category_id=4,
        category_name=category,
        retrieval_metrics=metrics,
    )


def test_aggregate_excludes_no_evidence_questions_but_counts_them() -> None:
    diagnostics = [
        _diagnostic(has_evidence=True, hit=1),
        _diagnostic(has_evidence=True, hit=0),
        _diagnostic(has_evidence=False, hit=1),
    ]

    aggregate = aggregate_retrieval_metrics(diagnostics)

    assert aggregate.questions_evaluated == 2
    assert aggregate.questions_excluded_no_evidence == 1
    assert aggregate.hit_at_1 == pytest.approx(0.5)


def test_group_by_category_only_includes_scored_questions() -> None:
    diagnostics = [
        _diagnostic(has_evidence=True, hit=1, category="single_hop"),
        _diagnostic(has_evidence=True, hit=1, category="temporal"),
    ]
    by_category = group_retrieval_by_category(diagnostics)

    assert set(by_category) == {"single_hop", "temporal"}
    assert by_category["single_hop"].questions_evaluated == 1
