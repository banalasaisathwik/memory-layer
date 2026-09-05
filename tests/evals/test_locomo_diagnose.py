"""Pure unit tests for evals/locomo/diagnose.py's stage classification.

No database, no provider calls: these exercise classify_stage() directly
against constructed QuestionDiagnostic objects.
"""

from __future__ import annotations

from evals.locomo.diagnose import classify_stage
from evals.locomo.schemas import QuestionDiagnostic, QuestionRetrievalMetrics


def _diagnostic(*, gold_evidence: list[str], rank: int | None) -> QuestionDiagnostic:
    return QuestionDiagnostic(
        sample_id="conv-test",
        question_index=0,
        question="q",
        category_id=4,
        category_name="single_hop",
        gold_evidence=gold_evidence,
        retrieval_metrics=QuestionRetrievalMetrics(
            rank=rank,
            hit_at_1=1 if rank == 1 else 0,
            hit_at_3=1 if rank is not None and rank <= 3 else 0,
            hit_at_5=1 if rank is not None and rank <= 5 else 0,
            hit_at_10=1 if rank is not None and rank <= 10 else 0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            reciprocal_rank=(1.0 / rank if rank else 0.0),
            has_evidence=bool(gold_evidence),
            gold_evidence_count=len(gold_evidence),
        ),
    )


def test_no_gold_memory_when_no_active_memory_covers_the_evidence() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4"], rank=None)
    assert classify_stage(diagnostic, covered_dia_ids=set()) == "NO_GOLD_MEMORY"


def test_retrieval_miss_when_gold_memory_exists_but_was_never_retrieved() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4"], rank=None)
    assert classify_stage(diagnostic, covered_dia_ids={"D6:4"}) == "RETRIEVAL_MISS"


def test_ranking_miss_when_gold_memory_retrieved_below_k5_cutoff() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4"], rank=8)
    assert classify_stage(diagnostic, covered_dia_ids={"D6:4"}, k=5) == "RANKING_MISS"


def test_success_when_gold_memory_retrieved_at_or_above_k5_cutoff() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4"], rank=3)
    assert classify_stage(diagnostic, covered_dia_ids={"D6:4"}, k=5) == "SUCCESS"


def test_rank_exactly_at_k_is_success() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4"], rank=5)
    assert classify_stage(diagnostic, covered_dia_ids={"D6:4"}, k=5) == "SUCCESS"


def test_questions_with_no_gold_evidence_are_not_classified() -> None:
    diagnostic = _diagnostic(gold_evidence=[], rank=None)
    assert classify_stage(diagnostic, covered_dia_ids=set()) is None


def test_partial_gold_overlap_still_counts_as_gold_memory_present() -> None:
    diagnostic = _diagnostic(gold_evidence=["D6:4", "D6:5"], rank=2)
    assert classify_stage(diagnostic, covered_dia_ids={"D6:5"}, k=5) == "SUCCESS"
