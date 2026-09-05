"""Aggregate QA scores, kept strictly separate from retrieval metrics.

F1 (categories 1-4: multi_hop/temporal/open_domain/single_hop) and
adversarial abstention accuracy (category 5) are two different measurements
of two different things and must never be averaged into one number -- see
evals/README.md.
"""

from __future__ import annotations

from .schemas import QAAggregate, QuestionDiagnostic


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_qa_metrics(diagnostics: list[QuestionDiagnostic]) -> QAAggregate:
    f1_scored = [d for d in diagnostics if d.qa_score is not None and d.category_id != 5]
    adversarial_scored = [d for d in diagnostics if d.qa_score is not None and d.category_id == 5]

    return QAAggregate(
        f1_questions=len(f1_scored),
        f1_mean=_mean([d.qa_score for d in f1_scored]),
        adversarial_questions=len(adversarial_scored),
        adversarial_abstention_accuracy=_mean([d.qa_score for d in adversarial_scored]),
    )


def group_qa_by_category(diagnostics: list[QuestionDiagnostic]) -> dict[str, QAAggregate]:
    by_category: dict[str, list[QuestionDiagnostic]] = {}
    for diagnostic in diagnostics:
        if diagnostic.qa_score is None:
            continue
        by_category.setdefault(diagnostic.category_name, []).append(diagnostic)
    return {category: aggregate_qa_metrics(items) for category, items in by_category.items()}
