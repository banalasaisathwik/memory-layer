"""Plain-text table rendering and JSON persistence for a LocomoRunReport."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import LocomoRunReport, QAAggregate, RetrievalAggregate


def _retrieval_block(title: str, aggregate: RetrievalAggregate) -> str:
    return "\n".join(
        [
            title,
            f"Questions evaluated: {aggregate.questions_evaluated}  "
            f"(excluded, no gold evidence: {aggregate.questions_excluded_no_evidence})",
            "",
            f"Hit@1:        {aggregate.hit_at_1:.2f}",
            f"Hit@3:        {aggregate.hit_at_3:.2f}",
            f"Hit@5:        {aggregate.hit_at_5:.2f}",
            f"Hit@10:       {aggregate.hit_at_10:.2f}",
            "",
            f"Recall@1:     {aggregate.recall_at_1:.2f}",
            f"Recall@3:     {aggregate.recall_at_3:.2f}",
            f"Recall@5:     {aggregate.recall_at_5:.2f}",
            f"Recall@10:    {aggregate.recall_at_10:.2f}",
            "",
            f"MRR:          {aggregate.mrr:.2f}",
            f"Isolation failures: {aggregate.isolation_failures}",
        ]
    )


def _qa_block(title: str, aggregate: QAAggregate) -> str:
    return "\n".join(
        [
            title,
            f"F1 questions: {aggregate.f1_questions}   F1 mean: {aggregate.f1_mean:.3f}",
            f"Adversarial questions: {aggregate.adversarial_questions}   "
            f"Abstention accuracy: {aggregate.adversarial_abstention_accuracy:.3f}",
        ]
    )


def render_report(report: LocomoRunReport) -> str:
    meta = report.metadata
    sections = [
        f"LoCoMo benchmark run  mode={meta.mode}  top_k={meta.top_k}  run_at={meta.run_at.isoformat()}",
        f"dataset={meta.dataset}  sha256={meta.dataset_sha256[:16]}...  git_commit={meta.git_commit}",
        f"LLM model: {meta.llm_model}  Embedding model: {meta.embedding_model}",
        f"Conversations ingested: {meta.conversations_ingested}   Questions evaluated: {meta.questions_evaluated}",
        f"NOTE: {meta.provenance_caveat}",
        "",
    ]

    if report.retrieval is not None:
        sections.append(_retrieval_block("Retrieval (overall)", report.retrieval))
        sections.append("")
        for category, aggregate in report.retrieval_by_category.items():
            sections.append(_retrieval_block(f"Retrieval ({category})", aggregate))
            sections.append("")

    if report.qa is not None:
        sections.append(_qa_block("QA (overall)", report.qa))
        sections.append("")
        for category, aggregate in report.qa_by_category.items():
            sections.append(_qa_block(f"QA ({category})", aggregate))
            sections.append("")

    sections.append("Retrieval metrics measure whether search() surfaced evidence-grounded memories.")
    sections.append("QA metrics measure whether answer() produced a correct final answer. Never compare the two directly.")

    return "\n".join(sections)


def save_report(report: LocomoRunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
