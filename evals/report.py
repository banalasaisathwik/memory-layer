"""Plain-text table rendering and JSON persistence for a RunReport."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import AggregateMetrics, RunReport


def _case_row(case_id: str, metrics) -> str:
    return (
        f"{case_id:<28} {metrics.hit_at_1:<6} {metrics.hit_at_3:<6} {metrics.hit_at_5:<6} "
        f"{metrics.reciprocal_rank:.2f}"
    )


def _aggregate_block(title: str, aggregate: AggregateMetrics) -> str:
    lines = [
        title,
        f"Cases: {aggregate.cases}",
        "",
        f"Hit@1:     {aggregate.hit_at_1:.2f}",
        f"Hit@3:     {aggregate.hit_at_3:.2f}",
        f"Hit@5:     {aggregate.hit_at_5:.2f}",
        "",
        f"Recall@1:  {aggregate.recall_at_1:.2f}",
        f"Recall@3:  {aggregate.recall_at_3:.2f}",
        f"Recall@5:  {aggregate.recall_at_5:.2f}",
        "",
        f"MRR:       {aggregate.mrr:.2f}",
        "",
        f"Isolation failures:  {aggregate.isolation_failures}",
        f"Superseded returned: {aggregate.superseded_returned}",
    ]
    return "\n".join(lines)


def render_table(report: RunReport) -> str:
    """Render per-case, category, and aggregate results as plain text."""

    header = f"{'Case':<28} {'Hit@1':<6} {'Hit@3':<6} {'Hit@5':<6} RR"
    divider = "-" * len(header)
    case_lines = [_case_row(result.case_id, result.metrics) for result in report.cases]

    sections = [
        f"Dataset: {report.dataset}  top_k={report.top_k}  run_at={report.run_at.isoformat()}",
        f"LLM model: {report.llm_model}  Embedding model: {report.embedding_model}",
        "",
        header,
        divider,
        *case_lines,
        "",
        _aggregate_block("Aggregate", report.aggregate),
    ]

    if report.by_category:
        sections.append("")
        sections.append("By category")
        sections.append("-" * len("By category"))
        for category, aggregate in report.by_category.items():
            sections.append("")
            sections.append(_aggregate_block(category, aggregate))

    return "\n".join(sections)


def save_report(report: RunReport, path: Path) -> None:
    """Write the full run report as JSON, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
