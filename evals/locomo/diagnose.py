"""V0-vs-V1 comparison and stage-level diagnostics for remaining retrieval failures.

Stage classification (K=5), for every question that has gold evidence:

    NO_GOLD_MEMORY   -- no active Memory's provenance overlaps any gold dia_id.
                        The failure happened before retrieval (extraction/write).
    RETRIEVAL_MISS   -- a gold-linked active memory exists, but search() never
                        surfaced it anywhere in the top-K retrieved set.
    RANKING_MISS     -- a gold-linked memory was retrieved, but only below the
                        K=5 cutoff.
    SUCCESS          -- a gold-linked memory was retrieved at rank <= 5.

This is deliberately conservative: current provenance is per-interaction, not
per-candidate (see LocomoRunMetadata.provenance_caveat), so "gold-linked" is
an approximation of true evidence attribution, not exact. This module does
not attempt a perfect automatic root-cause engine -- only enough signal to
see where Recall@5 is being lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Memory, User

from .schemas import LocomoRunReport, QuestionDiagnostic, RetrievalAggregate

StageOutcome = Literal["NO_GOLD_MEMORY", "RETRIEVAL_MISS", "RANKING_MISS", "SUCCESS"]


def active_memory_covered_dia_ids(
    db: Session,
    *,
    user_id: str,
    message_id_to_dia_id: dict[str, str],
) -> set[str]:
    """Every gold dia_id that at least one *active* memory's provenance resolves to."""

    user = db.scalar(select(User).where(User.external_id == user_id))
    if user is None:
        raise RuntimeError(f"No user exists for external ID {user_id!r}.")

    covered: set[str] = set()
    for memory in db.scalars(select(Memory).where(Memory.user_id == user.id, Memory.is_active.is_(True))):
        for message_id in memory.source_message_ids:
            dia_id = message_id_to_dia_id.get(message_id)
            if dia_id is not None:
                covered.add(dia_id)
    return covered


def classify_stage(
    diagnostic: QuestionDiagnostic,
    *,
    covered_dia_ids: set[str],
    k: int = 5,
) -> StageOutcome | None:
    """Classify one question's K=5 outcome; None if it has no gold evidence at all."""

    gold = set(diagnostic.gold_evidence)
    if not gold:
        return None
    if not (gold & covered_dia_ids):
        return "NO_GOLD_MEMORY"
    rank = diagnostic.retrieval_metrics.rank if diagnostic.retrieval_metrics is not None else None
    if rank is None:
        return "RETRIEVAL_MISS"
    if rank <= k:
        return "SUCCESS"
    return "RANKING_MISS"


@dataclass(frozen=True)
class StageBreakdown:
    overall: dict[str, int]
    by_category: dict[str, dict[str, int]]


def build_stage_breakdown(
    diagnostics: list[QuestionDiagnostic],
    *,
    covered_dia_ids: set[str],
    k: int = 5,
) -> StageBreakdown:
    overall: dict[str, int] = {"NO_GOLD_MEMORY": 0, "RETRIEVAL_MISS": 0, "RANKING_MISS": 0, "SUCCESS": 0}
    by_category: dict[str, dict[str, int]] = {}
    for diagnostic in diagnostics:
        outcome = classify_stage(diagnostic, covered_dia_ids=covered_dia_ids, k=k)
        if outcome is None:
            continue
        overall[outcome] += 1
        bucket = by_category.setdefault(
            diagnostic.category_name, {"NO_GOLD_MEMORY": 0, "RETRIEVAL_MISS": 0, "RANKING_MISS": 0, "SUCCESS": 0}
        )
        bucket[outcome] += 1
    return StageBreakdown(overall=overall, by_category=by_category)


def _retrieval_delta(v0: RetrievalAggregate, v1: RetrievalAggregate) -> dict[str, float]:
    fields = [
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
    return {field: getattr(v1, field) - getattr(v0, field) for field in fields}


@dataclass(frozen=True)
class ComparisonReport:
    overall_v0: RetrievalAggregate
    overall_v1: RetrievalAggregate
    overall_delta: dict[str, float]
    by_category: dict[str, dict[str, object]]

    def to_json_dict(self) -> dict:
        return {
            "overall": {
                "v0": self.overall_v0.model_dump(mode="json"),
                "v1": self.overall_v1.model_dump(mode="json"),
                "delta": self.overall_delta,
            },
            "by_category": self.by_category,
        }


def compare_reports(v0: LocomoRunReport, v1: LocomoRunReport) -> ComparisonReport:
    if v0.retrieval is None or v1.retrieval is None:
        raise ValueError("Both reports must include retrieval metrics to compare.")

    by_category: dict[str, dict[str, object]] = {}
    categories = sorted(set(v0.retrieval_by_category) | set(v1.retrieval_by_category))
    for category in categories:
        v0_cat = v0.retrieval_by_category.get(category)
        v1_cat = v1.retrieval_by_category.get(category)
        if v0_cat is None or v1_cat is None:
            continue
        by_category[category] = {
            "v0": v0_cat.model_dump(mode="json"),
            "v1": v1_cat.model_dump(mode="json"),
            "delta": _retrieval_delta(v0_cat, v1_cat),
        }

    return ComparisonReport(
        overall_v0=v0.retrieval,
        overall_v1=v1.retrieval,
        overall_delta=_retrieval_delta(v0.retrieval, v1.retrieval),
        by_category=by_category,
    )


def render_stage_breakdown(breakdown: StageBreakdown) -> str:
    lines = ["Stage diagnostics at K=5 (questions with gold evidence only)", "", "Overall:"]
    for key in ["SUCCESS", "NO_GOLD_MEMORY", "RETRIEVAL_MISS", "RANKING_MISS"]:
        lines.append(f"  {key:<16}{breakdown.overall[key]}")
    lines.append("")
    for category, counts in breakdown.by_category.items():
        lines.append(f"{category}:")
        for key in ["SUCCESS", "NO_GOLD_MEMORY", "RETRIEVAL_MISS", "RANKING_MISS"]:
            lines.append(f"  {key:<16}{counts[key]}")
        lines.append("")
    return "\n".join(lines)


def render_comparison(comparison: ComparisonReport) -> str:
    lines = ["V0 -> V1 retrieval comparison (overall)", ""]
    header = f"{'metric':<12}{'V0':>8}{'V1':>8}{'delta':>10}"
    lines.append(header)
    for field in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"]:
        v0_value = getattr(comparison.overall_v0, field)
        v1_value = getattr(comparison.overall_v1, field)
        delta = comparison.overall_delta[field]
        lines.append(f"{field:<12}{v0_value:>8.3f}{v1_value:>8.3f}{delta:>+10.3f}")
    lines.append("")
    for category, block in comparison.by_category.items():
        lines.append(f"-- {category} --")
        for field in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"]:
            v0_value = block["v0"][field]
            v1_value = block["v1"][field]
            delta = block["delta"][field]
            lines.append(f"{field:<12}{v0_value:>8.3f}{v1_value:>8.3f}{delta:>+10.3f}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    from src.config import configure, reset_config
    from src.database import SessionLocal, create_tables, reset_engine

    from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
    from .recover import build_dia_id_mapping, session_message_slices
    from .dataset import DEFAULT_DATASET_PATH, load_locomo_dataset

    parser = argparse.ArgumentParser(description="Compare a V0 and V1 LoCoMo retrieval report and diagnose K=5 failures.")
    parser.add_argument("--v0", required=True)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--conversation-id", default="conversation")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    v0_raw = _json.loads(Path(args.v0).read_text(encoding="utf-8"))
    v0_report = LocomoRunReport.model_validate(v0_raw)

    v1_raw = _json.loads(Path(args.v1).read_text(encoding="utf-8"))
    v1_report = LocomoRunReport.model_validate(v1_raw["report"] if "report" in v1_raw else v1_raw)

    comparison = compare_reports(v0_report, v1_report)
    print(render_comparison(comparison))

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

    try:
        with SessionLocal() as db:
            from src.database.models import Conversation, User as UserModel
            from sqlalchemy import select as _select

            user = db.scalar(_select(UserModel).where(UserModel.external_id == user_id))
            conversation = db.scalar(
                _select(Conversation).where(
                    Conversation.external_id == args.conversation_id, Conversation.user_id == user.id
                )
            )
            slices = session_message_slices(db, sample, conversation=conversation)
            _, message_id_to_dia_id = build_dia_id_mapping(sample, slices)
            covered_dia_ids = active_memory_covered_dia_ids(
                db, user_id=user_id, message_id_to_dia_id=message_id_to_dia_id
            )
            breakdown = build_stage_breakdown(v1_report.questions, covered_dia_ids=covered_dia_ids, k=5)
    finally:
        reset_engine()
        reset_config()

    print()
    print(render_stage_breakdown(breakdown))

    output_path = Path(args.output) if args.output else Path(args.v1).with_name(Path(args.v1).stem + "-comparison.json")
    output_path.write_text(
        _json.dumps(
            {
                "comparison": comparison.to_json_dict(),
                "stage_breakdown": {"overall": breakdown.overall, "by_category": breakdown.by_category},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved comparison + stage diagnostics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
