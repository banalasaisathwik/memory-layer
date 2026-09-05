"""Rank-only reciprocal rank fusion for independent retrieval branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.database.models import Memory, Message


RRF_K = 60


class _Identified(Protocol):
    id: object


def _rrf_branch_scores(
    branches: dict[str, list[_Identified]],
    *,
    k: int,
) -> dict[str, dict[str, object]]:
    """Compute score(item) = Σ 1/(k+rank_in_branch), keyed by str(item.id).

    Shared by every fusion caller so there is exactly one place that
    implements the RRF formula; only branch composition and tie-breaking
    differ between durable-memory fusion and older-message fusion.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    by_id: dict[str, dict[str, object]] = {}
    for branch_name, items in branches.items():
        for rank, item in enumerate(items, start=1):
            key = str(item.id)
            entry = by_id.setdefault(
                key,
                {"item": item, "score": 0.0, "ranks": {}},
            )
            entry["score"] = float(entry["score"]) + (1 / (k + rank))
            entry["ranks"][branch_name] = rank  # type: ignore[index]
    return by_id


@dataclass(frozen=True)
class FusedMemory:
    """One memory plus the rank contribution visible to retrieval callers."""

    memory: Memory
    score: float
    structured_rank: int | None
    lexical_rank: int | None
    vector_rank: int | None


def reciprocal_rank_fusion(
    *,
    structured: list[Memory],
    lexical: list[Memory],
    vector: list[Memory],
    k: int = RRF_K,
) -> list[FusedMemory]:
    """Fuse ordered result lists without combining incompatible raw scores."""

    by_id = _rrf_branch_scores(
        {"structured": structured, "lexical": lexical, "vector": vector},
        k=k,
    )
    fused = [
        FusedMemory(
            memory=entry["item"],  # type: ignore[arg-type]
            score=float(entry["score"]),
            structured_rank=entry["ranks"].get("structured"),  # type: ignore[union-attr]
            lexical_rank=entry["ranks"].get("lexical"),  # type: ignore[union-attr]
            vector_rank=entry["ranks"].get("vector"),  # type: ignore[union-attr]
        )
        for entry in by_id.values()
    ]
    return sorted(
        fused,
        key=lambda hit: (
            -hit.score,
            -hit.memory.importance,
            not hit.memory.is_active,
            -hit.memory.updated_at.timestamp(),
            str(hit.memory.id),
        ),
    )


@dataclass(frozen=True)
class FusedMessage:
    """One older conversation message plus its per-branch rank contribution."""

    message: Message
    score: float
    lexical_rank: int | None
    semantic_rank: int | None


def fuse_message_rankings(
    *,
    lexical: list[Message],
    semantic: list[Message],
    k: int = RRF_K,
) -> list[FusedMessage]:
    """Fuse lexical- and semantic-ranked older messages with equal-weight RRF.

    Callers must pass each branch in its own relevance-rank order (best
    match first), not pre-sorted chronologically -- fusing already
    chronologically-sorted branches discards the ranking signal this
    function exists to combine. The caller is responsible for capping to
    the extraction-context limit and for the final chronological sort used
    only for LLM readability; this function never reorders by recency or
    importance, since these are transient conversation rows, not durable
    memories.
    """

    by_id = _rrf_branch_scores({"lexical": lexical, "semantic": semantic}, k=k)
    fused = [
        FusedMessage(
            message=entry["item"],  # type: ignore[arg-type]
            score=float(entry["score"]),
            lexical_rank=entry["ranks"].get("lexical"),  # type: ignore[union-attr]
            semantic_rank=entry["ranks"].get("semantic"),  # type: ignore[union-attr]
        )
        for entry in by_id.values()
    ]

    def _best_branch_rank(item: FusedMessage) -> int:
        ranks = [rank for rank in (item.lexical_rank, item.semantic_rank) if rank is not None]
        return min(ranks)

    return sorted(
        fused,
        key=lambda item: (
            -item.score,
            _best_branch_rank(item),
            item.message.created_at,
            str(item.message.id),
        ),
    )
