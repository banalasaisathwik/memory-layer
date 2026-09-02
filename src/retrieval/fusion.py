"""Rank-only reciprocal rank fusion for independent retrieval branches."""

from __future__ import annotations

from dataclasses import dataclass

from src.database.models import Memory


RRF_K = 60


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

    if k < 1:
        raise ValueError("k must be greater than zero.")

    by_id: dict[str, dict[str, object]] = {}
    for branch_name, memories in (
        ("structured", structured),
        ("lexical", lexical),
        ("vector", vector),
    ):
        for rank, memory in enumerate(memories, start=1):
            key = str(memory.id)
            item = by_id.setdefault(
                key,
                {
                    "memory": memory,
                    "score": 0.0,
                    "structured_rank": None,
                    "lexical_rank": None,
                    "vector_rank": None,
                },
            )
            item["score"] = float(item["score"]) + (1 / (k + rank))
            item[f"{branch_name}_rank"] = rank

    fused = [
        FusedMemory(
            memory=item["memory"],  # type: ignore[arg-type]
            score=float(item["score"]),
            structured_rank=item["structured_rank"],  # type: ignore[arg-type]
            lexical_rank=item["lexical_rank"],  # type: ignore[arg-type]
            vector_rank=item["vector_rank"],  # type: ignore[arg-type]
        )
        for item in by_id.values()
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
