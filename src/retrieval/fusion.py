"""Rank-only reciprocal rank fusion for independent retrieval branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.database.models import Memory, Message


RRF_K = 60

# The project's current validated default for discounted_agreement_fusion,
# selected by the conv-30 fusion-policy ablation (evals/locomo/ablate_fusion.py):
# +3 net question-level gain over equal RRF on that one LoCoMo sample. Not
# claimed to be universally optimal -- just the current production default.
DEFAULT_AGREEMENT_DISCOUNT = 0.10


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


def _sort_fused_memories(fused: list[FusedMemory]) -> list[FusedMemory]:
    """The one, shared deterministic ordering every memory-fusion strategy uses.

    Fusion strategies below differ only in how ``score`` is computed; none of
    them may introduce their own tie-breaking, so gains are attributable to
    the score function alone.
    """

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


def reciprocal_rank_fusion(
    *,
    structured: list[Memory],
    lexical: list[Memory],
    vector: list[Memory],
    k: int = RRF_K,
) -> list[FusedMemory]:
    """Fuse ordered result lists without combining incompatible raw scores.

    Every branch contributes 1/(k+rank) with equal weight. This is the
    previous production default (``current_equal_rrf``) and remains
    available via ``fusion_strategy="rrf"`` for backward compatibility and
    ablation; it is no longer the default ``search_memories`` uses. Kept
    byte-for-byte behaviorally unchanged as the baseline other fusion
    strategies are compared against.
    """

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
    return _sort_fused_memories(fused)


# Alias so ablation/callers can name the production baseline explicitly
# without implying it is one strategy among equals in production code.
current_equal_rrf = reciprocal_rank_fusion


def _branch_rr_scores(
    branches: dict[str, list[_Identified]],
    *,
    k: int,
) -> dict[str, dict[str, object]]:
    """Per-candidate, per-branch reciprocal-rank values, kept separate (not summed).

    Unlike ``_rrf_branch_scores`` (which immediately sums every branch's
    contribution into one equal-weight score), this keeps each branch's
    1/(k+rank) value individually so weighted and discounted-agreement
    strategies can combine them differently after the fact.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    by_id: dict[str, dict[str, object]] = {}
    for branch_name, items in branches.items():
        for rank, item in enumerate(items, start=1):
            key = str(item.id)
            entry = by_id.setdefault(key, {"item": item, "ranks": {}, "rr": {}})
            entry["ranks"][branch_name] = rank  # type: ignore[index]
            entry["rr"][branch_name] = 1.0 / (k + rank)  # type: ignore[index]
    return by_id


def weighted_reciprocal_rank_fusion(
    *,
    structured: list[Memory],
    lexical: list[Memory],
    vector: list[Memory],
    k: int = RRF_K,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    structured_weight: float = 1.0,
) -> list[FusedMemory]:
    """RRF with a per-branch multiplier: score = Σ weight_branch × 1/(k+rank).

    ``bm25_weight`` controls only the lexical branch's contribution relative
    to vector (anchored at ``vector_weight=1.0``); structured is a no-op
    branch for benchmarks that supply no structured filter, and is weighted
    at 1.0 by default for parity with equal RRF when it is inactive.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")
    by_id = _branch_rr_scores({"structured": structured, "lexical": lexical, "vector": vector}, k=k)
    weights = {"structured": structured_weight, "lexical": bm25_weight, "vector": vector_weight}
    fused = [
        FusedMemory(
            memory=entry["item"],  # type: ignore[arg-type]
            score=sum(weights[name] * rr for name, rr in entry["rr"].items()),  # type: ignore[union-attr]
            structured_rank=entry["ranks"].get("structured"),  # type: ignore[union-attr]
            lexical_rank=entry["ranks"].get("lexical"),  # type: ignore[union-attr]
            vector_rank=entry["ranks"].get("vector"),  # type: ignore[union-attr]
        )
        for entry in by_id.values()
    ]
    return _sort_fused_memories(fused)


def discounted_agreement_fusion(
    *,
    structured: list[Memory],
    lexical: list[Memory],
    vector: list[Memory],
    k: int = RRF_K,
    lambda_: float = DEFAULT_AGREEMENT_DISCOUNT,
) -> list[FusedMemory]:
    """score = primary + lambda × secondary, where primary is the single
    strongest branch's reciprocal rank and secondary is the weakest of the
    remaining branches that also matched (0 if only one branch matched).

    This keeps a single excellent branch hit from being outscored by two
    branches that both agree only mediocrely, while still rewarding genuine
    multi-branch agreement with a bounded (lambda-scaled) bonus.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative.")
    by_id = _branch_rr_scores({"structured": structured, "lexical": lexical, "vector": vector}, k=k)
    fused = []
    for entry in by_id.values():
        rr_values = sorted(entry["rr"].values(), reverse=True)  # type: ignore[arg-type]
        primary = rr_values[0]
        secondary = min(rr_values[1:]) if len(rr_values) > 1 else 0.0
        fused.append(
            FusedMemory(
                memory=entry["item"],  # type: ignore[arg-type]
                score=primary + lambda_ * secondary,
                structured_rank=entry["ranks"].get("structured"),  # type: ignore[union-attr]
                lexical_rank=entry["ranks"].get("lexical"),  # type: ignore[union-attr]
                vector_rank=entry["ranks"].get("vector"),  # type: ignore[union-attr]
            )
        )
    return _sort_fused_memories(fused)


FusionStrategy = str  # "rrf" | "weighted_rrf" | "discounted_agreement"


def fuse_rankings(
    *,
    structured: list[Memory],
    lexical: list[Memory],
    vector: list[Memory],
    strategy: FusionStrategy = "discounted_agreement",
    k: int = RRF_K,
    bm25_weight: float = 1.0,
    lambda_: float = DEFAULT_AGREEMENT_DISCOUNT,
) -> list[FusedMemory]:
    """Small dispatcher over the fusion strategies evaluated in this milestone.

    Not a plugin framework -- just a name-to-function switch so callers can
    request a strategy by string without importing three separate functions.
    ``search_memories`` calls this dispatcher with its own
    ``fusion_strategy`` parameter, defaulting to ``"discounted_agreement"``
    (the project's current validated default); ``"rrf"`` reproduces the
    previous equal-weight behavior exactly for backward compatibility and
    ablation.
    """

    if strategy == "rrf":
        return reciprocal_rank_fusion(structured=structured, lexical=lexical, vector=vector, k=k)
    if strategy == "weighted_rrf":
        return weighted_reciprocal_rank_fusion(
            structured=structured, lexical=lexical, vector=vector, k=k, bm25_weight=bm25_weight
        )
    if strategy == "discounted_agreement":
        return discounted_agreement_fusion(
            structured=structured, lexical=lexical, vector=vector, k=k, lambda_=lambda_
        )
    raise ValueError(f"Unknown fusion strategy: {strategy!r}")


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
