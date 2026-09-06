"""Unit tests for rank-only reciprocal rank fusion."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from meminfra.database import Memory, MemoryType
from meminfra.database.models import Message, MessageRole, utcnow
from meminfra.retrieval import RRF_K, fuse_message_rankings, reciprocal_rank_fusion
from meminfra.retrieval.fusion import (
    current_equal_rrf,
    discounted_agreement_fusion,
    fuse_rankings,
    weighted_reciprocal_rank_fusion,
)


def _memory(*, importance: float = 0.5) -> Memory:
    timestamp = utcnow()
    return Memory(
        id=uuid4(),
        user_id=uuid4(),
        memory_type=MemoryType.SEMANTIC,
        memory_text="A durable memory",
        importance=importance,
        source_message_ids=[],
        valid_from=timestamp,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_rrf_uses_only_branch_ranks_and_exposes_each_rank() -> None:
    first = _memory()
    second = _memory()
    third = _memory()

    fused = reciprocal_rank_fusion(
        structured=[first, second],
        lexical=[second, third, first],
        vector=[third, first, second],
    )

    by_id = {str(hit.memory.id): hit for hit in fused}
    first_hit = by_id[str(first.id)]
    assert first_hit.score == pytest.approx(1 / 61 + 1 / 63 + 1 / 62)
    assert first_hit.structured_rank == 1
    assert first_hit.lexical_rank == 3
    assert first_hit.vector_rank == 2


def test_rrf_boosts_memories_that_appear_in_multiple_branches() -> None:
    shared = _memory()
    structured_only = _memory()

    fused = reciprocal_rank_fusion(
        structured=[structured_only, shared],
        lexical=[shared],
        vector=[shared],
    )

    assert fused[0].memory.id == shared.id
    assert fused[0].score > fused[1].score


def test_rrf_rejects_an_invalid_constant() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_fusion(structured=[], lexical=[], vector=[], k=0)


def _message(*, minute: int, content: str = "message") -> Message:
    return Message(
        id=uuid4(),
        conversation_id=uuid4(),
        role=MessageRole.USER,
        content=content,
        created_at=utcnow() + timedelta(minutes=minute),
    )


def test_fuse_message_rankings_matches_the_worked_example() -> None:
    m1, m2, m3, m4 = (_message(minute=minute) for minute in range(1, 5))

    fused = fuse_message_rankings(lexical=[m1, m2, m3], semantic=[m2, m4, m1])

    scores = {str(item.message.id): item.score for item in fused}
    assert scores[str(m1.id)] == pytest.approx(1 / 61 + 1 / 63)
    assert scores[str(m2.id)] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[str(m3.id)] == pytest.approx(1 / 63)
    assert scores[str(m4.id)] == pytest.approx(1 / 62)
    assert [item.message.id for item in fused] == [m2.id, m1.id, m4.id, m3.id]


def test_fuse_message_rankings_dedupes_by_message_id_and_combines_scores() -> None:
    shared = _message(minute=1)
    lexical_only = _message(minute=2)

    fused = fuse_message_rankings(lexical=[shared, lexical_only], semantic=[shared])

    assert {item.message.id for item in fused} == {shared.id, lexical_only.id}
    shared_hit = next(item for item in fused if item.message.id == shared.id)
    assert shared_hit.lexical_rank == 1
    assert shared_hit.semantic_rank == 1
    assert shared_hit.score == pytest.approx(1 / 61 + 1 / 61)
    assert fused[0].message.id == shared.id


def test_fuse_message_rankings_semantic_only_branch_works() -> None:
    only = _message(minute=1)

    fused = fuse_message_rankings(lexical=[], semantic=[only])

    assert len(fused) == 1
    assert fused[0].message.id == only.id
    assert fused[0].lexical_rank is None
    assert fused[0].semantic_rank == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_fuse_message_rankings_lexical_only_branch_works() -> None:
    only = _message(minute=1)

    fused = fuse_message_rankings(lexical=[only], semantic=[])

    assert len(fused) == 1
    assert fused[0].message.id == only.id
    assert fused[0].lexical_rank == 1
    assert fused[0].semantic_rank is None
    assert fused[0].score == pytest.approx(1 / 61)


def test_fuse_message_rankings_orders_by_relevance_not_recency() -> None:
    """Selection must happen before any chronological sort.

    These three messages are created oldest-to-newest as m_old, m_mid,
    m_new, but the lexical branch ranks them m_new > m_old > m_mid. The
    fused order must follow that relevance ranking, not creation time --
    chronological ordering is a separate step callers apply only after
    selection.
    """

    m_old = _message(minute=1)
    m_mid = _message(minute=2)
    m_new = _message(minute=3)

    fused = fuse_message_rankings(lexical=[m_new, m_old, m_mid], semantic=[])

    assert [item.message.id for item in fused] == [m_new.id, m_old.id, m_mid.id]


def test_fuse_message_rankings_breaks_ties_deterministically() -> None:
    """Equal scores fall back to best branch rank, then created_at, then id.

    Two single-branch, rank-1 messages score identically (both 1/61). With
    an equal best-branch-rank, the earlier-created message must sort first
    -- never by importance or recency preference, since those are
    explicitly out of scope for message context fusion.
    """

    earlier = _message(minute=1)
    later = _message(minute=2)

    fused = fuse_message_rankings(lexical=[earlier], semantic=[later])

    assert fused[0].score == pytest.approx(fused[1].score)
    assert [item.message.id for item in fused] == [earlier.id, later.id]


def test_fuse_message_rankings_uses_equal_branch_weights() -> None:
    """Both branches contribute 1/(k+rank) equally -- neither is weighted higher."""

    solo = _message(minute=1)
    lexical_only = fuse_message_rankings(lexical=[solo], semantic=[])
    semantic_only = fuse_message_rankings(lexical=[], semantic=[solo])
    assert lexical_only[0].score == pytest.approx(semantic_only[0].score)

    combined = _message(minute=2)
    both = fuse_message_rankings(lexical=[combined], semantic=[combined])
    assert both[0].score == pytest.approx(2 * lexical_only[0].score)


def test_fuse_message_rankings_rejects_an_invalid_constant() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        fuse_message_rankings(lexical=[], semantic=[], k=0)


def test_fuse_message_rankings_default_k_matches_memory_fusion() -> None:
    assert RRF_K == 60


# --------------------------------------------------------------------------
# Fusion strategy ablation: current_equal_rrf / weighted_rrf / discounted_agreement
# --------------------------------------------------------------------------


def test_current_equal_rrf_is_the_unchanged_production_baseline() -> None:
    """``current_equal_rrf`` must be the exact same function as reciprocal_rank_fusion.

    This is a named alias, not a reimplementation -- the baseline must never
    silently drift from what production ``search_memories`` calls.
    """

    assert current_equal_rrf is reciprocal_rank_fusion


def test_weighted_rrf_matches_equal_rrf_at_weight_one() -> None:
    first = _memory()
    second = _memory()

    equal = reciprocal_rank_fusion(structured=[], lexical=[first, second], vector=[second, first])
    weighted = weighted_reciprocal_rank_fusion(
        structured=[], lexical=[first, second], vector=[second, first], bm25_weight=1.0
    )

    equal_scores = {str(hit.memory.id): hit.score for hit in equal}
    weighted_scores = {str(hit.memory.id): hit.score for hit in weighted}
    assert weighted_scores == pytest.approx(equal_scores)


def test_weighted_rrf_reduces_bm25_branch_contribution() -> None:
    bm25_only = _memory()

    full_weight = weighted_reciprocal_rank_fusion(structured=[], lexical=[bm25_only], vector=[], bm25_weight=1.0)
    half_weight = weighted_reciprocal_rank_fusion(structured=[], lexical=[bm25_only], vector=[], bm25_weight=0.5)

    assert half_weight[0].score == pytest.approx(0.5 * full_weight[0].score)


def test_discounted_agreement_lets_strong_single_branch_beat_weak_dual_branch_agreement() -> None:
    """The motivating example from the fusion-ablation milestone.

    Gold: vector rank 2, absent from BM25. A generic competitor: BM25 rank
    20, vector rank 30. Under current equal RRF the competitor (two weak
    branch hits) outscores the gold (one strong branch hit); discounted
    agreement with lambda=0.25 must reverse that.
    """

    gold = _memory()
    competitor = _memory()

    # Competitor: BM25 rank 20, vector rank 30. Gold: vector rank 2, absent from BM25.
    bm25_branch = [_memory() for _ in range(19)] + [competitor]
    vector_branch_full = [_memory() for _ in range(29)] + [competitor]
    vector_branch_full[1] = gold  # keep gold at vector rank 2 in the same branch list

    current = reciprocal_rank_fusion(structured=[], lexical=bm25_branch, vector=vector_branch_full)
    current_scores = {str(hit.memory.id): hit.score for hit in current}
    assert current_scores[str(competitor.id)] > current_scores[str(gold.id)]
    assert current_scores[str(gold.id)] == pytest.approx(1 / 62)
    assert current_scores[str(competitor.id)] == pytest.approx(1 / 80 + 1 / 90)

    discounted = discounted_agreement_fusion(
        structured=[], lexical=bm25_branch, vector=vector_branch_full, lambda_=0.25
    )
    discounted_scores = {str(hit.memory.id): hit.score for hit in discounted}
    assert discounted_scores[str(gold.id)] == pytest.approx(1 / 62)
    assert discounted_scores[str(competitor.id)] == pytest.approx(1 / 80 + 0.25 * (1 / 90))
    assert discounted_scores[str(gold.id)] > discounted_scores[str(competitor.id)]


def test_discounted_agreement_still_rewards_excellent_dual_branch_agreement() -> None:
    """A BM25 #1 + vector #2 hit must still beat a weak single-branch-only result."""

    strong_agreement = _memory()
    weak_solo = _memory()

    fused = discounted_agreement_fusion(
        structured=[],
        lexical=[strong_agreement, _memory()],
        vector=[_memory(), strong_agreement],
        lambda_=0.25,
    )
    solo_fused = discounted_agreement_fusion(structured=[], lexical=[weak_solo], vector=[], lambda_=0.25)

    strong_score = next(hit.score for hit in fused if hit.memory.id == strong_agreement.id)
    solo_score = solo_fused[0].score
    assert strong_score > solo_score


def test_discounted_agreement_single_branch_has_no_secondary_bonus() -> None:
    only = _memory()
    fused = discounted_agreement_fusion(structured=[], lexical=[only], vector=[], lambda_=0.5)
    assert fused[0].score == pytest.approx(1 / 61)


def test_discounted_agreement_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_"):
        discounted_agreement_fusion(structured=[], lexical=[], vector=[], lambda_=-0.1)


def test_fuse_rankings_dispatches_by_strategy_name() -> None:
    only = _memory()

    via_dispatch = fuse_rankings(structured=[], lexical=[only], vector=[], strategy="rrf")
    via_direct = reciprocal_rank_fusion(structured=[], lexical=[only], vector=[])
    assert via_dispatch[0].score == pytest.approx(via_direct[0].score)

    via_weighted = fuse_rankings(structured=[], lexical=[only], vector=[], strategy="weighted_rrf", bm25_weight=0.5)
    assert via_weighted[0].score == pytest.approx(0.5 / 61)

    via_discounted = fuse_rankings(structured=[], lexical=[only], vector=[], strategy="discounted_agreement", lambda_=0.25)
    assert via_discounted[0].score == pytest.approx(1 / 61)


def test_fuse_rankings_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        fuse_rankings(structured=[], lexical=[], vector=[], strategy="not_a_strategy")


def test_weighted_and_discounted_fusion_keep_the_same_tie_break_order() -> None:
    """Equal scores must still fall back to importance/is_active/updated_at/id.

    Fusion gains must come from the score function, not a hidden tie-break
    change -- this pins the same deterministic ordering used by equal RRF.
    """

    higher_importance = _memory(importance=0.9)
    lower_importance = _memory(importance=0.1)

    weighted = weighted_reciprocal_rank_fusion(
        structured=[], lexical=[higher_importance, lower_importance], vector=[lower_importance, higher_importance]
    )
    assert weighted[0].score == pytest.approx(weighted[1].score)
    assert weighted[0].memory.id == higher_importance.id

    discounted = discounted_agreement_fusion(
        structured=[], lexical=[higher_importance, lower_importance], vector=[lower_importance, higher_importance]
    )
    assert discounted[0].score == pytest.approx(discounted[1].score)
    assert discounted[0].memory.id == higher_importance.id
