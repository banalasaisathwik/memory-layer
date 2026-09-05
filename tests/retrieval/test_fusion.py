"""Unit tests for rank-only reciprocal rank fusion."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from src.database import Memory, MemoryType
from src.database.models import Message, MessageRole, utcnow
from src.retrieval import RRF_K, fuse_message_rankings, reciprocal_rank_fusion


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
