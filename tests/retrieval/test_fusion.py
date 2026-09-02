"""Unit tests for rank-only reciprocal rank fusion."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.database import Memory, MemoryType
from src.database.models import utcnow
from src.retrieval import reciprocal_rank_fusion


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
