"""Unit checks for public retrieval structures and migration intent."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from meminfra.database import Memory, MemoryType
from meminfra.retrieval import SearchFilters
from meminfra.retrieval.vector import _normalize_embedding


def test_search_filters_support_explicit_structure_and_history() -> None:
    filters = SearchFilters(
        memory_type=MemoryType.SEMANTIC,
        predicate="location",
        subject_type="user",
        fact_key="user:user_123:location",
        conversation_external_id="conversation_123",
        include_history=True,
    )

    assert filters.memory_type is MemoryType.SEMANTIC
    assert filters.include_history is True


@pytest.mark.parametrize("field", ["predicate", "subject_type", "fact_key", "conversation_external_id"])
def test_search_filters_reject_blank_values(field: str) -> None:
    with pytest.raises(ValidationError):
        SearchFilters(**{field: " \t "})


def test_hybrid_migration_adds_embeddings_and_fts_index() -> None:
    migration = Path("src/meminfra/migrations/versions/0003_hybrid_memory_retrieval.py").read_text(encoding="utf-8")

    assert 'Column("embedding"' in migration
    assert 'Column("embedding_model"' in migration
    assert "ix_memories_memory_text_fts" in migration
    assert "postgresql_using=\"gin\"" in migration


def test_memory_metadata_matches_the_hybrid_fts_index_definition() -> None:
    index = next(index for index in Memory.__table__.indexes if index.name == "ix_memories_memory_text_fts")

    assert index.dialect_options["postgresql"]["using"] == "gin"
    assert len(index.expressions) == 1
    assert str(index.expressions[0]) == "to_tsvector('simple', memory_text)"


def test_message_embedding_migration_is_after_hybrid_memory_retrieval() -> None:
    migration = Path("src/meminfra/migrations/versions/0004_message_embedding_persistence.py").read_text(encoding="utf-8")

    assert 'down_revision = "0003_hybrid_memory_retrieval"' in migration
    assert 'add_column("messages", sa.Column("embedding"' in migration
    assert 'add_column("messages", sa.Column("embedding_model"' in migration
    assert 'revision = "0004_message_embeddings"' in migration
    assert len("0004_message_embeddings") <= 32


def test_embeddings_are_normalized_for_exact_inner_product_cosine_search() -> None:
    stored = _normalize_embedding([3.0, 4.0])
    query = _normalize_embedding([6.0, 8.0])

    assert stored.tolist() == pytest.approx([0.6, 0.8])
    assert float(stored @ query) == pytest.approx(1.0)
