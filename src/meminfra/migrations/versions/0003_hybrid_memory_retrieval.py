"""Add durable embeddings and PostgreSQL lexical memory indexing.

Revision ID: 0003_hybrid_memory_retrieval
Revises: 0002_conversation_summaries
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_hybrid_memory_retrieval"
down_revision = "0002_conversation_summaries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_names = set() if context.is_offline_mode() else set(sa.inspect(op.get_bind()).get_table_names())
    if not context.is_offline_mode() and "memories" not in table_names:
        raise RuntimeError("The Milestone 6 migration requires the existing memories table.")

    op.add_column("memories", sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("memories", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.create_index(
        "ix_memories_memory_text_fts",
        "memories",
        [sa.text("to_tsvector('simple', memory_text)")],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrading Milestone 6 would discard durable embeddings and is intentionally unsupported."
    )
