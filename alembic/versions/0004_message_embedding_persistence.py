"""Add durable raw Message embeddings for semantic extraction context.

Revision ID: 0004_message_embeddings
Revises: 0003_hybrid_memory_retrieval
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004_message_embeddings"
down_revision = "0003_hybrid_memory_retrieval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_names = set() if context.is_offline_mode() else set(sa.inspect(op.get_bind()).get_table_names())
    if not context.is_offline_mode() and "messages" not in table_names:
        raise RuntimeError("The Milestone 6.1 migration requires the existing messages table.")

    op.add_column("messages", sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("messages", sa.Column("embedding_model", sa.String(length=255), nullable=True))


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrading Milestone 6.1 would discard durable Message embeddings and is intentionally unsupported."
    )
