"""Enforce at most one active memory per (user_id, fact_key).

Revision ID: 0005_structured_fact_uniqueness
Revises: 0004_message_embeddings
Create Date: 2026-09-03
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0005_structured_fact_uniqueness"
down_revision = "0004_message_embeddings"
branch_labels = None
depends_on = None


INDEX_NAME = "ix_memories_active_user_fact_key"


def upgrade() -> None:
    table_names = set() if context.is_offline_mode() else set(sa.inspect(op.get_bind()).get_table_names())
    if not context.is_offline_mode() and "memories" not in table_names:
        raise RuntimeError("This migration requires the existing memories table.")

    # A plain application-level SELECT-then-write check cannot prevent two
    # concurrent transactions from both observing no active conflicting row
    # and both committing. This partial unique index makes PostgreSQL reject
    # the second commit. Open-semantic and episodic memories keep fact_key
    # NULL and never participate in this constraint.
    op.create_index(
        INDEX_NAME,
        "memories",
        ["user_id", "fact_key"],
        unique=True,
        postgresql_where=sa.text("is_active = true AND fact_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="memories")
