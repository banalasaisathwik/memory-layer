"""Add rolling conversation summaries and normalize importance as FLOAT.

Revision ID: 0002_conversation_summaries
Revises: 0001_initial_schema
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0002_conversation_summaries"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table_names = set() if context.is_offline_mode() else set(sa.inspect(op.get_bind()).get_table_names())
    if not context.is_offline_mode() and "memories" not in table_names:
        raise RuntimeError(
            "The 0001 baseline requires the Milestones 1-4 tables. Use `alembic upgrade head` for an empty database."
        )

    importance_type = None
    if not context.is_offline_mode():
        importance_type = next(
            column["type"]
            for column in sa.inspect(op.get_bind()).get_columns("memories")
            if column["name"] == "importance"
        )
    if context.is_offline_mode() or not isinstance(importance_type, sa.Float):
        op.alter_column(
            "memories",
            "importance",
            existing_type=sa.Integer(),
            type_=sa.Float(),
            existing_nullable=False,
            postgresql_using="importance::double precision",
        )

    if context.is_offline_mode() or "conversation_summaries" not in table_names:
        op.create_table(
            "conversation_summaries",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("conversation_id", sa.Uuid(), nullable=False),
            sa.Column("summary_text", sa.Text(), nullable=False),
            sa.Column("covered_through_message_id", sa.Uuid(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.ForeignKeyConstraint(["covered_through_message_id"], ["messages.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_conversation_summaries_conversation_id",
            "conversation_summaries",
            ["conversation_id"],
            unique=True,
        )


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrading this revision could lose fractional importance values and is intentionally unsupported."
    )
