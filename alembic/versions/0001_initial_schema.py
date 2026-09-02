"""Create the Milestones 1-4 schema baseline.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_external_id", "users", ["external_id"], unique=True)

    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_external_id", "conversations", ["external_id"], unique=False)
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", name="message_role", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)

    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column(
            "memory_type",
            sa.Enum("semantic", "episodic", name="memory_type", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column("memory_text", sa.Text(), nullable=False),
        sa.Column("subject_type", sa.String(length=100), nullable=True),
        sa.Column("subject_id", sa.String(length=255), nullable=True),
        sa.Column("predicate", sa.String(length=255), nullable=True),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("fact_key", sa.String(length=512), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("source_message_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by_id", sa.Uuid(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["superseded_by_id"], ["memories.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_conversation_id", "memories", ["conversation_id"], unique=False)
    op.create_index("ix_memories_fact_key", "memories", ["fact_key"], unique=False)
    op.create_index("ix_memories_is_active", "memories", ["is_active"], unique=False)
    op.create_index("ix_memories_user_id", "memories", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_index("ix_memories_is_active", table_name="memories")
    op.drop_index("ix_memories_fact_key", table_name="memories")
    op.drop_index("ix_memories_conversation_id", table_name="memories")
    op.drop_table("memories")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_index("ix_conversations_external_id", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("ix_users_external_id", table_name="users")
    op.drop_table("users")
