"""Current durable models for user-scoped long-term memory."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Enum as SqlEnum, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


def utcnow() -> datetime:
    """Produce timezone-aware timestamps for application-created records."""

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class shared by the small initial schema."""


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


message_role_enum = SqlEnum(
    MessageRole,
    name="message_role",
    native_enum=False,
    create_constraint=True,
    values_callable=lambda roles: [role.value for role in roles],
)
memory_type_enum = SqlEnum(
    MemoryType,
    name="memory_type",
    native_enum=False,
    create_constraint=True,
    values_callable=lambda types: [memory_type.value for memory_type in types],
)
source_message_ids_type = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    memories: Mapped[list["Memory"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    external_id: Mapped[str] = mapped_column(String(255), index=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    memories: Mapped[list["Memory"]] = relationship(back_populates="conversation")
    summary: Mapped["ConversationSummary | None"] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[MessageRole] = mapped_column(message_role_enum)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ConversationSummary(Base):
    """The one rolling, contextual summary retained for a conversation."""

    __tablename__ = "conversation_summaries"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id"),
        unique=True,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(Text)
    covered_through_message_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("messages.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="summary")


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    conversation_id: Mapped[UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)

    memory_type: Mapped[MemoryType] = mapped_column(memory_type_enum)
    memory_text: Mapped[str] = mapped_column(Text)

    subject_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    predicate: Mapped[str | None] = mapped_column(String(255), nullable=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    fact_key: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0)

    source_message_ids: Mapped[list[str]] = mapped_column(source_message_ids_type, default=list)

    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by_id: Mapped[UUID | None] = mapped_column(ForeignKey("memories.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="memories")
    conversation: Mapped[Conversation | None] = relationship(back_populates="memories")
    superseded_by: Mapped["Memory | None"] = relationship(remote_side="Memory.id", foreign_keys=[superseded_by_id])
