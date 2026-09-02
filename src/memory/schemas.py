"""Schemas for memory candidates before any persistence decision."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.database.models import MemoryType


class CandidateMemory(BaseModel):
    """A structurally valid memory candidate with no canonical subject identity."""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType = MemoryType.SEMANTIC
    memory_text: str

    subject_type: str | None = None
    subject_name: str | None = None

    predicate: str | None = None
    value: str | None = None

    confidence: float | None = Field(default=None, ge=0, le=1)
    importance: float = Field(default=0, ge=0, le=1)

    source_message_ids: list[str | int] = Field(default_factory=list)

    @field_validator("memory_text")
    @classmethod
    def validate_memory_text(cls, value: str) -> str:
        """Reject empty content while preserving meaningful extracted text."""

        if not value.strip():
            raise ValueError("memory_text must not be empty or whitespace-only.")
        return value

    @field_validator("subject_type", "subject_name", "predicate", "value")
    @classmethod
    def discard_whitespace_only_optional_strings(cls, value: str | None) -> str | None:
        """Represent absent optional fields consistently without altering meaningful values."""

        if value is not None and not value.strip():
            return None
        return value
