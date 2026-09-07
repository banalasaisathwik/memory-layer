"""Public input and output structures for memory retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from meminfra.database.models import MemoryType


class SearchFilters(BaseModel):
    """Caller-supplied deterministic constraints for one user-scoped search."""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType | None = None
    predicate: str | None = None
    subject_type: str | None = None
    fact_key: str | None = None
    conversation_external_id: str | None = None
    include_history: bool = False

    @field_validator("predicate", "subject_type", "fact_key", "conversation_external_id")
    @classmethod
    def reject_blank_filter_values(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Search filter values must not be empty or whitespace-only.")
        return value


class QueryIntent(BaseModel):
    """Optional LLM-derived hints for the structured retrieval branch only."""

    model_config = ConfigDict(extra="forbid")

    predicate: str | None = None
    value: str | None = None
    temporal_scope: Literal["current", "historical", "any"] = "current"

    @field_validator("predicate", "value")
    @classmethod
    def reject_blank_intent_values(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Query intent values must not be empty or whitespace-only.")
        return value


class SearchHit(BaseModel):
    """A durable-memory result with inspectable branch ranks and fused score."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_text: str
    memory_type: str

    fact_key: str | None = None
    predicate: str | None = None
    value: str | None = None

    is_active: bool

    confidence: float | None = None
    importance: float

    created_at: datetime
    valid_from: datetime
    valid_to: datetime | None = None

    score: float

    structured_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    vector_rank: int | None = Field(default=None, ge=1)
