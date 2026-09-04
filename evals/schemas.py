"""Small typed models for one evaluation case and its run output."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvalMessage(BaseModel):
    """One user or assistant message in an eval case's conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class ExpectedMemory(BaseModel):
    """A gold memory expressed as required terms rather than exact text.

    A retrieved memory matches when its normalized text contains every
    required term as a substring. This keeps gold matching deterministic and
    inspectable without exact-string equality, an LLM judge, or embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    required_terms: list[str] = Field(min_length=1)
    description: str | None = None

    @field_validator("required_terms")
    @classmethod
    def validate_required_terms(cls, value: list[str]) -> list[str]:
        if any(not term.strip() for term in value):
            raise ValueError("required_terms must not contain empty or whitespace-only entries.")
        return value


class EvalCase(BaseModel):
    """One deterministic memory case run through the production ingestion pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    messages: list[EvalMessage] = Field(min_length=1)
    query: str
    expected_memories: list[ExpectedMemory] = Field(min_length=1)
    description: str | None = None
    # Only the user-isolation case uses this: a second user's conversation,
    # ingested the same way, whose memories must never appear in this case's
    # search results for the primary user.
    isolation_messages: list[EvalMessage] | None = None


class RetrievedMemory(BaseModel):
    """A compact, report-friendly view of one search_memories() hit."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_text: str
    is_active: bool
    score: float


class CaseMetrics(BaseModel):
    """Retrieval metrics for one case's ranked results against its gold memories."""

    model_config = ConfigDict(extra="forbid")

    rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float


class CaseResult(BaseModel):
    """The full outcome of running one EvalCase through the pipeline."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    query: str
    retrieved: list[RetrievedMemory]
    metrics: CaseMetrics
    isolation_failures: int
    superseded_returned: int


class AggregateMetrics(BaseModel):
    """Mean metrics over one or more cases."""

    model_config = ConfigDict(extra="forbid")

    cases: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    isolation_failures: int
    superseded_returned: int


class RunReport(BaseModel):
    """One complete, JSON-serializable eval run."""

    model_config = ConfigDict(extra="forbid")

    run_at: datetime
    dataset: str
    top_k: int
    llm_model: str | None
    embedding_model: str | None
    aggregate: AggregateMetrics
    by_category: dict[str, AggregateMetrics]
    cases: list[CaseResult]
