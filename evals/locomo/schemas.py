"""Typed models for LoCoMo dataset content and one benchmark run's output."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Official LoCoMo category IDs. Do not substitute a different mapping.
CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


class LocomoTurn(BaseModel):
    """One dialog turn from one LoCoMo session, in original session order."""

    model_config = ConfigDict(extra="forbid")

    dia_id: str
    speaker: str
    text: str
    session_number: int
    session_date_time: str
    image_caption: str | None = None


class LocomoQA(BaseModel):
    """One LoCoMo QA item with dataset-fidelity evidence resolution."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str | None = None
    adversarial_answer: str | None = None
    category_id: int
    # Dialog IDs that could be matched to a real turn in this sample, deduplicated.
    evidence: list[str] = Field(default_factory=list)
    # The dataset's original, unparsed evidence strings, kept for diagnostics.
    evidence_raw: list[str] = Field(default_factory=list)
    # Raw evidence strings that did not resolve to any known dia_id in this
    # sample -- a small, documented upstream LoCoMo annotation quirk (see
    # evals/README.md), not something this adapter invents a fix for.
    unresolved_evidence: list[str] = Field(default_factory=list)

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES[self.category_id]


class LocomoSample(BaseModel):
    """One full LoCoMo conversation: every session's turns plus its QA set."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[LocomoTurn]
    qa: list[LocomoQA]

    def role_for_speaker(self, speaker: str) -> Literal["user", "assistant"]:
        """speaker_a -> user, speaker_b -> assistant; fixed for the whole sample."""

        if speaker == self.speaker_a:
            return "user"
        if speaker == self.speaker_b:
            return "assistant"
        raise ValueError(f"{self.sample_id}: unknown speaker {speaker!r} (expected {self.speaker_a!r}/{self.speaker_b!r}).")


class QuestionRetrievalMetrics(BaseModel):
    """Evidence-provenance retrieval metrics for one QA question."""

    model_config = ConfigDict(extra="forbid")

    rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    hit_at_10: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    has_evidence: bool
    gold_evidence_count: int


class QuestionDiagnostic(BaseModel):
    """Full per-question diagnostics, saved for later failure-pattern analysis."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question_index: int
    question: str
    category_id: int
    category_name: str

    gold_answer: str | None = None
    adversarial_trap_answer: str | None = None
    gold_evidence: list[str] = Field(default_factory=list)
    unresolved_evidence: list[str] = Field(default_factory=list)

    retrieved_memory_ids: list[str] = Field(default_factory=list)
    retrieved_memory_texts: list[str] = Field(default_factory=list)
    # Per retrieved memory (same order/rank as retrieved_memory_ids), the gold
    # LoCoMo dia_ids its Memory.source_message_ids resolve back to.
    retrieved_provenance_dia_ids: list[list[str]] = Field(default_factory=list)
    retrieval_metrics: QuestionRetrievalMetrics | None = None
    isolation_failures: int = 0

    predicted_answer: str | None = None
    abstained: bool | None = None
    qa_score: float | None = None
    is_correct_abstention: bool | None = None

    error: str | None = None


class RetrievalAggregate(BaseModel):
    """Mean evidence-retrieval metrics over questions that have gold evidence."""

    model_config = ConfigDict(extra="forbid")

    questions_evaluated: int
    questions_excluded_no_evidence: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    isolation_failures: int


class QAAggregate(BaseModel):
    """Mean QA scores, split between normal F1 categories and adversarial abstention."""

    model_config = ConfigDict(extra="forbid")

    f1_questions: int
    f1_mean: float
    adversarial_questions: int
    adversarial_abstention_accuracy: float


class LocomoRunMetadata(BaseModel):
    """Reproducibility metadata persisted with every run."""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["locomo10"] = "locomo10"
    dataset_source: str
    dataset_sha256: str
    git_commit: str | None
    run_at: datetime
    run_id: str
    llm_model: str | None
    embedding_model: str | None
    top_k: int
    mode: Literal["retrieval", "qa", "both"]
    conversations_ingested: int
    questions_evaluated: int
    resumed: bool = False
    ingestion_and_eval_duration_seconds: float | None = None
    message_index_full_rebuilds: int | None = None
    message_index_incremental_appends: int | None = None
    message_index_unchanged_hits: int | None = None
    sessions_ingested: int | None = None
    sessions_failed: int | None = None
    ingestion_warnings: list[str] = Field(default_factory=list)
    category_mapping: dict[int, str] = Field(default_factory=lambda: dict(CATEGORY_NAMES))
    provenance_caveat: str = (
        "Extraction currently attaches every candidate from one target interaction to that "
        "interaction's full source_message_ids, so provenance -- and therefore evidence "
        "matching here -- is per-interaction, not yet per-candidate precise."
    )


class LocomoRunReport(BaseModel):
    """One complete, JSON-serializable LoCoMo benchmark run."""

    model_config = ConfigDict(extra="forbid")

    metadata: LocomoRunMetadata
    retrieval: RetrievalAggregate | None
    retrieval_by_category: dict[str, RetrievalAggregate]
    qa: QAAggregate | None
    qa_by_category: dict[str, QAAggregate]
    questions: list[QuestionDiagnostic]
