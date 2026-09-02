"""Tests for the structural candidate-memory schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.database import MemoryType
from src.memory import CandidateMemory


def test_valid_structured_semantic_candidate() -> None:
    candidate = CandidateMemory(
        memory_text="User lives in Bangalore",
        subject_type="user",
        predicate="location",
        value="Bangalore",
        confidence=0.95,
        importance=0.7,
        source_message_ids=[12, 13],
    )

    assert candidate.memory_type is MemoryType.SEMANTIC
    assert candidate.source_message_ids == [12, 13]


def test_valid_unstructured_semantic_candidate_has_safe_defaults() -> None:
    candidate = CandidateMemory(memory_text="User prefers implementation-flow explanations.")

    assert candidate.memory_type is MemoryType.SEMANTIC
    assert candidate.predicate is None
    assert candidate.source_message_ids == []


def test_valid_episodic_candidate() -> None:
    candidate = CandidateMemory(
        memory_type="episodic",
        memory_text="User is debugging an authentication bug today.",
    )

    assert candidate.memory_type is MemoryType.EPISODIC


@pytest.mark.parametrize("memory_text", ["", "   \t\n"])
def test_empty_memory_text_is_rejected(memory_text: str) -> None:
    with pytest.raises(ValidationError, match="memory_text"):
        CandidateMemory(memory_text=memory_text)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError, match="confidence"):
        CandidateMemory(memory_text="A memory", confidence=confidence)


@pytest.mark.parametrize("importance", [-0.01, 1.01])
def test_invalid_importance_is_rejected(importance: float) -> None:
    with pytest.raises(ValidationError, match="importance"):
        CandidateMemory(memory_text="A memory", importance=importance)


def test_unknown_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        CandidateMemory(memory_text="A memory", unexpected="value")


def test_caller_supplied_fact_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="fact_key"):
        CandidateMemory(memory_text="A memory", fact_key="model-invented-key")
