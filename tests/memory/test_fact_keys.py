"""Tests for deterministic fact-key generation."""

from __future__ import annotations

from meminfra.memory import CandidateMemory, build_fact_key


def _semantic_candidate(**overrides: object) -> CandidateMemory:
    data: dict[str, object] = {
        "memory_text": "User lives in Hyderabad",
        "subject_type": "user",
        "predicate": "location",
        "value": "Hyderabad",
    }
    data.update(overrides)
    return CandidateMemory(**data)


def test_single_valued_location_key_uses_the_logical_slot() -> None:
    key = build_fact_key(_semantic_candidate(), subject_id="user_123")

    assert key == "user:user_123:location"


def test_single_valued_location_values_have_the_same_key() -> None:
    hyderabad = _semantic_candidate(value="Hyderabad")
    bangalore = _semantic_candidate(value="Bangalore")

    assert build_fact_key(hyderabad, subject_id="user_123") == build_fact_key(
        bangalore, subject_id="user_123"
    )


def test_multi_valued_fact_keys_differ_by_value_identity() -> None:
    python = _semantic_candidate(predicate="programming_language", value="Python")
    go = _semantic_candidate(predicate="programming_language", value="Go")

    assert build_fact_key(python, subject_id="user_123") == "user:user_123:programming_language:python"
    assert build_fact_key(go, subject_id="user_123") == "user:user_123:programming_language:go"


def test_multi_valued_identity_normalizes_case_and_whitespace() -> None:
    python = _semantic_candidate(predicate="programming_language", value="Python")
    spaced_python = _semantic_candidate(predicate="programming_language", value=" python ")

    assert build_fact_key(python, subject_id="user_123") == build_fact_key(
        spaced_python, subject_id="user_123"
    )


def test_multi_valued_special_characters_are_encoded_unambiguously() -> None:
    candidate = _semantic_candidate(predicate="programming_language", value="C++")

    assert build_fact_key(candidate, subject_id="user_123") == "user:user_123:programming_language:c%2B%2B"


def test_unknown_predicate_has_no_fact_key() -> None:
    candidate = _semantic_candidate(predicate="explanation_style_preference")

    assert build_fact_key(candidate, subject_id="user_123") is None


def test_missing_canonical_subject_id_has_no_fact_key() -> None:
    candidate = _semantic_candidate(subject_type="project", predicate="project_status", value="active")

    assert build_fact_key(candidate, subject_id=None) is None


def test_missing_subject_type_has_no_fact_key() -> None:
    candidate = _semantic_candidate(subject_type=None)

    assert build_fact_key(candidate, subject_id="user_123") is None


def test_multi_valued_predicate_without_value_has_no_fact_key() -> None:
    candidate = _semantic_candidate(predicate="programming_language", value=None)

    assert build_fact_key(candidate, subject_id="user_123") is None


def test_episodic_memory_has_no_fact_key() -> None:
    candidate = _semantic_candidate(memory_type="episodic")

    assert build_fact_key(candidate, subject_id="user_123") is None
