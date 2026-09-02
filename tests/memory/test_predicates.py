"""Tests for controlled predicate resolution."""

from __future__ import annotations

from src.memory import PREDICATES, get_predicate_cardinality, resolve_predicate


def test_canonical_predicate_resolves() -> None:
    assert resolve_predicate("location") == "location"
    assert get_predicate_cardinality("location") == "single"


def test_explicit_alias_resolves_to_canonical_predicate() -> None:
    assert resolve_predicate("preferred_database") == "database_preference"
    assert get_predicate_cardinality("db_preference") == "single"


def test_predicate_formatting_normalizes() -> None:
    assert resolve_predicate(" Database-Preference ") == "database_preference"
    assert resolve_predicate("DATABASE_PREFERENCE") == "database_preference"


def test_unknown_predicate_returns_none() -> None:
    assert resolve_predicate("explanation_style_preference") is None
    assert get_predicate_cardinality("explanation_style_preference") is None


def test_unknown_predicate_does_not_mutate_registry() -> None:
    before = dict(PREDICATES)

    assert resolve_predicate("explanation_style_preference") is None

    assert PREDICATES == before
