"""Controlled predicates that support deterministic memory identity."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal


Cardinality = Literal["single", "multi"]

# This intentionally small registry identifies only fact slots that need
# deterministic identity. Unknown predicates remain valid open-world memories.
PREDICATES: dict[str, dict[str, Cardinality]] = {
    "name": {"cardinality": "single"},
    "location": {"cardinality": "single"},
    "timezone": {"cardinality": "single"},
    "database_preference": {"cardinality": "single"},
    "framework_preference": {"cardinality": "multi"},
    "programming_language": {"cardinality": "multi"},
    "skill": {"cardinality": "multi"},
    "interest": {"cardinality": "multi"},
    "works_on": {"cardinality": "multi"},
    "project_status": {"cardinality": "single"},
}

# Aliases are intentionally explicit. Predicate resolution does not infer new
# labels or mutate the controlled registry.
PREDICATE_ALIASES = {
    "preferred_database": "database_preference",
    "db_preference": "database_preference",
}


def _normalize_predicate_name(name: str) -> str:
    """Normalize only superficial formatting differences in a predicate name."""

    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    return re.sub(r"[\s-]+", "_", normalized)


def resolve_predicate(name: str | None) -> str | None:
    """Return a known canonical predicate, or ``None`` for unknown input."""

    if name is None:
        return None

    normalized = _normalize_predicate_name(name)
    canonical = PREDICATE_ALIASES.get(normalized, normalized)
    return canonical if canonical in PREDICATES else None


def get_predicate_cardinality(name: str | None) -> Cardinality | None:
    """Return the cardinality for a known predicate without expanding the registry."""

    predicate = resolve_predicate(name)
    return PREDICATES[predicate]["cardinality"] if predicate is not None else None
