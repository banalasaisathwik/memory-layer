"""Deterministic logical identities for known structured memory candidates."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

from meminfra.database.models import MemoryType

from .predicates import get_predicate_cardinality, resolve_predicate
from .schemas import CandidateMemory


def _encode_component(value: str) -> str:
    """Encode a normalized component so fact-key delimiters stay unambiguous."""

    return quote(value, safe="-._~")


def normalize_subject_type(subject_type: str | None) -> str | None:
    """Turn a subject label into a stable identifier suitable for a fact key."""

    if subject_type is None:
        return None

    normalized = unicodedata.normalize("NFKC", subject_type).strip().casefold()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    return _encode_component(normalized) if normalized else None


def _normalize_subject_id(subject_id: str | None) -> str | None:
    """Preserve a canonical application ID while safely encoding key delimiters."""

    if subject_id is None:
        return None

    normalized = unicodedata.normalize("NFC", subject_id).strip()
    return _encode_component(normalized) if normalized else None


def normalize_value_identity(value: str | None) -> str | None:
    """Normalize only the identity component for a multi-valued predicate."""

    if value is None:
        return None

    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split()).casefold()
    return _encode_component(normalized) if normalized else None


def build_fact_key(candidate: CandidateMemory, *, subject_id: str | None) -> str | None:
    """Build a fact key when a candidate has a known, canonical structured identity."""

    if candidate.memory_type is not MemoryType.SEMANTIC:
        return None

    predicate = resolve_predicate(candidate.predicate)
    if predicate is None:
        return None

    subject_type = normalize_subject_type(candidate.subject_type)
    canonical_subject_id = _normalize_subject_id(subject_id)
    if subject_type is None or canonical_subject_id is None:
        return None

    if get_predicate_cardinality(predicate) == "single":
        return f"{subject_type}:{canonical_subject_id}:{predicate}"

    value_identity = normalize_value_identity(candidate.value)
    if value_identity is None:
        return None
    return f"{subject_type}:{canonical_subject_id}:{predicate}:{value_identity}"
