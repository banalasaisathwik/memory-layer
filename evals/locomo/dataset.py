"""Load the official LoCoMo dataset into typed, chronologically ordered samples.

Source: ``snap-research/locomo``, ``data/locomo10.json`` (10 conversations).
A copy is vendored at ``evals/data/locomo10.json`` so the benchmark and its
tests never depend on network access at run time; ``dataset_sha256()`` lets a
run record exactly which copy produced its numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .schemas import LocomoQA, LocomoSample, LocomoTurn

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"
DATASET_SOURCE_URL = "https://github.com/snap-research/locomo (data/locomo10.json)"

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_DIA_ID_RE = re.compile(r"D(\d+):(\d+)")


class LocomoDatasetError(ValueError):
    """Raised when a LoCoMo-shaped JSON file cannot be parsed as expected."""


def dataset_sha256(path: Path = DEFAULT_DATASET_PATH) -> str:
    """Return the SHA-256 of the raw dataset file for reproducibility records."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_dia_id(session: int, index: int) -> str:
    return f"D{session}:{index}"


def _normalize_evidence(raw_evidence: list[str], known_dia_ids: set[str]) -> tuple[list[str], list[str]]:
    """Resolve LoCoMo evidence strings to canonical dia_ids present in this sample.

    The released evidence lists contain a handful of upstream annotation
    quirks (a zero-padded index like ``D30:05``, or two ids concatenated into
    one string like ``"D8:6; D9:17"``). Extracting every ``D<session>:<index>``
    substring and normalizing away leading zeros recovers all but a couple of
    genuinely dangling references per the full LoCoMo-10 file; anything that
    still does not match a known turn is kept as unresolved for diagnostics
    rather than silently invented or dropped without a trace.
    """

    resolved: list[str] = []
    unresolved: list[str] = []
    for raw in raw_evidence:
        matches = list(_DIA_ID_RE.finditer(raw))
        found_known = False
        for match in matches:
            canonical = _canonical_dia_id(int(match.group(1)), int(match.group(2)))
            if canonical in known_dia_ids:
                found_known = True
                if canonical not in resolved:
                    resolved.append(canonical)
        if not found_known:
            unresolved.append(raw)
    return resolved, unresolved


def _parse_sample(raw: dict) -> LocomoSample:
    sample_id = raw.get("sample_id")
    if not sample_id:
        raise LocomoDatasetError("A LoCoMo sample is missing 'sample_id'.")

    conversation = raw.get("conversation")
    if not isinstance(conversation, dict):
        raise LocomoDatasetError(f"{sample_id}: missing or malformed 'conversation'.")

    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    if not speaker_a or not speaker_b:
        raise LocomoDatasetError(f"{sample_id}: missing speaker_a/speaker_b.")

    session_numbers = sorted(
        int(match.group(1)) for key in conversation if (match := _SESSION_KEY_RE.match(key)) is not None
    )
    if not session_numbers:
        raise LocomoDatasetError(f"{sample_id}: no session_N turn lists found.")

    turns: list[LocomoTurn] = []
    for session_number in session_numbers:
        date_time_key = f"session_{session_number}_date_time"
        session_date_time = conversation.get(date_time_key)
        if not session_date_time:
            raise LocomoDatasetError(f"{sample_id}: missing {date_time_key} for session_{session_number}.")
        for raw_turn in conversation[f"session_{session_number}"]:
            turns.append(
                LocomoTurn(
                    dia_id=raw_turn["dia_id"],
                    speaker=raw_turn["speaker"],
                    text=raw_turn["text"],
                    session_number=session_number,
                    session_date_time=session_date_time,
                    image_caption=raw_turn.get("blip_caption"),
                )
            )

    known_dia_ids = {turn.dia_id for turn in turns}
    qa: list[LocomoQA] = []
    for raw_qa in raw.get("qa", []):
        evidence_raw = list(raw_qa.get("evidence") or [])
        resolved, unresolved = _normalize_evidence(evidence_raw, known_dia_ids)
        raw_answer = raw_qa.get("answer")
        qa.append(
            LocomoQA(
                question=raw_qa["question"],
                answer=None if raw_answer is None else str(raw_answer),
                adversarial_answer=raw_qa.get("adversarial_answer"),
                category_id=raw_qa["category"],
                evidence=resolved,
                evidence_raw=evidence_raw,
                unresolved_evidence=unresolved,
            )
        )

    return LocomoSample(sample_id=sample_id, speaker_a=speaker_a, speaker_b=speaker_b, turns=turns, qa=qa)


def load_locomo_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[LocomoSample]:
    """Parse the LoCoMo JSON file into chronologically ordered samples."""

    raw_samples = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raise LocomoDatasetError(f"{path}: expected a top-level JSON list of samples.")
    return [_parse_sample(raw) for raw in raw_samples]
