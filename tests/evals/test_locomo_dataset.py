"""Pure, local-fixture tests for LoCoMo dataset parsing. No network, no DB."""

from __future__ import annotations

from pathlib import Path

from evals.locomo.dataset import load_locomo_dataset
from evals.locomo.schemas import CATEGORY_NAMES

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_sample.json"


def _load():
    samples = load_locomo_dataset(FIXTURE_PATH)
    assert len(samples) == 1
    return samples[0]


def test_loads_sample_identity_and_speakers() -> None:
    sample = _load()
    assert sample.sample_id == "conv-test-1"
    assert sample.speaker_a == "Alice"
    assert sample.speaker_b == "Bob"


def test_sessions_are_ordered_chronologically_with_turn_order_preserved() -> None:
    sample = _load()
    dia_ids = [turn.dia_id for turn in sample.turns]
    assert dia_ids == ["D1:1", "D1:2", "D1:3", "D2:1", "D2:2"]
    session_numbers = [turn.session_number for turn in sample.turns]
    assert session_numbers == sorted(session_numbers)


def test_session_timestamps_are_preserved_per_turn() -> None:
    sample = _load()
    session_1_turns = [turn for turn in sample.turns if turn.session_number == 1]
    session_2_turns = [turn for turn in sample.turns if turn.session_number == 2]
    assert all(turn.session_date_time == "10:00 am on 1 January, 2024" for turn in session_1_turns)
    assert all(turn.session_date_time == "3:00 pm on 5 January, 2024" for turn in session_2_turns)


def test_both_speakers_map_to_user_role() -> None:
    sample = _load()
    assert sample.role_for_speaker("Alice") == "user"
    assert sample.role_for_speaker("Bob") == "user"


def test_category_mapping_matches_official_locomo_ids() -> None:
    assert CATEGORY_NAMES == {
        1: "multi_hop",
        2: "temporal",
        3: "open_domain",
        4: "single_hop",
        5: "adversarial",
    }
    sample = _load()
    by_question = {qa.question: qa for qa in sample.qa}
    assert by_question["What hobby did Alice start again?"].category_name == "single_hop"
    assert by_question["When did Alice sell her painting?"].category_name == "temporal"
    assert by_question["Would Alice be interested in taking an art class?"].category_name == "open_domain"
    assert by_question["What did Alice paint and where did she sell it?"].category_name == "multi_hop"


def test_no_evidence_question_is_preserved_as_empty_evidence() -> None:
    sample = _load()
    open_domain = next(qa for qa in sample.qa if qa.category_id == 3)
    assert open_domain.evidence == []


def test_evidence_dia_ids_are_normalized_and_unresolved_ones_are_tracked() -> None:
    sample = _load()
    multi_hop = next(qa for qa in sample.qa if qa.category_id == 1)
    # "D1:01" (zero-padded) resolves to the real "D1:1" turn; "Dbogus" does not
    # match any dia_id pattern and is kept as an unresolved diagnostic instead
    # of being silently dropped or invented.
    assert multi_hop.evidence == ["D1:1"]
    assert multi_hop.unresolved_evidence == ["Dbogus"]
    assert multi_hop.evidence_raw == ["D1:01", "Dbogus"]
