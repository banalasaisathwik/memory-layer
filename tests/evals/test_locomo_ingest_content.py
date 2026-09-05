"""Pure unit tests for LoCoMo message-content construction. No network, no DB.

These test the deterministic representation choices documented in
evals/locomo/ingest.py and evals/README.md: speaker identity stays in the
text, the session timestamp is prefixed only on a session's first turn, and
an image caption (when present) is appended without downloading the image.
"""

from __future__ import annotations

from evals.locomo.dataset import load_locomo_dataset
from evals.locomo.ingest import _group_by_session, _turn_content
from evals.locomo.schemas import LocomoTurn
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "locomo_sample.json"


def test_speaker_name_is_preserved_in_message_content() -> None:
    turn = LocomoTurn(
        dia_id="D1:1",
        speaker="Caroline",
        text="I started painting again.",
        session_number=1,
        session_date_time="1:56 pm on 8 May, 2023",
    )
    content = _turn_content(turn, is_first_in_session=False)
    assert content == "Caroline: I started painting again."


def test_first_turn_of_a_session_is_prefixed_with_the_session_timestamp() -> None:
    turn = LocomoTurn(
        dia_id="D1:1",
        speaker="Caroline",
        text="I started painting again.",
        session_number=1,
        session_date_time="1:56 pm on 8 May, 2023",
    )
    content = _turn_content(turn, is_first_in_session=True)
    assert content == "[Session date: 1:56 pm on 8 May, 2023]\nCaroline: I started painting again."


def test_non_first_turns_do_not_repeat_the_timestamp() -> None:
    turn = LocomoTurn(
        dia_id="D1:2",
        speaker="Bob",
        text="Nice!",
        session_number=1,
        session_date_time="1:56 pm on 8 May, 2023",
    )
    content = _turn_content(turn, is_first_in_session=False)
    assert "Session date" not in content


def test_image_caption_is_appended_without_downloading_the_image() -> None:
    turn = LocomoTurn(
        dia_id="D1:5",
        speaker="Caroline",
        text="Look at this!",
        session_number=1,
        session_date_time="1:56 pm on 8 May, 2023",
        image_caption="a photo of a dog walking past a wall",
    )
    content = _turn_content(turn, is_first_in_session=False)
    assert content == "Caroline: Look at this! [shared image: a photo of a dog walking past a wall]"


def test_group_by_session_preserves_within_session_order() -> None:
    sample = load_locomo_dataset(FIXTURE_PATH)[0]
    sessions = _group_by_session(sample.turns)
    assert sorted(sessions) == [1, 2]
    assert [turn.dia_id for turn in sessions[1]] == ["D1:1", "D1:2", "D1:3"]
    assert [turn.dia_id for turn in sessions[2]] == ["D2:1", "D2:2"]
