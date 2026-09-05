"""Pure unit tests for the ported LoCoMo QA scorer. No network, no DB."""

from __future__ import annotations

import pytest

from evals.locomo.scoring import (
    ScoringError,
    f1_score,
    is_abstention_phrase,
    multi_hop_f1,
    normalize_answer,
    score_qa,
)


def test_normalize_answer_strips_articles_punctuation_case_and_whitespace() -> None:
    assert normalize_answer("The, Painting!") == "painting"
    assert normalize_answer("A dog and a cat") == "dog cat"
    assert normalize_answer("  Extra   spaces  ") == "extra spaces"


def test_f1_score_exact_match_is_one() -> None:
    assert f1_score("Painting", "painting") == 1.0


def test_f1_score_partial_overlap() -> None:
    # prediction tokens: {sunsets, over, lake} (3), gold tokens: {sunsets, lake} (2, article dropped)
    # common = {sunsets, lake} = 2 -> precision 2/3, recall 2/2 -> f1 = 0.8
    score = f1_score("sunsets over the lake", "the lake and sunsets")
    assert score == pytest.approx(0.8)


def test_f1_score_no_overlap_is_zero() -> None:
    assert f1_score("Painting", "Hiking") == 0.0


def test_f1_score_empty_prediction_is_zero() -> None:
    assert f1_score("", "Painting") == 0.0


def test_multi_hop_f1_matches_each_gold_subanswer_to_best_prediction() -> None:
    # Order-independent: each gold sub-answer finds its best-matching predicted
    # sub-answer, not a positional pairing.
    score = multi_hop_f1("friend, lake", "lake, friend")
    assert score == pytest.approx(1.0)


def test_multi_hop_f1_averages_partial_subanswer_matches() -> None:
    # "lake" only partially overlaps "sunsets over the lake" (article "the" is
    # dropped by normalization): precision 1/1, recall 1/3 -> f1 0.5 for that
    # sub-answer; "friend" is an exact match against the other -> f1 1.0.
    score = multi_hop_f1("lake, friend", "sunsets over the lake, friend")
    assert score == pytest.approx(0.75)


def test_is_abstention_phrase_matches_official_phrases_case_insensitively() -> None:
    assert is_abstention_phrase("There is NO INFORMATION AVAILABLE about that.")
    assert is_abstention_phrase("This is not mentioned in the conversation.")
    assert not is_abstention_phrase("The user prefers PostgreSQL.")


def test_score_qa_single_hop_and_temporal_use_plain_f1() -> None:
    score, is_correct_abstention = score_qa(
        category_id=4, predicted_answer="Painting", gold_answer="painting", abstained=False
    )
    assert score == 1.0
    assert is_correct_abstention is None


def test_score_qa_open_domain_strips_semicolon_alternatives_from_gold() -> None:
    score, _ = score_qa(
        category_id=3, predicted_answer="Yes", gold_answer="Yes; possibly", abstained=False
    )
    assert score == 1.0


def test_score_qa_multi_hop_uses_comma_split_subanswers() -> None:
    score, is_correct_abstention = score_qa(
        category_id=1,
        predicted_answer="friend, lake",
        gold_answer="lake, friend",
        abstained=False,
    )
    assert score == pytest.approx(1.0)
    assert is_correct_abstention is None


def test_score_qa_requires_a_gold_answer_for_non_adversarial_categories() -> None:
    with pytest.raises(ScoringError):
        score_qa(category_id=4, predicted_answer="Painting", gold_answer=None, abstained=False)


def test_score_qa_adversarial_correct_via_official_phrase() -> None:
    score, is_correct_abstention = score_qa(
        category_id=5,
        predicted_answer="There is no information available about that in the conversation.",
        gold_answer=None,
        abstained=False,
    )
    assert score == 1.0
    assert is_correct_abstention is True


def test_score_qa_adversarial_correct_via_our_structured_abstained_flag() -> None:
    # Our reader's fixed abstention sentence does not literally contain either
    # official phrase, so the structured `abstained` flag must also count as
    # a correct abstention -- this is the documented, deliberate deviation.
    score, is_correct_abstention = score_qa(
        category_id=5,
        predicted_answer="The retrieved memories do not contain enough information to answer this question.",
        gold_answer=None,
        abstained=True,
    )
    assert score == 1.0
    assert is_correct_abstention is True


def test_score_qa_adversarial_wrong_when_it_confidently_answers() -> None:
    score, is_correct_abstention = score_qa(
        category_id=5,
        predicted_answer="Self-care is important.",
        gold_answer=None,
        abstained=False,
    )
    assert score == 0.0
    assert is_correct_abstention is False
