"""Deterministic LoCoMo-compatible QA scoring, ported from the official eval.

Ported from ``task_eval/evaluation.py`` in ``snap-research/locomo``
(``normalize_answer``, ``f1_score``, ``f1``, and the category dispatch inside
``eval_question_answering``). Two deliberate deviations from the official
script, both because this project has no existing NLP-toolkit dependency and
should not add one just for this milestone:

1. No Porter stemming. The official script stems every normalized token
   before computing F1; this port compares normalized tokens directly. This
   only affects near-miss inflectional matches (e.g. "paints" vs "painted")
   and is not expected to change which system is better, only nudge absolute
   F1 slightly.
2. Adversarial (category 5) correctness accepts either of the official
   phrase checks (``"no information available"`` / ``"not mentioned"``
   appearing in the predicted answer) OR ``MemoryLayer.answer()``'s own
   structured ``abstained`` flag. Our reader abstains with a fixed sentence
   that does not literally contain either official phrase, so relying on
   phrase-matching alone would undercount correct abstentions our system can
   already report directly and truthfully.

Category IDs and names are the official mapping (see
``evals.locomo.schemas.CATEGORY_NAMES``): 1 multi_hop, 2 temporal,
3 open_domain, 4 single_hop, 5 adversarial.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the|and)\b")
ADVERSARIAL_ABSTENTION_PHRASES: tuple[str, ...] = ("no information available", "not mentioned")


class ScoringError(ValueError):
    """Raised when a QA item cannot be scored with the data it was given."""


def normalize_answer(text: str) -> str:
    """Match the official normalize_answer: strip commas/articles/punctuation/case/whitespace."""

    text = text.replace(",", "")
    text = _ARTICLE_RE.sub(" ", text.lower())
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 between a predicted and gold answer (single-hop shape)."""

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """Comma-split sub-answer F1, matching the official multi-hop ``f1()``.

    Both the prediction and the gold answer are split on ',' into
    sub-answers; the score is the mean, over gold sub-answers, of the best
    F1 against any predicted sub-answer.
    """

    predictions = [part.strip() for part in prediction.split(",") if part.strip()] or [prediction.strip()]
    ground_truths = [part.strip() for part in ground_truth.split(",") if part.strip()] or [ground_truth.strip()]
    return sum(max(f1_score(p, gt) for p in predictions) for gt in ground_truths) / len(ground_truths)


def is_abstention_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ADVERSARIAL_ABSTENTION_PHRASES)


def score_qa(
    *,
    category_id: int,
    predicted_answer: str,
    gold_answer: str | None,
    abstained: bool,
) -> tuple[float, bool | None]:
    """Score one prediction. Returns (score, is_correct_abstention).

    ``is_correct_abstention`` is populated only for category 5 (adversarial);
    it is ``None`` for every F1-scored category, keeping the two metric
    families never averaged together (see evals/README.md).
    """

    if category_id == 5:
        correct = abstained or is_abstention_phrase(predicted_answer)
        return (1.0 if correct else 0.0), correct

    if gold_answer is None:
        raise ScoringError(f"category {category_id} requires a gold answer to score F1.")

    if category_id == 3:
        gold_answer = gold_answer.split(";")[0].strip()

    if category_id == 1:
        return multi_hop_f1(predicted_answer, gold_answer), None

    if category_id in (2, 3, 4):
        return f1_score(predicted_answer, gold_answer), None

    raise ScoringError(f"Unknown LoCoMo category_id: {category_id}")
