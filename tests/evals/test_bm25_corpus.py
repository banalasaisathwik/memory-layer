"""Unit tests for the eval-only prepared BM25 corpus (Part C, no database)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from evals.locomo.bm25_corpus import prepare_bm25_corpus
from meminfra.retrieval.bm25 import bm25_rank


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


DOCS = [
    Doc("a", "Gina opened an online clothing store"),
    Doc("b", "Gina bought furniture for her store"),
    Doc("c", "Jon changed jobs"),
    Doc("d", "Gina filed ticket PROJ-42 for the store"),
    Doc("e", "Alex delivers orders through DoorDash every week"),
]


def _ids_and_scores(hits) -> list[tuple[str, float]]:
    return [(hit.item.id, hit.score) for hit in hits]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "PROJ-42",
        "What clothing business does Gina run?",
        "Gina clothing store business",
        "DoorDash",
        "Jon changed jobs",
        "totally unrelated term xyz",
    ],
)
def test_prepared_corpus_score_matches_bm25_rank(query: str) -> None:
    """bm25_rank(raw docs) == prepared_bm25.score(query) for a variety of query shapes."""

    reference = bm25_rank(query, DOCS, text_of=lambda d: d.text)
    prepared = prepare_bm25_corpus(DOCS, text_of=lambda d: d.text)
    optimized = prepared.score(query)

    reference_pairs = _ids_and_scores(reference)
    optimized_pairs = _ids_and_scores(optimized)

    assert [i for i, _ in optimized_pairs] == [i for i, _ in reference_pairs]
    for (_, ref_score), (_, opt_score) in zip(reference_pairs, optimized_pairs):
        assert opt_score == pytest.approx(ref_score, rel=1e-9, abs=1e-12)


def test_prepared_corpus_reused_across_many_queries_is_still_exact() -> None:
    """The whole point of Part C: one prepared corpus, many queries, same numbers."""

    prepared = prepare_bm25_corpus(DOCS, text_of=lambda d: d.text)
    queries = [
        "Gina",
        "clothing business",
        "Gina's retail store",
        "PROJ-42",
        "DoorDash delivery",
    ]
    for query in queries:
        reference = bm25_rank(query, DOCS, text_of=lambda d: d.text)
        optimized = prepared.score(query)
        assert [i for i, _ in _ids_and_scores(optimized)] == [i for i, _ in _ids_and_scores(reference)]
        for (_, ref_score), (_, opt_score) in zip(_ids_and_scores(reference), _ids_and_scores(optimized)):
            assert opt_score == pytest.approx(ref_score, rel=1e-9, abs=1e-12)


def test_prepared_corpus_empty_query_returns_no_results() -> None:
    prepared = prepare_bm25_corpus(DOCS, text_of=lambda d: d.text)
    assert prepared.score("") == []
    assert prepared.score("...") == []


def test_prepared_corpus_empty_document_list_returns_no_results() -> None:
    prepared = prepare_bm25_corpus([], text_of=lambda d: d.text)
    assert prepared.score("clothing") == []


def test_prepared_corpus_exposes_expected_fields() -> None:
    prepared = prepare_bm25_corpus(DOCS, text_of=lambda d: d.text)
    assert prepared.corpus_size == len(DOCS)
    assert prepared.document_ids == [d.id for d in DOCS]
    assert prepared.memory_texts == [d.text for d in DOCS]
    assert len(prepared.tokenized_docs) == len(DOCS)
    assert len(prepared.document_lengths) == len(DOCS)
    assert prepared.avgdl > 0
    assert isinstance(prepared.document_frequencies, dict)
