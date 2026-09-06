"""Unit tests for genuine Okapi BM25 lexical scoring (no database required)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from meminfra.retrieval import BM25_B, BM25_K1, bm25_rank, tokenize


@dataclass(frozen=True)
class Doc:
    """A minimal id+text document, independent of Message/Memory ORM models."""

    id: str
    text: str


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("PostgreSQL, DoorDash! Gina's Project-123.") == [
        "postgresql",
        "doordash",
        "gina",
        "s",
        "project",
        "123",
    ]


def test_tokenize_handles_empty_and_punctuation_only_text() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("...!?--") == []


def test_bm25_partial_matching_ranks_the_best_overlap_highest() -> None:
    d1 = Doc("d1", "Gina opened an online clothing store")
    d2 = Doc("d2", "Gina bought furniture")
    d3 = Doc("d3", "Jon works at a bank")

    hits = bm25_rank("Gina clothing business", [d1, d2, d3], text_of=lambda d: d.text)

    assert [hit.item.id for hit in hits] == ["d1", "d2"]
    assert hits[0].score > hits[1].score


def test_bm25_rare_terms_rank_strongly() -> None:
    matching = Doc("matching", "Alex delivers orders through DoorDash every week")
    common_words_only = Doc("common", "Alex delivers orders through the week")
    unrelated = Doc("unrelated", "The weather was nice today")

    hits = bm25_rank("DoorDash", [matching, common_words_only, unrelated], text_of=lambda d: d.text)

    assert [hit.item.id for hit in hits] == ["matching"]


def test_bm25_does_not_require_every_query_term() -> None:
    query = "What clothing business does Gina run these days"
    a = Doc("a", "Gina opened an online clothing store")
    b = Doc("b", "Gina bought furniture for her store")
    c = Doc("c", "Jon changed jobs")

    hits = bm25_rank(query, [a, b, c], text_of=lambda d: d.text)

    assert [hit.item.id for hit in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score


def test_bm25_normalizes_for_document_length() -> None:
    noisy_but_one_match = Doc(
        "noisy",
        "word " * 200 + "clothing",
    )
    concise_match = Doc("concise", "Gina clothing store")

    hits = bm25_rank("clothing", [noisy_but_one_match, concise_match], text_of=lambda d: d.text)

    assert [hit.item.id for hit in hits] == ["concise", "noisy"]
    assert hits[0].score > hits[1].score


def test_bm25_empty_query_returns_no_results() -> None:
    docs = [Doc("d1", "Gina opened a clothing store")]

    assert bm25_rank("", docs, text_of=lambda d: d.text) == []
    assert bm25_rank("   ", docs, text_of=lambda d: d.text) == []
    assert bm25_rank("...", docs, text_of=lambda d: d.text) == []


def test_bm25_empty_document_list_returns_no_results() -> None:
    assert bm25_rank("clothing", [], text_of=lambda d: d.text) == []


def test_bm25_documents_with_no_query_term_overlap_are_excluded() -> None:
    d1 = Doc("d1", "Gina opened an online clothing store")
    d2 = Doc("d2", "Jon changed jobs")

    hits = bm25_rank("Gina clothing", [d1, d2], text_of=lambda d: d.text)

    assert [hit.item.id for hit in hits] == ["d1"]


def test_bm25_is_deterministic_for_the_same_corpus_and_query() -> None:
    docs = [
        Doc("a", "Gina opened an online clothing store"),
        Doc("b", "Gina bought furniture"),
        Doc("c", "Jon works at a bank"),
    ]

    first = [(hit.item.id, hit.score) for hit in bm25_rank("Gina clothing business", docs, text_of=lambda d: d.text)]
    second = [(hit.item.id, hit.score) for hit in bm25_rank("Gina clothing business", docs, text_of=lambda d: d.text)]

    assert first == second


def test_bm25_breaks_score_ties_deterministically_by_id() -> None:
    tied_high_id = Doc("z", "clothing store")
    tied_low_id = Doc("a", "clothing store")

    hits = bm25_rank("clothing", [tied_high_id, tied_low_id], text_of=lambda d: d.text)

    assert hits[0].score == pytest.approx(hits[1].score)
    assert [hit.item.id for hit in hits] == ["a", "z"]


def test_bm25_documents_use_conventional_default_parameters() -> None:
    assert BM25_K1 == pytest.approx(1.5)
    assert BM25_B == pytest.approx(0.75)
