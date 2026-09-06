"""Genuine Okapi BM25 lexical scoring over an already scope-restricted corpus.

This is deliberately not a rename of PostgreSQL's ``ts_rank``/``ts_rank_cd``:
it implements the standard BM25 term-frequency-saturation + inverse-document-
frequency + document-length-normalization formula, computed in-process. It
never depends on a strict AND-style full-text-search candidate filter --
every document handed to :func:`bm25_rank` gets scored, so a document that
shares only some query terms with the query can still rank above a document
that shares none, exactly as BM25 is supposed to behave.

Scope safety (user, conversation, active/inactive, memory-type filters, ...)
is entirely the caller's responsibility: :func:`bm25_rank` scores exactly the
``documents`` it is given and never fetches anything else. Callers must
restrict ``documents`` to the authorized corpus *before* calling this
function.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Callable, Generic, Protocol, TypeVar


# Conventional Okapi BM25 defaults. Not tuned against any benchmark.
BM25_K1 = 1.5
BM25_B = 0.75

# Deterministic, simple normalization: lowercase word tokenization. This does
# not stem or remove stopwords -- BM25's own IDF term already down-weights
# common words, and aggressive normalization would risk losing exactly the
# rare, high-signal identifiers (e.g. "PostgreSQL", "DoorDash", "KAN-4") that
# make lexical search useful in the first place. Hyphenated identifiers such
# as "Project-123" tokenize as separate words ("project", "123"), which still
# lets a query containing either half match.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-word tokenization with no stemming or stopword removal."""

    return _TOKEN_PATTERN.findall(text.casefold())


class _HasId(Protocol):
    id: object


_T = TypeVar("_T", bound=_HasId)


@dataclass(frozen=True)
class BM25Hit(Generic[_T]):
    """One scored document; only documents with a positive score are returned."""

    item: _T
    score: float


def bm25_rank(
    query: str,
    documents: list[_T],
    *,
    text_of: Callable[[_T], str],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[BM25Hit[_T]]:
    """Rank every document in ``documents`` against ``query`` using Okapi BM25.

    ``documents`` must already be the caller's fully scoped, authorized
    corpus -- this function adds no filtering of its own beyond scoring.
    Returns only positive-score documents, most relevant first, with
    deterministic tie-breaking by ``str(item.id)``. An empty query, a
    query with no usable tokens (punctuation-only, etc.), or an empty
    document list all safely return ``[]`` rather than scoring everything
    equally.

    Formula, for each document D and each unique query term t::

        idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score(D) = sum over t of:
            idf(t) * f(t, D) * (k1 + 1)
                     / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))

    where N is the corpus size, df(t) the number of documents containing t,
    f(t, D) the term frequency of t in D, |D| the document's token count,
    and avgdl the average token count over non-empty documents.
    """

    query_terms = tokenize(query)
    if not query_terms or not documents:
        return []

    tokenized_docs = [tokenize(text_of(document)) for document in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    non_empty_lengths = [length for length in doc_lengths if length > 0]
    if not non_empty_lengths:
        return []
    avg_doc_length = sum(non_empty_lengths) / len(non_empty_lengths)

    document_count = len(documents)
    unique_query_terms = sorted(set(query_terms))
    document_frequency = {
        term: sum(1 for tokens in tokenized_docs if term in tokens) for term in unique_query_terms
    }
    idf = {
        term: log(1 + (document_count - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }

    hits: list[BM25Hit[_T]] = []
    for document, tokens, length in zip(documents, tokenized_docs, doc_lengths):
        if length == 0:
            continue
        term_counts = Counter(tokens)
        score = 0.0
        for term in unique_query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * (length / avg_doc_length))
            score += idf[term] * (numerator / denominator)
        if score > 0:
            hits.append(BM25Hit(item=document, score=score))

    hits.sort(key=lambda hit: (-hit.score, str(hit.item.id)))
    return hits
