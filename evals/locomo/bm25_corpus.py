"""A BM25 corpus prepared once per benchmark run (eval-only, Part C).

``meminfra.retrieval.bm25.bm25_rank`` deliberately recomputes tokenization, document
lengths, avgdl, and per-query document frequencies/IDF on every call -- exactly
right for production, where the scoped corpus can change between searches.
For a frozen LoCoMo benchmark run the corpus is the same set of active
memories for the whole run, so redoing that work for all ~105 questions is
pure waste: tokenizing the same memory text over and over, and rebuilding the
same avgdl/document-frequency tables each time.

``PreparedBM25Corpus`` precomputes everything that only depends on the corpus
(tokenized docs, per-doc term frequencies, document lengths, avgdl, and
corpus-wide document frequencies) exactly once, then ``score()`` repeats only
the truly query-dependent step (per-query-term IDF lookup and the scoring
loop) using the identical Okapi BM25 formula, k1, b, and tie-breaking as
``bm25_rank``. This module never changes the tokenizer, the IDF formula, or
the score formula -- it only caches the corpus-derived intermediates so they
are computed once instead of once per question. It is eval-only: production
``search_memories(..., lexical_backend="bm25")`` is untouched and still calls
``bm25_retrieve``/``bm25_rank`` directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log
from typing import Callable, Generic, Protocol, TypeVar

from meminfra.retrieval.bm25 import BM25_B, BM25_K1, BM25Hit, tokenize


class _HasId(Protocol):
    id: object


_T = TypeVar("_T", bound=_HasId)


@dataclass(frozen=True)
class PreparedBM25Corpus(Generic[_T]):
    """Corpus-derived BM25 intermediates, computed once and reused per query."""

    documents: list[_T] = field(repr=False)
    document_ids: list[str]
    memory_texts: list[str]
    tokenized_docs: list[list[str]] = field(repr=False)
    term_frequencies: list[Counter] = field(repr=False)
    document_lengths: list[int]
    avgdl: float
    document_frequencies: dict[str, int]
    corpus_size: int

    def score(self, query: str, *, k1: float = BM25_K1, b: float = BM25_B) -> list[BM25Hit[_T]]:
        """Score ``query`` against this prepared corpus with the exact bm25_rank formula.

        Only the query-dependent IDF lookup and per-document scoring loop run
        here; everything corpus-derived was already computed once in
        :func:`prepare_bm25_corpus`. Numerically identical to calling
        ``bm25_rank(query, self.documents, text_of=...)`` fresh each time, for
        the same reasons ``document_frequencies`` above is safe to precompute:
        a term's document frequency does not depend on the query, only on the
        (unchanged) corpus.
        """

        query_terms = tokenize(query)
        if not query_terms or not self.documents or self.avgdl == 0:
            return []

        unique_query_terms = sorted(set(query_terms))
        idf = {
            term: log(
                1
                + (self.corpus_size - self.document_frequencies.get(term, 0) + 0.5)
                / (self.document_frequencies.get(term, 0) + 0.5)
            )
            for term in unique_query_terms
        }

        hits: list[BM25Hit[_T]] = []
        for document, term_counts, length in zip(self.documents, self.term_frequencies, self.document_lengths):
            if length == 0:
                continue
            score = 0.0
            for term in unique_query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (k1 + 1)
                denominator = frequency + k1 * (1 - b + b * (length / self.avgdl))
                score += idf[term] * (numerator / denominator)
            if score > 0:
                hits.append(BM25Hit(item=document, score=score))

        hits.sort(key=lambda hit: (-hit.score, str(hit.item.id)))
        return hits


def prepare_bm25_corpus(documents: list[_T], *, text_of: Callable[[_T], str]) -> PreparedBM25Corpus[_T]:
    """Precompute every corpus-derived BM25 intermediate exactly once.

    ``documents`` must already be the caller's fully scoped, authorized
    corpus, exactly like ``bm25_rank`` -- this function adds no filtering of
    its own. Callers that need a different scope (a different user, a
    different filter set) must call this again with that scope's documents;
    never reuse one user's ``PreparedBM25Corpus`` for another user's query.
    """

    tokenized_docs = [tokenize(text_of(document)) for document in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    non_empty_lengths = [length for length in doc_lengths if length > 0]
    avgdl = sum(non_empty_lengths) / len(non_empty_lengths) if non_empty_lengths else 0.0

    term_frequencies = [Counter(tokens) for tokens in tokenized_docs]
    document_frequencies: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    return PreparedBM25Corpus(
        documents=list(documents),
        document_ids=[str(document.id) for document in documents],
        memory_texts=[text_of(document) for document in documents],
        tokenized_docs=tokenized_docs,
        term_frequencies=term_frequencies,
        document_lengths=doc_lengths,
        avgdl=avgdl,
        document_frequencies=document_frequencies,
        corpus_size=len(documents),
    )
