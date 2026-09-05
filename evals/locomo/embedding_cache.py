"""Eval-only, on-disk query-embedding cache for LoCoMo retrieval benchmarks (Part E).

Every benchmark question's text is fixed for a whole run (and across resumed
runs, since the vendored dataset is versioned by its sha256). Re-embedding
the same question text on every ablation/backend re-run wastes provider
calls and money for a value that never changes as long as the dataset,
embedding provider, model, and dimension all stay the same. This cache
exists to remove that waste.

Scope, deliberately narrow:
    * This module is never imported by ``src/`` and never touches
      ``MemoryLayer.search()``, ``search_memories()``, or ``vector_retrieve()``
      for arbitrary production queries. It is only for benchmark question
      embeddings computed by an eval runner.
    * The cache key includes the dataset hash, the question text, the
      embedding provider, the embedding model, and the embedding dimension --
      a change in any of those is a cache miss, never a stale hit.
    * A corrupt or unreadable cache entry is treated as a miss (safe
      recompute), never a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "query_embeddings"


@dataclass(frozen=True)
class EmbeddingCacheContext:
    """The non-question part of the cache key: what must match for a hit."""

    dataset_sha256: str
    provider: str
    model: str
    dimension: int


@dataclass
class EmbeddingCacheStats:
    """Simple hit/miss/call counters for a Part L run summary."""

    hits: int = 0
    misses: int = 0
    corrupt_entries: int = 0
    provider_batch_calls: int = 0
    embeddings_requested_from_provider: int = 0


def cache_key(question: str, *, context: EmbeddingCacheContext) -> str:
    """sha256(dataset_sha + question + provider + model + dimension)."""

    payload = "\x1f".join(
        [context.dataset_sha256, question, context.provider, context.model, str(context.dimension)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QueryEmbeddingCache:
    """A small, atomic, JSON-file-per-entry cache under ``evals/cache/``."""

    def __init__(self, *, context: EmbeddingCacheContext, cache_dir: Path | None = None) -> None:
        self.context = context
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = EmbeddingCacheStats()

    def _path(self, question: str) -> Path:
        return self.cache_dir / f"{cache_key(question, context=self.context)}.json"

    def get(self, question: str) -> np.ndarray | None:
        """Return the cached vector for ``question``, or None on any miss/corruption."""

        path = self._path(question)
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            vector = raw["vector"]
            if (
                raw.get("dataset_sha256") != self.context.dataset_sha256
                or raw.get("provider") != self.context.provider
                or raw.get("model") != self.context.model
                or raw.get("dimension") != self.context.dimension
                or not isinstance(vector, list)
                or len(vector) != self.context.dimension
            ):
                self.stats.corrupt_entries += 1
                self.stats.misses += 1
                return None
            array = np.asarray(vector, dtype=np.float32)
        except Exception:
            # Corrupt/unreadable cache file: safe fallback to a miss, never a crash.
            self.stats.corrupt_entries += 1
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return array

    def put(self, question: str, vector: np.ndarray) -> None:
        """Persist ``vector`` for ``question`` with replace-on-complete durability."""

        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size != self.context.dimension:
            raise ValueError(
                f"Embedding cache dimension mismatch: expected {self.context.dimension}, got {array.shape}."
            )
        path = self._path(question)
        payload = {
            "question": question,
            "dataset_sha256": self.context.dataset_sha256,
            "provider": self.context.provider,
            "model": self.context.model,
            "dimension": self.context.dimension,
            "vector": array.astype(float).tolist(),
        }
        handle = tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".json.tmp", delete=False, mode="w", encoding="utf-8"
        )
        temporary_path = Path(handle.name)
        try:
            json.dump(payload, handle)
            handle.close()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def ensure_batch(
        self,
        questions: list[str],
        *,
        embed_many: Callable[[list[str]], list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Load cached vectors for ``questions``; batch-embed only the misses.

        ``embed_many`` must accept a list of texts and return one vector per
        text, in the same order (the provider abstraction already supports
        this -- see ``embedding_response_vectors`` in
        ``src/retrieval/vector_support.py``). Distinct duplicate questions are
        embedded only once.
        """

        result: dict[str, np.ndarray] = {}
        misses: list[str] = []
        seen: set[str] = set()
        for question in questions:
            if question in seen:
                continue
            seen.add(question)
            cached = self.get(question)
            if cached is not None:
                result[question] = cached
            else:
                misses.append(question)

        if misses:
            self.stats.provider_batch_calls += 1
            self.stats.embeddings_requested_from_provider += len(misses)
            vectors = embed_many(misses)
            if len(vectors) != len(misses):
                raise ValueError("Embedding provider returned an unexpected number of vectors for a batch request.")
            for question, vector in zip(misses, vectors, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                if array.size != self.context.dimension:
                    raise ValueError(
                        f"Embedding provider returned dimension {array.size}, expected {self.context.dimension}."
                    )
                self.put(question, array)
                result[question] = array
        return result
