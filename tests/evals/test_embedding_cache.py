"""Unit tests for the eval-only query embedding cache (Part E, no database)."""

from __future__ import annotations

import numpy as np
import pytest

from evals.locomo.embedding_cache import EmbeddingCacheContext, QueryEmbeddingCache


def _context(**overrides) -> EmbeddingCacheContext:
    defaults = dict(dataset_sha256="dataset-abc", provider="openai", model="text-embedding-3-small", dimension=4)
    defaults.update(overrides)
    return EmbeddingCacheContext(**defaults)


def test_miss_calls_provider_and_persists(tmp_path) -> None:
    cache = QueryEmbeddingCache(context=_context(), cache_dir=tmp_path)
    calls: list[list[str]] = []

    def embed_many(texts: list[str]) -> list[np.ndarray]:
        calls.append(texts)
        return [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in texts]

    result = cache.ensure_batch(["What color is the sky?"], embed_many=embed_many)

    assert calls == [["What color is the sky?"]]
    assert result["What color is the sky?"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
    assert cache.stats.provider_batch_calls == 1


def test_hit_does_not_call_provider(tmp_path) -> None:
    context = _context()
    cache = QueryEmbeddingCache(context=context, cache_dir=tmp_path)
    cache.put("What color is the sky?", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def embed_many(texts: list[str]) -> list[np.ndarray]:
        raise AssertionError("provider must not be called on a cache hit")

    result = cache.ensure_batch(["What color is the sky?"], embed_many=embed_many)

    assert result["What color is the sky?"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


def test_different_model_is_a_miss(tmp_path) -> None:
    cache_a = QueryEmbeddingCache(context=_context(model="model-a"), cache_dir=tmp_path)
    cache_a.put("q", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    cache_b = QueryEmbeddingCache(context=_context(model="model-b"), cache_dir=tmp_path)
    assert cache_b.get("q") is None


def test_different_dimension_is_a_miss(tmp_path) -> None:
    cache_a = QueryEmbeddingCache(context=_context(dimension=4), cache_dir=tmp_path)
    cache_a.put("q", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    cache_b = QueryEmbeddingCache(context=_context(dimension=8), cache_dir=tmp_path)
    assert cache_b.get("q") is None


def test_different_dataset_hash_is_a_miss(tmp_path) -> None:
    cache_a = QueryEmbeddingCache(context=_context(dataset_sha256="dataset-abc"), cache_dir=tmp_path)
    cache_a.put("q", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    cache_b = QueryEmbeddingCache(context=_context(dataset_sha256="dataset-xyz"), cache_dir=tmp_path)
    assert cache_b.get("q") is None


def test_corrupt_cache_file_is_a_safe_miss_not_a_crash(tmp_path) -> None:
    context = _context()
    cache = QueryEmbeddingCache(context=context, cache_dir=tmp_path)
    cache.put("q", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # Corrupt the persisted file directly.
    path = cache._path("q")
    path.write_text("{not valid json", encoding="utf-8")

    assert cache.get("q") is None  # no crash
    assert cache.stats.corrupt_entries >= 1


def test_ensure_batch_only_calls_provider_for_misses(tmp_path) -> None:
    context = _context()
    cache = QueryEmbeddingCache(context=context, cache_dir=tmp_path)
    cache.put("cached question", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    calls: list[list[str]] = []

    def embed_many(texts: list[str]) -> list[np.ndarray]:
        calls.append(texts)
        return [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32) for _ in texts]

    result = cache.ensure_batch(["cached question", "new question"], embed_many=embed_many)

    assert calls == [["new question"]]
    assert result["cached question"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert result["new question"] == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_ensure_batch_deduplicates_repeated_questions(tmp_path) -> None:
    context = _context()
    cache = QueryEmbeddingCache(context=context, cache_dir=tmp_path)
    calls: list[list[str]] = []

    def embed_many(texts: list[str]) -> list[np.ndarray]:
        calls.append(texts)
        return [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32) for _ in texts]

    cache.ensure_batch(["repeat", "repeat", "repeat"], embed_many=embed_many)

    assert calls == [["repeat"]]


def test_put_rejects_wrong_dimension(tmp_path) -> None:
    cache = QueryEmbeddingCache(context=_context(dimension=4), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        cache.put("q", np.array([1.0, 0.0], dtype=np.float32))
