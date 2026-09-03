"""Small shared primitives for durable FAISS indexes.

The Memory and raw Message indexes intentionally have different ownership and
metadata contracts.  They share only vector validation, provider response
handling, safe fingerprints, and atomic-file preparation.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .errors import EmbeddingError, InvalidEmbeddingError


def safe_fingerprint(value: str) -> str:
    """Return a deterministic filesystem-safe identifier without exposing input."""

    return sha256(value.encode("utf-8")).hexdigest()


def normalize_embedding(values: object) -> np.ndarray:
    """Validate and normalize one vector for IndexFlatIP cosine search."""

    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise InvalidEmbeddingError("Embedding values must be a numeric vector.") from error
    if vector.ndim != 1 or vector.size == 0:
        raise InvalidEmbeddingError("Embedding vectors must be one-dimensional and non-empty.")
    if not np.isfinite(vector).all():
        raise InvalidEmbeddingError("Embedding vectors must contain only finite values.")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise InvalidEmbeddingError("Embedding vectors must have a non-zero norm.")
    return vector / norm


def embedding_response_vectors(
    texts: list[str],
    *,
    model: str,
    client_factory: Callable[[], Any],
) -> list[np.ndarray]:
    """Request one embedding batch and validate its provider-neutral shape."""

    if not texts:
        return []
    try:
        response = client_factory().embeddings.create(model=model, input=texts)
        data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
    except Exception as error:
        raise EmbeddingError("The configured embedding provider could not generate vectors.") from error
    if len(data) != len(texts):
        raise EmbeddingError("The embedding provider returned an unexpected number of vectors.")
    return [normalize_embedding(item.embedding) for item in data]


def temporary_path(directory: Path, suffix: str) -> Path:
    """Reserve a sibling temporary file for replace-on-complete persistence."""

    handle = tempfile.NamedTemporaryFile(dir=directory, suffix=suffix, delete=False)
    handle.close()
    return Path(handle.name)
