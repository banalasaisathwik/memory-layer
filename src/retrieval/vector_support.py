"""Small shared primitives for durable FAISS indexes.

The Memory and raw Message indexes intentionally have different ownership and
metadata contracts.  They share only vector validation, provider response
handling, safe fingerprints, and atomic-file preparation.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import faiss
import numpy as np

from .errors import EmbeddingError, IndexStateError, InvalidEmbeddingError


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


def persist_index(
    *,
    index: Any,
    index_path: Path,
    metadata_path: Path,
    metadata: dict[str, object],
    error_message: str,
) -> None:
    """Persist a FAISS index and its metadata with replace-on-complete durability.

    Both files are written to sibling temporary paths first and only replace
    the real files after both writes succeed, so a concurrent reader never
    observes a FAISS index without its matching position-mapping metadata (or
    vice versa). Identical for the Memory and Message indexes; only the
    metadata contents differ, and building that dict remains the caller's job.
    """

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = temporary_path(index_path.parent, ".faiss.tmp")
    temporary_metadata = temporary_path(index_path.parent, ".json.tmp")
    try:
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary_index, index_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception as error:
        raise IndexStateError(error_message) from error
    finally:
        for path in (temporary_index, temporary_metadata):
            if path.exists():
                path.unlink(missing_ok=True)


def load_and_validate_index(
    *,
    index_path: Path,
    metadata_path: Path,
    missing_message: str,
    corrupt_message: str,
    validate_metadata: Callable[[dict[str, Any]], tuple[list[str], int]],
) -> tuple[Any, list[str], int]:
    """Load one FAISS index plus its JSON metadata and cross-check their shape.

    ``validate_metadata`` receives the parsed metadata dict and must return
    ``(ids, dimension)`` after checking the caller's own scope/model/format
    fields, raising IndexStateError (or a subclass, e.g. for a model
    mismatch) for anything invalid. That keeps each index kind's specific
    checks and exception types next to its own metadata contract; only the
    identical file-existence check, JSON load, FAISS read, and final
    index-vs-mapping shape check live here.
    """

    if not index_path.is_file() or not metadata_path.is_file():
        raise IndexStateError(missing_message)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ids, dimension = validate_metadata(metadata)
        index = faiss.read_index(str(index_path))
        if index.d != dimension or index.ntotal != len(ids):
            raise IndexStateError(corrupt_message)
    except IndexStateError:
        raise
    except Exception as error:
        raise IndexStateError(corrupt_message) from error
    return index, ids, dimension
