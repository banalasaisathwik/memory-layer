"""Explicit failures for the small, user-scoped retrieval API."""


class RetrievalError(Exception):
    """Base class for retrieval failures that callers can handle deliberately."""


class InvalidSearchError(RetrievalError):
    """Raised when a caller supplies an invalid search query or limit."""


class UserNotFoundError(RetrievalError):
    """Raised when retrieval is requested for a user that does not exist."""


class InvalidFilterScopeError(RetrievalError):
    """Raised when a supplied conversation filter is not valid for the user."""


class EmbeddingError(RetrievalError):
    """Raised when the configured embedding provider cannot create a vector."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised for zero, non-finite, or inconsistent embedding vectors."""


class IndexStateError(RetrievalError):
    """Raised when persisted FAISS derived state cannot be trusted."""


class IndexModelMismatchError(IndexStateError):
    """Raised when an index cannot safely be used with the configured model."""


class IndexDimensionMismatchError(IndexStateError):
    """Raised when a query vector cannot safely be searched in a FAISS index."""
