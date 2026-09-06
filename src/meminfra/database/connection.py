"""Lazy SQLAlchemy connection helpers for Neon/PostgreSQL."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from meminfra.config import get_config


_engine: Engine | None = None
_engine_key: tuple[str, int, int] | None = None


def _sqlalchemy_url(database_url: str) -> str:
    """Use psycopg 3 when callers provide Neon's standard PostgreSQL URL."""

    if database_url.startswith("postgresql://"):
        return f"postgresql+psycopg://{database_url.removeprefix('postgresql://')}"
    if database_url.startswith("postgres://"):
        return f"postgresql+psycopg://{database_url.removeprefix('postgres://')}"
    return database_url


def get_engine() -> Engine:
    """Create an engine only when database access is first requested."""

    global _engine, _engine_key
    settings = get_config()
    database_url = settings.database_url
    if not database_url:
        raise RuntimeError("DATABASE_URL must be configured before accessing the database.")

    engine_key = (
        database_url,
        settings.database_pool_size,
        settings.database_max_overflow,
    )
    if _engine is not None and _engine_key == engine_key:
        return _engine

    if _engine is not None:
        _engine.dispose()

    _engine = create_engine(
        _sqlalchemy_url(database_url),
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
    _engine_key = engine_key
    return _engine


def SessionLocal() -> Session:
    """Return a new session bound to the configured lazy engine."""

    return Session(bind=get_engine(), expire_on_commit=False)


def create_tables() -> None:
    """Create the current schema in the configured database if it is absent."""

    from meminfra.database.models import Base

    Base.metadata.create_all(get_engine())


def reset_engine() -> None:
    """Dispose the cached engine so a later configuration can use another URL."""

    global _engine, _engine_key
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _engine_key = None
