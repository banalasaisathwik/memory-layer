"""Alembic environment using Memory Layer's existing SQLAlchemy metadata."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.config import get_migration_database_url
from src.database.connection import _sqlalchemy_url
from src.database.models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _migration_url() -> str:
    """Prefer DIRECT_URL and never render the resolved URL in Alembic output."""

    return _sqlalchemy_url(get_migration_database_url())


def run_migrations_offline() -> None:
    """Generate SQL without connecting, using the normal PostgreSQL dialect."""

    context.configure(
        url=_migration_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run versioned changes through DIRECT_URL when it is configured."""

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _migration_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
