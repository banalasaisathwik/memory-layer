"""Small installed-package command line interface."""

from __future__ import annotations

import argparse
from importlib import resources

from alembic import command
from alembic.config import Config

from meminfra.config import get_migration_database_url


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meminfra", description="meminfra package commands")
    parser.add_argument("command", choices=("migrate",), help="command to run")
    return parser


def _migration_config() -> Config:
    """Configure Alembic from the installed migration package, not cwd."""

    migration_path = resources.files("meminfra").joinpath("migrations")
    config = Config()
    config.set_main_option("script_location", str(migration_path))
    return config


def _migrate() -> None:
    # Validate before Alembic starts so a missing URL is concise and secret-free.
    get_migration_database_url()
    command.upgrade(_migration_config(), "head")


def main() -> None:
    """Run the supported meminfra command."""

    parser = _parser()
    parser.parse_args()
    try:
        _migrate()
    except RuntimeError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
