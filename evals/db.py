"""Database URL resolution for evaluation/benchmark workloads.

Evaluation workloads (LoCoMo ingestion, retrieval evaluation, lexical
ablation, recovery/diagnostic scripts) must resolve their own dedicated
database and must never silently reuse ``TEST_DATABASE_URL`` or
``DATABASE_URL``. Falling back to either would let a heavy eval run collide
with integration-test state or consume Neon network quota.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from dotenv import load_dotenv

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class EvalDatabaseConfigError(RuntimeError):
    """Raised when EVAL_DATABASE_URL is missing or fails a configured safety check."""


def get_eval_database_url() -> str:
    """Return EVAL_DATABASE_URL, failing closed if it is unset.

    Never falls back to TEST_DATABASE_URL or DATABASE_URL. If
    ``EVAL_REQUIRE_LOCAL_DB`` is truthy, also refuses a non-local host.
    """

    load_dotenv(override=False)
    value = os.getenv("EVAL_DATABASE_URL")
    if not value or not value.strip():
        raise EvalDatabaseConfigError(
            "EVAL_DATABASE_URL is required for evaluation workloads. "
            "It never falls back to TEST_DATABASE_URL or DATABASE_URL; set it to a "
            "dedicated eval PostgreSQL database (see .env.example)."
        )
    database_url = value.strip()

    if os.getenv("EVAL_REQUIRE_LOCAL_DB", "").strip().lower() in {"1", "true", "yes", "on"}:
        host = (urlsplit(database_url).hostname or "").lower()
        if host not in _LOCAL_HOSTS:
            raise EvalDatabaseConfigError(
                f"EVAL_REQUIRE_LOCAL_DB is set but EVAL_DATABASE_URL's host ({host!r}) is not "
                "local (localhost/127.0.0.1). Refusing to run against a remote database."
            )

    return database_url


def describe_eval_database(database_url: str) -> str:
    """A sanitized ``host:port/dbname`` summary safe to print -- never credentials."""

    parts = urlsplit(database_url)
    host = parts.hostname or "unknown-host"
    port = f":{parts.port}" if parts.port else ""
    db_name = parts.path.lstrip("/") or "unknown-db"
    return f"{host}{port}/{db_name}"


def eval_database_classification(database_url: str) -> str:
    """A one-line, best-effort safety classification for startup reporting."""

    host = (urlsplit(database_url).hostname or "").lower()
    if host in _LOCAL_HOSTS:
        return "local PostgreSQL"
    return "REMOTE database -- verify this is intentional"


def print_eval_database_banner(database_url: str) -> None:
    """Print a sanitized destination summary; never prints username/password."""

    print(f"Eval DB: {describe_eval_database(database_url)}")
    classification = eval_database_classification(database_url)
    print(f"Eval DB target: {classification}")
    if classification.startswith("REMOTE"):
        print("WARNING: eval DB target does not look local; this may use remote database quota.")
