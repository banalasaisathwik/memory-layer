"""Unit tests for evals.db: eval workloads must resolve EVAL_DATABASE_URL and
must never silently fall back to TEST_DATABASE_URL or DATABASE_URL."""

from __future__ import annotations

import pytest

import evals.db as eval_db_module
from evals.db import (
    EvalDatabaseConfigError,
    describe_eval_database,
    eval_database_classification,
    get_eval_database_url,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never let the repository's real .env (which sets EVAL_DATABASE_URL for
    # local development) leak into these fail-closed/precedence assertions.
    monkeypatch.setattr(eval_db_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("EVAL_REQUIRE_LOCAL_DB", raising=False)


def test_eval_runner_uses_eval_database_url_not_test_or_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5433/memory_layer_eval")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://user:pass@neon-test-host/db")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@neon-dev-host/db")

    resolved = get_eval_database_url()

    assert resolved == "postgresql+psycopg://user:pass@localhost:5433/memory_layer_eval"


def test_missing_eval_database_url_fails_closed_even_with_test_database_url_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://user:pass@neon-test-host/db")

    with pytest.raises(EvalDatabaseConfigError, match="EVAL_DATABASE_URL is required"):
        get_eval_database_url()


def test_missing_eval_database_url_fails_closed_even_with_database_url_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@neon-dev-host/db")

    with pytest.raises(EvalDatabaseConfigError, match="EVAL_DATABASE_URL is required"):
        get_eval_database_url()


def test_require_local_db_allows_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5433/memory_layer_eval")
    monkeypatch.setenv("EVAL_REQUIRE_LOCAL_DB", "true")

    assert get_eval_database_url() == "postgresql+psycopg://user:pass@localhost:5433/memory_layer_eval"


def test_require_local_db_rejects_remote_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql+psycopg://user:pass@ep-shiny.neon.tech/db")
    monkeypatch.setenv("EVAL_REQUIRE_LOCAL_DB", "true")

    with pytest.raises(EvalDatabaseConfigError, match="not local"):
        get_eval_database_url()


def test_describe_eval_database_never_includes_credentials() -> None:
    summary = describe_eval_database("postgresql+psycopg://memory_eval:memory_eval_pass@localhost:5433/memory_layer_eval")

    assert summary == "localhost:5433/memory_layer_eval"
    assert "memory_eval_pass" not in summary
    assert "@" not in summary


def test_eval_database_classification_local_vs_remote() -> None:
    assert eval_database_classification("postgresql+psycopg://u:p@localhost:5433/db") == "local PostgreSQL"
    assert eval_database_classification("postgresql+psycopg://u:p@127.0.0.1:5433/db") == "local PostgreSQL"
    assert eval_database_classification("postgresql+psycopg://u:p@ep-shiny.neon.tech/db").startswith("REMOTE")
