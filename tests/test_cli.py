"""Focused checks for the installed-package migration command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from meminfra import cli
from meminfra.config import configure, get_migration_database_url, reset_config


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    reset_config()
    yield
    reset_config()


def test_help_is_available(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["meminfra", "--help"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 0
    assert "migrate" in capsys.readouterr().out


def test_unknown_command_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["meminfra", "unknown"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code != 0


def test_migrate_requires_database_url_without_leaking_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure(direct_url=None, database_url=None)
    monkeypatch.setattr(sys, "argv", ["meminfra", "migrate"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    output = capsys.readouterr().err
    assert error.value.code != 0
    assert "DIRECT_URL or DATABASE_URL must be configured" in output


def test_migrate_uses_packaged_script_location_and_direct_url_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    configure(
        direct_url="postgresql://direct-user:direct-password@localhost/direct-db",
        database_url="postgresql://database-user:database-password@localhost/database-db",
    )
    monkeypatch.setattr(sys, "argv", ["meminfra", "migrate"])

    def fake_upgrade(config: object, revision: str) -> None:
        seen["location"] = config.get_main_option("script_location")  # type: ignore[attr-defined]
        seen["revision"] = revision

    monkeypatch.setattr(cli.command, "upgrade", fake_upgrade)
    cli.main()

    assert get_migration_database_url().endswith("/direct-db")
    assert seen["revision"] == "head"
    assert Path(str(seen["location"])).name == "migrations"
    assert "meminfra" in Path(str(seen["location"])).parts


def test_migrate_falls_back_to_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    configure(direct_url=None, database_url="postgresql://database-user:database-password@localhost/database-db")
    monkeypatch.setattr(sys, "argv", ["meminfra", "migrate"])
    monkeypatch.setattr(cli.command, "upgrade", lambda config, revision: None)

    cli.main()
    assert get_migration_database_url().endswith("/database-db")
