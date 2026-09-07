"""Test-specific environment initialization."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv


# Load the project .env before test modules are collected, while preserving
# process environment values supplied by CI or the shell.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@pytest.fixture(autouse=True)
def disable_live_query_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default search enhancement deterministic outside its own tests."""

    from meminfra.retrieval import QueryIntent

    monkeypatch.setattr("meminfra.retrieval.search.extract_query_intent", lambda query: QueryIntent())
