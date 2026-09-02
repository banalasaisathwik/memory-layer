"""Configuration and lazy provider-client unit tests."""

from __future__ import annotations

import pytest

from src.config import configure, get_config, get_migration_database_url, reset_config
from src.database.connection import get_engine, reset_engine
from src.providers import reset_embedding_client, reset_llm_client


@pytest.fixture(autouse=True)
def reset_global_configuration() -> None:
    reset_config()
    reset_engine()
    reset_llm_client()
    reset_embedding_client()
    yield
    reset_config()
    reset_engine()
    reset_llm_client()
    reset_embedding_client()


def test_configuration_loads_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test-user:test-pass@localhost/test-db")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/test-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("DEBUG", "true")

    settings = get_config()

    assert settings.database_url == "postgresql+psycopg://test-user:test-pass@localhost/test-db"
    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "anthropic/test-model"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.debug is True


def test_programmatic_configuration_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    settings = configure(
        database_url="postgresql+psycopg://test-user:test-pass@localhost/test-db",
        llm_provider="openrouter",
        llm_api_key="configured-key",
        llm_model="anthropic/configured-model",
    )

    assert settings.llm_provider == "openrouter"
    assert settings.llm_api_key == "configured-key"
    assert get_config().llm_model == "anthropic/configured-model"


def test_reset_config_reloads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "first-model")
    assert get_config().llm_model == "first-model"

    monkeypatch.setenv("LLM_MODEL", "second-model")
    reset_config()

    assert get_config().llm_model == "second-model"


def test_standard_postgresql_url_uses_psycopg_3_driver() -> None:
    configure(database_url="postgresql://test-user:test-pass@localhost/test-db")

    assert get_engine().url.drivername == "postgresql+psycopg"


def test_direct_url_does_not_replace_application_database_url() -> None:
    configure(
        database_url="postgresql://application-user:test-pass@localhost/application-db",
        direct_url="postgresql://migration-user:test-pass@localhost/direct-db",
    )

    assert get_engine().url.database == "application-db"


def test_migrations_prefer_direct_url_without_printing_it(capsys: pytest.CaptureFixture[str]) -> None:
    configure(
        database_url="postgresql://application-user:test-pass@localhost/application-db",
        direct_url="postgresql://migration-user:test-pass@localhost/direct-db",
    )

    assert get_migration_database_url().endswith("/direct-db")
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


def test_migrations_fall_back_to_database_url() -> None:
    configure(
        database_url="postgresql://application-user:test-pass@localhost/application-db",
        direct_url=None,
    )

    assert get_migration_database_url().endswith("/application-db")


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("summary_trigger_messages", 0),
        ("summary_recent_keep", 0),
        ("extraction_recent_messages", 0),
        ("extraction_lexical_messages", 0),
    ],
)
def test_context_settings_must_be_positive(setting: str, value: int) -> None:
    with pytest.raises(ValueError):
        configure(**{setting: value})


def test_metadata_includes_summary_table_and_float_importance() -> None:
    from sqlalchemy import Float

    from src.database.models import Base

    summary = Base.metadata.tables["conversation_summaries"]
    assert summary.c.conversation_id.unique is True
    assert summary.c.covered_through_message_id.nullable is True
    assert isinstance(Base.metadata.tables["memories"].c.importance.type, Float)


def test_engine_is_recreated_when_pool_configuration_changes() -> None:
    configure(
        database_url="postgresql://test-user:test-pass@localhost/test-db",
        database_pool_size=2,
        database_max_overflow=1,
    )
    first_engine = get_engine()

    configure(database_pool_size=3)
    second_engine = get_engine()

    assert second_engine is not first_engine
    assert second_engine.pool.size() == 3
    assert second_engine.pool._max_overflow == 1


def test_provider_clients_are_created_without_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    class RecordingClient:
        def __init__(self, **kwargs: str) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("src.providers.llm.OpenAI", RecordingClient)
    monkeypatch.setattr("src.providers.embeddings.OpenAI", RecordingClient)
    configure(
        llm_provider="openrouter",
        llm_api_key="llm-test-key",
        llm_model="anthropic/test-model",
        embedding_provider="openai_compatible",
        embedding_api_key="embedding-test-key",
        embedding_base_url="https://embeddings.example.test/v1",
        embedding_model="test-embedding-model",
    )

    from src.providers.embeddings import get_embedding_client
    from src.providers.llm import OPENROUTER_BASE_URL, get_llm_client

    get_llm_client()
    get_embedding_client()

    assert calls == [
        {"api_key": "llm-test-key", "base_url": OPENROUTER_BASE_URL},
        {"api_key": "embedding-test-key", "base_url": "https://embeddings.example.test/v1"},
    ]


def test_openai_uses_its_default_endpoint_without_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    class RecordingClient:
        def __init__(self, **kwargs: str) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("src.providers.llm.OpenAI", RecordingClient)
    configure(llm_provider="openai", llm_api_key="llm-test-key")

    from src.providers.llm import get_llm_client

    get_llm_client()

    assert calls == [{"api_key": "llm-test-key"}]


def test_custom_llm_base_url_is_used_without_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []

    class RecordingClient:
        def __init__(self, **kwargs: str) -> None:
            calls.append(kwargs)

    monkeypatch.setattr("src.providers.llm.OpenAI", RecordingClient)
    configure(
        llm_provider="openai_compatible",
        llm_api_key="llm-test-key",
        llm_base_url="https://llm.example.test/v1",
    )

    from src.providers.llm import get_llm_client

    get_llm_client()

    assert calls == [{"api_key": "llm-test-key", "base_url": "https://llm.example.test/v1"}]
