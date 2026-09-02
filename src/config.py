"""Central configuration for the current foundation milestone."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


_PROVIDER_ALIASES = {
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
}
_SUPPORTED_PROVIDERS = {"openai", "openrouter", "openai_compatible"}


class Settings(BaseModel):
    """Configuration shared by database and provider modules."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    database_url: str | None = None
    # DIRECT_URL is intentionally configuration-only for this milestone. Normal
    # application sessions always use database_url.
    direct_url: str | None = None
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=5, ge=0)

    llm_provider: str = "openai"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    embedding_provider: str = "openai"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = Field(default="text-embedding-3-small")

    debug: bool = False

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def validate_provider(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Provider names must be strings.")

        provider = _PROVIDER_ALIASES.get(value.strip().lower(), value.strip().lower())
        if provider not in _SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
            raise ValueError(f"Unsupported provider '{provider}'. Use one of: {supported}.")
        return provider


_settings: Settings | None = None


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _debug_from_environment() -> bool:
    """Interpret common truthy DEBUG values without leaking host labels into settings."""

    value = os.getenv("DEBUG", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _settings_from_environment() -> Settings:
    """Read .env without replacing environment values supplied by the caller."""

    load_dotenv(override=False)
    return Settings(
        database_url=_optional_env("DATABASE_URL"),
        direct_url=_optional_env("DIRECT_URL"),
        database_pool_size=_optional_env("DATABASE_POOL_SIZE") or 5,
        database_max_overflow=_optional_env("DATABASE_MAX_OVERFLOW") or 5,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_api_key=_optional_env("LLM_API_KEY"),
        llm_base_url=_optional_env("LLM_BASE_URL"),
        llm_model=_optional_env("LLM_MODEL"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_api_key=_optional_env("EMBEDDING_API_KEY"),
        embedding_base_url=_optional_env("EMBEDDING_BASE_URL"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        debug=_debug_from_environment(),
    )


def get_config() -> Settings:
    """Return the cached settings, loading the environment on first use."""

    global _settings
    if _settings is None:
        _settings = _settings_from_environment()
    return _settings


def configure(**overrides: object) -> Settings:
    """Apply explicit settings, keeping unspecified values from the environment."""

    global _settings
    _settings = Settings.model_validate({**get_config().model_dump(), **overrides})
    return _settings


def reset_config() -> None:
    """Clear cached configuration; useful for isolated tests and process setup."""

    global _settings
    _settings = None
