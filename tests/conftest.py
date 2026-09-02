"""Test-specific environment initialization."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


# Load the project .env before test modules are collected, while preserving
# process environment values supplied by CI or the shell.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
