"""Registry of named eval datasets."""

from __future__ import annotations

from evals.schemas import EvalCase

from .smoke import SMOKE_CASES


DATASETS: dict[str, list[EvalCase]] = {
    "smoke": SMOKE_CASES,
}

__all__ = ["DATASETS", "SMOKE_CASES"]
