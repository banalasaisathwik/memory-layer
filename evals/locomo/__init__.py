"""LoCoMo long-term conversational-memory benchmark integration.

This package is a dedicated adapter for the official SNAP Research LoCoMo
release (``snap-research/locomo``, ``data/locomo10.json``). It is
intentionally separate from ``evals/datasets`` (the hand-written smoke
cases): one LoCoMo sample is one long multi-session conversation ingested
once through :class:`src.memory_layer.MemoryLayer`, with every one of its QA
questions evaluated against that same resulting memory state, rather than
being forced into the smoke harness's "one isolated conversation per case"
shape. See ``evals/README.md`` for the full methodology.
"""

from __future__ import annotations

__all__: list[str] = []
