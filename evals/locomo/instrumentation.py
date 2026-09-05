"""Lightweight eval-only timing (Part K) and DB/network counters (Part L).

No telemetry framework: this is a tiny stopwatch dataclass plus a counters
dataclass, both printed as a plain summary table at the end of a run. Nothing
here is imported by ``src/`` or touches production retrieval code paths.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timings:
    """Accumulated wall-clock time (seconds) and call counts per named stage."""

    total_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.total_seconds[stage] += elapsed
            self.call_counts[stage] += 1

    def average_seconds(self, stage: str) -> float:
        count = self.call_counts.get(stage, 0)
        if count == 0:
            return 0.0
        return self.total_seconds.get(stage, 0.0) / count

    def render(self) -> str:
        stages = sorted(self.total_seconds)
        if not stages:
            return "Timings: (nothing measured)"
        width = max(len(stage) for stage in stages)
        lines = ["Timings (stage, calls, total_s, avg_s):"]
        for stage in stages:
            total = self.total_seconds[stage]
            calls = self.call_counts[stage]
            avg = total / calls if calls else 0.0
            lines.append(f"  {stage:<{width}}  calls={calls:<6} total={total:8.3f}s  avg={avg:8.4f}s")
        return "\n".join(lines)


@dataclass
class DbNetworkCounters:
    """Simple expected-vs-actual counters for repeated corpus loads / provider calls."""

    full_memory_corpus_loads: int = 0
    bm25_corpus_builds: int = 0
    query_embedding_provider_calls: int = 0
    vector_searches: int = 0
    fts_queries: int = 0
    diagnostic_reruns: int = 0

    def render(self, *, question_count: int) -> str:
        lines = [
            f"DB/network counters (question_count={question_count} from actual dataset size):",
            f"  full_memory_corpus_loads     = {self.full_memory_corpus_loads}  (expected: 1)",
            f"  bm25_corpus_builds           = {self.bm25_corpus_builds}  (expected: 1)",
            f"  query_embedding_provider_calls = {self.query_embedding_provider_calls}  "
            f"(expected: <= question_count, cold cache; 0, warm cache)",
            f"  vector_searches               = {self.vector_searches}  (expected: question_count)",
            f"  fts_queries                   = {self.fts_queries}  (expected: question_count)",
            f"  diagnostic_reruns             = {self.diagnostic_reruns}  (expected: 0)",
        ]
        return "\n".join(lines)
