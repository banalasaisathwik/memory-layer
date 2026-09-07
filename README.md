# Memory Layer / `meminfra`

Memory Layer is a Python library for durable, user-scoped long-term memory in LLM applications. The GitHub project is [`memory-layer`](https://github.com/banalasaisathwik/memory-layer); the PyPI package and Python import are `meminfra`.

It is memory infrastructure, not a chatbot, agent framework, hosted API, document RAG system, or full context builder.

## Installation

```bash
pip install meminfra
```

For development from a checkout:

```bash
pip install -e ".[dev]"
```

Configure the runtime database and providers, then apply the packaged migrations:

```bash
meminfra migrate
```

`DATABASE_URL` is the PostgreSQL runtime connection. `DIRECT_URL`, when set, is used by `meminfra migrate`. Configure LLM and embedding providers independently with the `LLM_*` and `EMBEDDING_*` environment variables; never commit credentials.

## Quickstart

```python
from meminfra import MemoryLayer
from meminfra.database import SessionLocal

with SessionLocal() as db:
    memory = MemoryLayer(db)
    memory.add(
        user_id="user-123",
        conversation_id="conversation-1",
        messages=[
            {"role": "user", "content": "I prefer PostgreSQL for backend projects."},
        ],
    )

    hits = memory.search(
        user_id="user-123",
        query="What database do I prefer?",
    )

for hit in hits:
    print(hit.memory_text)
```

`add()` creates the user and conversation scope when needed. `search()` returns ranked `SearchHit` records for that user. `answer()` is available when an application wants a configured LLM to answer only from retrieved memory.

## What happens when you add memory

The write path is:

```text
messages -> memory extractor -> controlled predicate canonicalization where applicable
-> deterministic validation -> ADD / NOOP / SUPERSEDE -> PostgreSQL durable state
-> FAISS derived vector state
```

Controlled predicates give supported single-value facts deterministic identity and lifecycle behavior. For example, an active location can be added, repeated as a `NOOP`, or superseded by a newer durable location. Unknown or open-world facts remain valid semantic memory; they are not forced into the controlled ontology, and not every memory has a predicate.

`SUPERSEDE` keeps the earlier memory for history and provenance while default retrieval returns active memories. Raw-message context is conversation-scoped; durable memory is user-scoped and can be retrieved across that user's conversations.

## What happens when you search

The read path is:

```text
natural-language query -> QueryIntent
                         -> structured retrieval
                       + BM25 lexical retrieval
                       + FAISS vector retrieval
                         -> discounted-agreement fusion -> ranked SearchHit[]
```

The three retrieval branches preserve the original natural-language query for BM25 and FAISS. The production fusion is discounted agreement with `lambda = 0.10`: a strong rank in one branch remains competitive, while agreement across branches receives a discounted bonus. Equal reciprocal-rank fusion is retained for compatibility and ablation, not as the default.

## Memory lifecycle

| Existing state | New statement | Result |
| --- | --- | --- |
| No active database preference | "I use PostgreSQL" | `ADD` |
| Same active preference | "I use PostgreSQL" | `NOOP` |
| Active database preference | "I switched to SQLite" | `SUPERSEDE` |

Lifecycle rules apply only where deterministic structured identity is supported. Open semantic and episodic memories retain conservative duplicate handling and provenance.

## Structured, lexical, and vector retrieval

Structured retrieval can provide exact, lifecycle-aware evidence. BM25 contributes lexical matches, and FAISS contributes semantic vector matches. Their ranks, rather than incomparable raw scores, are fused into each `SearchHit`.

PostgreSQL is the durable source of truth for memories, embeddings, temporal state, and provenance. Per-user FAISS indexes are rebuildable derived state; PostgreSQL resolves and state-filters vector candidates before they become results.

## QueryIntent

`QueryIntent` is an optional best-effort LLM-derived structured hint with three fields:

- `predicate`
- `value`
- `temporal_scope` (`current`, `historical`, or `any`)

For example:

```python
memory.search(
    user_id="user-123",
    query="Where do I live?",
)
```

can infer `predicate="location"` and `temporal_scope="current"`, allowing the structured branch to contribute relevant evidence without the application manually supplying `SearchFilters(predicate="location")`. Inference is not guaranteed, and unknown predicates simply add no inferred structure.

If optional intent inference fails, search continues with the existing hard structured filters plus BM25 and FAISS. Set `infer_query_intent=False` on `search()` or `answer()` for lower latency or fully deterministic retrieval.

## SearchFilters

`SearchFilters` are explicit caller-supplied hard constraints. They apply to every retrieval branch. `QueryIntent` is inferred soft structured evidence only: it never becomes a filter and never constrains BM25 or FAISS. Memory Layer does not infer semantic or episodic memory type on reads.

## Benchmarks

Internal baseline comparison under the same evaluation protocol:

| Metric | Internal baseline | meminfra 0.2.0 candidate |
| --- | ---: | ---: |
| Hit@5 | 0.457 | 0.657 |
| Recall@5 | 0.426 | 0.618 |
| MRR | 0.376 | 0.544 |

Protocol: LoCoMo · `conv-30` · 19 sessions · 369 turns · 105 QA · `k=5`.

Hit@5 means at least one relevant memory appears in the top five. Recall@5 is the fraction of relevant memories recovered in the top five. MRR measures how early the first relevant memory appears.

### QueryIntent engineering ablation

| Setting | Hit@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| QueryIntent OFF | 0.657 | 0.618 | 0.544 |
| QueryIntent ON | 0.657 | 0.618 | 0.544 |

Across 105 questions, 105/105 intent calls validated; 3/105 returned controlled predicates, 1/105 included a value, and 1/105 activated structured retrieval. LoCoMo `conv-30` has little overlap with the current controlled ontology, so this dataset does not meaningfully demonstrate the value of QueryIntent. It does not imply a metric gain or a feature failure.

### Write-path diagnostic ablation

| Setting | Hit@5 | Recall@5 | MRR | Sessions ingested | Extraction failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLD_WRITE | 0.524 | 0.499 | 0.462 | 14/19 | 5 |
| CURRENT_WRITE | 0.629 | 0.589 | 0.502 | 19/19 | 0 |

The old-write side failed to ingest five sessions, so this experiment is diagnostic evidence rather than the primary public benchmark comparison.

## Public API

`MemoryLayer` is the high-level facade:

```python
MemoryLayer.add(*, user_id: str, conversation_id: str, messages: list[dict[str, str]]) -> AddResult
MemoryLayer.search(*, user_id: str, query: str, limit: int = 10,
                   filters: SearchFilters | None = None,
                   infer_query_intent: bool = True) -> list[SearchHit]
MemoryLayer.answer(*, user_id: str, query: str, limit: int = 5,
                   filters: SearchFilters | None = None,
                   infer_query_intent: bool = True) -> AnswerResult
```

`answer()` retrieves once through `search()`, then asks the configured LLM to answer only from the retrieved memory. It abstains without an answer-generation call when no memory is found.

## Migrations and recovery

Run `meminfra migrate` after configuring `DATABASE_URL` (or `DIRECT_URL`). The command uses Alembic migrations packaged with the wheel and is safe to run repeatedly. The current schema head is `0005_structured_fact_uniqueness`.

PostgreSQL is recoverable durable state. If a local FAISS index is missing, stale, or corrupt, it is rebuilt from PostgreSQL.

## Development and testing

```bash
python -m pytest
python -m compileall -q src tests
python -m build
python -m twine check dist/*
```

Database integration tests run only when `TEST_DATABASE_URL` is configured; they never fall back to `DATABASE_URL`. Evaluation workloads use `EVAL_DATABASE_URL` and are not part of the normal test suite.

## License

[MIT](LICENSE)
