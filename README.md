# Memory Layer

A Python library for durable, user-scoped long-term memory in LLM applications.

Applications send conversations to Memory Layer. It extracts durable memories, updates existing facts deterministically, stores them in PostgreSQL, and retrieves relevant memory with hybrid search. It is memory infrastructure—not a chatbot, agent framework, hosted API, or full context builder.

## Install

The intended PyPI installation for the 0.1.0 release is:

```bash
pip install meminfra
```

Until the package is published, install from source for development:

```bash
git clone https://github.com/banalasaisathwik/memory-layer.git
cd memory-layer
pip install -e .
```

## Quickstart

Set the required database and provider configuration, then apply the packaged migrations:

```bash
meminfra migrate
```

```python
from meminfra import MemoryLayer
from meminfra.database import SessionLocal

db = SessionLocal()
memory = MemoryLayer(db)

memory.add(
    user_id="user-123",
    conversation_id="conversation-1",
    messages=[
        {
            "role": "user",
            "content": "I prefer PostgreSQL for backend projects.",
        }
    ],
)

hits = memory.search(
    user_id="user-123",
    query="What database do I prefer?",
)

for hit in hits:
    print(hit.memory_text)
```

`add()` persists the messages, extracts candidates, writes durable memory, and may update the conversation summary. It creates the user and conversation scope when needed. `search()` returns ranked `SearchHit` records for one user.

To generate an answer grounded in retrieved memory:

```python
result = memory.answer(
    user_id="user-123",
    query="What database do I prefer?",
)

print(result.answer)
```

`answer()` retrieves memory before asking the configured LLM to answer from it. If search finds no memory, it abstains without making an answer-generation call.

## What happens when you add memory?

Memory Layer stores semantic and episodic memories. Structured, single-value semantic facts are updated deterministically; open semantic and episodic memories use conservative duplicate handling. Persisted memories retain source-message provenance.

| Existing state | New statement | Action |
| --- | --- | --- |
| No active database preference | “I use PostgreSQL” | ADD |
| Same active preference | “I use PostgreSQL” | NOOP |
| Active database preference | “I switched to SQLite” | SUPERSEDE |

`SUPERSEDE` applies when a supported single-value structured fact changes. The prior memory stays stored for history and provenance, but default retrieval returns active memories only.

Durable memory is user-scoped. Raw-message context is conversation-scoped, while durable memory can be retrieved across that user’s conversations.

## Retrieval

Natural-language retrieval combines BM25 lexical search with FAISS vector search, then ranks candidates with discounted agreement fusion:

```text
BM25 + FAISS vector search → discounted agreement fusion → ranked memories
```

The default fusion strategy rewards agreement while allowing a strong result from one branch to remain competitive. Equal reciprocal-rank fusion (RRF) remains available for compatibility and ablation, but is not the default. When you supply structured filters, exact structured lookup is used as an additional retrieval branch; it is not inferred from every query.

PostgreSQL is the durable source of truth for memories and embeddings. Local FAISS indexes are derived retrieval state and rebuild from PostgreSQL when missing, stale, or corrupt.

## Benchmark

Frozen internal LoCoMo results for `conv-30`:

| Metric | Baseline | Current |
| --- | ---: | ---: |
| Hit@5 | 0.457 | 0.686 |
| Recall@5 | 0.426 | 0.657 |
| MRR | 0.376 | 0.595 |

LoCoMo · `conv-30` · 105 QA · same evaluation protocol.

This is an internal baseline comparison, not a cross-system comparison against another memory product. Hit@5 measures whether relevant memory appears in the top five; Recall@5 measures how much gold evidence appears there; MRR measures how early the first relevant result appears. See [`evals/`](evals/) for the harnesses and adapters.

## Configuration

Configure connections and providers with environment variables. Do not commit credentials.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL runtime connection |
| `DIRECT_URL` | Direct PostgreSQL URL for migrations; takes precedence for `meminfra migrate` |
| `LLM_PROVIDER` | LLM provider (`openai`, `openrouter`, or `openai_compatible`) |
| `LLM_API_KEY` | LLM provider credential |
| `LLM_BASE_URL` | Optional custom/OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | LLM model used for extraction, summaries, and answers |
| `EMBEDDING_PROVIDER` | Embedding provider |
| `EMBEDDING_API_KEY` | Embedding provider credential |
| `EMBEDDING_BASE_URL` | Optional custom/OpenAI-compatible embedding endpoint |
| `EMBEDDING_MODEL` | Embedding model |
| `FAISS_INDEX_DIR` | Local directory for derived FAISS indexes |

Minimal example:

```env
DATABASE_URL=postgresql+psycopg://user:password@host/database
LLM_PROVIDER=openai
LLM_API_KEY=your-llm-key
LLM_MODEL=your-llm-model
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=text-embedding-3-small
```

`LLM_BASE_URL` and `EMBEDDING_BASE_URL` are optional and are useful for custom or OpenAI-compatible endpoints.

## Database setup

Run migrations after configuring `DATABASE_URL` (or `DIRECT_URL` when the migration connection should differ):

```bash
meminfra migrate
```

This applies the packaged Alembic migrations to the configured PostgreSQL database. For repository development, `python -m alembic upgrade head` also works.

## Public API

`MemoryLayer` is the high-level facade:

```python
MemoryLayer.add(user_id=..., conversation_id=..., messages=...)
MemoryLayer.search(user_id=..., query=..., limit=10, filters=None)
MemoryLayer.answer(user_id=..., query=..., limit=5, filters=None)
```

- `add` persists messages and extracts or updates durable memory.
- `search` retrieves ranked memory for one user.
- `answer` retrieves memory and produces a grounded answer.

The application remains responsible for deciding how retrieved memory is inserted into its final prompt or context.

## Architecture

```mermaid
flowchart TD
    A[Conversation] --> B[Bounded extraction context]
    B --> C[LLM extraction]
    C --> D[Deterministic writer<br/>ADD / NOOP / SUPERSEDE]
    D --> E[PostgreSQL]
    E --> F[BM25 + FAISS]
    F --> G[Discounted agreement fusion]
    G --> H[Search or grounded answer]
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python -m compileall -q src tests
```

Database integration tests run only when `TEST_DATABASE_URL` is configured; they never fall back to `DATABASE_URL`. The core memory, retrieval, CLI, and package paths are covered by automated tests.

## License

[MIT](LICENSE)
