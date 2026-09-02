# Memory Layer

Memory Layer is a small, reusable foundation for applications that need durable, user-scoped long-term memory. It implements the Milestone 1 foundation (configuration, lazy OpenAI-compatible provider clients, a Neon/PostgreSQL connection layer, and the initial SQLAlchemy schema), Milestone 2 deterministic candidate-memory identity, Milestone 3 bounded LLM extraction into validated candidates, Milestone 4 validated PostgreSQL writes with temporal supersession, Milestone 5 bounded conversation context with rolling summaries, and Milestone 6 hybrid memory retrieval.

`extract_memories()` accepts one current user/assistant interaction and returns validated `CandidateMemory` proposals. It can additionally receive a `ConversationContext` containing a rolling summary, recent pre-target messages, and a small optional PostgreSQL lexical-match window from older raw messages. That context may resolve references, but the target interaction remains the only source of evidence for a new memory. It does not write to PostgreSQL, generate fact keys or canonical subject IDs, decide mutations, generate embeddings, or retrieve memories. Source message IDs remain application-owned provenance and are deterministically attached after model output is validated.

`write_memories()` validates the existing user, optional conversation, and message provenance before deterministically writing a batch. Known structured facts use exact scoped identity to ADD, NOOP, or SUPERSEDE while preserving historical rows. Open semantic memories use exact normalized-text deduplication only. The write path never calls retrieval, FAISS, or an embedding provider.

`search_memories()` reads the durable `Memory` table through deterministic structured lookup, PostgreSQL full-text search, and exact per-user FAISS cosine search. The results are combined with Reciprocal Rank Fusion, and every public search is scoped to one existing user.

## Prerequisites

- Python 3.11 or newer
- A Neon PostgreSQL database (or another PostgreSQL database) for local development

## Setup

Create a virtual environment, install the project with test dependencies, then create `.env` from the example.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `DATABASE_URL` in `.env` to your Neon connection string. Standard `postgresql://...` Neon URLs are accepted and are routed through psycopg 3 automatically. `DATABASE_POOL_SIZE` and `DATABASE_MAX_OVERFLOW` default to 5 to keep the normal Neon connection pool small. Application sessions always use `DATABASE_URL`; Alembic migrations prefer `DIRECT_URL` and otherwise fall back to `DATABASE_URL`. Provider clients are configured independently through the `LLM_*` and `EMBEDDING_*` variables; creating a client never sends a provider request.

## Migrations

Use Alembic for PostgreSQL schema changes. A new empty database can be initialized with:

```powershell
python -m alembic upgrade head
```

For an existing Milestones 1-4 development database that was created with `create_tables()`, take a backup, verify it has the current pre-Alembic tables, then establish the known baseline before applying the Milestone 5 change:

```powershell
python -m alembic stamp 0001_initial_schema
python -m alembic upgrade head
```

This adds `conversation_summaries` and safely normalizes `memories.importance` to `FLOAT`; it does not drop or recreate data. Do not stamp a database whose schema has not been verified as the Milestones 1-4 baseline. The Milestone 5 downgrade is intentionally unsupported because converting fractional importance values back to integers would lose data. The Milestone 6 migration adds nullable `memories.embedding` and `memories.embedding_model` columns plus a PostgreSQL `simple`-configuration GIN full-text index; it does not generate embeddings or make provider requests.

## Search

```python
from src.retrieval import search_memories

hits = search_memories(
    db,
    "Which database does the user prefer?",
    user_external_id="user_123",
    limit=5,
)
```

Search defaults to active memories. Pass `SearchFilters(include_history=True)` to include superseded rows, or a `conversation_external_id` filter to restrict results to memories that explicitly originated in that user-owned conversation. Memories with `conversation_id=NULL` are user-level and do not match a conversation filter.

FAISS files are local derived state in `FAISS_INDEX_DIR` (default `.memory-layer/faiss`). PostgreSQL stores the durable text, structure, temporal state, embeddings, and embedding model, so missing, stale, or corrupt local files are rebuilt. Automatic synchronization embeds only rows that do not have the currently configured embedding model; it never changes the write pipeline.

## Tests

```bash
python -m pytest
```

Database integration tests run only when `TEST_DATABASE_URL` is set. They never fall back to `DATABASE_URL`, so use a disposable or dedicated test database rather than your production Neon database.

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://user:password@host/database"
python -m pytest
```

See [the architecture notes](docs/architecture.md) for the current boundary and the explicitly planned next stages.

## Manual provider smoke test

No provider network request runs as part of the test suite. Once the independently configured `LLM_*` and `EMBEDDING_*` values are present, intentionally run the following command to make one small completion request and one embedding request:

```powershell
python scripts/provider_smoke.py
```

The script prints only model metadata, generated content, and embedding dimensions. It never prints configuration values or API keys.
