# Memory Layer

Memory Layer is a small, reusable foundation for applications that need durable, user-scoped long-term memory. It implements the Milestone 1 foundation (configuration, lazy OpenAI-compatible provider clients, a Neon/PostgreSQL connection layer, and the initial SQLAlchemy schema), Milestone 2 deterministic candidate-memory identity, Milestone 3 bounded LLM extraction into validated candidates, Milestone 4 validated PostgreSQL writes with temporal supersession, Milestone 5 bounded conversation context with rolling summaries, Milestone 6 hybrid memory retrieval, Milestone 6.1 semantic old-message context retrieval, and Milestone 7 the public `MemoryLayer` facade.

`extract_memories()` accepts one current user/assistant interaction and returns validated `CandidateMemory` proposals. It can additionally receive a `ConversationContext` containing a rolling summary, recent pre-target messages, and a bounded deduplicated union of lexical and semantic older raw-message matches. That context may resolve references, but the target interaction remains the only source of evidence for a new memory. It does not write to PostgreSQL, generate fact keys or canonical subject IDs, decide mutations, generate embeddings, or retrieve long-term memories. Source message IDs remain application-owned provenance and are deterministically attached after model output is validated.

`write_memories()` validates the existing user, optional conversation, and message provenance before deterministically writing a batch. Known structured facts use exact scoped identity to ADD, NOOP, or SUPERSEDE while preserving historical rows. Open semantic memories use exact normalized-text deduplication only. The write path never calls retrieval, FAISS, or an embedding provider.

`search_memories()` reads the durable `Memory` table through deterministic structured lookup, BM25/PostgreSQL full-text search, and exact per-user FAISS cosine search. The results are combined with discounted rank agreement fusion (equal-weight Reciprocal Rank Fusion remains available via `fusion_strategy="rrf"`), and every public search is scoped to one existing user.

## Quick Start

Most applications do not need to call extraction, context, write, and summary functions individually. `MemoryLayer` is a small facade over that existing pipeline:

```python
from src import MemoryLayer
from src.database import SessionLocal

db = SessionLocal()
memory = MemoryLayer(db)

result = memory.add(
    user_id="user_123",
    conversation_id="conv_1",
    messages=[{"role": "user", "content": "I prefer PostgreSQL."}],
)

hits = memory.search(
    user_id="user_123",
    query="Which database does the user prefer?",
    limit=5,
)

answer = memory.answer(
    user_id="user_123",
    query="Which database does the user prefer?",
    limit=5,
)
print(answer.answer)
```

- `add()` = ingest chat messages, extract candidate memories, write them, and update the rolling summary when enough new history has accumulated. It creates the `User`/`Conversation` scope automatically if it does not already exist, and returns an `AddResult` (`message_ids`, `write_results`, `summary_updated`, `warnings`, `extracted_candidate_count`).
- `search()` = raw memory retrieval; a thin wrapper over `search_memories()` with unchanged ranking.
- `answer()` = a convenience LLM reader over `search()` results. It abstains (`AnswerResult.abstained = True`) without an LLM call when nothing is retrieved, and its reader prompt is grounded strictly in the retrieved memory text.

`MemoryLayer` does not replace the lower-level functions below; it calls them. Use `build_extraction_context()`, `extract_memories()`, `write_memories()`, `update_conversation_summary()`, and `search_memories()` directly when an application needs finer control over one step.

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

This adds `conversation_summaries` and safely normalizes `memories.importance` to `FLOAT`; it does not drop or recreate data. Do not stamp a database whose schema has not been verified as the Milestones 1-4 baseline. The Milestone 5 downgrade is intentionally unsupported because converting fractional importance values back to integers would lose data. The Milestone 6 migration adds nullable `memories.embedding` and `memories.embedding_model` columns plus a PostgreSQL `simple`-configuration GIN full-text index; it does not generate embeddings or make provider requests. The Milestone 6.1 migration adds nullable `messages.embedding` and `messages.embedding_model` columns for durable semantic context rebuilding, also without provider requests.

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

## Extraction context

`build_extraction_context()` keeps the extractor input bounded and target-authoritative:

```text
rolling summary + recent messages + lexical old + semantic old
    -> deduplicated relevant older context
    -> target extraction
```

Semantic raw-message lookup is per conversation, not per user and not a substitute for `search_memories()`. PostgreSQL persists Message embeddings and remains authoritative; local per-conversation FAISS files are rebuildable derived state. If this optional semantic branch fails, the returned context exposes a sanitized `semantic_retrieval_error` while retaining safe summary, lexical, and recent context.

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
