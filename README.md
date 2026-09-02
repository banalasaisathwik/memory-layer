# Memory Layer

Memory Layer is a small, reusable foundation for applications that need durable, user-scoped long-term memory. It implements the Milestone 1 foundation (configuration, lazy OpenAI-compatible provider clients, a Neon/PostgreSQL connection layer, and the initial SQLAlchemy schema), Milestone 2 deterministic candidate-memory identity, and Milestone 3 bounded LLM extraction into validated candidates.

`extract_memories()` accepts one current user/assistant interaction and returns validated `CandidateMemory` proposals. It does not write to PostgreSQL, generate fact keys or canonical subject IDs, decide mutations, generate embeddings, or retrieve memories. Source message IDs remain application-owned provenance and are deterministically attached after model output is validated.

Write lifecycle decisions, retrieval, embeddings generation, FAISS, and supersession behavior are deliberately not implemented yet.

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

Set `DATABASE_URL` in `.env` to your Neon connection string. Standard `postgresql://...` Neon URLs are accepted and are routed through psycopg 3 automatically. `DATABASE_POOL_SIZE` and `DATABASE_MAX_OVERFLOW` default to 5 to keep the normal Neon connection pool small. `DIRECT_URL` is retained only as configuration for future migration or administration work; Milestone 1 application sessions never use it. Provider clients are configured independently through the `LLM_*` and `EMBEDDING_*` variables; creating a client never sends a provider request.

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
