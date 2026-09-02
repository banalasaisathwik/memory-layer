# Memory Layer — Codex Instructions

## Engineering style

Keep the project simple, readable, and modular.

* Do not add abstractions unless they solve a current problem.
* Avoid unnecessary Repository, Service, Manager, Factory, Adapter, or dependency-injection patterns.
* Prefer direct functions and small modules.
* Do not create placeholder files for future milestones.
* Do not expand the requested scope without a concrete reason.
* Preserve the current architecture unless a change fixes an actual issue.

## Safety

* Never commit `.env`.
* Never print API keys, database passwords, or full database URLs containing credentials.
* Never run destructive operations against the development or production database.
* Database integration tests MUST use `TEST_DATABASE_URL`.
* Never silently fall back from `TEST_DATABASE_URL` to `DATABASE_URL`.
* Do not modify external resources unless explicitly requested.

## Testing

For every implementation or verification task:

1. Run the relevant unit tests.
2. Run database integration tests when `TEST_DATABASE_URL` is available.
3. Run `python -m compileall -q src tests`.
4. Run `git diff --check`.
5. Report exact pass/fail/skip counts.
6. Investigate failures instead of merely reporting them.
7. Do not hide skipped tests; explain why each class of test was skipped.

Prefer concise command output where possible.

## Current architecture

This repository is a reusable long-term memory infrastructure package.

It is NOT:

* a chatbot
* an agent framework
* a Context Builder
* a document RAG system

Current milestone contains only:

* configuration
* provider clients
* Neon PostgreSQL connectivity
* SQLAlchemy models
* tests
* architecture documentation

Do not implement future memory extraction or retrieval functionality unless explicitly asked.

## Database principles

* Neon PostgreSQL is the primary database.
* SQLAlchemy 2.x is the Python database layer.
* psycopg 3 is the PostgreSQL driver.
* `user_id` is the long-term memory scope.
* `conversation_id` is session/provenance context.
* Memories may exist without a conversation.
* Structured and unstructured memories must both remain supported.

## Provider principles

* LLM and embedding providers are independently configurable.
* Package users supply their own provider credentials.
* OpenAI, OpenRouter, and OpenAI-compatible endpoints should remain supported.
* Provider/client initialization should remain lazy.
* Tests must not perform real provider API calls unless explicitly designated as manual smoke tests.
