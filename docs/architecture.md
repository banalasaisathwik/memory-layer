# Architecture

## Purpose

Memory Layer is a reusable memory infrastructure package for applications using LLMs. It is not an agent framework, a chatbot, a Context Builder, or a document RAG platform.

Milestone 1 establishes configuration, provider client construction, PostgreSQL connectivity, and durable relational models. No memory is extracted or retrieved in this milestone.

## Current boundary

An application explicitly calls the memory layer. The package does not own application conversations or decide when an application should write or read memory.

```text
Application
    |
Memory Layer
    |- Write Pipeline (planned)
    `- Retrieval Pipeline (planned)
```

`user_id` is the long-term memory scope. `conversation_id` is source and session context: a user can have many conversations, and a memory may optionally refer to the conversation that produced it. A conversation must not become the long-term memory boundary.

## Current foundation

- `src.config` holds environment and programmatic settings.
- `src.providers` creates OpenAI-compatible LLM and embedding clients lazily and independently.
- `src.database` creates a lazy SQLAlchemy/psycopg connection to PostgreSQL or Neon and defines the current durable models.
- A memory can be structured (subject, predicate, value, and optional fact key) or open/unstructured (`memory_text` alone). JSON provenance stores the source message IDs.

`DATABASE_URL` is the only URL used for normal application sessions. `DIRECT_URL` is configuration-only in Milestone 1 and is reserved for future migration or administration work. Database integration tests explicitly configure `TEST_DATABASE_URL`; they never fall back to `DATABASE_URL`.

## Planned write architecture

The following is a design target, not an implemented feature:

```text
Conversation
    |
LLM extraction
    |
candidate memory
    |
deterministic validation
    |
existing-memory resolution
    |
ADD / UPDATE / SUPERSEDE / NOOP
    |
PostgreSQL
```

LLMs will make semantic judgments, while deterministic code will own validation and execution. Temporal fields already model memory history but no supersession behavior exists yet.

## Planned retrieval architecture

The following is also planned only:

```text
query
 |- structured PostgreSQL retrieval
 |- PostgreSQL lexical retrieval
 `- FAISS semantic retrieval
             |
           fusion
             |
        final results
```

## Design principles

- LLMs perform semantic judgment; deterministic code validates and executes decisions.
- Structured memory is useful where deterministic identity matters, while unstructured memory remains valid for open-world information.
- PostgreSQL stores durable memory state; vector retrieval will supplement, not replace, it.
- The architecture stays small until added complexity solves a real problem.
