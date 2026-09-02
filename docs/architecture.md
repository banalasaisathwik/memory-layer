# Architecture

## Purpose

Memory Layer is a reusable memory infrastructure package for applications using LLMs. It is not an agent framework, a chatbot, a Context Builder, or a document RAG platform.

Milestone 1 established configuration, provider client construction, PostgreSQL connectivity, and durable relational models. Milestone 2 adds deterministic candidate-memory identity; no memory is extracted, persisted through a write pipeline, or retrieved yet.

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
- `src.memory` validates structural `CandidateMemory` data and produces deterministic fact keys for a small set of known predicates.

`DATABASE_URL` is the only URL used for normal application sessions. `DIRECT_URL` is configuration-only in Milestone 1 and is reserved for future migration or administration work. Database integration tests explicitly configure `TEST_DATABASE_URL`; they never fall back to `DATABASE_URL`.

## Candidate-memory identity

Milestone 2 represents extracted meaning as `CandidateMemory` before any persistence or mutation decision. A candidate contains no canonical `subject_id` and rejects caller-provided `fact_key`; applications or later deterministic entity resolution supply a canonical subject ID when one is available.

```text
CandidateMemory
      |
Predicate Resolver
      |
      |-- known
      |      |
      |  cardinality
      |      |
      |   fact_key
      |
      `-- unknown
             |
      open semantic memory
      fact_key = NULL
```

`fact_key` is not primarily a search string. It is the stable identity of a logical fact slot, enabling later exact structured lookup, duplicate detection, and update/supersession detection. Mutation and supersession behavior are not implemented in this milestone.

Known predicates receive deterministic structured identity. Useful facts with unknown predicates remain valid open-world semantic memories with no fact key, so the registry never rejects knowledge merely because it does not recognize a slot.

Cardinality determines the key shape. A single-valued fact omits its value, such as `user:user_123:location`. A multi-valued fact includes a normalized and safely encoded value identity, such as `user:user_123:programming_language:python`.

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
