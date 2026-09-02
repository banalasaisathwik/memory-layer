# Architecture

## Purpose

Memory Layer is a reusable memory infrastructure package for applications using LLMs. It is not an agent framework, a chatbot, a Context Builder, or a document RAG platform.

Milestone 1 established configuration, provider client construction, PostgreSQL connectivity, and durable relational models. Milestone 2 adds deterministic candidate-memory identity. Milestone 3 adds bounded LLM extraction into validated `CandidateMemory` objects. Milestone 4 adds validated, deterministic PostgreSQL writes and temporal supersession. Milestone 5 adds bounded conversation context, a single rolling summary per conversation, and Alembic schema migrations; retrieval remains unimplemented.

## Current boundary

An application explicitly calls the memory layer. The package does not own application conversations or decide when an application should write or read memory.

```text
Application
    |
Memory Layer
    |- Write Pipeline
    `- Retrieval Pipeline (planned)
```

`user_id` is the long-term memory scope. `conversation_id` is source and session context: a user can have many conversations, and a memory may optionally refer to the conversation that produced it. A conversation must not become the long-term memory boundary.

## Current foundation

- `src.config` holds environment and programmatic settings.
- `src.providers` creates OpenAI-compatible LLM and embedding clients lazily and independently.
- `src.database` creates a lazy SQLAlchemy/psycopg connection to PostgreSQL or Neon and defines the current durable models, including one `ConversationSummary` per conversation.
- A memory can be structured (subject, predicate, value, and optional fact key) or open/unstructured (`memory_text` alone). JSON provenance stores the source message IDs.
- `src.memory` validates structural `CandidateMemory` data, extracts candidates from a current user/assistant interaction, and produces deterministic fact keys for a small set of known predicates.

`DATABASE_URL` is the only URL used for normal application sessions. Alembic migrations use `DIRECT_URL` when configured, otherwise `DATABASE_URL`; neither path uses `TEST_DATABASE_URL`. This allows Neon migrations to use a direct connection while normal application traffic keeps its pooled connection. Database integration tests explicitly configure `TEST_DATABASE_URL`; they never fall back to `DATABASE_URL`.

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

`fact_key` is not primarily a search string. It is the stable identity of a logical fact slot, enabling exact structured lookup, duplicate detection, and deterministic supersession.

Known predicates receive deterministic structured identity. Useful facts with unknown predicates remain valid open-world semantic memories with no fact key, so the registry never rejects knowledge merely because it does not recognize a slot.

Cardinality determines the key shape. A single-valued fact omits its value, such as `user:user_123:location`. A multi-valued fact includes a normalized and safely encoded value identity, such as `user:user_123:programming_language:python`.

## Conversation context and rolling summaries

Milestone 5 adds only bounded context for interpreting a current interaction. It does not add a Context Builder framework, long-term memory retrieval, or automatic memory mutation.

```text
Conversation history
        |
        |-- rolling summary ---------+
        |-- recent messages ---------+
        `-- older lexical matches ---+
                                  | CONTEXT ONLY
Latest interaction ----------------+
                                  |
                            LLM extractor
                                  |
                           CandidateMemory
                                  |
                         deterministic writer
                                  |
                             PostgreSQL
```

`ConversationSummary` is a compact conversational aid, not durable memory. In particular, **summary != memory**: the summary is neither searched as long-term memory nor a replacement for the `Memory` table.

`build_extraction_context()` requires both user and conversation external IDs, resolves one unambiguous conversation within that user scope, loads its one persisted summary if present, and selects only a small chronological window of messages that precede the designated target IDs. Its optional `older_lexical_query` overrides the deterministic default query made from up to twelve de-duplicated non-filler target terms joined with lexical OR. The query performs a bounded PostgreSQL full-text lexical lookup over older raw messages, excluding both targets and the recent window; selected matches are then ordered chronologically for prompt readability. This deliberately lexical approach cannot resolve references without shared terms (for example, "that deployment thing" versus "Atlas service"). It preserves the conceptual order: rolling summary, recent raw messages, older lexical matches, then target interaction. Defaults are six recent messages and three lexical matches, configured with `EXTRACTION_RECENT_MESSAGES` and `EXTRACTION_LEXICAL_MESSAGES`.

The extractor receives four explicitly labeled sections: `CONVERSATION SUMMARY — CONTEXT ONLY`, `RECENT CONTEXT — CONTEXT ONLY`, `OLDER LEXICAL CONTEXT — CONTEXT ONLY`, and `TARGET INTERACTION`. Context helps interpretation; the target provides all new-memory evidence. Therefore only application-supplied target IDs become candidate provenance—summary, recent context, and lexical-match IDs are never attached to a new `Memory`.

This is conversation-context retrieval only. It intentionally uses no embeddings, FAISS, or semantic raw-message retrieval; semantic older-message retrieval remains a Milestone 6 concern.

`update_conversation_summary()` requires the same user and conversation scope and maintains one row per conversation. It retains the newest `SUMMARY_RECENT_KEEP` messages outside the summary (default 6), and calls the provider only once the remaining eligible history reaches `SUMMARY_TRIGGER_MESSAGES` (default 20). An update sends the prior summary plus only messages that became newly eligible, then advances `covered_through_message_id`; it never resends the entire conversation after the initial summary. Empty output, provider failure, an invalid coverage marker, or persistence failure raises `SummaryError` and rolls back the attempted write.

## Alembic migration strategy

`alembic/` is a conventional Alembic environment wired to `Base.metadata`. A new PostgreSQL/Neon database uses `python -m alembic upgrade head`. For an existing, verified Milestones 1-4 database created by `create_tables()`, first make a backup, run `python -m alembic stamp 0001_initial_schema`, then run `python -m alembic upgrade head`. The second revision adds `conversation_summaries` and converts `memories.importance` with `importance::double precision`; it does not recreate existing data.

The Milestone 5 downgrade is intentionally unsupported because a FLOAT-to-integer conversion could discard fractional importance values. Do not use migration commands against an unverified database. Application migrations never read `TEST_DATABASE_URL`; any migration verification against a dedicated test database must explicitly supply that test URL.

## Implemented extraction boundary

`extract_memories()` receives one bounded target interaction: one or two normal chat-style `user` and/or `assistant` messages. It optionally receives the narrow `ConversationContext` described above, but it does not accept arbitrary conversation history, query stored memories, or make mutation decisions.

```text
context only (optional)     target messages
        |                         |
        +----------+--------------+
                   |
             LLM extractor
                   |
             strict JSON parsing
                   |
          CandidateMemory validation
```

The LLM proposes only semantic content. The prompt requires strict JSON and instructs it to avoid unsupported inference, assistant speculation, and secrets. It cannot supply canonical subject IDs or fact keys because `CandidateMemory` forbids them. The application owns message provenance: supplied target `source_message_ids` never reach the LLM and are deterministically attached to each validated candidate after parsing. Summary and recent-context messages never become provenance by virtue of appearing in context.

The extractor uses the existing lazy OpenAI-compatible chat client with an ordinary chat completion and portable `json.loads` plus Pydantic validation. It intentionally does not depend on vendor-specific structured-output SDK helpers, so OpenAI, OpenRouter, and custom OpenAI-compatible endpoints use the same path.

Malformed JSON, empty model content, provider failures, and schema violations raise an explicit extraction error; malformed output is never converted into a valid candidate.

## Implemented write lifecycle

`write_memories()` accepts a batch of validated candidates and owns one PostgreSQL commit or rollback for the batch.

```text
Target interaction
      |
LLM extraction
      |
CandidateMemory
      |
deterministic identity
      |
validated write lifecycle
      |
ADD / NOOP / SUPERSEDE
      |
PostgreSQL
```

The writer resolves the requested `user_external_id` to an existing user. An optional conversation must exist and belong to that user. Every supplied source message must exist, belong to the same user, and, when a conversation is supplied, belong to that conversation. User scope is included in every active-memory lookup.

For a candidate with `subject_type = user`, deterministic application context supplies `User.external_id` as `subject_id`. Known predicates are canonicalized through the predicate registry and can produce a fact key. Unknown predicates remain open semantic memories: their persisted `predicate`, `value`, and `fact_key` are `NULL` while the original `memory_text` is retained.

### Structured memory

When `fact_key` is available, the writer performs an exact active-fact lookup in the user scope. The first fact is an ADD. A value equivalent under the shared deterministic value-identity normalization is a NOOP, and its source-message provenance is merged in first-seen order. A changed value for a single-valued predicate is a SUPERSEDE. Multi-valued predicates incorporate normalized value identity in the key, so different values coexist and repeating the same value is a NOOP.

SUPERSEDE is not DELETE. The writer inserts the new active row, then marks the old row inactive with `valid_to` equal to the new row's `valid_from` and sets `superseded_by_id`. Both changes are part of the same transaction, so the old fact remains historically queryable.

### Open semantic and episodic memory

Open semantic memory has no fact key. It uses only exact normalized `memory_text` matching within the user scope: an exact duplicate is a NOOP and every other text is an ADD. Semantic duplicate and contradiction detection are intentionally deferred until richer retrieval exists.

Episodic memory also has no fact key, but repeated text can describe distinct events. It is a NOOP only when both normalized text and the set of validated source message IDs match; otherwise it is added as another episode.

### Trust boundary

The LLM proposes `CandidateMemory`. Deterministic code owns user scope, conversation and provenance validation, predicate resolution, fact-key generation, active-memory lookup, mutation decisions, and transaction execution. The write path makes no LLM, embedding, FAISS, or retrieval call.

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
