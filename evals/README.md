# Memory layer evaluation harness

This evaluates the memory layer's **retrieval quality**, not an application LLM's
answers. It measures whether `search_memories()` ranks the right memories near the
top after a conversation has gone through the real ingestion pipeline. There is no
LLM judge and no answer-generation step in this milestone.

## Flow

```text
EvalCase.messages
    -> Message rows persisted per conversation
    -> build_extraction_context() for each user turn
    -> extract_memories()
    -> write_memories()
EvalCase.query
    -> search_memories()
    -> gold matching (evals/metrics.py)
    -> Hit@K / Recall@K / MRR + hard invariants
    -> CaseResult -> RunReport
```

Every stage after "persist messages" calls the existing `src.memory` /
`src.retrieval` APIs directly. The harness never inserts `Memory` rows by hand and
never changes extraction, writing, or retrieval behavior to make a case score
better.

Each user message is treated as its own one-message target interaction, with
everything earlier in the conversation available as extraction context (recent
messages, plus lexical/semantic older matches). Assistant messages are persisted so
that context is available, but are never extraction targets themselves.

## Current dataset: `smoke`

`evals/datasets/smoke.py` has 10 hand-written, deterministic cases in
`evals/schemas.py`'s `EvalCase` shape: `single_fact`, `update` (x2),
`multi_value`, `multi_fact`, `duplicate_paraphrase`, `cross_turn_reference`,
`irrelevant_distractor`, and `user_isolation`. Each case is small enough to read
end to end and predict the intended gold match by eye.

### One eval case

```python
EvalCase(
    id="preference_001",
    category="single_fact",
    messages=[EvalMessage(role="user", content="I prefer PostgreSQL over MongoDB.")],
    query="Which database does the user prefer?",
    expected_memories=[ExpectedMemory(required_terms=["postgresql"])],
)
```

`expected_memories` are gold memories expressed as required terms, not exact
sentences: a retrieved memory matches when its case- and whitespace-normalized text
contains every required term as a substring. This is deterministic, inspectable,
and does not call an LLM or an embedding model.

## Metrics

- **Hit@K** (K = 1, 3, 5): 1 if any gold memory is matched by a hit at or before
  rank K, else 0. Averaged across cases.
- **Recall@K**: for a case with multiple gold memories, the fraction of them
  matched by some hit within the top K. Averaged across cases.
- **MRR**: mean of `1 / rank_of_first_relevant_result` (0 if no gold memory was
  matched anywhere in the returned results).

## Hard invariants

- **User isolation**: for every hit, the harness loads the corresponding `Memory`
  row and asserts `Memory.user_id == primary_user.id`. A hit whose row belongs to
  another user (or is missing) is an isolation failure, checked for every case
  (not only `user_isolation_001`) since every case runs against a freshly
  created, uniquely named user.
- **No superseded memories by default**: since search runs without
  `include_history=True`, every hit's `is_active` should be `True`. A `False` hit
  is counted as `superseded_returned`.

Both are summed (not averaged) in the aggregate report; either one should read 0.

## Running it

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://user:password@host/database"
python -m evals.runner --dataset smoke --top-k 5
```

This calls the real, independently configured LLM and embedding providers (the
same `LLM_*` / `EMBEDDING_*` variables described in the project README), so it is
a deliberate, credentialed run rather than part of the normal test suite. It
always uses `TEST_DATABASE_URL`, matching the rest of the project's database
integration tests, and never falls back to `DATABASE_URL`. If `TEST_DATABASE_URL`,
`LLM_MODEL`, `LLM_API_KEY`, or `EMBEDDING_API_KEY` is missing, the command exits
with a clear message naming exactly what to configure instead of running.

### Provider configuration must be internally consistent

The harness only checks that `LLM_MODEL`, `LLM_API_KEY`, and `EMBEDDING_API_KEY`
are *present* before running; it does not (and should not) guess whether a key
actually belongs to the provider it's labeled for, since that would mean
inventing provider-specific validation the rest of the project doesn't have. For
each of the LLM and embedding provider, these three settings must all refer to
the same real account/endpoint:

- `LLM_PROVIDER` / `EMBEDDING_PROVIDER` (`openai`, `openrouter`, or
  `openai_compatible`)
- `LLM_API_KEY` / `EMBEDDING_API_KEY` — issued by that same provider
- `LLM_BASE_URL` / `EMBEDDING_BASE_URL` — required for `openai_compatible`;
  optional for `openrouter` (defaults to `https://openrouter.ai/api/v1`); left
  blank for `openai` to use OpenAI's own endpoint

A mismatch (for example, an OpenRouter-issued key with `EMBEDDING_PROVIDER=openai`
and no `EMBEDDING_BASE_URL` override) reaches OpenAI's real endpoint with the
wrong key and fails with a 401 `AuthenticationError` from the `openai` SDK, not a
harness error — the traceback names the exact request and rejected key, which is
the fastest way to spot this class of mismatch.

## Results

Each run prints a per-case table, an aggregate summary, and a per-category
breakdown to the terminal, then saves the same data as JSON under
`evals/results/`. That directory is gitignored (only `evals/results/.gitkeep` is
committed) so individual run outputs are not committed; the harness and dataset
themselves are.

## What's next

This milestone is the harness and a hand-written smoke dataset only. Integrating
the LoCoMo benchmark, adding an answer-generation/LLM-judge evaluation, and any
retrieval-quality changes (semantic dedup, reranking, recency, etc.) are later
milestones.
