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
$env:EVAL_DATABASE_URL = "postgresql+psycopg://user:password@host/database"
python -m evals.runner --dataset smoke --top-k 5
```

This calls the real, independently configured LLM and embedding providers (the
same `LLM_*` / `EMBEDDING_*` variables described in the project README), so it is
a deliberate, credentialed run rather than part of the normal test suite. It
always uses `EVAL_DATABASE_URL` (see `evals/db.py`), a dedicated eval database kept
separate from both `DATABASE_URL` (application/Neon) and `TEST_DATABASE_URL`
(integration tests), and never falls back to either. If `EVAL_DATABASE_URL`,
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

The `smoke` dataset above is a hand-written, deterministic sanity check. The
`locomo` benchmark below is the first public, external baseline. Any
retrieval-quality changes (semantic dedup, reranking, recency, per-candidate
provenance, etc.) are later milestones -- this integration is evaluation only.

# LoCoMo benchmark

`evals/locomo/` integrates the official **LoCoMo** long-term
conversational-memory benchmark. Unlike `smoke`, it is not built on
`EvalCase`: one LoCoMo sample is one long, multi-session, two-speaker
conversation, ingested **once**, with every one of its QA questions evaluated
against that same resulting memory state -- not exploded into isolated
one-question conversations.

```text
LoCoMo sample
    -> sessions (chronological, session_1..session_N)
        -> MemoryLayer.add() once per session
    -> final long-term memory state for that sample
    -> every qa[] question evaluated against that one state
        -> MemoryLayer.search()  -> evidence-provenance retrieval metrics
        -> MemoryLayer.answer()  -> LoCoMo-compatible QA score
```

## 1. Dataset source

The official SNAP Research release: `snap-research/locomo`,
`data/locomo10.json` (10 conversations, ~2,000 QA questions). A copy is
vendored at `evals/data/locomo10.json` so the benchmark and its tests never
need network access; `evals.locomo.dataset.dataset_sha256()` records exactly
which copy produced a given run's numbers. No fork or modified copy is used,
and no images are downloaded -- only the released text and, where present,
`blip_caption` image-caption text.

A few source-dataset annotation quirks are handled without inventing a fix:
a handful of evidence strings are zero-padded (`D30:05`) or have two ids
concatenated into one string (`"D8:6; D9:17"`); these are recovered by
extracting every `D<session>:<index>` substring and normalizing away leading
zeros. Two references (of ~2,800 in the full file) still don't resolve to any
real turn and are kept as `unresolved_evidence` for diagnostics rather than
silently dropped or guessed.

## 2. Ingestion: how a conversation reaches `MemoryLayer.add()`

For one LoCoMo sample, `evals/locomo/ingest.py` calls `MemoryLayer.add()`
once per session (not once for the whole conversation, and not once per QA
question), in chronological session order, so rolling summaries and
memory state evolve the same way a real application's history would:

- **Speaker identity** is preserved in the message text itself --
  `"{speaker}: {text}"` -- not discarded in favor of only the mapped chat
  role, since some questions name a speaker directly (e.g. "What hobby did
  Caroline start again?").
- **Speaker -> role mapping** is fixed for the whole sample:
  `speaker_a -> "user"`, `speaker_b -> "assistant"`.
- **Session timestamps**: `MemoryLayer.add()` has no application-supplied
  timestamp parameter, and this milestone does not redesign that API.
  Instead, only the *first* turn of each session is prefixed with
  `"[Session date: <session_date_time>]"`; later turns in the same session
  don't repeat it, since they share one timestamp. Example:

  ```text
  [Session date: 1:56 pm on 8 May, 2023]
  Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
  ```

- **Image captions**: a turn with a released `blip_caption` gets it appended
  as `"... [shared image: <caption>]"`. The image itself is never fetched.

## 3. Evidence mapping: dia_id -> Message UUID -> provenance scoring

`ingest_sample()` uses `MemoryLayer.add()`'s own returned `message_ids`
(guaranteed to be in the same order as the input messages for that call) to
build a `dia_id <-> Message.id` mapping -- no Message rows are inserted by
hand, and no separate provenance scheme is invented.

For each retrieved `SearchHit`, the underlying `Memory` row's
`source_message_ids` are looked up (`db.get(Memory, hit.memory_id)`, the same
pattern the `smoke` harness's isolation check uses) and mapped back through
that dictionary to the LoCoMo dia_ids they provide provenance for. Comparing
that per-rank provenance against a question's gold `evidence` dia_ids answers:
*did the retriever return memories actually grounded in the dialog that
contains the answer?*

**Provenance caveat**: extraction currently attaches every candidate memory
from one target interaction to that whole interaction's `source_message_ids`,
so provenance is per-interaction, not yet per-candidate precise. This makes
evidence matching somewhat coarse -- it is an existing property of the system,
reported in every run's metadata, and deliberately not changed for this
baseline. A future milestone can implement per-candidate provenance and rerun.

## 4. Retrieval metrics

Computed in `evals/locomo/metrics.py`, for K = 1, 3, 5, 10:

- **Hit@K**: 1 if any gold evidence dia_id is covered by a hit's provenance
  at or before rank K, else 0.
- **Evidence Recall@K**: the fraction of a question's gold evidence dia_ids
  covered by the union of provenance across the top K hits (e.g. evidence
  `{E1, E2, E3}` with top-5 provenance covering `{E1, E3}` -> Recall@5 = 2/3).
- **MRR**: `1 / rank` of the first hit whose provenance intersects the gold
  evidence (0 if none does).

Questions with `evidence == []` (LoCoMo's adversarial and some open-domain
questions are intentionally unanswerable from the conversation) are excluded
from these aggregates -- recall against zero gold items isn't meaningful --
but every run reports `questions_excluded_no_evidence` so the exclusion is
never silent.

## 5. QA scoring

`evals/locomo/scoring.py` ports the scoring logic in
`task_eval/evaluation.py` from the official `snap-research/locomo` repository
(`normalize_answer`, `f1_score`, the multi-hop `f1`, and the category
dispatch in `eval_question_answering`), rather than inventing a new protocol:

- **Categories 2 (temporal) and 4 (single_hop)**: plain normalized token F1
  between the predicted and gold answer.
- **Category 3 (open_domain)**: same F1, against only the first `;`-separated
  alternative in the gold answer (matching the official script).
- **Category 1 (multi_hop)**: both the prediction and the gold answer are
  split on `,` into sub-answers; the score is the mean, over gold
  sub-answers, of the best F1 against any predicted sub-answer.
- **Category 5 (adversarial)**: not F1-scored. LoCoMo's adversarial questions
  presuppose something false; correct behavior is abstention. The official
  script scores this as 1 if the prediction contains `"no information
  available"` or `"not mentioned"`, else 0.

Two deliberate deviations from the official script, both documented in
`evals/locomo/scoring.py`:

1. **No Porter stemming.** The official script stems tokens before comparing;
   this port compares normalized tokens directly, to avoid adding an `nltk`
   dependency for this milestone. This only affects near-miss inflectional
   matches and is not expected to change which system looks better, only
   nudge absolute F1 slightly.
2. **Adversarial correctness also honors `AnswerResult.abstained`.**
   `MemoryLayer.answer()` abstains with one fixed sentence ("The retrieved
   memories do not contain enough information to answer this question.")
   that does not literally contain either official phrase. Scoring an
   abstention our system can already report structurally as *wrong* just
   because of surface wording would be a worse, less honest baseline, so
   `abstained is True` also counts as a correct adversarial response.

## 6. Retrieval and QA are always reported separately

A LoCoMo run's `--mode` is `retrieval`, `qa`, or `both`. The two are never
averaged into one number: retrieval metrics say whether `search()` surfaced
evidence-grounded memories; QA metrics say whether `answer()`'s final text
was correct. A low Evidence Recall@5 with a reasonable QA F1 (or vice versa)
is itself a diagnostic finding -- distinguishing retrieval failure from
answer-reader failure is the point of keeping them apart.

## 7. Category mapping

The official LoCoMo category IDs, used as-is (never a different
industry/fork mapping): `1 -> multi_hop`, `2 -> temporal`,
`3 -> open_domain`, `4 -> single_hop`, `5 -> adversarial`. Every result
carries both `category_id` and `category_name`.

## 8. Isolation and reproducibility

Each LoCoMo sample gets its own isolated `user_id`
(`locomo-<sample_id>-<run_id>`, where `run_id` is fresh per CLI invocation)
and a single fixed `conversation_id`. `isolation_failures` -- the same
`Memory.user_id` check the `smoke` harness uses -- is expected to be `0`.

Every run's JSON output includes reproducibility metadata: dataset name,
source, and SHA-256; git commit SHA; run timestamp and run ID; configured LLM
and embedding models; top_k; mode; conversation/question counts; the category
mapping; and the provenance caveat above. No API keys are ever recorded.

By default (without `--resume`), re-running the benchmark always re-ingests
from scratch under a new, isolated `run_id`. This means every such run pays
full extraction/embedding provider cost again; use `--conversation` and
`--max-questions` during development to bound that cost. See section 11 for
`--resume`, which lets a long ingestion survive a crash (e.g. a dropped
database connection) without starting over.

## 9. Running it

```powershell
$env:EVAL_DATABASE_URL = "postgresql+psycopg://user:password@host/database"

# Cheap development subset: one conversation, first 20 questions, retrieval only.
python -m evals.locomo --conversation 0 --max-questions 20 --mode retrieval

# Full retrieval baseline across all 10 conversations.
python -m evals.locomo --mode retrieval

# QA baseline (calls MemoryLayer.answer(), i.e. an extra LLM call per question).
python -m evals.locomo --mode qa

# A long conversation, safe to resume if the process dies mid-ingestion.
python -m evals.locomo --conversation 1 --mode retrieval --resume
```

Same provider-configuration requirements as `evals.runner` above
(`EVAL_DATABASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `EMBEDDING_API_KEY`), never
falling back to `TEST_DATABASE_URL` or `DATABASE_URL`. `--top-k` defaults to 10 (the largest K this
milestone reports). Output is a per-run JSON report under `evals/results/`
(gitignored, same as `smoke`'s reports) plus a printed summary table.

**`--max-questions` only bounds QA/retrieval *evaluation* cost, never
ingestion cost.** Every session of every selected conversation is still fully
ingested (`MemoryLayer.add()` once per session) regardless of
`--max-questions`; the flag only stops evaluating further questions once the
cap is reached. Use `--conversation` to actually bound ingestion cost.

## 10. Tests

`tests/evals/test_locomo_*.py` use a tiny local fixture
(`tests/evals/fixtures/locomo_sample.json`: 2 sessions, 2 speakers, 5 turns,
4 QA questions covering all but the adversarial category) and mocked
LLM/embedding clients. Dataset parsing, scoring, and metrics tests need
neither a database nor a provider and always run; ingestion/runner
integration tests are gated behind `TEST_DATABASE_URL` like the rest of the
project's database tests and are skipped without it. This is intentional and
distinct from `EVAL_DATABASE_URL`: these tests exercise the harness's
functions directly against a disposable database via `configure()`, the same
way every other integration test in `tests/` does, so the regular test suite
never needs the Docker eval Postgres instance. Only `EVAL_DATABASE_URL`
resolution itself (`evals/db.py`, `tests/evals/test_eval_db.py`) and the CLI
entry points (`evals.runner`, `evals.locomo`, `ablate_lexical`, `recover`,
`diagnose`, `rerun`) use `EVAL_DATABASE_URL`. No test downloads the
dataset or calls a live provider.

## 11. Incremental per-conversation Message FAISS sync

A real LoCoMo run (hundreds of turns, dozens of sessions) exposed an
`O(N²)`-shaped bottleneck: every session's first extraction call triggered
`retrieve_semantic_message_context()` -> `_ensure_conversation_message_index()`
in `src/retrieval/message_vector.py`, which detected that the DB's message
set had grown since the last persisted index and responded by re-fetching
*every* message in the conversation (content and embedding included),
re-validating all of them, rebuilding the whole `faiss.IndexFlatIP` from
scratch, and rewriting the index and its metadata file -- even though only
that session's few new messages actually needed embedding or adding. Across
~19 growing sessions this meant thousands of redundant message re-fetches
and vector re-adds, and was the dominant cost behind a single conversation
taking 1-2 hours to ingest.

`_ensure_conversation_message_index()` now tries three paths, cheapest and
most common first:

1. **Unchanged** (`_fast_index_match`): compare the persisted index's
   ordered message IDs against a cheap id-only DB query (no content or
   embedding payload). If they match exactly, return the already-loaded
   index -- no re-embedding, no rebuild.
2. **Incremental append** (`_try_incremental_append`): if the persisted IDs
   are an exact, ordered *prefix* of the DB's current IDs (pure append
   growth), embed only the new suffix (reusing any embedding already
   persisted on the `Message` row), `index.add()` only those new vectors
   onto the already-loaded FAISS index, and persist the result with the
   project's existing atomic replace-on-complete write.
3. **Full validate-or-rebuild** (unchanged): the original, authoritative
   logic -- full re-fetch, full re-validation, full rebuild if anything
   doesn't line up. This is still exactly what runs for anything that isn't
   provably safe to fast-path: a missing/corrupt index, a model or dimension
   mismatch, a deleted/reordered message, or any other ID divergence. Its
   correctness is unchanged; the first two paths only skip it when they can
   prove it's unnecessary.

This is retrieval-*performance* only: ranking, RRF, provenance, and every
other retrieval-quality behavior are untouched, and the full pre-existing
`tests/memory/test_semantic_context.py` suite (corruption/model/dimension
recovery included) still passes unmodified. `src/retrieval.get_message_index_sync_stats()`
exposes process-wide `full_rebuilds` / `incremental_appends` / `unchanged_hits`
counters (reset with `reset_message_index_sync_stats()`) purely as
diagnostics; `evals/locomo/runner.py` prints and records them so a run can
show its hot path actually changed. See
`tests/retrieval/test_message_vector_sync.py` for the append/rebuild-fallback
test matrix (empty index, single/repeated appends, model mismatch, dimension
divergence, metadata ID divergence, a message becoming ineligible, and index
corruption).

## 12. Checkpoint/resume (`--resume`)

Ingesting a long conversation session-by-session is a real-money, real-time
operation; a dropped Neon/Postgres connection partway through should not
mean starting over. `--resume` (`evals/locomo/checkpoint.py`,
`evals/locomo/runner.py`, `evals/locomo/ingest.py`) adds session-level
checkpointing, kept entirely inside the benchmark harness -- it never
touches production `MemoryLayer`, its schema, or its transaction semantics.

- **Deterministic identity.** With `--resume`, a sample's `user_id` is
  `locomo-resume-<sample_id>` (not the normal `locomo-<sample_id>-<run_id>`
  tagged with a fresh random ID per run), so a later invocation resolves to
  the *same* conversation and can safely continue it.
- **Checkpoint only after a session fully completes.** After each session's
  `MemoryLayer.add()` call returns successfully, `ingest_sample()` calls an
  `on_session_complete` hook that writes
  `evals/checkpoints/<sample_id>.json` (atomically, temp file + replace) with
  `last_completed_session`, the dataset SHA-256, the configured LLM/embedding
  models, and the git commit. A session that raises never reaches this hook,
  so a partially-failed session is never checkpointed as done.
- **Resuming verifies compatibility first.** Before reusing a checkpoint, the
  runner compares its `sample_id`, `dataset_sha256`, `llm_model`, and
  `embedding_model` (plus `user_id`/`conversation_id`) against the current
  run. Any mismatch raises a clear `LocomoEnvironmentError` naming every
  field that differs and refuses to resume -- it never silently continues
  with stale benchmark state.
- **Partial-session safety.** `ingest_sample()` never blindly replays a
  session on resume. Before skipping the sessions a checkpoint claims are
  complete, it verifies the conversation's *actual* persisted message count
  exactly equals what those completed sessions alone account for. A mismatch
  (almost always a crash that persisted a session's messages but died before
  `add()` returned, e.g. mid-connection-drop) raises
  `LocomoResumeAmbiguousError` and refuses to guess -- calling `add()` again
  for that session would duplicate its messages and re-run extraction on
  them. This is the simplest safe policy for this milestone: completed,
  checkpointed sessions resume automatically; anything ambiguous stops and is
  reported rather than auto-replayed.
- **No new production API.** `MemoryLayer.add()` is unchanged; checkpointing
  lives entirely in `resume_from_session` / `on_session_complete` parameters
  added to `ingest_sample()`, both optional and defaulted so every existing
  non-resume caller and test is unaffected.

`evals/checkpoints/` is gitignored the same way as `evals/results/` (only
`.gitkeep` is committed).

## 13. Database connection robustness

`src/database/connection.py` already creates its SQLAlchemy engine with
`pool_pre_ping=True`, so a stale/dropped connection is detected and
transparently replaced before a query runs on it, rather than surfacing as a
mid-transaction failure. This milestone did not add any retry loop around
`MemoryLayer.add()` or any other write path: a connection failure genuinely
mid-transaction (for example, while `add()` is between persisting messages
and finishing extraction/writes) must **not** be silently retried, since
`add()` is not idempotent -- replaying it could duplicate messages and
memories. That is exactly why checkpoint/resume (section 12) exists as a
harness-level safety net instead: `--resume` only ever continues from a
session boundary it can prove is safe, never by retrying an in-flight call.

For a genuinely long benchmark run against a remote database, prefer Neon's
**pooled** connection endpoint (its hostname contains `-pooler`) for
`EVAL_DATABASE_URL` if your Neon project has one -- it tolerates many
short-lived connections better than the direct endpoint. This project does
not hard-code or rewrite any Neon-specific URL; it only reuses whatever
`EVAL_DATABASE_URL` you configure. During normal development, `EVAL_DATABASE_URL`
should instead point at a local, dedicated PostgreSQL instance (see
`.env.example`), keeping benchmark workloads off Neon entirely.

## 14. LoCoMo Subset Baseline V0

Rather than immediately running all 10 LoCoMo conversations, this milestone
fixes one full conversation as the standing development/public subset:
**`conv-30`** (dataset index `1`), chosen deterministically as the smallest
vendored conversation by turn count (369 turns, 19 sessions, 105 QA
questions) -- not a hand-picked "easy" sample, and the same conversation
every time this subset is referenced.

```powershell
python -m evals.locomo --conversation 1 --mode retrieval --resume
```

This ingests the *entire* conversation (all 369 turns across 19 sessions,
never a truncated prefix) and evaluates all 105 of its QA questions in
`retrieval` mode (the metrics section 22 of this milestone asks for --
Hit@K, Evidence Recall@K, MRR, isolation failures, extraction warnings --
are all retrieval metrics; QA F1 is not part of this baseline). The result
is saved under `evals/results/` and is never overwritten by a later run
(each save uses a fresh timestamped filename); it is explicitly labeled
**"LoCoMo Subset Baseline V0"** in this README and in any report generated
from it -- never "LoCoMo Baseline", "Full LoCoMo", or "LoCoMo score", since
it covers exactly one of the ten released conversations.
