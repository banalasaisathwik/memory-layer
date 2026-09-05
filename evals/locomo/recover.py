"""Recover extraction/write processing for LoCoMo sessions that failed mid-ingestion.

Context: ``evals/locomo/ingest.py`` persists every message for a session
before extraction begins, then calls ``MemoryLayer.add()``'s per-interaction
extraction/write loop for that session's messages. If one interaction's
``extract_memories()`` call raises, the loop stops immediately -- interactions
already processed earlier in that same session keep whatever they wrote
(``write_memories()`` commits per call), but every interaction from the
failure point onward in that session was never attempted at all.

This module recovers exactly those never-attempted interactions using the
already-persisted ``Message`` rows -- it never calls ``MemoryLayer.add()``
again and never inserts a ``Message`` row. It reuses the production
``build_extraction_context`` / ``extract_memories`` / ``write_memories``
functions and the same interaction-grouping policy as
``MemoryLayer.add()`` (``src.memory_layer._group_interactions``), so recovered
memories are written through the exact same deterministic ADD/NOOP/SUPERSEDE
rules as a normal ingestion run.

Boundary detection
-------------------
A session's persisted messages are split into interactions in original
order. For each interaction, "covered" means some existing ``Memory`` row for
this user already has ``source_message_ids`` that are a superset of this
interaction's message IDs -- proof that this interaction's extraction was
attempted and (successfully) written by the original run. An *uncovered*
interaction is ambiguous by itself: it may mean extraction legitimately found
no durable memory in that interaction, or it may mean extraction never ran.

That ambiguity is resolved using the loop's strict ordering: because
interactions are attempted strictly in order and a failure stops the loop
outright, any covered interaction at index i proves every interaction before
it (0..i-1) was also attempted. So the last covered interaction in a session
proves forward progress up to that point; every interaction after it was
*never* attempted (uncovered interactions before it were attempted but
produced no durable memory, and are correctly left alone). The recovery
boundary for a session is therefore one past its last covered interaction --
0 if no interaction is covered at all (the whole session failed on its first
interaction).

Known limitation (context contamination)
-----------------------------------------
``build_extraction_context()`` bounds recent/lexical/semantic message context
to strictly predate the target interaction using existing, unmodified APIs.
It does **not** bound the persisted rolling ``ConversationSummary`` the same
way: that summary reflects whatever the *original* (mostly successful)
ingestion run had summarized by the time this recovery runs, which may
already cover sessions chronologically after the interaction being
recovered. This is a pre-existing limitation of the production context API,
not something this recovery utility introduces or works around -- fixing it
would mean redesigning ``build_extraction_context()``, which is out of scope
for this recovery milestone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, Message, User
from src.memory import ExtractionError, WriteError, WriteResult, build_extraction_context, extract_memories, write_memories
from src.memory_layer import _group_interactions

from .schemas import LocomoSample

try:
    import openai
except ImportError:  # pragma: no cover - openai is a hard runtime dependency elsewhere
    openai = None  # type: ignore[assignment]


ErrorCategory = Literal[
    "rate_limit",
    "timeout",
    "connection",
    "invalid_json",
    "schema_validation",
    "other",
]

# A pure extraction request may be retried once for an explicit transient
# provider condition, and only before any write has happened for that
# interaction. This is intentionally not a general retry framework: no
# backoff, no retrying schema-invalid output, no retrying write_memories(),
# and never retrying MemoryLayer.add().
MAX_TRANSIENT_EXTRACTION_RETRIES = 1
_TRANSIENT_CATEGORIES: frozenset[str] = frozenset({"rate_limit", "timeout", "connection"})


@dataclass(frozen=True)
class InteractionPlan:
    """One reconstructed interaction and whether it already has a covering memory."""

    index: int
    message_ids: list[str]
    roles: list[str]
    covered: bool


@dataclass(frozen=True)
class SessionRecoveryPlan:
    """The dry-run-safe recovery plan for one session: no LLM calls, no writes."""

    session_number: int
    persisted_message_count: int
    interactions: list[InteractionPlan]
    boundary_index: int

    @property
    def already_complete(self) -> list[InteractionPlan]:
        return self.interactions[: self.boundary_index]

    @property
    def recovery_candidates(self) -> list[InteractionPlan]:
        return self.interactions[self.boundary_index :]


@dataclass(frozen=True)
class InteractionRecoveryError:
    """One classified extraction/write failure recorded during live recovery."""

    session_number: int
    interaction_index: int
    message_ids: list[str]
    error_category: ErrorCategory
    error_message: str
    attempts: int


@dataclass(frozen=True)
class InteractionRecoveryResult:
    """The outcome of attempting to recover one previously-unattempted interaction."""

    session_number: int
    interaction_index: int
    message_ids: list[str]
    status: Literal["recovered", "failed"]
    write_results: list[WriteResult] = field(default_factory=list)
    error: InteractionRecoveryError | None = None


@dataclass(frozen=True)
class SessionRecoveryOutcome:
    """A session's plan plus, when not a dry run, its live recovery results."""

    plan: SessionRecoveryPlan
    results: list[InteractionRecoveryResult] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryReport:
    """The full multi-session recovery run, in the chronological order it ran."""

    sample_id: str
    user_id: str
    conversation_id: str
    dry_run: bool
    sessions: list[SessionRecoveryOutcome]

    def to_json_dict(self) -> dict:
        def interaction_plan_dict(plan: InteractionPlan) -> dict:
            return {
                "index": plan.index,
                "message_ids": plan.message_ids,
                "roles": plan.roles,
                "covered": plan.covered,
            }

        def result_dict(result: InteractionRecoveryResult) -> dict:
            return {
                "interaction_index": result.interaction_index,
                "message_ids": result.message_ids,
                "status": result.status,
                "write_results": [
                    {
                        "action": write_result.action.value,
                        "memory_id": write_result.memory_id,
                        "fact_key": write_result.fact_key,
                        "superseded_memory_id": write_result.superseded_memory_id,
                        "reason": write_result.reason,
                    }
                    for write_result in result.write_results
                ],
                "error": (
                    None
                    if result.error is None
                    else {
                        "error_category": result.error.error_category,
                        "error_message": result.error.error_message,
                        "attempts": result.error.attempts,
                    }
                ),
            }

        return {
            "sample_id": self.sample_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "dry_run": self.dry_run,
            "sessions": [
                {
                    "session_number": outcome.plan.session_number,
                    "persisted_message_count": outcome.plan.persisted_message_count,
                    "interactions_total": len(outcome.plan.interactions),
                    "boundary_index": outcome.plan.boundary_index,
                    "already_complete_count": len(outcome.plan.already_complete),
                    "recovery_candidate_count": len(outcome.plan.recovery_candidates),
                    "interactions": [interaction_plan_dict(plan) for plan in outcome.plan.interactions],
                    "results": [result_dict(result) for result in outcome.results],
                }
                for outcome in self.sessions
            ],
        }


def session_message_slices(
    db: Session,
    sample: LocomoSample,
    *,
    conversation: Conversation,
) -> dict[int, list[Message]]:
    """Positionally map persisted messages back to LoCoMo sessions.

    ``MemoryLayer.add()`` persists one session's messages, in original turn
    order, in exactly one contiguous append before extraction for that
    session begins (see evals/locomo/ingest.py). Sessions are ingested in
    ascending session-number order, so the conversation's full persisted
    message history -- read back in chronological (created_at, id) order --
    is exactly the concatenation of each session's turns in that same order.
    This never re-derives a Message row's identity: it only slices the
    existing rows using the dataset's own known per-session turn counts, the
    same technique already used by ``evals/locomo/ingest.py``'s resume path.
    """

    turn_counts: dict[int, int] = {}
    for turn in sample.turns:
        turn_counts[turn.session_number] = turn_counts.get(turn.session_number, 0) + 1

    messages = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    )
    total_expected = sum(turn_counts.values())
    if len(messages) != total_expected:
        raise RuntimeError(
            f"{sample.sample_id}: expected {total_expected} persisted messages for the full sample, "
            f"found {len(messages)}. Refusing to guess session boundaries against unexpected state."
        )

    slices: dict[int, list[Message]] = {}
    cursor = 0
    for session_number in sorted(turn_counts):
        count = turn_counts[session_number]
        slices[session_number] = messages[cursor : cursor + count]
        cursor += count
    return slices


def build_dia_id_mapping(
    sample: LocomoSample,
    session_slices: dict[int, list[Message]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Rebuild the dia_id <-> Message.id mapping from already-persisted rows.

    Positional, in original per-session turn order -- the same technique
    ``evals/locomo/ingest.py`` already trusts for its resume path. Used by
    the retrieval rerun so it never has to re-ingest to know which memory
    provenance corresponds to which LoCoMo dia_id.
    """

    turns_by_session: dict[int, list] = {}
    for turn in sample.turns:
        turns_by_session.setdefault(turn.session_number, []).append(turn)

    dia_id_to_message_id: dict[str, str] = {}
    for session_number, turns in turns_by_session.items():
        messages = session_slices[session_number]
        if len(messages) != len(turns):
            raise RuntimeError(
                f"{sample.sample_id} session {session_number}: expected {len(turns)} persisted messages, "
                f"found {len(messages)}."
            )
        for turn, message in zip(turns, messages):
            dia_id_to_message_id[turn.dia_id] = str(message.id)

    message_id_to_dia_id = {message_id: dia_id for dia_id, message_id in dia_id_to_message_id.items()}
    return dia_id_to_message_id, message_id_to_dia_id


def _resolve_user_and_conversation(
    db: Session,
    *,
    user_id: str,
    conversation_id: str,
) -> tuple[User, Conversation]:
    user = db.scalar(select(User).where(User.external_id == user_id))
    if user is None:
        raise RuntimeError(f"No user exists for external ID {user_id!r}; nothing to recover.")
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.external_id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise RuntimeError(
            f"No conversation exists for external ID {conversation_id!r} under user {user_id!r}; nothing to recover."
        )
    return user, conversation


def _existing_source_id_sets(db: Session, *, user: User) -> list[frozenset[str]]:
    """Every existing memory's provenance set, regardless of active/superseded status."""

    return [
        frozenset(memory.source_message_ids)
        for memory in db.scalars(select(Memory).where(Memory.user_id == user.id))
    ]


def _boundary_index(covered: list[bool]) -> int:
    """One past the last covered interaction; 0 if none is covered at all."""

    last_covered = -1
    for index, is_covered in enumerate(covered):
        if is_covered:
            last_covered = index
    return last_covered + 1


def plan_session_recovery(
    db: Session,
    sample: LocomoSample,
    session_number: int,
    *,
    user_id: str,
    conversation_id: str,
) -> SessionRecoveryPlan:
    """Build the dry-run-safe recovery plan for one session: no LLM calls, no writes."""

    user, conversation = _resolve_user_and_conversation(db, user_id=user_id, conversation_id=conversation_id)
    session_slices = session_message_slices(db, sample, conversation=conversation)
    if session_number not in session_slices:
        raise RuntimeError(f"{sample.sample_id}: session {session_number} does not exist in this dataset.")

    messages = session_slices[session_number]
    interactions = _group_interactions(messages)
    existing_sources = _existing_source_id_sets(db, user=user)

    plans: list[InteractionPlan] = []
    covered_flags: list[bool] = []
    for index, interaction in enumerate(interactions):
        ordered_ids = [str(message.id) for message in interaction]
        is_covered = any(set(ordered_ids) <= source_set for source_set in existing_sources)
        covered_flags.append(is_covered)
        plans.append(
            InteractionPlan(
                index=index,
                message_ids=ordered_ids,
                roles=[message.role.value for message in interaction],
                covered=is_covered,
            )
        )

    return SessionRecoveryPlan(
        session_number=session_number,
        persisted_message_count=len(messages),
        interactions=plans,
        boundary_index=_boundary_index(covered_flags),
    )


def _classify_extraction_error(error: ExtractionError) -> ErrorCategory:
    """Classify by the actual provider/parsing exception, never by guessing."""

    cause = error.__cause__
    if openai is not None:
        # APITimeoutError subclasses APIConnectionError, so it must be checked first.
        if isinstance(cause, openai.APITimeoutError):
            return "timeout"
        if isinstance(cause, openai.RateLimitError):
            return "rate_limit"
        if isinstance(cause, openai.APIConnectionError):
            return "connection"
    if isinstance(cause, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(cause, PydanticValidationError):
        return "schema_validation"
    return "other"


def _is_transient(category: ErrorCategory) -> bool:
    return category in _TRANSIENT_CATEGORIES


def _extract_with_bounded_retry(
    interaction: list[Message],
    *,
    target_ids: list[str],
    context,
) -> tuple[list, InteractionRecoveryError | None, int]:
    """Run extract_memories() with at most one retry for an explicit transient error.

    Returns (candidates, error_or_none, attempts). Only ever retries the pure
    extraction LLM call itself, before anything is written -- never retries a
    write, and never coerces or fabricates output on failure.
    """

    attempts = 0
    last_category: ErrorCategory = "other"
    last_message = ""
    while True:
        attempts += 1
        try:
            candidates = extract_memories(
                [{"role": message.role.value, "content": message.content} for message in interaction],
                source_message_ids=target_ids,
                context=context,
            )
            return candidates, None, attempts
        except ExtractionError as error:
            last_category = _classify_extraction_error(error)
            last_message = str(error)
            if _is_transient(last_category) and attempts <= MAX_TRANSIENT_EXTRACTION_RETRIES:
                continue
            return [], InteractionRecoveryError(
                session_number=-1,  # filled in by the caller
                interaction_index=-1,
                message_ids=target_ids,
                error_category=last_category,
                error_message=last_message,
                attempts=attempts,
            ), attempts


def _recover_interaction(
    db: Session,
    interaction: list[Message],
    *,
    index: int,
    session_number: int,
    user_id: str,
    conversation_id: str,
) -> InteractionRecoveryResult:
    target_ids = [str(message.id) for message in interaction]
    context = build_extraction_context(
        db,
        user_external_id=user_id,
        conversation_external_id=conversation_id,
        target_message_ids=target_ids,
    )
    candidates, error, attempts = _extract_with_bounded_retry(interaction, target_ids=target_ids, context=context)
    if error is not None:
        return InteractionRecoveryResult(
            session_number=session_number,
            interaction_index=index,
            message_ids=target_ids,
            status="failed",
            error=InteractionRecoveryError(
                session_number=session_number,
                interaction_index=index,
                message_ids=target_ids,
                error_category=error.error_category,
                error_message=error.error_message,
                attempts=attempts,
            ),
        )

    write_results: list[WriteResult] = []
    if candidates:
        try:
            write_results = write_memories(
                db,
                candidates,
                user_external_id=user_id,
                conversation_external_id=conversation_id,
            )
        except WriteError as error:
            return InteractionRecoveryResult(
                session_number=session_number,
                interaction_index=index,
                message_ids=target_ids,
                status="failed",
                error=InteractionRecoveryError(
                    session_number=session_number,
                    interaction_index=index,
                    message_ids=target_ids,
                    error_category="other",
                    error_message=str(error),
                    attempts=1,
                ),
            )

    return InteractionRecoveryResult(
        session_number=session_number,
        interaction_index=index,
        message_ids=target_ids,
        status="recovered",
        write_results=write_results,
    )


def recover_session(
    db: Session,
    sample: LocomoSample,
    session_number: int,
    *,
    user_id: str,
    conversation_id: str,
    dry_run: bool,
) -> SessionRecoveryOutcome:
    """Recover one session's never-attempted interactions, in chronological order.

    Only interactions at or after the detected boundary are ever sent to
    ``extract_memories()``. Already-complete interactions (attempted by the
    original run, whether or not they produced a durable memory) are never
    replayed.
    """

    plan = plan_session_recovery(db, sample, session_number, user_id=user_id, conversation_id=conversation_id)
    if dry_run:
        return SessionRecoveryOutcome(plan=plan, results=[])

    user, conversation = _resolve_user_and_conversation(db, user_id=user_id, conversation_id=conversation_id)
    session_slices = session_message_slices(db, sample, conversation=conversation)
    messages = session_slices[session_number]
    interactions = _group_interactions(messages)

    results: list[InteractionRecoveryResult] = []
    for interaction_plan in plan.recovery_candidates:
        interaction = interactions[interaction_plan.index]
        results.append(
            _recover_interaction(
                db,
                interaction,
                index=interaction_plan.index,
                session_number=session_number,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        )
    return SessionRecoveryOutcome(plan=plan, results=results)


def recover_sessions(
    db: Session,
    sample: LocomoSample,
    session_numbers: list[int],
    *,
    user_id: str,
    conversation_id: str,
    dry_run: bool,
) -> RecoveryReport:
    """Recover every listed session strictly in chronological (ascending) order.

    Chronological order matters: a later failed session's extraction context
    can legitimately include memories recovered from an earlier failed
    session, so sessions must never be recovered out of order or in parallel.
    """

    outcomes = [
        recover_session(
            db,
            sample,
            session_number,
            user_id=user_id,
            conversation_id=conversation_id,
            dry_run=dry_run,
        )
        for session_number in sorted(set(session_numbers))
    ]
    return RecoveryReport(
        sample_id=sample.sample_id,
        user_id=user_id,
        conversation_id=conversation_id,
        dry_run=dry_run,
        sessions=outcomes,
    )


def default_recovery_report_path(sample_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[1] / "results" / f"locomo-recovery-{sample_id}-{timestamp}.json"


def save_recovery_report(report: RecoveryReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), indent=2), encoding="utf-8")


def render_dry_run_report(report: RecoveryReport) -> str:
    lines = [f"LoCoMo recovery dry-run: sample={report.sample_id} user_id={report.user_id}", ""]
    for outcome in report.sessions:
        plan = outcome.plan
        lines.append(
            f"Session {plan.session_number}: {plan.persisted_message_count} persisted messages, "
            f"{len(plan.interactions)} interactions"
        )
        lines.append(
            f"  already-complete interactions: {len(plan.already_complete)} (indices 0..{plan.boundary_index - 1})"
            if plan.boundary_index > 0
            else "  already-complete interactions: 0"
        )
        lines.append(f"  candidate recovery interactions: {len(plan.recovery_candidates)}")
        for interaction_plan in plan.recovery_candidates:
            lines.append(
                f"    interaction {interaction_plan.index}: roles={interaction_plan.roles} "
                f"message_ids={interaction_plan.message_ids}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.config import configure, get_config, reset_config
    from src.database import SessionLocal, create_tables, reset_engine

    from ..db import EvalDatabaseConfigError, get_eval_database_url, print_eval_database_banner
    from .dataset import DEFAULT_DATASET_PATH, load_locomo_dataset

    parser = argparse.ArgumentParser(description="Recover failed LoCoMo extraction/write processing.")
    parser.add_argument("--sample", required=True, help="LoCoMo sample_id, e.g. conv-30.")
    parser.add_argument("--sessions", required=True, nargs="+", type=int, help="Failed session numbers to recover.")
    parser.add_argument(
        "--user-id",
        default=None,
        help="Defaults to the resumable checkpoint convention locomo-resume-<sample_id>.",
    )
    parser.add_argument("--conversation-id", default="conversation")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None, help="Path for the JSON recovery report.")
    args = parser.parse_args(argv)

    try:
        eval_database_url = get_eval_database_url()
    except EvalDatabaseConfigError as error:
        print(f"error: {error}")
        return 1

    print_eval_database_banner(eval_database_url)
    reset_config()
    reset_engine()
    configure(database_url=eval_database_url)
    if not args.dry_run:
        settings = get_config()
        missing = [
            name
            for name, value in [
                ("LLM_MODEL", settings.llm_model),
                ("LLM_API_KEY", settings.llm_api_key),
                ("EMBEDDING_API_KEY", settings.embedding_api_key),
            ]
            if not value
        ]
        if missing:
            print(f"error: missing required provider configuration: {', '.join(missing)}.")
            return 1
    create_tables()

    samples = load_locomo_dataset(args.dataset_path)
    sample = next((candidate for candidate in samples if candidate.sample_id == args.sample), None)
    if sample is None:
        print(f"error: sample {args.sample!r} not found in {args.dataset_path}.")
        return 1

    user_id = args.user_id or f"locomo-resume-{args.sample}"

    try:
        with SessionLocal() as db:
            report = recover_sessions(
                db,
                sample,
                args.sessions,
                user_id=user_id,
                conversation_id=args.conversation_id,
                dry_run=args.dry_run,
            )
    finally:
        reset_engine()
        reset_config()

    if args.dry_run:
        print(render_dry_run_report(report))
    else:
        for outcome in report.sessions:
            recovered = sum(1 for r in outcome.results if r.status == "recovered")
            failed = sum(1 for r in outcome.results if r.status == "failed")
            print(
                f"Session {outcome.plan.session_number}: {len(outcome.results)} interactions recovered attempt, "
                f"{recovered} recovered, {failed} failed"
            )
            for result in outcome.results:
                if result.error is not None:
                    print(
                        f"  interaction {result.interaction_index} FAILED "
                        f"[{result.error.error_category}]: {result.error.error_message}"
                    )

    output_path = Path(args.output) if args.output else default_recovery_report_path(args.sample)
    save_recovery_report(report, output_path)
    print(f"\nSaved recovery report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
