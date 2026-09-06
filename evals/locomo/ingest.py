"""Ingest one LoCoMo conversation through the real MemoryLayer.add() pipeline.

Representation choices (see evals/README.md for the full rationale):

- Speaker identity is preserved in the message text itself, as
  ``"{speaker}: {text}"``, rather than discarded in favor of only the mapped
  chat role. Some LoCoMo questions name a speaker directly (e.g. "What hobby
  did Caroline start again?"), so the speaker name must remain part of what
  extraction actually sees.
- Both speaker_a and speaker_b map to "user" (LocomoSample.role_for_speaker).
  LoCoMo conversations are between two human participants, not a user and an
  assistant -- mapping speaker_b to "assistant" made the production
  extractor's assistant-claims-are-not-evidence rule silently discard every
  fact speaker_b stated about themselves. Speaker names in the text (above)
  are what let extraction still tell the two participants apart.
- A turn's released image caption (``blip_caption``), when present, is
  appended to its text as ``"... [shared image: <caption>]"``. Images
  themselves are never downloaded.
- MemoryLayer.add() has no application-supplied timestamp parameter and this
  milestone does not redesign that API. Instead, every turn is prefixed with
  ``"[Session date: <session_date_time>]"`` so the released session timestamp
  is not silently discarded for any turn, not just the first one in its
  session.

A LoCoMo conversation is long enough (hundreds of turns, many extraction
calls) that a single interaction's malformed LLM response -- for example, a
provider returning a JSON boolean where the extractor's schema requires a
string -- is a realistic, honest baseline outcome, not a hypothetical. When
that happens mid-session, MemoryLayer.add() has already durably persisted
every Message for that session (message persistence commits before
extraction begins) but never returns, so its message_ids are lost. Rather
than aborting the whole sample's ingestion, this module records the failure
as a warning and recovers that session's dia_id -> Message.id mapping by
re-reading the just-persisted rows -- never inserting Memory or Message rows
by hand, only reading back what MemoryLayer.add() itself already wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from meminfra.database.models import Conversation, Message, User
from meminfra.memory import ExtractionError, WriteError
from meminfra.memory_layer import AddResult, MemoryLayer

from .schemas import LocomoSample, LocomoTurn


class LocomoResumeAmbiguousError(RuntimeError):
    """Raised when a conversation's persisted message count doesn't match a checkpoint.

    This means a previous run left behind messages that are not accounted
    for by any fully-completed, checkpointed session -- almost always a
    process that died mid-session (for example, a dropped database
    connection) after persisting that session's messages but before
    ``MemoryLayer.add()`` returned. Blindly calling ``add()`` again for that
    session would duplicate its messages and re-run extraction on them, so
    resuming refuses outright instead of guessing which messages are safe to
    keep.
    """


@dataclass(frozen=True)
class IngestOutcome:
    """The result of ingesting one LoCoMo sample: provenance mapping plus counters."""

    dia_id_to_message_id: dict[str, str]
    message_id_to_dia_id: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    sessions_ingested: int = 0
    sessions_failed: int = 0
    messages_persisted: int = 0


def _turn_content(turn: LocomoTurn) -> str:
    text = turn.text
    if turn.image_caption:
        text = f"{text} [shared image: {turn.image_caption}]"
    return f"[Session date: {turn.session_date_time}] {turn.speaker}: {text}"


def _group_by_session(turns: list[LocomoTurn]) -> dict[int, list[LocomoTurn]]:
    sessions: dict[int, list[LocomoTurn]] = {}
    for turn in turns:
        sessions.setdefault(turn.session_number, []).append(turn)
    return sessions


def _resolve_conversation_row_id(db: Session, *, user_id: str, conversation_id: str) -> object | None:
    return db.scalar(
        select(Conversation.id)
        .join(User, Conversation.user_id == User.id)
        .where(User.external_id == user_id, Conversation.external_id == conversation_id)
    )


def _last_n_persisted_messages(db: Session, *, conversation_row_id: object, n: int) -> list[Message]:
    """Read back the n most recently persisted messages, in chronological order.

    Safe to use as a fallback because MemoryLayer.add() persists and commits
    every message for one call before extraction for that call begins, so a
    mid-call extraction/write failure still leaves exactly that call's
    messages as the newest rows in the conversation.
    """

    rows = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_row_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(n)
        )
    )
    rows.reverse()
    return rows


def _first_n_persisted_messages(db: Session, *, conversation_row_id: object, n: int) -> list[Message]:
    """Read back the n earliest persisted messages, in chronological order.

    Used only when resuming: sessions the checkpoint already covers are not
    re-ingested, so their dia_id<->Message.id mapping is recovered by
    re-reading what an earlier run already durably persisted, in the same
    chronological order those sessions were originally ingested in.
    """

    return list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_row_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(n)
        )
    )


def _persisted_message_count(db: Session, *, conversation_row_id: object) -> int:
    count_query = select(func.count()).select_from(Message).where(Message.conversation_id == conversation_row_id)
    return db.scalar(count_query) or 0


def ingest_sample(
    db: Session,
    sample: LocomoSample,
    *,
    user_id: str,
    conversation_id: str,
    resume_from_session: int = 0,
    on_session_complete: Callable[[int, AddResult], None] | None = None,
    on_session_failed: Callable[[int, str], None] | None = None,
) -> IngestOutcome:
    """Ingest every session of one LoCoMo sample, session-by-session, via MemoryLayer.add().

    This exercises the real ingestion/extraction/write/summary pipeline once
    per session (matching LoCoMo's natural session structure and exercising
    rolling summaries), never inserting Memory or Message rows by hand.
    Returns the dia_id <-> persisted Message.id mapping built from
    MemoryLayer.add()'s own returned ``message_ids`` when a session ingests
    cleanly, or recovered by re-reading the persisted rows when it doesn't
    (see module docstring).

    ``resume_from_session`` skips every session numbered at or below it
    (sessions a caller has already checkpointed as fully complete) and
    recovers their dia_id<->Message.id mapping by re-reading the rows an
    earlier run already persisted. Before doing that, this verifies the
    conversation's total persisted message count exactly matches what those
    completed sessions alone account for; a mismatch means a previous run
    left ambiguous partial state (almost always a crash mid-session, after
    messages were persisted but before ``add()`` returned), and this raises
    ``LocomoResumeAmbiguousError`` rather than guessing -- resuming past that
    point could duplicate messages or re-run extraction on them.

    ``on_session_complete``, if given, is called with a session's number and
    its ``AddResult`` immediately after ``MemoryLayer.add()`` returns
    successfully for it -- never for a session that raised, so a caller
    checkpointing progress or printing it from this hook never marks (or
    reports) a partially-failed session as done. ``on_session_failed``, if
    given, is called instead (with the session's number and the recorded
    warning message) when a session's ``add()`` raised and its provenance
    was recovered -- so a long run's live progress output can show a failure
    as it happens rather than only in the final returned warnings list.
    """

    memory = MemoryLayer(db)
    dia_id_to_message_id: dict[str, str] = {}
    warnings: list[str] = []
    sessions_ingested = 0
    sessions_failed = 0
    messages_persisted = 0

    sessions = sorted(_group_by_session(sample.turns).items())
    already_done_turns = [
        turn for session_number, turns in sessions if session_number <= resume_from_session for turn in turns
    ]
    conversation_row_id = _resolve_conversation_row_id(db, user_id=user_id, conversation_id=conversation_id)

    if conversation_row_id is not None:
        actual_count = _persisted_message_count(db, conversation_row_id=conversation_row_id)
        if actual_count != len(already_done_turns):
            raise LocomoResumeAmbiguousError(
                f"{sample.sample_id}: expected exactly {len(already_done_turns)} persisted messages for "
                f"sessions completed through checkpoint {resume_from_session}, found {actual_count} "
                "persisted for this conversation."
            )
        if already_done_turns:
            recovered = _first_n_persisted_messages(
                db, conversation_row_id=conversation_row_id, n=len(already_done_turns)
            )
            if len(recovered) != len(already_done_turns):
                raise LocomoResumeAmbiguousError(
                    f"{sample.sample_id}: expected to recover {len(already_done_turns)} previously persisted "
                    f"messages for sessions through checkpoint {resume_from_session}, found {len(recovered)}."
                )
            for turn, message in zip(already_done_turns, recovered):
                dia_id_to_message_id[turn.dia_id] = str(message.id)
            messages_persisted += len(recovered)
    elif already_done_turns:
        raise LocomoResumeAmbiguousError(
            f"{sample.sample_id}: resume_from_session={resume_from_session} but no persisted conversation "
            f"exists for user_id={user_id!r}, conversation_id={conversation_id!r} to resume from."
        )

    for session_number, turns in sessions:
        if session_number <= resume_from_session:
            continue
        messages = [
            {
                "role": sample.role_for_speaker(turn.speaker),
                "content": _turn_content(turn),
            }
            for turn in turns
        ]
        try:
            result = memory.add(user_id=user_id, conversation_id=conversation_id, messages=messages)
        except (ExtractionError, WriteError) as error:
            # Defensive: a write-path failure may leave an open, unusable
            # transaction; a clean session tolerates this no-op safely.
            db.rollback()
            sessions_failed += 1
            failure_message = (
                f"session {session_number}: MemoryLayer.add() failed ({error.__class__.__name__}: {error}); "
                "recovering message provenance from persisted rows, memory extraction for this "
                "session is incomplete."
            )
            warnings.append(failure_message)
            if on_session_failed is not None:
                on_session_failed(session_number, failure_message)
            conversation_row_id = _resolve_conversation_row_id(
                db, user_id=user_id, conversation_id=conversation_id
            )
            if conversation_row_id is None:
                raise RuntimeError(
                    f"{sample.sample_id} session {session_number}: add() failed before any message "
                    "in this conversation was ever persisted; nothing to recover."
                ) from error
            recovered = _last_n_persisted_messages(db, conversation_row_id=conversation_row_id, n=len(turns))
            if len(recovered) != len(turns):
                raise RuntimeError(
                    f"{sample.sample_id} session {session_number}: expected to recover {len(turns)} "
                    f"persisted messages after a failed add(), found {len(recovered)}."
                ) from error
            message_ids = [str(message.id) for message in recovered]
        else:
            warnings.extend(result.warnings)
            message_ids = result.message_ids
            if len(message_ids) != len(turns):
                raise RuntimeError(
                    f"{sample.sample_id} session {session_number}: expected {len(turns)} persisted "
                    f"message IDs, got {len(message_ids)}."
                )
            sessions_ingested += 1
            if on_session_complete is not None:
                on_session_complete(session_number, result)

        for turn, message_id in zip(turns, message_ids):
            dia_id_to_message_id[turn.dia_id] = message_id
        messages_persisted += len(message_ids)

    message_id_to_dia_id = {message_id: dia_id for dia_id, message_id in dia_id_to_message_id.items()}
    return IngestOutcome(
        dia_id_to_message_id=dia_id_to_message_id,
        message_id_to_dia_id=message_id_to_dia_id,
        warnings=warnings,
        sessions_ingested=sessions_ingested,
        sessions_failed=sessions_failed,
        messages_persisted=messages_persisted,
    )
