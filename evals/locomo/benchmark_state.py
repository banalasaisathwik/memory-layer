"""Load one benchmark user's frozen retrieval state once per ablation run (Part G).

A retrieval-only ablation over an already-ingested LoCoMo sample never
mutates the database, so the corpus it reads -- this user's active
memories, their provenance, and the BM25 corpus built from their text -- is
identical for every one of the ~105 questions in a run. Re-fetching that
corpus from Neon per question (the original ``ablate_lexical.py`` shape) is
pure repeated network cost for data that cannot have changed.

``BenchmarkState`` loads that corpus exactly once per (user, sample) and is
then reused read-only for every question. The vector branch deliberately
does *not* preload embedding vectors here: it still goes through
``src.retrieval.vector``'s FAISS index + cheap id-only sync check (the
Part A fast path), which is already the cheap way to serve per-query vector
search without re-downloading every embedding on each call.

Every ``BenchmarkState`` is scoped to exactly one ``user_id`` -- callers must
never merge two users' memories into one corpus or one BM25 index (G1).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Conversation, Memory, User

from .bm25_corpus import PreparedBM25Corpus, prepare_bm25_corpus
from .diagnose import active_memory_covered_dia_ids
from .recover import build_dia_id_mapping, session_message_slices
from .schemas import LocomoSample


@dataclass(frozen=True)
class GoldMemoryLookup:
    """Every active memory this user has, plus which gold dia_ids it covers."""

    memories: list[Memory]
    dia_ids_by_memory_id: dict[str, set[str]]


def build_gold_memory_lookup(db: Session, *, user: User, message_id_to_dia_id: dict[str, str]) -> GoldMemoryLookup:
    memories = list(db.scalars(select(Memory).where(Memory.user_id == user.id, Memory.is_active.is_(True))))
    dia_ids_by_memory_id: dict[str, set[str]] = {}
    for memory in memories:
        dia_ids = {message_id_to_dia_id[mid] for mid in memory.source_message_ids if mid in message_id_to_dia_id}
        dia_ids_by_memory_id[str(memory.id)] = dia_ids
    return GoldMemoryLookup(memories=memories, dia_ids_by_memory_id=dia_ids_by_memory_id)


@dataclass(frozen=True)
class BenchmarkState:
    """One user's frozen, read-only retrieval state for a whole ablation run."""

    user_id: str
    user: User
    conversation: Conversation | None
    active_memories: list[Memory]
    active_memory_by_id: dict[str, Memory]
    bm25_corpus: PreparedBM25Corpus
    message_id_to_dia_id: dict[str, str]
    covered_dia_ids: set[str]
    gold_lookup: GoldMemoryLookup


def load_benchmark_state(
    db: Session,
    sample: LocomoSample,
    *,
    user_id: str,
    conversation_external_id: str,
) -> BenchmarkState:
    """Load every stable, user-scoped read this ablation needs -- exactly once.

    Raises ``RuntimeError`` if the user or conversation does not exist (the
    same failure mode the original per-question code had, just surfaced up
    front instead of on the first question).
    """

    user = db.scalar(select(User).where(User.external_id == user_id))
    if user is None:
        raise RuntimeError(f"No user exists for external ID {user_id!r}; nothing to ablate.")
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.external_id == conversation_external_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise RuntimeError(
            f"No conversation exists for external ID {conversation_external_id!r} under user {user_id!r}."
        )

    active_memories = list(
        db.scalars(
            select(Memory)
            .where(Memory.user_id == user.id, Memory.is_active.is_(True))
            .order_by(Memory.created_at.asc(), Memory.id.asc())
        )
    )
    active_memory_by_id = {str(memory.id): memory for memory in active_memories}
    bm25_corpus = prepare_bm25_corpus(active_memories, text_of=lambda memory: memory.memory_text)

    slices = session_message_slices(db, sample, conversation=conversation)
    _, message_id_to_dia_id = build_dia_id_mapping(sample, slices)
    covered_dia_ids = active_memory_covered_dia_ids(db, user_id=user_id, message_id_to_dia_id=message_id_to_dia_id)
    gold_lookup = build_gold_memory_lookup(db, user=user, message_id_to_dia_id=message_id_to_dia_id)

    return BenchmarkState(
        user_id=user_id,
        user=user,
        conversation=conversation,
        active_memories=active_memories,
        active_memory_by_id=active_memory_by_id,
        bm25_corpus=bm25_corpus,
        message_id_to_dia_id=message_id_to_dia_id,
        covered_dia_ids=covered_dia_ids,
        gold_lookup=gold_lookup,
    )
