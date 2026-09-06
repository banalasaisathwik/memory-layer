"""Real concurrent-transaction regression test for the active-fact-key race.

Milestone 6.1's audit found that the writer's SELECT-then-write check alone
cannot prevent two concurrent transactions from both observing no conflicting
active memory and both committing one, leaving two active rows for the same
(user_id, fact_key). This module reproduces that race with two independent
SQLAlchemy sessions on two threads, synchronized so both reach the decision
window before either commits, and asserts the database-level invariant holds
regardless of which writer loses.
"""

from __future__ import annotations

import os
import threading
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from meminfra.config import configure, reset_config
from meminfra.database import Memory, SessionLocal, User, create_tables, reset_engine
from meminfra.memory import CandidateMemory, WriteAction, WriteConflictError, write_memories
from meminfra.memory import writer as writer_module


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; writer concurrency tests never use DATABASE_URL.",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL)
    create_tables()
    yield
    reset_engine()
    reset_config()


def _user() -> User:
    with SessionLocal() as session:
        user = User(external_id=f"user-{uuid4().hex}")
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def _location_candidate(*, value: str) -> CandidateMemory:
    return CandidateMemory(
        memory_text=f"User lives in {value}",
        subject_type="user",
        predicate="location",
        value=value,
        confidence=0.9,
        importance=0.5,
    )


def _run_write(
    *,
    user_external_id: str,
    candidate: CandidateMemory,
    start_barrier: threading.Barrier,
    outcomes: list[object],
    index: int,
) -> None:
    """Write one candidate on its own session/transaction, released by a barrier."""

    with SessionLocal() as session:
        try:
            outcomes[index] = write_memories(session, [candidate], user_external_id=user_external_id)
        except Exception as error:  # noqa: BLE001 - the raised type is asserted by the caller
            outcomes[index] = error
        finally:
            # Release the other thread even if this one raised before reaching
            # the synchronized lookup, so the test cannot hang.
            if not start_barrier.broken:
                try:
                    start_barrier.wait(timeout=5)
                except threading.BrokenBarrierError:
                    pass


def test_concurrent_writes_for_the_same_fact_key_never_leave_two_active_rows(monkeypatch) -> None:
    """Two independent transactions racing to ADD the same fact_key.

    _active_memories_for_fact_key is the writer's read-then-decide checkpoint.
    Both threads are forced to complete that read (observing zero active rows)
    before either is allowed to proceed to its INSERT, which is exactly the
    interleaving the old code was vulnerable to: without the partial unique
    index, both would proceed to ADD and both would commit.
    """

    user = _user()
    first_candidate = _location_candidate(value="Bangalore")
    second_candidate = _location_candidate(value="Hyderabad")

    release_readers = threading.Barrier(2, timeout=5)
    original_lookup = writer_module._active_memories_for_fact_key

    def synchronized_lookup(db, *, user, fact_key):
        result = original_lookup(db, user=user, fact_key=fact_key)
        release_readers.wait()
        return result

    monkeypatch.setattr(writer_module, "_active_memories_for_fact_key", synchronized_lookup)

    outcomes: list[object] = [None, None]
    threads = [
        threading.Thread(
            target=_run_write,
            kwargs={
                "user_external_id": user.external_id,
                "candidate": candidate,
                "start_barrier": release_readers,
                "outcomes": outcomes,
                "index": index,
            },
        )
        for index, candidate in enumerate([first_candidate, second_candidate])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads), "a writer thread did not finish"

    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WriteConflictError)]
    successes = [outcome for outcome in outcomes if isinstance(outcome, list)]
    assert len(conflicts) == 1, f"expected exactly one WriteConflictError, got: {outcomes!r}"
    assert len(successes) == 1, f"expected exactly one successful write, got: {outcomes!r}"
    assert successes[0][0].action is WriteAction.ADD

    with SessionLocal() as session:
        fact_key = f"user:{user.external_id}:location"
        active_count = session.scalar(
            select(func.count())
            .select_from(Memory)
            .where(Memory.user_id == user.id, Memory.fact_key == fact_key, Memory.is_active.is_(True))
        )
        total_count = session.scalar(
            select(func.count()).select_from(Memory).where(Memory.user_id == user.id, Memory.fact_key == fact_key)
        )
        assert active_count == 1
        # The losing transaction's INSERT was rolled back entirely, not just
        # deactivated: the invariant is enforced before a second row ever
        # persists, so exactly one row should exist for this fact key at all.
        assert total_count == 1


def test_losing_writer_can_safely_retry_after_a_conflict(monkeypatch) -> None:
    """A WriteConflictError batch is fully rolled back, so retrying is safe.

    This validates the docstring's claim: retrying the same candidate after a
    conflict re-reads the winner's committed row and resolves deterministically
    (NOOP for the same value, SUPERSEDE for a changed one) instead of raising
    again.
    """

    user = _user()
    same_candidate = _location_candidate(value="Bangalore")

    release_readers = threading.Barrier(2, timeout=5)
    original_lookup = writer_module._active_memories_for_fact_key

    def synchronized_lookup(db, *, user, fact_key):
        result = original_lookup(db, user=user, fact_key=fact_key)
        release_readers.wait()
        return result

    monkeypatch.setattr(writer_module, "_active_memories_for_fact_key", synchronized_lookup)

    outcomes: list[object] = [None, None]
    threads = [
        threading.Thread(
            target=_run_write,
            kwargs={
                "user_external_id": user.external_id,
                "candidate": same_candidate,
                "start_barrier": release_readers,
                "outcomes": outcomes,
                "index": index,
            },
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WriteConflictError)]
    assert len(conflicts) == 1

    monkeypatch.setattr(writer_module, "_active_memories_for_fact_key", original_lookup)
    with SessionLocal() as session:
        retried = write_memories(session, [same_candidate], user_external_id=user.external_id)

    assert retried[0].action is WriteAction.NOOP
    with SessionLocal() as session:
        fact_key = f"user:{user.external_id}:location"
        active_count = session.scalar(
            select(func.count())
            .select_from(Memory)
            .where(Memory.user_id == user.id, Memory.fact_key == fact_key, Memory.is_active.is_(True))
        )
        assert active_count == 1
