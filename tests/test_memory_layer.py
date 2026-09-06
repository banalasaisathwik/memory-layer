"""PostgreSQL integration tests for the public MemoryLayer facade.

These tests verify facade orchestration -- persistence, interaction
grouping, provenance, automatic summary updates, and delegation to
search_memories()/an LLM reader. They intentionally do not re-verify
extraction, write, or retrieval correctness, which already have their own
test suites; every LLM and embedding call here is mocked so the suite never
depends on a live provider.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select

from meminfra.config import configure, reset_config
from meminfra.database import (
    Conversation,
    ConversationSummary,
    Memory,
    Message,
    SessionLocal,
    User,
    create_tables,
    reset_engine,
)
from meminfra.memory_layer import AnswerError, MemoryLayer, MemoryLayerError
from meminfra.retrieval import UserNotFoundError


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; memory-layer facade integration tests never use DATABASE_URL.",
    ),
]


class FakeCompletions:
    """Record chat-completion requests while returning a local fake response.

    ``responses``, when set, is a queue of ``{"content": ...}``/``{"error": ...}``
    dicts consumed one per call (repeating the last entry once exhausted), so
    a test can simulate a malformed-then-valid sequence across the
    extractor's own internal repair-retry attempts.
    """

    def __init__(
        self,
        *,
        content: str | None = None,
        error: Exception | None = None,
        responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.error = error
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.responses is not None:
            step = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
            if "error" in step:
                raise step["error"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=step.get("content")))])
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeEmbeddingClient:
    """A constant-vector embedding client; only avoids live calls, not ranking quality."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index, _ in enumerate(input)]
        )


def _extraction_payload(*, memory_text: str, predicate: str | None = None, value: str | None = None) -> str:
    return json.dumps(
        {
            "memories": [
                {
                    "memory_type": "semantic",
                    "memory_text": memory_text,
                    "subject_type": "user",
                    "subject_name": None,
                    "predicate": predicate,
                    "value": value,
                    "confidence": 0.9,
                    "importance": 0.8,
                }
            ]
        }
    )


_EMPTY_EXTRACTION_PAYLOAD = json.dumps({"memories": []})


def _target_interaction_json(call: dict[str, Any]) -> str:
    """Pull the raw JSON line out of extractor._request_content()'s labeled TARGET INTERACTION section."""

    content = call["messages"][1]["content"]
    _, _, remainder = content.partition("TARGET INTERACTION\n")
    _, _, json_line = remainder.partition("\n")
    return json_line


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL, llm_model="test-model", embedding_model="fake-model")
    create_tables()
    yield
    reset_engine()
    reset_config()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
        session.rollback()


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeEmbeddingClient:
    """Keep every semantic-context and vector-search branch fully local."""

    client = FakeEmbeddingClient()
    configure(embedding_model="fake-model", faiss_index_dir=tmp_path, vector_candidate_multiplier=5)
    monkeypatch.setattr("meminfra.retrieval.message_vector.get_embedding_client", lambda: client)
    monkeypatch.setattr("meminfra.retrieval.vector.get_embedding_client", lambda: client)
    return client


@pytest.fixture
def fake_extraction(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions(content=_EMPTY_EXTRACTION_PAYLOAD)
    monkeypatch.setattr("meminfra.memory.extractor.get_llm_client", lambda: FakeClient(completions))
    return completions


@pytest.fixture
def fake_summary(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions(content="Updated summary.")
    monkeypatch.setattr("meminfra.memory.summaries.get_llm_client", lambda: FakeClient(completions))
    return completions


@pytest.fixture
def fake_answer(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions(content="UNKNOWN")
    monkeypatch.setattr("meminfra.memory_layer.get_llm_client", lambda: FakeClient(completions))
    return completions


@pytest.fixture
def low_summary_threshold():
    """Make a couple of messages enough to trigger a summary, then restore defaults."""

    configure(summary_trigger_messages=1, summary_recent_keep=1)
    yield
    configure(summary_trigger_messages=20, summary_recent_keep=6)


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


# -- add(): new/existing scope, cross-user isolation, interaction grouping -----------


def test_add_creates_missing_user_and_conversation_and_writes_a_memory(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(
        memory_text="User prefers PostgreSQL.",
        predicate="database_preference",
        value="PostgreSQL",
    )

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "I prefer PostgreSQL."}],
    )

    assert len(result.message_ids) == 1
    assert result.extracted_candidate_count == 1
    assert len(result.write_results) == 1

    user = db.scalar(select(User).where(User.external_id == user_id))
    conversation = db.scalar(select(Conversation).where(Conversation.external_id == conversation_id))
    assert user is not None
    assert conversation is not None
    assert conversation.user_id == user.id

    written_memory = db.get(Memory, result.write_results[0].memory_id)
    assert written_memory is not None
    assert written_memory.user_id == user.id
    assert written_memory.source_message_ids == result.message_ids


def test_add_reuses_an_existing_user_and_conversation(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    memory = MemoryLayer(db)

    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "Hello."}])
    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "Hi again."}])

    users = list(db.scalars(select(User).where(User.external_id == user_id)))
    conversations = list(db.scalars(select(Conversation).where(Conversation.external_id == conversation_id)))
    assert len(users) == 1
    assert len(conversations) == 1


def test_add_scopes_the_same_conversation_id_independently_per_user(db, fake_extraction) -> None:
    shared_conversation_id = _unique("main")
    user_a = _unique("user-a")
    user_b = _unique("user-b")
    memory = MemoryLayer(db)

    memory.add(user_id=user_a, conversation_id=shared_conversation_id, messages=[{"role": "user", "content": "A"}])
    memory.add(user_id=user_b, conversation_id=shared_conversation_id, messages=[{"role": "user", "content": "B"}])

    conversations = list(db.scalars(select(Conversation).where(Conversation.external_id == shared_conversation_id)))
    assert len(conversations) == 2
    owner_ids = {conversation.user_id for conversation in conversations}
    user_ids = {
        db.scalar(select(User).where(User.external_id == user_a)).id,
        db.scalar(select(User).where(User.external_id == user_b)).id,
    }
    assert owner_ids == user_ids


def test_add_persists_a_user_assistant_interaction_and_uses_both_as_provenance(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(memory_text="User is building MemoryLayer with PostgreSQL.")

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": "I'm using PostgreSQL for my project."},
            {"role": "assistant", "content": "Sounds good."},
        ],
    )

    assert len(result.message_ids) == 2
    assert len(fake_extraction.calls) == 1
    written_memory = db.get(Memory, result.write_results[0].memory_id)
    assert set(written_memory.source_message_ids) == set(result.message_ids)


def test_add_splits_longer_history_into_bounded_sequential_interactions(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": "I know Python."},
            {"role": "assistant", "content": "Nice."},
            {"role": "user", "content": "I also work with TypeScript."},
            {"role": "assistant", "content": "Great combo."},
        ],
    )

    assert len(result.message_ids) == 4
    assert len(fake_extraction.calls) == 2

    first_target = json.loads(_target_interaction_json(fake_extraction.calls[0]))
    second_target = json.loads(_target_interaction_json(fake_extraction.calls[1]))
    assert [message["content"] for message in first_target["messages"]] == ["I know Python.", "Nice."]
    assert [message["content"] for message in second_target["messages"]] == [
        "I also work with TypeScript.",
        "Great combo.",
    ]


def test_add_treats_a_trailing_unanswered_user_message_as_its_own_interaction(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")

    memory = MemoryLayer(db)
    memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": "I live in Hyderabad."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "I moved to Bengaluru."},
        ],
    )

    assert len(fake_extraction.calls) == 2
    last_target = json.loads(_target_interaction_json(fake_extraction.calls[1]))
    assert [message["content"] for message in last_target["messages"]] == ["I moved to Bengaluru."]


def test_add_rejects_empty_messages(db, fake_extraction) -> None:
    memory = MemoryLayer(db)
    with pytest.raises(Exception):
        memory.add(user_id=_unique("user"), conversation_id=_unique("conv"), messages=[])


def test_add_rejects_a_blank_user_id(db) -> None:
    memory = MemoryLayer(db)
    with pytest.raises(MemoryLayerError):
        memory.add(user_id="  ", conversation_id=_unique("conv"), messages=[{"role": "user", "content": "hi"}])


# -- add(): summary orchestration ------------------------------------------------------


def test_add_automatically_updates_the_summary_when_eligible(db, fake_extraction, fake_summary, low_summary_threshold) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": "I live in Pune."},
            {"role": "assistant", "content": "Noted."},
        ],
    )

    assert result.summary_updated is True
    assert result.warnings == []
    assert len(fake_summary.calls) == 1

    conversation = db.scalar(select(Conversation).where(Conversation.external_id == conversation_id))
    summary = db.scalar(select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id))
    assert summary is not None
    assert summary.summary_text == "Updated summary."


def test_add_surfaces_a_summary_failure_without_losing_the_memory_write(
    db, fake_extraction, fake_summary, low_summary_threshold
) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(memory_text="User's name is Asha.")
    fake_summary.error = RuntimeError("provider outage")

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            {"role": "user", "content": "My name is Asha."},
            {"role": "assistant", "content": "Nice to meet you."},
        ],
    )

    assert result.summary_updated is False
    assert result.warnings == ["summary_update_failed"]
    assert len(result.write_results) == 1

    written_memory = db.get(Memory, result.write_results[0].memory_id)
    assert written_memory is not None
    assert written_memory.memory_text == "User's name is Asha."

    conversation = db.scalar(select(Conversation).where(Conversation.external_id == conversation_id))
    summary = db.scalar(select(ConversationSummary).where(ConversationSummary.conversation_id == conversation.id))
    assert summary is None


def test_add_propagates_extraction_failure_without_losing_persisted_messages(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.error = RuntimeError("provider outage")

    memory = MemoryLayer(db)
    with pytest.raises(Exception):
        memory.add(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[{"role": "user", "content": "This will fail to extract."}],
        )

    conversation = db.scalar(select(Conversation).where(Conversation.external_id == conversation_id))
    assert conversation is not None
    persisted = list(db.scalars(select(Message).where(Message.conversation_id == conversation.id)))
    assert len(persisted) == 1


def test_add_writes_zero_memories_when_extraction_exhausts_its_repair_retries(db, fake_extraction) -> None:
    """A response that stays malformed across all internal repair attempts must

    still leave zero Memory rows for the interaction -- write_memories() is
    only ever called by MemoryLayer.add() after extract_memories() returns
    successfully (src/memory_layer.py), so a final extraction failure means no
    partial write happened.
    """

    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = "not json"  # malformed on every one of the extractor's 3 internal attempts

    memory = MemoryLayer(db)
    with pytest.raises(Exception):
        memory.add(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[{"role": "user", "content": "This will never parse."}],
        )

    user = db.scalar(select(User).where(User.external_id == user_id))
    assert user is not None
    written = list(db.scalars(select(Memory).where(Memory.user_id == user.id)))
    assert written == []
    # The extractor retried internally up to its 3-attempt cap.
    assert len(fake_extraction.calls) == 3


def test_add_writes_exactly_one_memory_batch_when_extraction_needed_an_internal_retry(
    db, fake_extraction
) -> None:
    """One MemoryLayer.add() call that required an internal extraction repair

    retry must still produce exactly one write batch for the interaction, not
    two -- extract_memories() is called once per interaction and only returns
    once, fully resolved.
    """

    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.responses = [
        {"content": "not json"},
        {
            "content": _extraction_payload(
                memory_text="User prefers PostgreSQL.",
                predicate="database_preference",
                value="PostgreSQL",
            )
        },
    ]

    memory = MemoryLayer(db)
    result = memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "I prefer PostgreSQL."}],
    )

    assert len(fake_extraction.calls) == 2
    assert result.extracted_candidate_count == 1
    assert len(result.write_results) == 1

    user = db.scalar(select(User).where(User.external_id == user_id))
    written = list(db.scalars(select(Memory).where(Memory.user_id == user.id)))
    assert len(written) == 1
    assert written[0].source_message_ids == result.message_ids


# -- search(): thin wrapper over search_memories() -------------------------------------


def test_search_delegates_to_hybrid_retrieval_within_user_scope(db, fake_extraction) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(
        memory_text="User prefers PostgreSQL.",
        predicate="database_preference",
        value="PostgreSQL",
    )

    memory = MemoryLayer(db)
    memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "I prefer PostgreSQL."}],
    )

    hits = memory.search(user_id=user_id, query="Which database does the user prefer?", limit=5)

    assert len(hits) == 1
    assert "PostgreSQL" in hits[0].memory_text
    assert hits[0].is_active is True


def test_search_raises_for_an_unknown_user(db) -> None:
    memory = MemoryLayer(db)
    with pytest.raises(UserNotFoundError):
        memory.search(user_id=_unique("nobody"), query="anything")


# -- answer(): grounded reader over search() --------------------------------------------


def test_answer_grounds_its_response_in_retrieved_memories(db, fake_extraction, fake_answer) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(
        memory_text="User prefers PostgreSQL.",
        predicate="database_preference",
        value="PostgreSQL",
    )
    fake_answer.content = "The user prefers PostgreSQL."

    memory = MemoryLayer(db)
    memory.add(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "I prefer PostgreSQL."}],
    )

    result = memory.answer(user_id=user_id, query="Which database does the user prefer?", limit=5)

    assert result.answer == "The user prefers PostgreSQL."
    assert result.abstained is False
    assert len(result.supporting_memory_ids) == 1
    assert len(result.retrieved_memories) == 1
    assert len(fake_answer.calls) == 1


def test_answer_abstains_without_an_llm_call_when_no_memories_exist(db, fake_extraction, fake_answer) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    # Empty extraction keeps this user/conversation scope real with zero memories.
    memory = MemoryLayer(db)
    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "Hello!"}])

    result = memory.answer(user_id=user_id, query="What does the user prefer?")

    assert result.abstained is True
    assert result.supporting_memory_ids == []
    assert result.retrieved_memories == []
    assert fake_answer.calls == []


def test_answer_raises_a_clear_error_on_provider_failure(db, fake_extraction, fake_answer) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    fake_extraction.content = _extraction_payload(memory_text="User prefers PostgreSQL.")
    fake_answer.error = RuntimeError("provider outage")

    memory = MemoryLayer(db)
    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "I prefer PostgreSQL."}])

    with pytest.raises(AnswerError):
        memory.answer(user_id=user_id, query="Which database does the user prefer?")


def test_answer_presents_multiple_retrieved_memories_as_a_bounded_numbered_list(
    db, fake_extraction, fake_answer
) -> None:
    user_id = _unique("user")
    conversation_id = _unique("conv")
    memory = MemoryLayer(db)

    fake_extraction.content = _extraction_payload(memory_text="User's name is Asha.")
    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "My name is Asha."}])
    fake_extraction.content = _extraction_payload(memory_text="User lives in Pune.")
    memory.add(user_id=user_id, conversation_id=conversation_id, messages=[{"role": "user", "content": "I live in Pune."}])

    fake_answer.content = "Asha lives in Pune."
    result = memory.answer(user_id=user_id, query="What do we know about the user?", limit=5)

    request_payload = json.loads(fake_answer.calls[0]["messages"][1]["content"])
    assert [entry["ref"] for entry in request_payload["memories"]] == ["M0", "M1"]
    assert len(result.supporting_memory_ids) == 2
