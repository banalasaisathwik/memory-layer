"""Tests for portable, mocked LLM memory extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import configure, reset_config
from src.database import MemoryType
from src.memory.context import ChatMessage, ConversationContext
from src.memory.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionError,
    extract_memories,
)


class FakeCompletions:
    """Record completion requests while returning a fully local fake response."""

    def __init__(self, *, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


class FakeClient:
    """Expose only the OpenAI-compatible chat-completions surface the extractor uses."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture(autouse=True)
def configured_extractor() -> None:
    """Keep tests independent from environment credentials and provider clients."""

    reset_config()
    configure(llm_model="test-model")
    yield
    reset_config()


@pytest.fixture
def fake_completions(monkeypatch: pytest.MonkeyPatch) -> FakeCompletions:
    completions = FakeCompletions()
    monkeypatch.setattr("src.memory.extractor.get_llm_client", lambda: FakeClient(completions))
    return completions


def _extract(fake_completions: FakeCompletions, payload: dict[str, object]) -> list[object]:
    fake_completions.content = json.dumps(payload)
    return extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_valid_semantic_memory_parses_and_attaches_provenance(fake_completions: FakeCompletions) -> None:
    fake_completions.content = json.dumps(
        {
            "memories": [
                {
                    "memory_type": "semantic",
                    "memory_text": "User prefers PostgreSQL",
                    "subject_type": "user",
                    "subject_name": None,
                    "predicate": "database_preference",
                    "value": "PostgreSQL",
                    "confidence": 0.95,
                    "importance": 0.8,
                }
            ]
        }
    )

    memories = extract_memories(
        [{"role": "user", "content": "I prefer PostgreSQL."}],
        source_message_ids=[101, 102],
    )

    assert len(memories) == 1
    assert memories[0].memory_type is MemoryType.SEMANTIC
    assert memories[0].predicate == "database_preference"
    assert memories[0].source_message_ids == [101, 102]
    assert json.loads(fake_completions.calls[0]["messages"][1]["content"]) == {
        "messages": [{"role": "user", "content": "I prefer PostgreSQL."}]
    }


def test_context_is_labeled_separately_from_target_and_keeps_target_only_provenance(
    fake_completions: FakeCompletions,
) -> None:
    fake_completions.content = json.dumps({"memories": [{"memory_text": "User chose PostgreSQL"}]})

    memories = extract_memories(
        [{"role": "user", "content": "Yes, I'll use that."}],
        source_message_ids=["target-89", "target-90"],
        context=ConversationContext(
            summary="User knows Python. User lives in Bangalore.",
            recent_messages=[
                ChatMessage(role="assistant", content="PostgreSQL may fit your workload."),
            ],
            older_lexical_messages=[
                ChatMessage(role="user", content="The deployment target is PostgreSQL."),
            ],
            older_semantic_messages=[
                ChatMessage(role="user", content="User lives in Bangalore."),
            ],
        ),
    )

    prompt = fake_completions.calls[0]["messages"][1]["content"]
    assert "CONVERSATION SUMMARY — CONTEXT ONLY" in prompt
    assert "Do not create memories solely from this section." in prompt
    assert "RECENT CONTEXT — CONTEXT ONLY" in prompt
    assert "RELEVANT OLDER CONTEXT — CONTEXT ONLY" in prompt
    assert "User lives in Bangalore." in prompt
    assert "TARGET INTERACTION" in prompt
    assert "treat the target interaction as authoritative" in EXTRACTION_SYSTEM_PROMPT
    assert "Yes, I'll use that." in prompt
    assert memories[0].source_message_ids == ["target-89", "target-90"]


def test_multiple_candidates_parse(fake_completions: FakeCompletions) -> None:
    memories = _extract(
        fake_completions,
        {
            "memories": [
                {
                    "memory_text": "User knows Python",
                    "subject_type": "user",
                    "predicate": "programming_language",
                    "value": "Python",
                },
                {
                    "memory_text": "User knows Go",
                    "subject_type": "user",
                    "predicate": "programming_language",
                    "value": "Go",
                },
            ]
        },
    )

    assert [memory.value for memory in memories] == ["Python", "Go"]


def test_episodic_candidate_parses(fake_completions: FakeCompletions) -> None:
    memories = _extract(
        fake_completions,
        {
            "memories": [
                {
                    "memory_type": "episodic",
                    "memory_text": "User is debugging a memory retrieval bug today",
                    "subject_type": "user",
                    "subject_name": None,
                    "predicate": None,
                    "value": None,
                    "confidence": 0.9,
                    "importance": 0.5,
                }
            ]
        },
    )

    assert memories[0].memory_type is MemoryType.EPISODIC
    assert memories[0].predicate is None


def test_empty_memories_is_a_successful_extraction(fake_completions: FakeCompletions) -> None:
    assert _extract(fake_completions, {"memories": []}) == []


def test_unknown_predicate_is_preserved(fake_completions: FakeCompletions) -> None:
    memories = _extract(
        fake_completions,
        {
            "memories": [
                {
                    "memory_text": "User prefers implementation-first explanations",
                    "subject_type": "user",
                    "predicate": "explanation_style_preference",
                    "value": "implementation-first",
                }
            ]
        },
    )

    assert memories[0].predicate == "explanation_style_preference"


def test_invalid_json_raises_an_extraction_error(fake_completions: FakeCompletions) -> None:
    fake_completions.content = "not json"

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_fenced_json_with_language_tag_is_parsed(fake_completions: FakeCompletions) -> None:
    body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.content = f"```json\n{body}\n```"

    memories = extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(memories) == 1
    assert memories[0].memory_text == "User prefers PostgreSQL"


def test_fenced_json_without_language_tag_is_parsed(fake_completions: FakeCompletions) -> None:
    body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.content = f"```\n{body}\n```"

    memories = extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(memories) == 1
    assert memories[0].memory_text == "User prefers PostgreSQL"


def test_fenced_json_tolerates_surrounding_whitespace(fake_completions: FakeCompletions) -> None:
    body = json.dumps({"memories": []})
    fake_completions.content = f"  \n```json\n{body}\n```\n  "

    assert extract_memories([{"role": "user", "content": "Thanks!"}]) == []


def test_malformed_fenced_json_still_fails(fake_completions: FakeCompletions) -> None:
    fake_completions.content = "```json\nnot valid json\n```"

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_valid_json_with_trailing_prose_still_fails(fake_completions: FakeCompletions) -> None:
    fake_completions.content = json.dumps({"memories": []}) + "\nHope this helps!"

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_valid_json_with_leading_prose_still_fails(fake_completions: FakeCompletions) -> None:
    fake_completions.content = "Here is the JSON:\n" + json.dumps({"memories": []})

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_fenced_json_followed_by_trailing_prose_still_fails(fake_completions: FakeCompletions) -> None:
    body = json.dumps({"memories": []})
    fake_completions.content = f"```json\n{body}\n```\nHope this helps!"

    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_missing_top_level_memories_fails(fake_completions: FakeCompletions) -> None:
    with pytest.raises(ExtractionError, match="required schema"):
        _extract(fake_completions, {"candidates": []})


def test_invalid_candidate_fails(fake_completions: FakeCompletions) -> None:
    with pytest.raises(ExtractionError, match="required schema"):
        _extract(fake_completions, {"memories": [{"memory_text": "   "}]})


@pytest.mark.parametrize("forbidden_field", ["fact_key", "subject_id"])
def test_llm_cannot_supply_deterministic_identity_fields(
    fake_completions: FakeCompletions,
    forbidden_field: str,
) -> None:
    with pytest.raises(ExtractionError, match="required schema"):
        _extract(
            fake_completions,
            {"memories": [{"memory_text": "User lives in Bangalore", forbidden_field: "invented"}]},
        )


def test_llm_cannot_supply_source_message_ids(fake_completions: FakeCompletions) -> None:
    with pytest.raises(ExtractionError, match="must not include source_message_ids"):
        _extract(
            fake_completions,
            {
                "memories": [
                    {"memory_text": "User lives in Bangalore", "source_message_ids": [999]}
                ]
            },
        )


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_empty_provider_response_fails(fake_completions: FakeCompletions, content: str | None) -> None:
    fake_completions.content = content

    with pytest.raises(ExtractionError, match="empty response"):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])


def test_provider_exception_is_wrapped(fake_completions: FakeCompletions) -> None:
    fake_completions.error = RuntimeError("provider unavailable")

    with pytest.raises(ExtractionError, match="request failed") as error:
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert isinstance(error.value.__cause__, RuntimeError)


def test_invalid_messages_are_rejected_before_a_provider_request(fake_completions: FakeCompletions) -> None:
    with pytest.raises(ValidationError, match="role"):
        extract_memories([{"role": "system", "content": "not supported"}])

    assert fake_completions.calls == []


def test_interaction_is_bounded_to_two_messages(fake_completions: FakeCompletions) -> None:
    with pytest.raises(ValidationError, match="at most 2"):
        extract_memories(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ]
        )

    assert fake_completions.calls == []


def test_prompt_excludes_assistant_speculation() -> None:
    assert "Assistant messages may provide conversational context" in EXTRACTION_SYSTEM_PROMPT
    assert "never treat assistant-generated claims, guesses, or speculation as user facts" in EXTRACTION_SYSTEM_PROMPT
    assert "CONVERSATION SUMMARY — CONTEXT ONLY" in EXTRACTION_SYSTEM_PROMPT
    assert "RELEVANT OLDER CONTEXT — CONTEXT ONLY" in EXTRACTION_SYSTEM_PROMPT
