"""Tests for portable, mocked LLM memory extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from meminfra.config import configure, reset_config
from meminfra.database import MemoryType
from meminfra.memory.context import ChatMessage, ConversationContext
from meminfra.memory.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionError,
    extract_memories,
)


class FakeCompletions:
    """Record completion requests while returning a fully local fake response.

    ``content``/``error`` still work as fixed single values used for every
    call (unchanged behavior for all pre-existing tests). ``responses``, when
    set, is a queue of ``{"content": ...}`` or ``{"error": ...}`` dicts
    consumed one per call -- once exhausted, the last entry repeats for any
    further calls, so a test only needs to specify the responses it cares
    about.
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
            index = min(len(self.calls) - 1, len(self.responses) - 1)
            step = self.responses[index]
            if "error" in step:
                raise step["error"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=step.get("content")))],
            )

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
    monkeypatch.setattr("meminfra.memory.extractor.get_llm_client", lambda: FakeClient(completions))
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


def test_extraction_prompt_renders_controlled_predicates_and_location_guidance(
    fake_completions: FakeCompletions,
) -> None:
    _extract(fake_completions, {"memories": []})
    sent_system_prompt = fake_completions.calls[0]["messages"][0]["content"]

    assert sent_system_prompt == EXTRACTION_SYSTEM_PROMPT
    for predicate in ("location", "database_preference", "programming_language"):
        assert f"- {predicate}:" in sent_system_prompt

    assert "use that canonical predicate name exactly" in sent_system_prompt
    assert 'subject_type "user"' in sent_system_prompt
    assert '"I live in Hyderabad."' in sent_system_prompt
    assert '"I now live in Delhi."' in sent_system_prompt
    assert '"I\'m visiting Mumbai for two days."' in sent_system_prompt
    assert 'do not use the durable "location" predicate' in sent_system_prompt


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


# --- Bounded repair-retry for malformed structured output ---------------------------------


def test_valid_first_response_costs_exactly_one_provider_call(fake_completions: FakeCompletions) -> None:
    memories = _extract(fake_completions, {"memories": [{"memory_text": "User prefers PostgreSQL"}]})

    assert len(memories) == 1
    assert len(fake_completions.calls) == 1


def test_invalid_json_then_valid_response_succeeds_after_one_repair_retry(
    fake_completions: FakeCompletions,
) -> None:
    valid_body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.responses = [
        {"content": "not json"},
        {"content": valid_body},
    ]

    memories = extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(memories) == 1
    assert memories[0].memory_text == "User prefers PostgreSQL"
    assert len(fake_completions.calls) == 2
    # The retry must carry a repair note in addition to the original request content.
    second_call_messages = fake_completions.calls[1]["messages"]
    assert len(second_call_messages) == 3
    assert "REPAIR INSTRUCTION" in second_call_messages[2]["content"]
    # The original evidence must be unchanged between attempts.
    assert fake_completions.calls[0]["messages"][1] == fake_completions.calls[1]["messages"][1]


def test_schema_invalid_enum_then_valid_response_succeeds_after_one_repair_retry(
    fake_completions: FakeCompletions,
) -> None:
    invalid_body = json.dumps(
        {"memories": [{"memory_text": "User prefers PostgreSQL", "memory_type": "unknown_type"}]}
    )
    valid_body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.responses = [
        {"content": invalid_body},
        {"content": valid_body},
    ]

    memories = extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(memories) == 1
    assert len(fake_completions.calls) == 2
    second_call_messages = fake_completions.calls[1]["messages"]
    assert "memory_type" in second_call_messages[2]["content"]


def test_out_of_range_importance_then_valid_response_succeeds_after_one_repair_retry(
    fake_completions: FakeCompletions,
) -> None:
    invalid_body = json.dumps(
        {"memories": [{"memory_text": "User prefers PostgreSQL", "importance": 5.0}]}
    )
    valid_body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.responses = [
        {"content": invalid_body},
        {"content": valid_body},
    ]

    memories = extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(memories) == 1
    assert len(fake_completions.calls) == 2


def test_all_attempts_fail_raises_typed_extraction_error_with_attempt_count(
    fake_completions: FakeCompletions,
) -> None:
    invalid_enum_body = json.dumps(
        {"memories": [{"memory_text": "User prefers PostgreSQL", "memory_type": "unknown_type"}]}
    )
    fake_completions.responses = [
        {"content": invalid_enum_body},
        {"content": invalid_enum_body},
        {"content": "not json"},
    ]

    with pytest.raises(ExtractionError) as error:
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert len(fake_completions.calls) == 3
    assert error.value.attempt_count == 3
    # The last attempt failed with invalid JSON, so the reported category reflects that.
    assert error.value.category == "invalid_json"


def test_provider_exception_fails_immediately_without_repair_retry(fake_completions: FakeCompletions) -> None:
    fake_completions.error = RuntimeError("provider unavailable")

    with pytest.raises(ExtractionError, match="request failed") as error:
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert isinstance(error.value.__cause__, RuntimeError)
    assert len(fake_completions.calls) == 1


def test_extraction_failure_never_reaches_a_write_call(
    fake_completions: FakeCompletions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extract_memories() must raise before any write path could be invoked.

    extract_memories() never imports or calls write_memories() itself, so this
    spies on the module to prove the assumption end-to-end for this file's
    scope; the call-site guarantee (write_memories() only runs after
    extract_memories() returns successfully) is exercised in
    tests/memory/test_writer.py and the MemoryLayer facade tests.
    """

    from meminfra.memory import writer as writer_module

    write_calls: list[Any] = []
    monkeypatch.setattr(writer_module, "write_memories", lambda *args, **kwargs: write_calls.append(1))

    fake_completions.content = "not json"

    with pytest.raises(ExtractionError):
        extract_memories([{"role": "user", "content": "I prefer PostgreSQL."}])

    assert write_calls == []


def test_successful_extraction_after_retry_returns_exactly_once(fake_completions: FakeCompletions) -> None:
    """One extract_memories() call that needs an internal repair retry still

    returns exactly one candidate list from exactly one logical invocation --
    no duplicate internal success paths.
    """

    valid_body = json.dumps({"memories": [{"memory_text": "User prefers PostgreSQL"}]})
    fake_completions.responses = [
        {"content": "not json"},
        {"content": valid_body},
    ]

    memories = extract_memories(
        [{"role": "user", "content": "I prefer PostgreSQL."}],
        source_message_ids=[42],
    )

    assert len(memories) == 1
    assert memories[0].source_message_ids == [42]
    assert len(fake_completions.calls) == 2
