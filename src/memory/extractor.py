"""Portable LLM extraction of validated candidate memories for one interaction."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, TypeAdapter, ValidationError, field_validator

from src.config import get_config
from src.providers import get_llm_client

from .prompts import EXTRACTION_SYSTEM_PROMPT
from .schemas import CandidateMemory


MAX_EXTRACTION_MESSAGES = 2


class ExtractionError(Exception):
    """Raised when a provider response cannot safely become candidate memories."""


class ExtractionMessage(BaseModel):
    """One user or assistant message from the designated extraction interaction."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject empty message content without changing meaningful user wording."""

        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only.")
        return value


class ExtractionResult(BaseModel):
    """The strict top-level response contract requested from the LLM."""

    model_config = ConfigDict(extra="forbid")

    memories: list[CandidateMemory]


_MessageList = Annotated[list[ExtractionMessage], Field(min_length=1, max_length=MAX_EXTRACTION_MESSAGES)]
_SourceMessageIds = list[StrictStr | StrictInt] | None
_message_list_adapter = TypeAdapter(_MessageList)
_source_message_ids_adapter = TypeAdapter(_SourceMessageIds)


def _request_content(messages: list[ExtractionMessage]) -> str:
    """Serialize target messages as data, not as additional chat instructions."""

    return json.dumps(
        {"messages": [message.model_dump() for message in messages]},
        ensure_ascii=False,
    )


def _response_content(response: Any) -> str | None:
    """Read normal OpenAI-compatible chat-completion content defensively."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _parse_result(content: str) -> ExtractionResult:
    """Parse strict JSON and reject any response field outside the LLM contract."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ExtractionError("LLM extraction returned invalid JSON.") from error

    if not isinstance(payload, dict):
        raise ExtractionError("LLM extraction response must be a JSON object.")

    raw_memories = payload.get("memories")
    if isinstance(raw_memories, list):
        for memory in raw_memories:
            if isinstance(memory, dict) and "source_message_ids" in memory:
                raise ExtractionError("LLM extraction response must not include source_message_ids.")

    try:
        return ExtractionResult.model_validate(payload)
    except ValidationError as error:
        raise ExtractionError("LLM extraction response does not match the required schema.") from error


def extract_memories(
    messages: list[ExtractionMessage | dict[str, object]],
    *,
    source_message_ids: list[str | int] | None = None,
) -> list[CandidateMemory]:
    """Extract validated candidates from at most one current user/assistant interaction.

    Source IDs are application-owned provenance: they are never shown to the LLM
    and are attached to each successfully parsed candidate after validation.
    """

    target_messages = _message_list_adapter.validate_python(messages)
    validated_source_ids = _source_message_ids_adapter.validate_python(source_message_ids)

    settings = get_config()
    if not settings.llm_model:
        raise ExtractionError("LLM_MODEL must be configured before extracting memories.")

    try:
        response = get_llm_client().chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _request_content(target_messages)},
            ],
            temperature=0,
        )
    except Exception as error:
        raise ExtractionError("LLM extraction request failed.") from error

    content = _response_content(response)
    if content is None or not content.strip():
        raise ExtractionError("LLM extraction returned an empty response.")

    result = _parse_result(content)
    provenance = list(validated_source_ids or [])
    return [candidate.model_copy(update={"source_message_ids": provenance}) for candidate in result.memories]
