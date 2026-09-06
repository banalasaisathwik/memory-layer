"""Portable LLM extraction of validated candidate memories for one interaction."""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, TypeAdapter, ValidationError

from meminfra.config import get_config
from meminfra.providers import get_llm_client

from .context import ChatMessage, ConversationContext
from .prompts import EXTRACTION_SYSTEM_PROMPT
from .schemas import CandidateMemory


logger = logging.getLogger(__name__)

MAX_EXTRACTION_MESSAGES = 2

# Bounded repair-retry: only response parsing/validation failures (invalid JSON,
# schema-invalid payloads) are retried, and only by re-sending the exact same
# evidence plus a short repair note describing what was wrong. Provider-level
# request failures (network, auth, rate limit, timeout, ...) are never retried
# here -- they raise immediately, exactly as before this loop existed.
MAX_EXTRACTION_ATTEMPTS = 3


class ExtractionError(Exception):
    """Raised when a provider response cannot safely become candidate memories.

    ``category`` and ``attempt_count`` are optional, purely informational
    attributes for callers that want to inspect why extraction ultimately
    failed (e.g. ``evals/locomo/recover.py``'s own classification). They do
    not change how this exception is normally raised or stringified: existing
    call sites that do ``raise ExtractionError("message")`` or
    ``except ExtractionError as error: str(error)`` keep working unchanged.
    """

    def __init__(self, message: str, *, category: str = "other", attempt_count: int = 1) -> None:
        super().__init__(message)
        self.category = category
        self.attempt_count = attempt_count


class _ParseFailure(Exception):
    """Internal: one attempt's response failed to parse/validate.

    Never raised out of this module -- ``extract_memories()`` catches it to
    decide whether a repair-retry attempt remains, and only ever raises the
    public ``ExtractionError`` to callers.
    """

    def __init__(self, message: str, category: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.cause = cause


def _format_validation_error(error: dict[str, Any]) -> str:
    """Render one pydantic error as a concise ``item[1].memory_type: <msg>`` path."""

    loc_parts: list[str] = []
    for segment in error.get("loc", ()):
        if isinstance(segment, int):
            if loc_parts:
                loc_parts[-1] = f"{loc_parts[-1]}[{segment}]"
            else:
                loc_parts.append(f"[{segment}]")
        elif segment == "memories" and not loc_parts:
            loc_parts.append("item")
        else:
            loc_parts.append(str(segment))
    loc = ".".join(loc_parts) if loc_parts else "<root>"
    return f"{loc}: {error.get('msg', error.get('type', 'invalid value'))}"


def _repair_instruction(failure: _ParseFailure) -> str:
    """A short, concrete correction note built from the actual validation error.

    Never a raw traceback -- field paths and error types only, so it stays a
    small addition to the request rather than a second copy of the evidence.
    """

    if isinstance(failure.cause, ValidationError):
        details = "; ".join(_format_validation_error(error) for error in failure.cause.errors())
    elif isinstance(failure.cause, json.JSONDecodeError):
        details = f"invalid JSON: {failure.cause}"
    else:
        details = str(failure)

    return (
        "REPAIR INSTRUCTION\n"
        "Your previous response was rejected for this reason: "
        f"{details}\n"
        "Return exactly one corrected JSON object matching the required schema below, "
        "with no markdown fence, no surrounding prose, and no source_message_ids field. "
        "Use the same target interaction and context as before -- do not invent new evidence."
    )


ExtractionMessage = ChatMessage


class ExtractionResult(BaseModel):
    """The strict top-level response contract requested from the LLM."""

    model_config = ConfigDict(extra="forbid")

    memories: list[CandidateMemory]


_MessageList = Annotated[list[ExtractionMessage], Field(min_length=1, max_length=MAX_EXTRACTION_MESSAGES)]
_SourceMessageIds = list[StrictStr | StrictInt] | None
_message_list_adapter = TypeAdapter(_MessageList)
_source_message_ids_adapter = TypeAdapter(_SourceMessageIds)


def _request_content(
    messages: list[ExtractionMessage],
    context: ConversationContext | None,
) -> str:
    """Keep optional context separate from target-only extraction evidence."""

    target_data = {"messages": [message.model_dump() for message in messages]}
    if context is None:
        return json.dumps(target_data, ensure_ascii=False)

    summary = context.summary if context.summary is not None else "(no persisted summary)"
    older_context = json.dumps(
        {"messages": [message.model_dump() for message in context.effective_older_messages()]},
        ensure_ascii=False,
    )
    recent_messages = json.dumps(
        {"messages": [message.model_dump() for message in context.recent_messages]},
        ensure_ascii=False,
    )
    return (
        "CONVERSATION SUMMARY — CONTEXT ONLY\n"
        "Do not create memories solely from this section.\n"
        f"{summary}\n\n"
        "RELEVANT OLDER CONTEXT — CONTEXT ONLY\n"
        "Use only to resolve older references and meaning.\n"
        f"{older_context}\n\n"
        "RECENT CONTEXT — CONTEXT ONLY\n"
        "Use only to resolve references and meaning.\n"
        f"{recent_messages}\n\n"
        "TARGET INTERACTION\n"
        "Extract new memories only from evidence in this section.\n"
        f"{json.dumps(target_data, ensure_ascii=False)}"
    )


def _response_content(response: Any) -> str | None:
    """Read normal OpenAI-compatible chat-completion content defensively."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


_OUTER_MARKDOWN_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(?P<body>.*)\n```", re.IGNORECASE | re.DOTALL)


def _strip_outer_markdown_fence(content: str) -> str:
    """Remove exactly one outer ```json / ``` fence; anything else is left untouched.

    This is a narrow portability accommodation: some OpenAI-compatible models
    wrap otherwise-valid JSON in a single markdown code fence despite being
    instructed to return raw JSON. Content must fully match "one fence wrapping
    everything" to be touched at all -- prose before or after the fence, extra
    fences, or an unterminated fence all fail to match and are passed through
    unchanged, so they still fail json.loads() as a genuine extraction error.
    """

    match = _OUTER_MARKDOWN_FENCE.fullmatch(content)
    return match.group("body") if match else content


def _parse_result(content: str) -> ExtractionResult:
    """Parse strict JSON and reject any response field outside the LLM contract.

    Raises the internal ``_ParseFailure`` (never ``ExtractionError`` directly)
    so ``extract_memories()`` can decide whether a repair-retry attempt
    remains before ever raising to its own caller.
    """

    candidate = _strip_outer_markdown_fence(content.strip())
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise _ParseFailure("LLM extraction returned invalid JSON.", "invalid_json", error) from error

    if not isinstance(payload, dict):
        raise _ParseFailure("LLM extraction response must be a JSON object.", "invalid_json")

    raw_memories = payload.get("memories")
    if isinstance(raw_memories, list):
        for memory in raw_memories:
            if isinstance(memory, dict) and "source_message_ids" in memory:
                raise _ParseFailure(
                    "LLM extraction response must not include source_message_ids.", "schema_validation"
                )

    try:
        return ExtractionResult.model_validate(payload)
    except ValidationError as error:
        raise _ParseFailure(
            "LLM extraction response does not match the required schema.", "schema_validation", error
        ) from error


def extract_memories(
    messages: list[ExtractionMessage | dict[str, object]],
    *,
    source_message_ids: list[str | int] | None = None,
    context: ConversationContext | None = None,
) -> list[CandidateMemory]:
    """Extract validated candidates from at most one current user/assistant interaction.

    Optional context may resolve references but is visibly separated from the
    target interaction, which remains the only evidence for a new memory. Source
    IDs are application-owned provenance: they are never shown to the LLM and
    are attached to each successfully parsed candidate after validation.
    """

    target_messages = _message_list_adapter.validate_python(messages)
    validated_source_ids = _source_message_ids_adapter.validate_python(source_message_ids)

    settings = get_config()
    if not settings.llm_model:
        raise ExtractionError("LLM_MODEL must be configured before extracting memories.")

    base_messages: list[dict[str, str]] = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": _request_content(target_messages, context)},
    ]

    last_failure: _ParseFailure | None = None
    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        request_messages = list(base_messages)
        if last_failure is not None:
            request_messages.append({"role": "user", "content": _repair_instruction(last_failure)})

        try:
            response = get_llm_client().chat.completions.create(
                model=settings.llm_model,
                messages=request_messages,
                temperature=0,
            )
        except Exception as error:
            # Provider/request-level failures (network, auth, rate limit,
            # timeout, ...) are out of scope for this repair-retry loop and
            # raise immediately, exactly as before this loop existed.
            raise ExtractionError("LLM extraction request failed.") from error

        content = _response_content(response)
        if content is None or not content.strip():
            last_failure = _ParseFailure("LLM extraction returned an empty response.", "invalid_json")
        else:
            try:
                result = _parse_result(content)
            except _ParseFailure as failure:
                last_failure = failure
            else:
                if attempt > 1:
                    logger.info("extraction succeeded after %d attempt(s)", attempt)
                provenance = list(validated_source_ids or [])
                return [
                    candidate.model_copy(update={"source_message_ids": provenance}) for candidate in result.memories
                ]

        if attempt < MAX_EXTRACTION_ATTEMPTS:
            logger.warning(
                "extraction attempt %d failed (%s), retrying (repair attempt %d/%d): %s",
                attempt,
                last_failure.category,
                attempt + 1,
                MAX_EXTRACTION_ATTEMPTS,
                last_failure,
            )

    assert last_failure is not None  # the loop only exits without returning after a recorded failure
    logger.warning(
        "extraction failed after %d attempts (%s): %s",
        MAX_EXTRACTION_ATTEMPTS,
        last_failure.category,
        last_failure,
    )
    final_error = ExtractionError(str(last_failure), category=last_failure.category, attempt_count=MAX_EXTRACTION_ATTEMPTS)
    if last_failure.cause is not None:
        raise final_error from last_failure.cause
    raise final_error
