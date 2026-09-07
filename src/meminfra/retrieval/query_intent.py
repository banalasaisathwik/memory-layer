"""Best-effort LLM query understanding for the optional structured search branch."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from meminfra.config import get_config
from meminfra.providers import get_llm_client

from .schemas import QueryIntent


logger = logging.getLogger(__name__)
MAX_QUERY_INTENT_ATTEMPTS = 3
_OUTER_MARKDOWN_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(?P<body>.*)\n```", re.IGNORECASE | re.DOTALL)


class QueryIntentError(Exception):
    """Raised when optional query understanding cannot produce a valid intent."""

    def __init__(self, message: str, *, category: str = "other") -> None:
        super().__init__(message)
        self.category = category


def _response_content(response: Any) -> str | None:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _parse_intent(content: str) -> QueryIntent:
    candidate = content.strip()
    match = _OUTER_MARKDOWN_FENCE.fullmatch(candidate)
    if match:
        candidate = match.group("body")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise QueryIntentError("LLM query intent returned invalid JSON.", category="invalid_json") from error
    if not isinstance(payload, dict):
        raise QueryIntentError("LLM query intent response must be a JSON object.", category="invalid_json")
    try:
        return QueryIntent.model_validate(payload)
    except ValidationError as error:
        raise QueryIntentError("LLM query intent response does not match the required schema.", category="schema_validation") from error


def _repair_instruction(error: QueryIntentError) -> str:
    return (
        "Your previous response could not be used (" + error.category + "). "
        "Return exactly one corrected JSON object matching the requested schema, with no markdown or prose."
    )


def extract_query_intent(query: str) -> QueryIntent:
    """Return a validated structured hint, retrying only malformed model output.

    This intentionally has no access to a database or caller filters. Its result
    is evidence for a later retrieval branch, never a replacement for filters.
    """

    settings = get_config()
    if not settings.llm_model:
        raise QueryIntentError("LLM_MODEL must be configured before inferring query intent.", category="configuration")

    # Deferred to avoid coupling retrieval module import order to memory's
    # public package exports, while still sharing the one predicate registry.
    from meminfra.memory.prompts import build_query_intent_system_prompt

    base_messages = [
        {"role": "system", "content": build_query_intent_system_prompt()},
        {"role": "user", "content": json.dumps({"query": query}, ensure_ascii=False)},
    ]
    last_error: QueryIntentError | None = None
    for attempt in range(1, MAX_QUERY_INTENT_ATTEMPTS + 1):
        messages = list(base_messages)
        if last_error is not None:
            messages.append({"role": "user", "content": _repair_instruction(last_error)})
        try:
            response = get_llm_client().chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0,
            )
        except Exception as error:
            raise QueryIntentError("LLM query intent request failed.", category="provider") from error

        content = _response_content(response)
        if content is not None and content.strip():
            try:
                return _parse_intent(content)
            except QueryIntentError as error:
                last_error = error
        else:
            last_error = QueryIntentError("LLM query intent returned an empty response.", category="invalid_json")

        if attempt < MAX_QUERY_INTENT_ATTEMPTS:
            logger.warning("query intent attempt %d failed (%s); retrying", attempt, last_error.category)

    assert last_error is not None
    raise last_error
