"""Unit coverage for strict, registry-backed query intent extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from meminfra.config import configure, reset_config
from meminfra.memory.predicates import PREDICATES
from meminfra.memory.prompts import build_query_intent_system_prompt
from meminfra.retrieval import QueryIntentError, extract_query_intent


class FakeCompletions:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


@pytest.fixture
def fake_query_provider(monkeypatch: pytest.MonkeyPatch):
    reset_config()
    configure(llm_model="fake-query-model")
    completions = FakeCompletions(["{}"])
    monkeypatch.setattr("meminfra.retrieval.query_intent.get_llm_client", lambda: FakeClient(completions))
    yield completions
    reset_config()


@pytest.mark.parametrize(
    ("query", "payload", "predicate", "value", "temporal_scope"),
    [
        ("Where do I live?", {"predicate": "location", "value": None, "temporal_scope": "current"}, "location", None, "current"),
        ("What database do I prefer?", {"predicate": "database_preference", "value": None, "temporal_scope": "current"}, "database_preference", None, "current"),
        ("Do I know Python?", {"predicate": "programming_language", "value": "Python", "temporal_scope": "current"}, "programming_language", "Python", "current"),
        ("Where did I live before Delhi?", {"predicate": "location", "value": None, "temporal_scope": "historical"}, "location", None, "historical"),
        ("What was I debugging yesterday?", {"predicate": None, "value": None, "temporal_scope": "current"}, None, None, "current"),
    ],
)
def test_extract_query_intent_parses_canonical_examples(
    fake_query_provider: FakeCompletions,
    query: str,
    payload: dict[str, object],
    predicate: str | None,
    value: str | None,
    temporal_scope: str,
) -> None:
    fake_query_provider.responses = [json.dumps(payload)]

    intent = extract_query_intent(query)

    assert (intent.predicate, intent.value, intent.temporal_scope) == (predicate, value, temporal_scope)


def test_query_intent_prompt_uses_the_controlled_registry_dynamically(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(PREDICATES, "dynamic_test_predicate", {"cardinality": "single"})

    prompt = build_query_intent_system_prompt()

    assert "location" in prompt
    assert "database_preference" in prompt
    assert "programming_language" in prompt
    assert "dynamic_test_predicate" in prompt
    assert "exact canonical name" in prompt


def test_invalid_json_is_repaired_once_valid_output_arrives(fake_query_provider: FakeCompletions) -> None:
    fake_query_provider.responses = ["not json", json.dumps({"predicate": "location", "value": None, "temporal_scope": "current"})]

    assert extract_query_intent("Where do I live?").predicate == "location"
    assert len(fake_query_provider.calls) == 2


def test_provider_failure_is_typed(fake_query_provider: FakeCompletions) -> None:
    fake_query_provider.responses = [RuntimeError("unavailable")]

    with pytest.raises(QueryIntentError, match="request failed") as error:
        extract_query_intent("Where do I live?")

    assert error.value.category == "provider"
