"""Unit coverage for query-intent flow through the public facade."""

from __future__ import annotations

from meminfra.memory_layer import MemoryLayer


class StubMemoryLayer(MemoryLayer):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
        filters=None,
        infer_query_intent: bool = True,
    ) -> list:
        self.calls.append(
            {
                "user_id": user_id,
                "query": query,
                "limit": limit,
                "filters": filters,
                "infer_query_intent": infer_query_intent,
            }
        )
        return []


def test_answer_calls_search_once_and_forwards_query_intent_switch() -> None:
    memory = StubMemoryLayer()

    result = memory.answer(
        user_id="user-123",
        query="Where do I live?",
        infer_query_intent=False,
    )

    assert result.abstained is True
    assert memory.calls == [
        {
            "user_id": "user-123",
            "query": "Where do I live?",
            "limit": 5,
            "filters": None,
            "infer_query_intent": False,
        }
    ]
