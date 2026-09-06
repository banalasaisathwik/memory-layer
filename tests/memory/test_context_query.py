"""Unit tests for deterministic target-term lexical query construction."""

from meminfra.database import Message, MessageRole
from meminfra.memory.context import _target_lexical_query


def test_target_lexical_query_uses_bounded_distinctive_terms_with_or() -> None:
    targets = [
        Message(
            role=MessageRole.USER,
            content="The Atlas deployment is failing again, Atlas.",
        )
    ]

    assert _target_lexical_query(targets) == "atlas | deployment | failing | again"
