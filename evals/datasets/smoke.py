"""Hand-written, deterministic smoke cases exercising the current memory layer.

These cases are read-only data: the runner ingests each case's messages
through the real extraction/write pipeline and queries search_memories(). No
production retrieval or write behavior is changed to make a case pass.
"""

from __future__ import annotations

from evals.schemas import EvalCase, EvalMessage, ExpectedMemory


def _messages(*turns: tuple[str, str]) -> list[EvalMessage]:
    return [EvalMessage(role=role, content=content) for role, content in turns]


SMOKE_CASES: list[EvalCase] = [
    EvalCase(
        id="preference_001",
        category="single_fact",
        description="A single, directly stated preference.",
        messages=_messages(("user", "I prefer PostgreSQL over MongoDB.")),
        query="Which database does the user prefer?",
        expected_memories=[ExpectedMemory(required_terms=["postgresql"])],
    ),
    EvalCase(
        id="location_001",
        category="single_fact",
        description="A single, directly stated personal location.",
        messages=_messages(("user", "I currently live in Hyderabad.")),
        query="Where does the user live?",
        expected_memories=[ExpectedMemory(required_terms=["hyderabad"])],
    ),
    EvalCase(
        id="location_update_001",
        category="update",
        description="A single-valued fact changes; only the newest value should be current.",
        messages=_messages(
            ("user", "I live in Hyderabad."),
            ("user", "I moved to Bengaluru."),
        ),
        query="Where does the user live now?",
        expected_memories=[ExpectedMemory(required_terms=["bengaluru"])],
    ),
    EvalCase(
        id="timezone_update_001",
        category="update",
        description="A second single-valued supersession case with a different predicate.",
        messages=_messages(
            ("user", "I'm currently in the IST timezone."),
            ("user", "Actually, I've relocated and I'm now in the PST timezone."),
        ),
        query="What timezone is the user in now?",
        expected_memories=[ExpectedMemory(required_terms=["pst"])],
    ),
    EvalCase(
        id="multi_value_001",
        category="multi_value",
        description="Two distinct values for a multi-valued predicate should both remain retrievable.",
        messages=_messages(
            ("user", "I know Python."),
            ("user", "I also work with TypeScript."),
        ),
        query="Which programming languages does the user know?",
        expected_memories=[
            ExpectedMemory(required_terms=["python"]),
            ExpectedMemory(required_terms=["typescript"]),
        ],
    ),
    EvalCase(
        id="multi_fact_001",
        category="multi_fact",
        description="Two unrelated facts stated in separate turns should both be retrievable.",
        messages=_messages(
            ("user", "My name is Asha."),
            ("user", "I live in Pune."),
        ),
        query="What do we know about the user?",
        expected_memories=[
            ExpectedMemory(required_terms=["asha"]),
            ExpectedMemory(required_terms=["pune"]),
        ],
    ),
    EvalCase(
        id="paraphrase_duplicate_001",
        category="duplicate_paraphrase",
        description=(
            "The same fact stated twice in different words. This is a baseline observation for "
            "later semantic-dedup work, not something this milestone fixes."
        ),
        messages=_messages(
            ("user", "I enjoy Python programming."),
            ("user", "By the way, Python is one of my favorite languages to work with."),
        ),
        query="What programming language does the user enjoy?",
        expected_memories=[ExpectedMemory(required_terms=["python"])],
    ),
    EvalCase(
        id="cross_turn_reference_001",
        category="cross_turn_reference",
        description="Resolving 'it' requires the extraction context, not just the target message.",
        messages=_messages(
            ("user", "I'm building a project called MemoryLayer."),
            ("assistant", "Sounds interesting."),
            ("user", "I'm using PostgreSQL for it."),
        ),
        query="Which database is used for MemoryLayer?",
        expected_memories=[
            ExpectedMemory(required_terms=["postgresql"]),
            ExpectedMemory(
                required_terms=["memorylayer", "postgresql"],
                description="Stricter gold: did the memory text itself resolve the reference?",
            ),
        ],
    ),
    EvalCase(
        id="distractor_001",
        category="irrelevant_distractor",
        description="Small talk surrounds the one fact the query actually asks about.",
        messages=_messages(
            ("user", "Good morning!"),
            ("user", "It's really sunny today."),
            ("user", "I really enjoy playing chess in my free time."),
            ("user", "Anyway, have a good one."),
        ),
        query="What is the user's hobby?",
        expected_memories=[ExpectedMemory(required_terms=["chess"])],
    ),
    EvalCase(
        id="user_isolation_001",
        category="user_isolation",
        description="A second user's memories must never appear in this user's search results.",
        messages=_messages(("user", "I prefer PostgreSQL over MongoDB.")),
        isolation_messages=_messages(("user", "I prefer MongoDB over PostgreSQL.")),
        query="Which database does the user prefer?",
        expected_memories=[ExpectedMemory(required_terms=["postgresql"])],
    ),
]
