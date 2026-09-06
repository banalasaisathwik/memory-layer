"""FTS-vs-BM25 lexical ablation over a small, deterministic fixture.

Compares the retained ``lexical_retrieve`` (PostgreSQL
``websearch_to_tsquery``/``ts_rank_cd``) branch against ``bm25_retrieve``
(genuine Okapi BM25) on the same scoped corpus and queries, across the
categories called for in the BM25 milestone: exact entity lookup,
natural-language question, partial keyword overlap, paraphrase with partial
lexical overlap, and an identifier-heavy query. This is not a new benchmark
framework -- just a focused, printable comparison over one fixture.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from meminfra.config import configure, reset_config
from meminfra.database import Conversation, Memory, MemoryType, SessionLocal, User, create_tables, reset_engine
from meminfra.retrieval import SearchFilters, bm25_retrieve, lexical_retrieve
from meminfra.retrieval.search import search_memories


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="TEST_DATABASE_URL is not set; lexical ablation never uses DATABASE_URL.",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def configured_test_database() -> None:
    reset_config()
    reset_engine()
    configure(database_url=TEST_DATABASE_URL)
    create_tables()
    yield
    reset_engine()
    reset_config()


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session
        session.rollback()


def _user(db) -> User:
    user = User(external_id=f"ablation-user-{uuid4().hex}")
    db.add(user)
    db.commit()
    return user


def _memory(db, user: User, text: str) -> Memory:
    memory = Memory(
        user_id=user.id,
        memory_type=MemoryType.SEMANTIC,
        memory_text=text,
        source_message_ids=[],
    )
    db.add(memory)
    db.commit()
    return memory


def _ranked_ids(db, user: User, query: str, *, backend) -> list[str]:
    filters = SearchFilters()
    hits = backend(db, query, user=user, filters=filters, conversation=None, limit=10)
    return [str(memory.id) for memory in hits]


@pytest.fixture
def gina_fixture(db) -> dict[str, Memory | User]:
    user = _user(db)
    a = _memory(db, user, "Gina opened an online clothing store.")
    b = _memory(db, user, "Gina bought furniture for her store.")
    c = _memory(db, user, "Jon changed jobs.")
    d = _memory(db, user, "Gina filed ticket PROJ-42 for the store.")
    return {"user": user, "a": a, "b": b, "c": c, "d": d}


def test_exact_entity_lookup_both_backends_find_gina(db, gina_fixture) -> None:
    # "d" also mentions Gina ("Gina filed ticket PROJ-42..."), so all three
    # Gina memories -- not just A and B -- are the expected match set here.
    user, a, b, c, d = (
        gina_fixture["user"],
        gina_fixture["a"],
        gina_fixture["b"],
        gina_fixture["c"],
        gina_fixture["d"],
    )

    fts = _ranked_ids(db, user, "Gina", backend=lexical_retrieve)
    bm25 = _ranked_ids(db, user, "Gina", backend=bm25_retrieve)

    assert set(fts) == {str(a.id), str(b.id), str(d.id)}
    assert set(bm25) == {str(a.id), str(b.id), str(d.id)}
    assert str(c.id) not in fts and str(c.id) not in bm25


def test_natural_language_question_fts_fails_bm25_ranks_relevant_memories(db, gina_fixture) -> None:
    """The exact scenario motivating Problem B: FTS's implicit AND over every
    word (including "what"/"does", since the "simple" dictionary keeps
    stopwords) matches nothing; BM25 still ranks A over B and excludes C."""

    user, a, b, c = gina_fixture["user"], gina_fixture["a"], gina_fixture["b"], gina_fixture["c"]
    query = "What clothing business does Gina run?"

    fts = _ranked_ids(db, user, query, backend=lexical_retrieve)
    bm25 = _ranked_ids(db, user, query, backend=bm25_retrieve)

    assert fts == []
    assert bm25[0] == str(a.id)
    assert str(b.id) in bm25
    assert str(c.id) not in bm25


def test_partial_keyword_overlap_fts_fails_bm25_still_ranks_the_match(db, gina_fixture) -> None:
    """Neither memory contains the literal word "business"; FTS's AND
    semantics reject every candidate, while BM25 still credits the
    "clothing" overlap and ranks A above unrelated memories."""

    user, a = gina_fixture["user"], gina_fixture["a"]
    query = "clothing business"

    fts = _ranked_ids(db, user, query, backend=lexical_retrieve)
    bm25 = _ranked_ids(db, user, query, backend=bm25_retrieve)

    assert fts == []
    assert bm25 == [str(a.id)]


def test_paraphrase_with_partial_overlap_fts_fails_bm25_succeeds(db, gina_fixture) -> None:
    """"retail" appears in none of the memories; FTS's AND over all query
    terms rejects everything, while BM25 still matches on "gina" and
    "store" -- which A, B, and D (also "...for the store") all contain."""

    user, a, b, d = gina_fixture["user"], gina_fixture["a"], gina_fixture["b"], gina_fixture["d"]
    query = "Gina's retail store"

    fts = _ranked_ids(db, user, query, backend=lexical_retrieve)
    bm25 = _ranked_ids(db, user, query, backend=bm25_retrieve)

    assert fts == []
    assert set(bm25) == {str(a.id), str(b.id), str(d.id)}


def test_identifier_heavy_query_both_backends_agree(db, gina_fixture) -> None:
    """Both backends handle a punctuation-heavy identifier: FTS via its
    literal ILIKE fallback, BM25 via plain word tokenization of "PROJ-42"."""

    user, d = gina_fixture["user"], gina_fixture["d"]

    fts = _ranked_ids(db, user, "PROJ-42", backend=lexical_retrieve)
    bm25 = _ranked_ids(db, user, "PROJ-42", backend=bm25_retrieve)

    assert fts == [str(d.id)]
    assert bm25[0] == str(d.id)


def test_ablation_report(db, gina_fixture, capsys) -> None:
    """Not an assertion-bearing test -- prints the side-by-side ranking table
    the milestone's final report references (run with `pytest -s` to see it)."""

    user = gina_fixture["user"]
    names = {
        str(gina_fixture["a"].id): "A (clothing store)",
        str(gina_fixture["b"].id): "B (furniture)",
        str(gina_fixture["c"].id): "C (Jon)",
        str(gina_fixture["d"].id): "D (PROJ-42)",
    }
    queries = [
        "Gina",
        "What clothing business does Gina run?",
        "clothing business",
        "Gina's retail store",
        "PROJ-42",
    ]
    print("\nFTS vs BM25 lexical ablation")
    for query in queries:
        fts = [names[i] for i in _ranked_ids(db, user, query, backend=lexical_retrieve)]
        bm25 = [names[i] for i in _ranked_ids(db, user, query, backend=bm25_retrieve)]
        print(f"  query={query!r}\n    FTS:  {fts}\n    BM25: {bm25}")


# --------------------------------------------------------------------------
# Part B parity: evaluate_question_shared()'s fused results must exactly
# match what search_memories() itself would return for the same query and
# default filters, for both lexical backends. Embeddings are mocked so the
# comparison is deterministic and makes no real provider calls.
# --------------------------------------------------------------------------


def _deterministic_vector(text: str, *, dimension: int = 8):
    import numpy as np

    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension).astype(np.float32)
    return vector


@pytest.fixture(autouse=False)
def deterministic_embeddings(monkeypatch):
    """Replace the real embedding provider with a deterministic, offline stand-in.

    Patches ``meminfra.retrieval.vector._embedding_response_vectors`` -- the single
    choke point both ``vector_retrieve`` (used by production ``search_memories``)
    and this ablation's shared vector branch call through -- so both sides of
    every parity comparison see identical embeddings with no network call.
    """

    import numpy as np

    import meminfra.retrieval.vector as vector_module

    def fake_embedding_response_vectors(texts: list[str]) -> list["np.ndarray"]:
        return [_deterministic_vector(text) / np.linalg.norm(_deterministic_vector(text)) for text in texts]

    monkeypatch.setattr(vector_module, "_embedding_response_vectors", fake_embedding_response_vectors)
    return _deterministic_vector


@pytest.fixture
def parity_fixture(db) -> dict:
    user = _user(db)
    a = _memory(db, user, "Gina opened an online clothing store.")
    b = _memory(db, user, "Gina bought furniture for her store.")
    c = _memory(db, user, "Jon changed jobs.")
    d = _memory(db, user, "Gina filed ticket PROJ-42 for the store.")
    return {"user": user, "a": a, "b": b, "c": c, "d": d}


def _shared_benchmark_state(db, *, user):
    from evals.locomo.benchmark_state import BenchmarkState, GoldMemoryLookup
    from evals.locomo.bm25_corpus import prepare_bm25_corpus
    from sqlalchemy import select as sa_select

    from meminfra.database.models import Memory as MemoryModel

    active_memories = list(
        db.scalars(
            sa_select(MemoryModel)
            .where(MemoryModel.user_id == user.id, MemoryModel.is_active.is_(True))
            .order_by(MemoryModel.created_at.asc(), MemoryModel.id.asc())
        )
    )
    bm25_corpus = prepare_bm25_corpus(active_memories, text_of=lambda m: m.memory_text)
    return BenchmarkState(
        user_id=user.external_id,
        user=user,
        conversation=None,
        active_memories=active_memories,
        active_memory_by_id={str(m.id): m for m in active_memories},
        bm25_corpus=bm25_corpus,
        message_id_to_dia_id={},
        covered_dia_ids=set(),
        gold_lookup=GoldMemoryLookup(memories=active_memories, dia_ids_by_memory_id={}),
    )


@pytest.mark.parametrize(
    "query",
    [
        "Gina",
        "What clothing business does Gina run?",
        "clothing business",
        "Gina's retail store",
        "PROJ-42",
    ],
)
def test_shared_ablation_fts_matches_production_search_memories(
    db, parity_fixture, deterministic_embeddings, query
) -> None:
    from evals.locomo.ablate_lexical import evaluate_question_shared
    from evals.locomo.schemas import CATEGORY_NAMES, LocomoQA, LocomoSample, LocomoTurn

    user = parity_fixture["user"]
    state = _shared_benchmark_state(db, user=user)
    sample = LocomoSample(
        sample_id="parity-sample",
        speaker_a="A",
        speaker_b="B",
        turns=[LocomoTurn(dia_id="d1#1", speaker="A", text="hi", session_number=1, session_date_time="1:00 pm on 1 January, 2024")],
        qa=[LocomoQA(question=query, category_id=4, evidence=[])],
    )
    query_vector = deterministic_embeddings(query)
    import numpy as np

    query_vector = query_vector / np.linalg.norm(query_vector)

    result = evaluate_question_shared(
        db, sample=sample, question_index=0, state=state, top_k=10, query_vector=query_vector
    )

    production_hits = search_memories(
        db, query, user_external_id=user.external_id, limit=10, lexical_backend="postgres_fts", fusion_strategy="rrf"
    )
    production_ids = [hit.memory_id for hit in production_hits]

    assert result.fts_fused_ids == production_ids


@pytest.mark.parametrize(
    "query",
    [
        "Gina",
        "What clothing business does Gina run?",
        "clothing business",
        "Gina's retail store",
        "PROJ-42",
    ],
)
def test_shared_ablation_bm25_matches_production_search_memories(
    db, parity_fixture, deterministic_embeddings, query
) -> None:
    from evals.locomo.ablate_lexical import evaluate_question_shared
    from evals.locomo.schemas import LocomoQA, LocomoSample, LocomoTurn

    user = parity_fixture["user"]
    state = _shared_benchmark_state(db, user=user)
    sample = LocomoSample(
        sample_id="parity-sample",
        speaker_a="A",
        speaker_b="B",
        turns=[LocomoTurn(dia_id="d1#1", speaker="A", text="hi", session_number=1, session_date_time="1:00 pm on 1 January, 2024")],
        qa=[LocomoQA(question=query, category_id=4, evidence=[])],
    )
    query_vector = deterministic_embeddings(query)
    import numpy as np

    query_vector = query_vector / np.linalg.norm(query_vector)

    result = evaluate_question_shared(
        db, sample=sample, question_index=0, state=state, top_k=10, query_vector=query_vector
    )

    production_hits = search_memories(
        db, query, user_external_id=user.external_id, limit=10, lexical_backend="bm25", fusion_strategy="rrf"
    )
    production_ids = [hit.memory_id for hit in production_hits]

    assert result.bm25_fused_ids == production_ids


def test_shared_ablation_computes_vector_branch_once_per_question(
    db, parity_fixture, deterministic_embeddings, monkeypatch
) -> None:
    """The whole point of Part B: one vector branch call feeds both backends."""

    from evals.locomo.ablate_lexical import evaluate_question_shared
    from evals.locomo.schemas import LocomoQA, LocomoSample, LocomoTurn
    import evals.locomo.ablate_lexical as ablate_module
    import numpy as np

    user = parity_fixture["user"]
    state = _shared_benchmark_state(db, user=user)
    query = "Gina's retail store"
    sample = LocomoSample(
        sample_id="parity-sample",
        speaker_a="A",
        speaker_b="B",
        turns=[LocomoTurn(dia_id="d1#1", speaker="A", text="hi", session_number=1, session_date_time="1:00 pm on 1 January, 2024")],
        qa=[LocomoQA(question=query, category_id=4, evidence=[])],
    )

    call_count = {"n": 0}
    original = ablate_module._vector_candidates_from_embedding

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ablate_module, "_vector_candidates_from_embedding", counting_wrapper)

    query_vector = deterministic_embeddings(query)
    query_vector = query_vector / np.linalg.norm(query_vector)
    ablate_module.evaluate_question_shared(
        db, sample=sample, question_index=0, state=state, top_k=10, query_vector=query_vector
    )

    assert call_count["n"] == 1
