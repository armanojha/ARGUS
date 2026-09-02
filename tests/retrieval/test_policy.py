"""Phase 06 acceptance tests: adaptive retrieval policy (06.1) and active evidence seeking (06.2).

Covers the PHASE_06 spec's acceptance criteria:
- each question pattern routes to its documented V2 §5.2 / V3 §5 retrieval mix;
- retrieval executes through the policy router and returns real `EvidenceRef`s;
- WEB degrades to hybrid with an explicit fallback marker (never fabricated);
- the evidence-gap detector flags missing/low-quality evidence and formulates a
  targeted follow-up action without any LLM call;
- Obsidian hypotheses are exposed only as *research tasks*, never as evidence.

All policy logic is deterministic: a gateway failure must never change results.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, EvidenceRef, Source, SourceType
from app.evidence.store import EvidenceStore
from app.orchestration.models import ResearchPlan
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.policy import (
    DEFAULT_RETRIEVAL_POLICY,
    QuestionPattern,
    RetrievalMethod,
    RetrievalMix,
    RetrievalPolicy,
    get_default_retrieval_policy,
)
from app.retrieval.router import (
    RetrievalPolicyRouter,
    get_retrieval_policy_router,
    load_retrieval_policy,
)
from app.retrieval.seeking import AdaptiveEvidenceGapDetector, ObsidianHypothesisSeeker
from app.retrieval.vector import assign_embedding_indices


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def store(temp_dir: Path) -> EvidenceStore:
    return EvidenceStore(
        db_path=temp_dir / "evidence.db",
        bm25_index_path=temp_dir / "bm25.pkl",
        faiss_index_path=temp_dir / "faiss.index",
    )


@pytest.fixture
def populated_store(store: EvidenceStore) -> EvidenceStore:
    source = Source(type=SourceType.TEXT, path="/test/corpus.txt", checksum="c1")
    store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d1", chunking_strategy="fixed")
    store.insert_document(doc)
    store.insert_chunks(
        [
            Chunk(document_id=doc.id, ordinal=0, text="The quick brown fox jumps over the lazy dog.", token_count=10),
            Chunk(document_id=doc.id, ordinal=1, text="Foxes are members of the Canidae family.", token_count=8),
        ]
    )
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)
    return store


@pytest.fixture
def retriever(populated_store: EvidenceStore) -> HybridRetriever:
    return HybridRetriever(populated_store)


@pytest.fixture
def router(retriever: HybridRetriever) -> RetrievalPolicyRouter:
    return get_retrieval_policy_router(settings=Settings(_env_file=None))


# -- 06.1 classification + policy table ---------------------------------------

PATTERN_CASES = [
    ('What does "carbon footprint" mean?', QuestionPattern.EXACT_TERM),
    ("Define the term photosynthesis.", QuestionPattern.EXACT_TERM),
    ("How does photosynthesis work?", QuestionPattern.CONCEPTUAL),
    ("Why does the fox hunt at night?", QuestionPattern.CONCEPTUAL),
    ("What is the relationship between inflation and unemployment?", QuestionPattern.ENTITY_RELATIONSHIP),
    ("Explain the history of Rome before the empire era.", QuestionPattern.HISTORICAL),
    ("Give me a comprehensive report on quantum computing.", QuestionPattern.LONG_REPORT),
    ("Compare the revenue chart and the expense table for 2024.", QuestionPattern.MULTIMODAL),
    ("What are the latest updates on the 2024 elections?", QuestionPattern.FRESH_MISSING),
    ("How does the fox's behavior vary across seasons?", QuestionPattern.CONCEPTUAL),
    ("What is France's GDP growth rate?", QuestionPattern.CONCEPTUAL),
    ("Compare 'revenue' and 'expenses' in the chart.", QuestionPattern.MULTIMODAL),
]


@pytest.mark.parametrize("query,expected", PATTERN_CASES)
def test_classify_question_maps_each_pattern(query: str, expected: QuestionPattern):
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    assert router.classify_question(query) == expected


def test_get_retrieval_mix_matches_policy_table():
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    mix = router.get_retrieval_mix(QuestionPattern.CONCEPTUAL)
    assert isinstance(mix, RetrievalMix)
    assert RetrievalMethod.VECTOR in mix.methods
    assert RetrievalMethod.HYBRID in mix.methods


def test_uncovered_mechanisms_limits_hybrid_pass():
    """HARDEN-06.5.5: a hybrid method skips mechanisms an explicit method already covers."""
    exact_mix = RetrievalMix(methods=[RetrievalMethod.BM25, RetrievalMethod.HYBRID])
    conceptual_mix = RetrievalMix(methods=[RetrievalMethod.VECTOR, RetrievalMethod.HYBRID])
    hybrid_only_mix = RetrievalMix(methods=[RetrievalMethod.HYBRID])
    both_mix = RetrievalMix(methods=[RetrievalMethod.BM25, RetrievalMethod.VECTOR, RetrievalMethod.HYBRID])

    needed = RetrievalPolicyRouter._needed_mechanisms
    # EXACT_TERM is lexical-only: hybrid runs BM25 only, no query embedding.
    assert needed(exact_mix, 20) == {"bm25"}
    # CONCEPTUAL: VECTOR covers dense; hybrid only adds lexical.
    assert needed(conceptual_mix, 20) == {"bm25"}
    # HYBRID alone -> full hybrid (base case unchanged).
    assert needed(hybrid_only_mix, 20) is None
    # Both explicit -> hybrid fully redundant.
    assert needed(both_mix, 20) == set()


def test_default_policy_covers_all_documented_patterns():
    policy = get_default_retrieval_policy()
    table = {entry.pattern: entry.retrieval_mix for entry in policy.entries}
    # The V2 §5.2 documented patterns must each map to a distinct mix, else
    # the pattern silently falls back on the default hybrid mix (wrong route).
    documented = {
        QuestionPattern.EXACT_TERM,
        QuestionPattern.CONCEPTUAL,
        QuestionPattern.ENTITY_RELATIONSHIP,
        QuestionPattern.HISTORICAL,
        QuestionPattern.LONG_REPORT,
        QuestionPattern.FRESH_MISSING,
        QuestionPattern.MULTIMODAL,
    }
    assert documented <= set(table)
    # Undocumented enum members deterministically resolve to the default mix.
    for pattern in QuestionPattern:
        if pattern not in table:
            assert policy.get_mix_for_pattern(pattern) is policy.default_mix


def test_load_retrieval_policy_reads_config_yaml():
    policy = load_retrieval_policy(settings=Settings(_env_file=None))
    assert isinstance(policy, RetrievalPolicy)
    assert len(policy.entries) >= 6
    assert policy.get_mix_for_pattern(QuestionPattern.HISTORICAL) is not DEFAULT_RETRIEVAL_POLICY.default_mix


def test_missing_config_falls_back_to_default(tmp_path: Path):
    settings = Settings(_env_file=None, retrieval_policy_config_path=str(tmp_path / "missing.yaml"))
    assert load_retrieval_policy(settings=settings) is DEFAULT_RETRIEVAL_POLICY


# -- 06.1 retrieval dispatch ----------------------------------------------------


async def test_execute_retrieval_conceptual_returns_real_evidence(populated_store: EvidenceStore):
    retriever = HybridRetriever(populated_store)
    retriever.ensure_indexes()
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    results = await router.execute_retrieval("What does the fox do?", QuestionPattern.CONCEPTUAL, retriever)
    assert len(results) >= 1
    assert all(isinstance(r, EvidenceRef) for r in results)
    assert all(r.source_path.endswith("corpus.txt") for r in results)
    assert len({r.chunk_id for r in results}) == len(results)  # deduped by chunk_id


async def test_execute_retrieval_dedups_and_weights_are_consistent(populated_store: EvidenceStore):
    retriever = HybridRetriever(populated_store)
    retriever.ensure_indexes()
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    results = await router.execute_retrieval("foxes family canidae", QuestionPattern.EXACT_TERM, retriever)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # descending by fused score
    assert all("policy_score" in r.metadata for r in results)


async def test_exact_term_lookup_skips_vector_embedding(populated_store: EvidenceStore):
    """HARDEN-06.5.5: an EXACT_TERM pattern (BM25 + HYBRID) never embeds the query.

    The policy router reports it used only lexical coverage; the embedding call
    count must stay zero across the whole execution.
    """
    import numpy as np

    class CountingEmbedder:
        def __init__(self):
            self.embed_texts_calls = 0

        def embed_texts(self, texts):
            self.embed_texts_calls += len(texts)
            return np.zeros((len(texts), 384), dtype=np.float32)

        def embed_chunks(self, chunks):
            return np.zeros((len(chunks), 384), dtype=np.float32)

    retriever = HybridRetriever(populated_store)
    retriever.ensure_indexes()

    spy = CountingEmbedder()
    retriever.embedder = spy
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    results = await router.execute_retrieval("canidae", QuestionPattern.EXACT_TERM, retriever)
    assert results
    assert spy.embed_texts_calls == 0, "exact-term lookup must not pay for vector embedding"


async def test_web_method_degrades_to_hybrid_with_marker(populated_store: EvidenceStore):
    retriever = HybridRetriever(populated_store)
    retriever.ensure_indexes()
    router = get_retrieval_policy_router(settings=Settings(_env_file=None))
    results = await router.execute_retrieval(
        "What are the latest updates on foxes?", QuestionPattern.FRESH_MISSING, retriever
    )
    assert results
    # The WEB method must be explicitly marked as degraded so downstream
    # code never mistakes the result for fresh web evidence.
    assert any(r.metadata.get("policy_fallback") == "web->hybrid" for r in results)


async def test_empty_policy_mix_falls_back_to_hybrid(populated_store: EvidenceStore):
    retriever = HybridRetriever(populated_store)
    retriever.ensure_indexes()
    empty = RetrievalPolicy(entries=[])
    router = RetrievalPolicyRouter(policy=empty, settings=Settings(_env_file=None))
    results = await router.execute_retrieval("fox", QuestionPattern.CONCEPTUAL, retriever)
    assert results  # never empty-handed


# -- 06.2 evidence gap detection ----------------------------------------------


def _plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Explain fox behavior",
        entities=["fox"],
        time_window=None,
        subquestions=["fox behavior"],
        evidence_type="factual",
        preferred_retrieval_methods=["hybrid"],
        required_sources=[],
        risk_level="low",
        token_budget=1000,
        iteration_budget=2,
        stopping_condition="Stop when supported.",
    )


def _refs(scores: list[float]) -> list[EvidenceRef]:
    out = []
    for idx, score in enumerate(scores, 1):
        out.append(
            EvidenceRef(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_id=uuid4(),
                source_path="/test/corpus.txt",
                source_type=SourceType.TEXT,
                text="fox evidence",
                score=score,
                rank=idx,
                metadata={},
            )
        )
    return out


def _state(query: str = "Does the fox hibernate?", issued: list[str] | None = None) -> dict:
    return {
        "request_id": None,
        "query": query,
        "max_iterations": 2,
        "token_budget": 1000,
        "query_analysis": None,
        "plan": None,
        "pending_subquestions": [],
        "issued_subqueries": issued or [query],
        "evidence": [],
        "consecutive_empty_retrievals": 0,
        "iteration": 1,
        "tokens_used": 10,
        "sufficient": False,
        "stop_reason": None,
        "answer": None,
        "warnings": [],
        "question_pattern": "conceptual",
        "retrieval_gain_history": [1.0],
        "user_early_stop": False,
        "contradiction_signals": [],
        "stop_conditions_checked": [],
        "stop_condition_fired": None,
        "evidence_tasks": [],
    }


def test_gap_detector_quality_gap_formulates_targeted_action():
    settings = Settings(_env_file=None, active_seeking_quality_threshold=0.4, active_seeking_min_priority=0.5)
    detector = AdaptiveEvidenceGapDetector(settings=settings)
    state = _state()
    gaps = detector.detect_gaps(state, _plan(), _refs(scores=[0.12, 0.08]))
    by_type = {g["gap_type"]: g for g in gaps}
    assert "evidence_quality" in by_type
    assert by_type["evidence_quality"]["suggested_query"]  # targeted follow-up action
    assert detector.should_re_retrieve(gaps) is True


def test_gap_detector_missing_subquestion_evidence():
    settings = Settings(_env_file=None, active_seeking_min_priority=0.9)
    detector = AdaptiveEvidenceGapDetector(settings=settings)
    state = _state(issued=["unrelated query"])
    gaps = detector.detect_gaps(state, _plan(), _refs(scores=[0.9]))
    assert any(g["gap_type"] == "missing_evidence" for g in gaps)
    # Priority bar above the gap priorities -> policy says do not re-retrieve.
    assert detector.should_re_retrieve(gaps) is False


def test_gap_detector_no_evidence_flagged():
    settings = Settings(_env_file=None)
    detector = AdaptiveEvidenceGapDetector(settings=settings)
    gaps = detector.detect_gaps(_state(), _plan(), [])
    assert "no_evidence" in {g["gap_type"] for g in gaps}


def test_gap_detector_unresolved_contradiction():
    settings = Settings(_env_file=None)
    detector = AdaptiveEvidenceGapDetector(settings=settings)
    state = _state()
    state["contradiction_signals"] = [{"severity": 0.9, "resolved": False, "critical": True}]
    gaps = detector.detect_gaps(state, _plan(), _refs(scores=[0.8]))
    assert any(g["gap_type"] == "contradiction_unresolved" for g in gaps)


def test_obsidian_hypothesis_seeker_returns_tasks_not_evidence(store: EvidenceStore):
    source = Source(type=SourceType.TEXT, path="/vault/note.md", checksum="c9")
    store.upsert_source(source)
    doc = Document(
        source_id=source.id,
        version=1,
        checksum="d9",
        chunking_strategy="obsidian_sections_v1",
        metadata={"note_type": "personal_context", "vault_relative_path": "notes/my_hypothesis.md"},
    )
    store.insert_document(doc)
    store.insert_chunks(
        [
            Chunk(
                document_id=doc.id,
                ordinal=0,
                text="Hypothesis: foxes dig dens under south-facing slopes.",
                token_count=8,
                metadata={"vault_relative_path": "notes/my_hypothesis.md"},
            )
        ]
    )

    seeker = ObsidianHypothesisSeeker(store=store, task_priority=0.6)
    tasks = seeker.seed_evidence_tasks("Where do foxes build their dens?")
    assert tasks, "expected at least one hypothesis task for an overlapping query"
    task = tasks[0]
    assert task["gap_type"] == "obsidian_hypothesis"
    assert task["suggested_query"]
    assert task["note_path"] == "notes/my_hypothesis.md"
    # A task is a query to run — it must not carry hypothesis text as evidence.
    assert "hypothesis_text" in task


def test_obsidian_seeker_ignores_unrelated_query(store: EvidenceStore):
    source = Source(type=SourceType.TEXT, path="/vault/note.md", checksum="c9")
    store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d9", chunking_strategy="fixed")
    store.insert_document(doc)
    store.insert_chunks(
        [Chunk(document_id=doc.id, ordinal=0, text="Hypothesis: foxes dig dens under south-facing slopes.", token_count=8)]
    )
    seeker = ObsidianHypothesisSeeker(store=store)
    assert seeker.seed_evidence_tasks("pythagorean theorem") == []


def test_obsidian_seeker_uses_phase09_hypothesis_classification(store: EvidenceStore):
    """Phase 09 classifier-annotated chunks seed tasks even without an inline marker."""
    source = Source(type=SourceType.TEXT, path="/vault/note.md", checksum="e1")
    store.upsert_source(source)
    doc = Document(
        source_id=source.id,
        version=1,
        checksum="e2",
        chunking_strategy="obsidian_sections_v1",
        metadata={"note_type": "personal_context", "vault_relative_path": "notes/idea.md"},
    )
    store.insert_document(doc)
    store.insert_chunks(
        [
            Chunk(
                document_id=doc.id,
                ordinal=0,
                text="Foxes dig dens under south-facing slopes.",
                token_count=8,
                metadata={"knowledge_class": "hypothesis", "vault_relative_path": "notes/idea.md"},
            )
        ]
    )

    seeker = ObsidianHypothesisSeeker(store=store)
    tasks = seeker.seed_evidence_tasks("Where do foxes build their dens?")
    assert tasks, "expected a task from a Phase 09-classified hypothesis chunk"
    task = tasks[0]
    assert task["gap_type"] == "obsidian_hypothesis"
    assert "Foxes dig dens" in task["hypothesis_text"]
    assert task["suggested_query"].startswith("Investigate whether:")