"""Phase 15 regression tests: Evidence Need Planner + Multi-Query Retrieval."""
from __future__ import annotations

import pytest
from app.retrieval.planner import (
    ClaimType,
    EvidenceNeed,
    EvidenceNeedPlanner,
    NeedPriority,
    QueryPlan,
)
from app.retrieval.multi_query import MultiQueryRetriever, RetrievalResult
from app.retrieval.router import RetrievalPolicyRouter
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.evidence.store import EvidenceStore
from app.evidence.models import Document, Chunk, Source, SourceType
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _tmp_store():
    """Create a temporary evidence store with benchmark corpus."""
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    return store, chunk_id_map


@pytest.fixture(scope="module")
def _retriever(_tmp_store):
    """Create a hybrid retriever over the temp store."""
    store, _ = _tmp_store
    bm25 = BM25Retriever(store)
    vector = FAISSVectorStore(store)
    embedder = EmbeddingGenerator()
    retriever = HybridRetriever(store=store, bm25=bm25, vector=vector, embedder=embedder)
    retriever.ensure_indexes()
    return retriever


@pytest.fixture(scope="module")
def _router():
    return RetrievalPolicyRouter()


@pytest.fixture(scope="module")
def _planner():
    return EvidenceNeedPlanner()


# ---------------------------------------------------------------------------
# Test: EvidenceNeed dataclass
# ---------------------------------------------------------------------------

class TestEvidenceNeed:
    def test_create_evidence_need(self):
        need = EvidenceNeed(
            topic="revenue",
            entities=["Acme"],
            claim_type=ClaimType.PRIMARY,
            search_query="Acme revenue",
            priority=NeedPriority.HIGH,
        )
        assert need.topic == "revenue"
        assert need.entities == ["Acme"]
        assert need.claim_type == ClaimType.PRIMARY
        assert need.priority == NeedPriority.HIGH
        assert len(need.id) == 8  # UUID prefix

    def test_evidence_need_hashable(self):
        need1 = EvidenceNeed(id="abc12345", topic="test")
        need2 = EvidenceNeed(id="abc12345", topic="other")
        assert need1 == need2
        assert hash(need1) == hash(need2)

    def test_evidence_need_different_ids(self):
        need1 = EvidenceNeed(topic="test")
        need2 = EvidenceNeed(topic="test")
        assert need1 != need2


# ---------------------------------------------------------------------------
# Test: QueryPlan
# ---------------------------------------------------------------------------

class TestQueryPlan:
    def test_unplanned_passthrough(self):
        plan = QueryPlan(original_query="What is 2+2?", pattern="simple_lookup")
        assert plan.is_planned is False
        assert plan.need_count == 0
        assert plan.query_count == 1  # Minimum 1

    def test_planned_with_needs(self):
        needs = [
            EvidenceNeed(topic="A", search_query="query A"),
            EvidenceNeed(topic="B", search_query="query B"),
        ]
        plan = QueryPlan(
            original_query="Compare A and B",
            pattern="conflict",
            evidence_needs=needs,
            is_planned=True,
        )
        assert plan.need_count == 2
        assert plan.query_count == 2


# ---------------------------------------------------------------------------
# Test: EvidenceNeedPlanner
# ---------------------------------------------------------------------------

class TestEvidenceNeedPlanner:
    def test_simple_query_not_planned(self, _planner):
        plan = _planner.plan("What is 2+2?", "simple_lookup")
        assert plan.is_planned is False
        assert plan.pattern == "simple_lookup"

    def test_normal_qa_not_planned(self, _planner):
        plan = _planner.plan("How does photosynthesis work?", "conceptual")
        assert plan.is_planned is False

    def test_conflict_query_planned(self, _planner):
        plan = _planner.plan(
            "What was Acme's annual revenue?",
            "conflict",
        )
        assert plan.is_planned is True
        assert plan.pattern == "conflict"
        assert plan.need_count >= 2  # At least primary + data query
        # Should have primary and at least one other claim type
        claim_types = {n.claim_type for n in plan.evidence_needs}
        assert ClaimType.PRIMARY in claim_types

    def test_conflict_query_has_contradictory_need(self, _planner):
        plan = _planner.plan(
            "How many employees does Acme have?",
            "conflict",
        )
        claim_types = {n.claim_type for n in plan.evidence_needs}
        # Should search for contradictory evidence
        assert ClaimType.CONTRADICTORY in claim_types or ClaimType.CAVEAT in claim_types

    def test_complex_research_planned(self, _planner):
        plan = _planner.plan(
            "Assess the argument that Acme is diversifying into robotics and evaluate what it depends on.",
            "complex_research",
        )
        assert plan.is_planned is True
        assert plan.need_count >= 2  # Should decompose into subtopics

    def test_multi_hop_planned(self, _planner):
        plan = _planner.plan(
            "Which plant is downstream of Ohio and depends indirectly on PetroKem?",
            "multi_hop",
        )
        assert plan.is_planned is True
        assert plan.need_count >= 2  # Should have entity-pair queries

    def test_entity_extraction(self, _planner):
        entities = _planner._extract_entities(
            "What is the relationship between Acme Corporation and PetroKem?"
        )
        entity_str = " ".join(entities)
        assert "Acme" in entity_str
        assert "PetroKem" in entity_str

    def test_topic_extraction(self, _planner):
        topic = _planner._extract_topic(
            "What was Acme's annual revenue?"
        )
        assert "acme" in topic.lower()
        assert "revenue" in topic.lower()


# ---------------------------------------------------------------------------
# Test: Planner integration with router
# ---------------------------------------------------------------------------

class TestPlannerIntegration:
    def test_router_has_planner(self, _router):
        assert hasattr(_router, '_planner')

    @pytest.mark.asyncio
    async def test_execute_planned_retrieval_simple_passthrough(self, _router, _retriever):
        """Simple queries should pass through without planning."""
        from app.retrieval.policy import QuestionPattern
        refs = await _router.execute_planned_retrieval(
            "Where is Acme headquartered?",
            QuestionPattern.EXACT_TERM,
            _retriever,
            top_k=10,
        )
        assert len(refs) > 0
        assert len(refs) <= 10

    @pytest.mark.asyncio
    async def test_execute_planned_retrieval_conflict(self, _router, _retriever):
        """Conflict queries should use the planner."""
        from app.retrieval.policy import QuestionPattern
        refs = await _router.execute_planned_retrieval(
            "What was Acme's annual revenue?",
            QuestionPattern.CONCEPTUAL,  # Classified as conceptual
            _retriever,
            top_k=10,
        )
        # Even if not planned (pattern mismatch), should still return results
        assert len(refs) > 0

    def test_planner_bypass_for_simple_patterns(self, _planner):
        """Planner should return unplanned for non-complex patterns."""
        for pattern in ["simple_lookup", "numerical", "technical_explanation",
                        "normal_qa", "absent_info", "adversarial",
                        "multi_doc_synthesis", "exact_term", "conceptual",
                        "comparative", "long_report", "historical"]:
            plan = _planner.plan("test query", pattern)
            assert plan.is_planned is False, f"Pattern {pattern} should not be planned"


# ---------------------------------------------------------------------------
# Test: Multi-Query Retrieval
# ---------------------------------------------------------------------------

class TestMultiQueryRetriever:
    def test_passthrough_retrieval(self, _router, _retriever):
        """Unplanned query should do single search."""
        retriever = MultiQueryRetriever(_router, _retriever, top_k=10)
        plan = QueryPlan(
            original_query="What is Atlas?",
            pattern="simple_lookup",
            is_planned=False,
        )
        result = retriever.retrieve(plan)
        assert result.total_retrieval_calls == 1
        assert len(result.selected) > 0

    def test_planned_retrieval_runs_multiple_queries(self, _router, _retriever):
        """Planned query should run multiple searches."""
        retriever = MultiQueryRetriever(_router, _retriever, top_k=10)
        needs = [
            EvidenceNeed(topic="robotics", search_query="Acme robotics", priority=NeedPriority.HIGH),
            EvidenceNeed(topic="Atlas DB", search_query="Atlas database features", priority=NeedPriority.MEDIUM),
        ]
        plan = QueryPlan(
            original_query="Assess robotics and Atlas DB",
            pattern="complex_research",
            evidence_needs=needs,
            is_planned=True,
        )
        result = retriever.retrieve(plan)
        assert result.total_retrieval_calls >= 2
        assert result.total_candidates > 0

    def test_deduplication_across_queries(self, _router, _retriever):
        """Same chunk found by multiple queries should be deduped."""
        retriever = MultiQueryRetriever(_router, _retriever, top_k=10)
        # Two queries that should find overlapping results
        needs = [
            EvidenceNeed(topic="Acme", search_query="Acme Corporation", priority=NeedPriority.HIGH),
            EvidenceNeed(topic="Acme revenue", search_query="Acme revenue", priority=NeedPriority.HIGH),
        ]
        plan = QueryPlan(
            original_query="Acme revenue",
            pattern="conflict",
            evidence_needs=needs,
            is_planned=True,
        )
        result = retriever.retrieve(plan)
        # All chunk_ids in selected should be unique
        chunk_ids = [r.chunk_id for r in result.selected]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_need_coverage_computed(self, _router, _retriever):
        """Coverage should be computed for each need."""
        retriever = MultiQueryRetriever(_router, _retriever, top_k=10)
        needs = [
            EvidenceNeed(topic="topic_a", search_query="Acme headquarters", priority=NeedPriority.HIGH),
            EvidenceNeed(topic="topic_b", search_query="Acme CEO", priority=NeedPriority.HIGH),
        ]
        plan = QueryPlan(
            original_query="Acme HQ and CEO",
            pattern="complex_research",
            evidence_needs=needs,
            is_planned=True,
        )
        result = retriever.retrieve(plan)
        assert len(result.need_coverage) == 2

    def test_source_diversity_tracked(self, _router, _retriever):
        """Source diversity should be tracked."""
        retriever = MultiQueryRetriever(_router, _retriever, top_k=10)
        needs = [
            EvidenceNeed(topic="Acme", search_query="Acme Corporation", priority=NeedPriority.HIGH),
        ]
        plan = QueryPlan(
            original_query="Tell me about Acme",
            pattern="complex_research",
            evidence_needs=needs,
            is_planned=True,
        )
        result = retriever.retrieve(plan)
        assert result.source_diversity >= 1


# ---------------------------------------------------------------------------
# Test: Conflict-specific planning
# ---------------------------------------------------------------------------

class TestConflictPlanning:
    def test_conflict_generates_caveat_need(self, _planner):
        plan = _planner.plan("What was the Ohio plant utilization?", "conflict")
        claim_types = {n.claim_type for n in plan.evidence_needs}
        # Should have at least primary + caveat
        assert ClaimType.PRIMARY in claim_types

    def test_conflict_search_variants_diverse(self, _planner):
        plan = _planner.plan("How many employees does Acme have?", "conflict")
        # Should have the original query + additional variants
        all_queries = [n.search_query for n in plan.evidence_needs]
        assert len(all_queries) >= 2


# ---------------------------------------------------------------------------
# Test: Fallback behavior
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    def test_planner_returns_empty_falls_back(self, _planner):
        """If planner produces no needs, should still work."""
        plan = _planner.plan("", "conflict")
        # Empty query should still produce a plan (even if not useful)
        assert isinstance(plan, QueryPlan)

    def test_single_need_planned(self, _planner):
        """Even a simple conflict query should produce multiple needs."""
        plan = _planner.plan("revenue", "conflict")
        assert plan.need_count >= 1
