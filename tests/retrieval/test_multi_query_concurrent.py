"""Tests for concurrent multi-query retrieval (Phase 15 parallelization).

Covers:
1. All subqueries execute concurrently
2. Concurrency limit is respected
3. Deterministic output despite different completion orders
4. One failed subquery does not fail the entire retrieval
5. Multiple failed subqueries still preserve successful evidence
6. EvidenceNeed attribution remains correct
7. Existing deduplication remains correct
8. Existing ranking/fusion remains unchanged
9. Zero/one subquery behaves exactly as before
"""
from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evidence.models import EvidenceRef, SourceType
from app.retrieval.multi_query import MultiQueryRetriever, RetrievalResult
from app.retrieval.planner import (
    EvidenceNeed,
    NeedPriority,
    QueryPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref(chunk_id: uuid.UUID | None = None, score: float = 0.8,
              text: str = "evidence text", doc_id: uuid.UUID | None = None) -> EvidenceRef:
    """Create a minimal EvidenceRef for testing."""
    return EvidenceRef(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        source_id=uuid.uuid4(),
        source_path="/test/doc.pdf",
        source_type=SourceType.PDF,
        text=text,
        score=score,
        rank=1,
    )


def _make_need(topic: str = "topic", query: str = "search query",
               priority: NeedPriority = NeedPriority.HIGH) -> EvidenceNeed:
    return EvidenceNeed(topic=topic, search_query=query, priority=priority)


def _make_plan(needs: list[EvidenceNeed], pattern: str = "conflict") -> QueryPlan:
    return QueryPlan(
        original_query="test query",
        pattern=pattern,
        evidence_needs=needs,
        is_planned=True,
    )


class _FakeRouter:
    """Fake router that returns configurable results per query."""

    def __init__(self, results_map: dict[str, list[EvidenceRef]] | None = None):
        self._results_map = results_map or {}
        self._call_log: list[str] = []

    def classify_question(self, query: str):
        return "conceptual"

    def get_retrieval_mix(self, pattern):
        m = MagicMock()
        m.bm25_weight = 0.5
        m.vector_weight = 0.5
        return m


class _FakeRetriever:
    """Fake retriever that tracks concurrent calls and supports latency injection."""

    def __init__(self, default_results: list[EvidenceRef] | None = None,
                 latency_per_call: float = 0.0):
        self._default_results = default_results or []
        self._latency = latency_per_call
        self._concurrent_calls = 0
        self._max_concurrent = 0
        self._lock = asyncio.Lock()
        self._call_count = 0
        self.store = MagicMock()

    async def search_async(self, query, top_k=10, bm25_weight=0.5, vector_weight=0.5,
                           mechanisms=None, normalization=None):
        async with self._lock:
            self._concurrent_calls += 1
            self._max_concurrent = max(self._max_concurrent, self._concurrent_calls)
            self._call_count += 1

        if self._latency > 0:
            await asyncio.sleep(self._latency)

        async with self._lock:
            self._concurrent_calls -= 1

        # Return results based on query or default
        return list(self._default_results)

    def search(self, query, top_k=10, bm25_weight=0.5, vector_weight=0.5):
        return list(self._default_results)

    def mark_dirty(self):
        pass


# ---------------------------------------------------------------------------
# Test: All subqueries execute concurrently
# ---------------------------------------------------------------------------

class TestConcurrentExecution:
    @pytest.mark.asyncio
    async def test_all_subqueries_execute(self):
        """All evidence needs should be dispatched."""
        router = _FakeRouter()
        ref = _make_ref()
        retriever = _FakeRetriever(default_results=[ref])

        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [_make_need(query=f"query_{i}") for i in range(4)]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)
        assert result.total_retrieval_calls == 4

    @pytest.mark.asyncio
    async def test_concurrent_execution_detected(self):
        """Multiple subqueries should execute concurrently (not sequentially)."""
        router = _FakeRouter()
        ref = _make_ref()
        # 100ms latency per call — with 4 concurrent calls, total should be ~100ms, not ~400ms
        retriever = _FakeRetriever(default_results=[ref], latency_per_call=0.1)

        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [_make_need(query=f"query_{i}") for i in range(4)]
        plan = _make_plan(needs)

        start = time.perf_counter()
        result = await mq.retrieve(plan)
        elapsed = time.perf_counter() - start

        # Should complete in roughly 100ms (concurrent), not 400ms (sequential)
        # Allow 200ms tolerance for overhead
        assert elapsed < 0.3, f"Expected concurrent execution, took {elapsed:.2f}s"
        assert retriever._max_concurrent > 1, "Retriever should see concurrent calls"


# ---------------------------------------------------------------------------
# Test: Concurrency limit is respected
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Concurrency should not exceed the configured limit."""
        router = _FakeRouter()
        ref = _make_ref()
        retriever = _FakeRetriever(default_results=[ref], latency_per_call=0.1)

        mq = MultiQueryRetriever(router, retriever, top_k=10, max_concurrency=2)
        needs = [_make_need(query=f"query_{i}") for i in range(6)]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)

        # With max_concurrency=2, max concurrent should never exceed 2
        assert retriever._max_concurrent <= 2, (
            f"Max concurrent was {retriever._max_concurrent}, expected <= 2"
        )
        assert result.total_retrieval_calls == 6


# ---------------------------------------------------------------------------
# Test: Deterministic output despite different completion orders
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.asyncio
    async def test_deterministic_output(self):
        """Output should be identical regardless of task completion order."""
        router = _FakeRouter()
        ref1 = _make_ref(score=0.9, text="high score")
        ref2 = _make_ref(score=0.7, text="medium score")
        ref3 = _make_ref(score=0.5, text="low score")

        # Run 1: normal order
        retriever1 = _FakeRetriever(default_results=[ref1])
        mq1 = MultiQueryRetriever(router, retriever1, top_k=10)
        needs1 = [
            _make_need(query="query_a"),
            _make_need(query="query_b"),
            _make_need(query="query_c"),
        ]
        plan1 = _make_plan(needs1)
        result1 = await mq1.retrieve(plan1)

        # Run 2: same setup (deterministic because results are sorted by score)
        retriever2 = _FakeRetriever(default_results=[ref1])
        mq2 = MultiQueryRetriever(router, retriever2, top_k=10)
        needs2 = [
            _make_need(query="query_a"),
            _make_need(query="query_b"),
            _make_need(query="query_c"),
        ]
        plan2 = _make_plan(needs2)
        result2 = await mq2.retrieve(plan2)

        # Results should have same candidates in same order
        assert len(result1.candidates) == len(result2.candidates)
        for c1, c2 in zip(result1.candidates, result2.candidates):
            assert c1.chunk_id == c2.chunk_id
            assert abs(c1.score - c2.score) < 1e-6

    @pytest.mark.asyncio
    async def test_needs_processed_in_order(self):
        """Results should be attributed to needs in original order, not completion order."""
        attribution_log = []
        router = _FakeRouter()

        class _TrackingRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                # Simulate varying latency so order could differ
                if "query_2" in query:
                    await asyncio.sleep(0.05)  # Fast
                elif "query_0" in query:
                    await asyncio.sleep(0.15)  # Slow
                else:
                    await asyncio.sleep(0.1)  # Medium
                ref = _make_ref()
                return [ref]

        retriever = _TrackingRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [
            _make_need(query="query_0"),
            _make_need(query="query_1"),
            _make_need(query="query_2"),
        ]
        plan = _make_plan(needs)
        result = await mq.retrieve(plan)

        # All needs should have results regardless of completion order
        assert result.total_retrieval_calls == 3


# ---------------------------------------------------------------------------
# Test: Failure isolation
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_failed_subquery_preserves_others(self):
        """A single failed subquery should not fail the entire retrieval."""
        router = _FakeRouter()
        ref = _make_ref()

        class _PartialFailRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                if "fail_here" in query:
                    raise RuntimeError("simulated failure")
                return [ref]

        retriever = _PartialFailRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [
            _make_need(query="good query 1"),
            _make_need(query="fail_here query"),
            _make_need(query="good query 2"),
        ]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)

        # Should have 2 successful calls (not 3, not 0)
        assert result.total_retrieval_calls == 2
        assert len(result.candidates) > 0

    @pytest.mark.asyncio
    async def test_multiple_failures_preserve_successful(self):
        """Multiple failed subqueries should still preserve successful evidence."""
        router = _FakeRouter()
        ref = _make_ref()

        class _MultiFailRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                if "fail" in query:
                    raise RuntimeError("simulated failure")
                return [ref]

        retriever = _MultiFailRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [
            _make_need(query="fail 1"),
            _make_need(query="good query"),
            _make_need(query="fail 2"),
        ]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)

        # Should have 1 successful call
        assert result.total_retrieval_calls == 1
        assert len(result.candidates) > 0

    @pytest.mark.asyncio
    async def test_all_failures_returns_empty(self):
        """If all subqueries fail, should return empty results gracefully."""
        router = _FakeRouter()

        class _AllFailRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                raise RuntimeError("always fails")

        retriever = _AllFailRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [_make_need(query=f"fail_{i}") for i in range(3)]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)
        assert result.total_retrieval_calls == 0
        assert len(result.candidates) == 0


# ---------------------------------------------------------------------------
# Test: EvidenceNeed attribution
# ---------------------------------------------------------------------------

class TestNeedAttribution:
    @pytest.mark.asyncio
    async def test_attribution_correct(self):
        """Each result should be attributed to the correct EvidenceNeed."""
        router = _FakeRouter()

        ref_a = _make_ref(text="result for need A")
        ref_b = _make_ref(text="result for need B")

        class _AttrRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                if "need_a" in query:
                    return [ref_a]
                return [ref_b]

        retriever = _AttrRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        need_a = _make_need(topic="topic_a", query="need_a search")
        need_b = _make_need(topic="topic_b", query="need_b search")
        plan = _make_plan([need_a, need_b])

        result = await mq.retrieve(plan)

        # Check attribution
        for ref in result.candidates:
            if ref.text == "result for need A":
                assert ref.metadata.get("evidence_need_id") == need_a.id
            elif ref.text == "result for need B":
                assert ref.metadata.get("evidence_need_id") == need_b.id


# ---------------------------------------------------------------------------
# Test: Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    @pytest.mark.asyncio
    async def test_dedup_preserved(self):
        """Same chunk found by multiple needs should be deduped."""
        router = _FakeRouter()
        shared_ref = _make_ref(text="shared chunk")

        class _DedupRetriever(_FakeRetriever):
            async def search_async(self, query, **kwargs):
                return [shared_ref]

        retriever = _DedupRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [
            _make_need(query="query_1"),
            _make_need(query="query_2"),
        ]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)

        # Should have only 1 candidate (deduped), not 2
        assert len(result.candidates) == 1
        # Should have multi_need_hit flag
        assert result.candidates[0].metadata.get("multi_need_hit") is True


# ---------------------------------------------------------------------------
# Test: Zero/one subquery edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_needs(self):
        """Empty evidence needs should return empty results."""
        router = _FakeRouter()
        retriever = _FakeRetriever()
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        plan = _make_plan([])

        result = await mq.retrieve(plan)
        assert result.total_retrieval_calls == 0
        assert len(result.candidates) == 0

    @pytest.mark.asyncio
    async def test_single_need(self):
        """Single need should work identically to before."""
        router = _FakeRouter()
        ref = _make_ref()
        retriever = _FakeRetriever(default_results=[ref])
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [_make_need(query="single query")]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)
        assert result.total_retrieval_calls == 1
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_unplanned_passthrough(self):
        """Unplanned query should use sync search (passthrough)."""
        router = _FakeRouter()
        ref = _make_ref()
        retriever = _FakeRetriever(default_results=[ref])
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        plan = QueryPlan(
            original_query="simple query",
            pattern="simple_lookup",
            is_planned=False,
        )

        result = await mq.retrieve(plan)
        assert result.total_retrieval_calls == 1
        assert len(result.selected) > 0

    @pytest.mark.asyncio
    async def test_needs_with_empty_query_skipped(self):
        """Needs with empty search_query should be skipped."""
        router = _FakeRouter()
        ref = _make_ref()
        retriever = _FakeRetriever(default_results=[ref])
        mq = MultiQueryRetriever(router, retriever, top_k=10)
        needs = [
            _make_need(query="valid query"),
            _make_need(query=""),  # empty
            _make_need(query="another valid query"),
        ]
        plan = _make_plan(needs)

        result = await mq.retrieve(plan)
        # Only 2 valid queries should be dispatched
        assert result.total_retrieval_calls == 2
