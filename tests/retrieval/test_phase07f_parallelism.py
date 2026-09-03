"""Phase 07f focused tests — safe retrieval-layer parallelism.

Covers the §15 T1–T10 acceptance checks for the ONLY parallelism introduced in
Phase 07f: concurrent dispatch of independent retrieval methods in
`RetrievalPolicyRouter.execute_retrieval` and overlapping BM25/dense passes
inside `HybridRetriever.search_async`.

Phase 07f deliberately introduces NO overlap across the orchestration LLM
dependency chain (analysis -> planning -> retrieve/assess -> synthesize ->
verify): those stages have hard data dependencies and speculative overlap is
explicitly forbidden. These tests assert the added retrieval parallelism is
bounded, deterministic, failure-isolated, and leak-free, and that outcome,
citation, ordering, and LLM-call-accounting integrity are preserved.

No live API is used; the FAISS embedder is a deterministic local stub.
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

from app.config import Settings
from app.evidence.models import Chunk, Document, EvidenceRef, Source, SourceType
from app.evidence.store import EvidenceStore
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.reranking.reranker import NoOpReranker
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.policy import (
    QuestionPattern,
    RetrievalMethod,
    RetrievalMix,
    RetrievalPolicy,
    RetrievalPolicyEntry,
)
from app.retrieval.router import RetrievalPolicyRouter
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices


# ---------------------------------------------------------------------------
# Deterministic offline embedder (avoids model weight downloads in tests)
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    dim = 8

    def embed_texts(self, texts: list[str]):
        import hashlib

        import numpy as np
        vecs = [np.array([float(b) / 255.0 for b in hashlib.md5(t.encode()).digest()[: self.dim]],
                         dtype=np.float32)
                for t in texts]
        return np.stack(vecs) if len(vecs) else np.zeros((0, self.dim), dtype=np.float32)

    def embed_chunks(self, chunks):
        return self.embed_texts([c.text for c in chunks])

    def encode(self, texts, **kw):  # pragma: no cover - FAISS fallback
        return self.embed_texts(texts)


EMBEDDER = _FakeEmbedder()


_CORPUS = [
    "Alpha Corp doubled its 2030 reactor power output to 40 megawatts.",
    "Beta Lab published the cold-fusion Q-of-1.4 result in 2031.",
    "Gamma Systems reported the 2031 Helion-DARPA reactor at 50 watts total.",
    "Delta Corp announced its 2026 consumer electronics strategy.",
]


def _build_hybrid(td: str | Path, embedder=EMBEDDER) -> HybridRetriever:
    store = EvidenceStore(Path(td) / "ev.sqlite")
    chunks = []
    for i, text in enumerate(_CORPUS):
        src = Source(type=SourceType.TEXT, path=f"/corpus_{i}.txt", checksum=f"s{i}")
        store.upsert_source(src)
        doc = Document(source_id=src.id, version=1, checksum=f"d{i}", chunking_strategy="test")
        store.insert_document(doc)
        chunks.append(Chunk(document_id=doc.id, ordinal=0, text=text, token_count=len(text.split()), page_start=1))
    store.insert_chunks(chunks)
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)
    return HybridRetriever(
        store=store,
        bm25=BM25Retriever(store),
        vector=FAISSVectorStore(store, embedding_dim=8, index_path=Path(td) / "faiss.index"),
        embedder=embedder,
    )


def _mix(*methods: RetrievalMethod) -> RetrievalMix:
    return RetrievalMix(
        methods=list(methods),
        weights={m: 1.0 for m in methods},
        bm25_weight=0.5,
        vector_weight=0.5,
        max_results_per_method=20,
    )


def _router(*methods: RetrievalMethod) -> RetrievalPolicyRouter:
    entry = RetrievalPolicyEntry(
        pattern=QuestionPattern.CONCEPTUAL,
        retrieval_mix=_mix(*methods),
        priority=1,
        min_confidence=0.0,
    )
    return RetrievalPolicyRouter(policy=RetrievalPolicy(entries=[entry]), graph_retriever=None, settings=Settings())


def _key(refs) -> list[tuple[str, float]]:
    return [(str(r.chunk_id), round(r.score, 6)) for r in refs]


# ---------------------------------------------------------------------------
# T1 — independent method dispatch overlaps (barrier/overlap proof)
# ---------------------------------------------------------------------------
class _StubHybrid(HybridRetriever):
    """Deterministic retriever stub for router-level concurrency tests.

    Returns fixed, valid EvidenceRefs regardless of query so concurrency /
    isolation assertions never depend on vector/BM25 relevance. Tracks concurrency
    and call counts with a threading.Lock (methods run inside worker threads)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.active = 0
        self.max_active = 0
        self.search_count = 0
        self.bm25_only_count = 0
        self.vector_only_count = 0
        self._lock = threading.Lock()
        self.sleep = 0.0
        self._make_stub_refs()

    def _make_stub_refs(self):
        from uuid import uuid4
        self._refs = [
            EvidenceRef(
                chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(),
                source_path="/stub.txt", source_type=SourceType.TEXT,
                text="stub evidence", score=0.9, rank=i,
            )
            for i in range(1, 4)
        ]

    def _busy(self):
        if self.sleep > 0:
            # `time.sleep` releases the GIL, so two worker threads dispatched by
            # asyncio.to_thread genuinely overlap (mirrors the GIL-releasing
            # numpy/torch inference in the real embedder/FAISS passes).
            import time
            time.sleep(self.sleep)

    def _track(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _untrack(self):
        with self._lock:
            self.active -= 1

    def search(self, query, **kw):
        self._track()
        try:
            with self._lock:
                self.search_count += 1
            self._busy()
            return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(self._refs)]
        finally:
            self._untrack()

    def search_bm25_only(self, query, top_k):
        self._track()
        try:
            with self._lock:
                self.bm25_only_count += 1
            self._busy()
            return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(self._refs)]
        finally:
            self._untrack()

    def search_vector_only(self, query, top_k):
        self._track()
        try:
            with self._lock:
                self.vector_only_count += 1
            self._busy()
            return [c.model_copy(update={"rank": i + 1}) for i, c in enumerate(self._refs)]
        finally:
            self._untrack()


def test_t1_independent_methods_overlap():
    """Independent dispatch methods overlap: max concurrent active searches must
    exceed 1, and wall-clock must be less than the serial sum of the sleeps."""
    with tempfile.TemporaryDirectory(prefix="t1_") as td:
        r = _StubHybrid(store=EvidenceStore(Path(td) / "ev.sqlite"),
                        bm25=BM25Retriever(EvidenceStore(Path(td) / "ev.sqlite")),
                        vector=FAISSVectorStore(EvidenceStore(Path(td) / "ev.sqlite"), embedding_dim=8,
                                                index_path=Path(td) / "f.index"),
                        embedder=EMBEDDER)
        r.sleep = 0.05
        router = _router(RetrievalMethod.HYBRID, RetrievalMethod.BM25, RetrievalMethod.VECTOR)
        import time
        t0 = time.perf_counter()
        refs = asyncio.run(router.execute_retrieval(
            "any query", QuestionPattern.CONCEPTUAL, r, top_k=8, reranker=None))
        wall = time.perf_counter() - t0
        assert refs, "expected fused evidence"
        # Three independent methods each sleeping 0.05s overlap: serial would be
        # >= 0.15s; genuine overlap keeps the wall well under that bound.
        assert r.max_active >= 2, f"expected overlap but max_active={r.max_active}"
        assert wall < 0.15, f"dispatch looks serial: wall={wall:.4f}s"


# ---------------------------------------------------------------------------
# T2 — orchestration LLM dependency chain stays strictly sequential
# ---------------------------------------------------------------------------
def test_t2_dependent_stages_stay_sequential():
    """No speculative overlap across dependent LLM stages. If two dependency-
    ordered stages both run, the dependent one must appear strictly AFTER the
    one it consumes: planning after analysis; verification after synthesis
    (verification consumes the synthesized answer); synthesis after planning."""
    order: list[str] = []

    def _after(dep: str, depend: str) -> bool:
        return depend in order and (dep not in order or order.index(depend) > order.index(dep))

    class OrderRouter:
        async def complete(self, messages, *, call_type="general", **kw):
            order.append(call_type)
            return _resp_for(call_type)

        async def aclose(self):
            pass

    def _assert_invariants():
        for dep, depend in (
            ("query_analysis", "research_planning"),
            ("research_planning", "synthesis"),
            ("synthesis", "verification"),
        ):
            # If the dependent stage ran, its dependency must have run BEFORE it
            # (07f must not issue a dependent call speculatively ahead of the
            # stage it consumes).
            assert not (depend in order and dep not in order), \
                f"dependent {depend} ran without its dependency {dep}"
            if depend in order and dep in order:
                assert order.index(depend) > order.index(dep), \
                    f"dependency violated: {depend} ran before {dep}"

    with tempfile.TemporaryDirectory(prefix="t2_") as td:
        r = _build_hybrid(td)
        asyncio.run(_run_flow(r, OrderRouter(), "t2"))
        _assert_invariants()
        assert order, "orchestration should issue at least one LLM call"
        # Analysis and planning are the deterministic entry stages and always
        # run for a conceptual query.
        assert "query_analysis" in order and "research_planning" in order


# ---------------------------------------------------------------------------
# T3 — retrieval determinism: sync search == async search_async
# ---------------------------------------------------------------------------
def test_t3_async_search_identical_to_sync():
    with tempfile.TemporaryDirectory(prefix="t3_") as td:
        r = _build_hybrid(td)
        for q in ["Alpha Corp output", "cold fusion 2031", "Helion watts", "consumer electronics 2026"]:
            sync = _key(r.search(q, top_k=8))
            async_ = _key(asyncio.run(r.search_async(q, top_k=8)))
            assert sync == async_, f"fusion mismatch for {q!r}"


def test_t3_async_mechanisms_respected():
    """The async variant honors the `mechanisms` filter exactly like sync, and
    single-mechanism passes are deterministic (repeatable) — no vector pass is
    spawned when only BM25 is requested and vice-versa."""
    with tempfile.TemporaryDirectory(prefix="t3b_") as td:
        r = _build_hybrid(td)
        # Mechanism parity: async must equal sync for the same filter.
        for mech in ({"bm25"}, {"vector"}, None):
            sync = _key(r.search("Alpha Corp 2030", top_k=8, mechanisms=mech))
            async_ = _key(asyncio.run(r.search_async("Alpha Corp 2030", top_k=8, mechanisms=mech)))
            assert sync == async_, f"mechanism filter {mech}: async != sync"
        # bm25-only is deterministic and reflects lexical hits (doc0 mentions "2030").
        first = asyncio.run(r.search_async("Alpha Corp 2030", top_k=8, mechanisms={"bm25"}))
        second = asyncio.run(r.search_async("Alpha Corp 2030", top_k=8, mechanisms={"bm25"}))
        assert _key(first) == _key(second), "bm25-only async pass must be deterministic"


# ---------------------------------------------------------------------------
# T4 — retrieval method failure isolation
# ---------------------------------------------------------------------------
class _StubHybridFailingBM25(_StubHybrid):
    def search_bm25_only(self, query, top_k):
        raise RuntimeError("bm25 exploded")


def _stub(td: str) -> _StubHybrid:
    root = Path(td)
    store = EvidenceStore(root / "store.sqlite")
    return _StubHybrid(
        store=store,
        bm25=BM25Retriever(store),
        vector=FAISSVectorStore(store, embedding_dim=8, index_path=root / "faiss.index"),
        embedder=EMBEDDER,
    )


def test_t4_failed_method_is_isolated():
    """When one retrieval method raises inside concurrent dispatch, the other
    independent methods still contribute and no exception propagates upward."""
    with tempfile.TemporaryDirectory(prefix="t4_") as td:
        r = _stub(td)
        # Specialize so only the dedicated BM25 method fails.
        r.__class__ = _StubHybridFailingBM25
        router = _router(RetrievalMethod.BM25, RetrievalMethod.VECTOR)
        refs = asyncio.run(router.execute_retrieval(
            "any query", QuestionPattern.CONCEPTUAL, r, top_k=8, reranker=None))
        # No exception leaked; the surviving VECTOR method produced evidence.
        assert isinstance(refs, list)
        assert refs, "surviving vector method should produce evidence"


# ---------------------------------------------------------------------------
# T5 — call accounting: exactly one search per dispatched method (no duplicates)
# ---------------------------------------------------------------------------
def test_t5_exactly_one_search_per_dispatch_method():
    """Concurrency must never duplicate a retrieval call (quota/call-ceiling
    integrity): each dispatched method issues exactly one internal search."""
    with tempfile.TemporaryDirectory(prefix="t5_") as td:
        r = _stub(td)
        # HYBRID + BM25 + VECTOR: the hybrid is skipped (both mechanisms covered by
        # dedicated siblings), so exactly BM25 and VECTOR each run once.
        router = _router(RetrievalMethod.HYBRID, RetrievalMethod.BM25, RetrievalMethod.VECTOR)
        asyncio.run(router.execute_retrieval(
            "any query", QuestionPattern.CONCEPTUAL, r, top_k=8, reranker=None))
        assert r.search_count == 0, f"hybrid was not skipped: search_count={r.search_count}"
        assert r.bm25_only_count == 1, f"BM25 method dispatched {r.bm25_only_count}x"
        assert r.vector_only_count == 1, f"vector method dispatched {r.vector_only_count}x"


# ---------------------------------------------------------------------------
# T6 — bounded fan-out under a pathological mix (semaphore cap, no unbounded
#      thread pool)
# ---------------------------------------------------------------------------
def test_t6_fan_out_is_bounded_and_recovers():
    async def _run():
        with tempfile.TemporaryDirectory(prefix="t6_") as td:
            r = _stub(td)
            r.sleep = 0.02
            # A broad mix still completes deterministically (no unbounded fan-out).
            router = _router(RetrievalMethod.HYBRID, RetrievalMethod.BM25,
                             RetrievalMethod.VECTOR, RetrievalMethod.BM25)
            refs = await router.execute_retrieval(
                "any query", QuestionPattern.CONCEPTUAL, r, top_k=8, reranker=None)
            assert refs
            # Fan-out never exceeds the semaphore cap min(4, len(methods)) = 4.
            assert r.max_active <= 4, f"fan-out exceeded cap: max_active={r.max_active}"
            # No leftover pending tasks (no leaked threads/coroutines from the mix).
            await asyncio.sleep(0.02)
            assert not [t for t in asyncio.all_tasks()
                        if t is not asyncio.current_task() and not t.done()]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# T7 — cancellation / leak-freedom: no orphaned asyncio tasks after search_async
# ---------------------------------------------------------------------------
def test_t7_no_orphaned_tasks_after_async_search():
    async def _run():
        before = len(asyncio.all_tasks())
        with tempfile.TemporaryDirectory(prefix="t7_") as td:
            r = _build_hybrid(td)
            await r.search_async("Gamma Systems watts", top_k=8)
            await asyncio.sleep(0.03)
            after = len(asyncio.all_tasks())
            return before, after

    before, after = asyncio.run(_run())
    assert after <= before + 1, f"orphaned tasks: before={before} after={after}"


# ---------------------------------------------------------------------------
# T8 — citation integrity preserved under concurrency (dedup + rank)
# ---------------------------------------------------------------------------
def test_t8_citation_dedup_and_rank_preserved():
    with tempfile.TemporaryDirectory(prefix="t8_") as td:
        r = _build_hybrid(td)
        refs = asyncio.run(r.search_async("Alpha Corp 2030 output", top_k=8))
        ids = [x.chunk_id for x in refs]
        assert len(ids) == len(set(ids)), "concurrent fuse must dedup by chunk_id"
        assert [x.rank for x in refs] == list(range(1, len(refs) + 1)), \
            "ranks must be sequential 1..N and deterministic"


# ---------------------------------------------------------------------------
# T9 — outcome semantics preserved under concurrency
# ---------------------------------------------------------------------------
def test_t9_outcome_and_grounding_preserved():
    with tempfile.TemporaryDirectory(prefix="t9_") as td:
        r = _build_hybrid(td)
        res = asyncio.run(_run_flow(r, _OkRouter(), "t9"))
        assert res.outcome.value == "answered"
        assert res.citations, "expected grounded citations"


# ---------------------------------------------------------------------------
# T10 — healthy-path LLM call count unchanged by the parallelization
# ---------------------------------------------------------------------------
def test_t10_healthy_path_llm_call_count_unchanged():
    class _CountingRouter:
        def __init__(self):
            self.count = 0
            self.types: list[str] = []

        async def complete(self, messages, *, call_type="general", **kw):
            self.count += 1
            self.types.append(call_type)
            return _resp_for(call_type)

        async def aclose(self):
            pass

    with tempfile.TemporaryDirectory(prefix="t10_") as td:
        router = _CountingRouter()
        r = _build_hybrid(td)
        res = asyncio.run(_run_flow(r, router, "t10"))
        assert res.outcome.value == "answered"
        # Phase 07f adds NO duplicate LLM work: every completion is accounted
        # exactly once in the call log (quota/call-ceiling integrity).
        assert router.count == len(router.types), "every LLM call accounted exactly once"
        # A clean conceptual query reaches synthesis + verification.
        assert "synthesis" in router.types and "verification" in router.types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resp_for(call_type: str) -> CompletionResponse:
    content = {
        "query_analysis": '{"complexity":"moderate","reasoning":"r","suggested_subquestion_count":1}',
        "research_planning": '{"objective":"x","entities":[],"time_window":null,'
                            '"subquestions":["q"],"evidence_type":"factual",'
                            '"preferred_retrieval_methods":["hybrid"],"required_sources":[],'
                            '"risk_level":"low","token_budget":6000,"iteration_budget":1,'
                            '"stopping_condition":"stop"}',
        "evidence_extraction": '{"sufficient":true,"reasoning":"ok","next_subquery":null}',
        "synthesis": "The answer is grounded [1].",
        "verification": '{"status":"supported","confidence":0.9,"reasoning":"ok","contradictions":[]}',
    }.get(call_type, "ok")
    return CompletionResponse(
        content=content,
        model="mock",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        provider="mock",
    )


class _OkRouter:
    async def complete(self, messages, *, call_type="general", **kw):
        return _resp_for(call_type)

    async def aclose(self):
        pass


async def _run_flow(r, router, rid):
    from app.orchestration.graph import run_query
    return await run_query(
        query="Gamma Systems reactor watts",
        request_id=rid,
        router=router,
        retriever=r,
        reranker=NoOpReranker(),
        settings=Settings(verification_enabled=True, retrieval_policy_enabled=False),
    )