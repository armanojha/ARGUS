#!/usr/bin/env python3
"""Phase 33: System-Level Performance Profiler.

Diagnostic benchmark that instruments the real ARGUS pipeline to measure:
- End-to-end latency per query
- Per-stage latency (classification, retrieval, assessment, synthesis)
- LLM call count and latency
- Embedding call count and latency
- Retrieval call count and latency
- Evidence selector call count and latency
- Cache hit/miss rates
- Duplicate work detection

Uses monkey-patching to instrument internal functions WITHOUT modifying
production code. This is a read-only diagnostic tool.
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import sys
import time
import threading
from pathlib import Path
from typing import Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from benchmarks.benchmark_fusion import build_benchmark_store


# ── Query set ────────────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {"id": "P33-Q01", "class": "simple_lookup", "query": "Where is Acme Corporation headquartered?"},
    {"id": "P33-Q02", "class": "multi_doc_synthesis", "query": "What are the key differences between Atlas v1 and Atlas v2?"},
    {"id": "P33-Q03", "class": "multi_hop", "query": "Who manages the team that developed Atlas?"},
    {"id": "P33-Q04", "class": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?"},
    {"id": "P33-Q05", "class": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?"},
    {"id": "P33-Q06", "class": "technical_explanation", "query": "How does the Atlas database achieve low latency?"},
    {"id": "P33-Q07", "class": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?"},
    {"id": "P33-Q08", "class": "absent_info", "query": "What is Acme's market share in the European robotics market?"},
    {"id": "P33-Q09", "class": "absent_info", "query": "What is the salary range for Acme's software engineers?"},
    {"id": "P33-Q10", "class": "adversarial", "query": "What undisclosed legal issues has Acme faced?"},
    {"id": "P33-Q11", "class": "adversarial", "query": "What internal documents reveal Acme's planned layoffs?"},
]


# ── Instrumentation state ────────────────────────────────────────────────────

class InstrumentState:
    """Thread-safe instrumentation state for one query execution."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.llm_calls = 0
            self.llm_latencies_ms: list[float] = []
            self.llm_call_types: list[str] = []
            self.embedding_calls = 0
            self.embedding_latencies_ms: list[float] = []
            self.retrieval_calls = 0
            self.retrieval_latencies_ms: list[float] = []
            self.bm25_calls = 0
            self.bm25_latencies_ms: list[float] = []
            self.vector_calls = 0
            self.vector_latencies_ms: list[float] = []
            self.rerank_calls = 0
            self.rerank_latencies_ms: list[float] = []
            self.evidence_selector_calls = 0
            self.evidence_selector_latencies_ms: list[float] = []
            self.evidence_selector_input_sizes: list[int] = []
            self.evidence_selector_output_sizes: list[int] = []
            self.cache_hits = 0
            self.cache_misses = 0
            self.duplicate_embeddings = 0
            self.duplicate_retrievals = 0
            self._seen_embeddings: set[str] = set()
            self._seen_retrievals: set[str] = set()
            self.stage_latencies: dict[str, float] = {}

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "llm_latencies_ms": list(self.llm_latencies_ms),
                "llm_call_types": list(self.llm_call_types),
                "embedding_calls": self.embedding_calls,
                "embedding_latencies_ms": list(self.embedding_latencies_ms),
                "retrieval_calls": self.retrieval_calls,
                "retrieval_latencies_ms": list(self.retrieval_latencies_ms),
                "bm25_calls": self.bm25_calls,
                "bm25_latencies_ms": list(self.bm25_latencies_ms),
                "vector_calls": self.vector_calls,
                "vector_latencies_ms": list(self.vector_latencies_ms),
                "rerank_calls": self.rerank_calls,
                "rerank_latencies_ms": list(self.rerank_latencies_ms),
                "evidence_selector_calls": self.evidence_selector_calls,
                "evidence_selector_latencies_ms": list(self.evidence_selector_latencies_ms),
                "evidence_selector_input_sizes": list(self.evidence_selector_input_sizes),
                "evidence_selector_output_sizes": list(self.evidence_selector_output_sizes),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "duplicate_embeddings": self.duplicate_embeddings,
                "duplicate_retrievals": self.duplicate_retrievals,
                "stage_latencies": dict(self.stage_latencies),
            }


# Global instrumentation state (reset per query)
_instrument = InstrumentState()


# ── Monkey-patches ───────────────────────────────────────────────────────────

def _patch_llm_router(router):
    """Instrument router.complete() to count and time LLM calls."""
    original_complete = router.complete

    async def instrumented_complete(messages, **kwargs):
        t0 = time.perf_counter()
        call_type = kwargs.get("call_type", "unknown")
        try:
            result = await original_complete(messages, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000
            with _instrument._lock:
                _instrument.llm_calls += 1
                _instrument.llm_latencies_ms.append(latency_ms)
                _instrument.llm_call_types.append(call_type)
            return result
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            with _instrument._lock:
                _instrument.llm_calls += 1
                _instrument.llm_latencies_ms.append(latency_ms)
                _instrument.llm_call_types.append(f"{call_type}_error")
            raise

    router.complete = instrumented_complete
    return original_complete


def _patch_embedder(embedder):
    """Instrument embedder.embed_texts() to count and time embedding calls."""
    original_embed = embedder.embed_texts

    def instrumented_embed(texts, **kwargs):
        t0 = time.perf_counter()
        result = original_embed(texts, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        with _instrument._lock:
            _instrument.embedding_calls += 1
            _instrument.embedding_latencies_ms.append(latency_ms)
            # Track duplicate embeddings
            for text in texts:
                key = text.strip().lower()
                if key in _instrument._seen_embeddings:
                    _instrument.duplicate_embeddings += 1
                else:
                    _instrument._seen_embeddings.add(key)
        return result

    embedder.embed_texts = instrumented_embed
    return original_embed


def _patch_bm25(bm25):
    """Instrument BM25Retriever.search() to count and time BM25 calls."""
    original_search = bm25.search

    def instrumented_search(query, **kwargs):
        t0 = time.perf_counter()
        result = original_search(query, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        with _instrument._lock:
            _instrument.bm25_calls += 1
            _instrument.bm25_latencies_ms.append(latency_ms)
        return result

    bm25.search = instrumented_search
    return original_search


def _patch_vector(vector):
    """Instrument FAISSVectorStore.search() to count and time vector calls."""
    original_search = vector.search

    def instrumented_search(embedding, **kwargs):
        t0 = time.perf_counter()
        result = original_search(embedding, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        with _instrument._lock:
            _instrument.vector_calls += 1
            _instrument.vector_latencies_ms.append(latency_ms)
        return result

    vector.search = instrumented_search
    return original_search


def _patch_reranker(reranker):
    """Instrument reranker.rerank() to count and time rerank calls."""
    original_rerank = reranker.rerank

    def instrumented_rerank(query, results, top_k, **kwargs):
        t0 = time.perf_counter()
        result = original_rerank(query, results, top_k, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        with _instrument._lock:
            _instrument.rerank_calls += 1
            _instrument.rerank_latencies_ms.append(latency_ms)
        return result

    reranker.rerank = instrumented_rerank
    return original_rerank


def _patch_evidence_selector(selector):
    """Instrument EvidenceSelector.select() to count and time calls."""
    original_select = selector.select

    def instrumented_select(evidence, *args, **kwargs):
        input_size = len(evidence)
        t0 = time.perf_counter()
        result = original_select(evidence, *args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        output_size = len(result)
        with _instrument._lock:
            _instrument.evidence_selector_calls += 1
            _instrument.evidence_selector_latencies_ms.append(latency_ms)
            _instrument.evidence_selector_input_sizes.append(input_size)
            _instrument.evidence_selector_output_sizes.append(output_size)
        return result

    selector.select = instrumented_select
    return original_select


def _patch_hybrid_search_async(retriever):
    """Instrument search_async() to detect cache bypass."""
    original_search_async = retriever.search_async

    async def instrumented_search_async(query, **kwargs):
        cache_key = f"{query}:{kwargs.get('top_k', retriever.settings.retrieval_top_k)}"
        t0 = time.perf_counter()
        result = await original_search_async(query, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Check if this query was seen before (duplicate detection)
        with _instrument._lock:
            _instrument.retrieval_calls += 1
            _instrument.retrieval_latencies_ms.append(latency_ms)
            if cache_key in _instrument._seen_retrievals:
                _instrument.duplicate_retrievals += 1
            else:
                _instrument._seen_retrievals.add(cache_key)
        return result

    retriever.search_async = instrumented_search_async
    return original_search_async


# ── Benchmark runner ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class QueryResult:
    query_id: str
    query_class: str
    query: str
    total_latency_ms: float
    answer: str
    stop_reason: str
    iterations_used: int
    outcome: str
    # Per-stage latency
    classify_latency_ms: float = 0.0
    plan_latency_ms: float = 0.0
    retrieve_total_latency_ms: float = 0.0
    assess_latency_ms: float = 0.0
    synthesize_latency_ms: float = 0.0
    # Call counts
    llm_calls: int = 0
    embedding_calls: int = 0
    bm25_calls: int = 0
    vector_calls: int = 0
    rerank_calls: int = 0
    evidence_selector_calls: int = 0
    retrieval_calls: int = 0
    # Duplicate work
    duplicate_embeddings: int = 0
    duplicate_retrievals: int = 0
    # Evidence selector
    evidence_selector_input_sizes: list[int] = dataclasses.field(default_factory=list)
    evidence_selector_output_sizes: list[int] = dataclasses.field(default_factory=list)
    evidence_selector_latencies_ms: list[float] = dataclasses.field(default_factory=list)
    # LLM details
    llm_latencies_ms: list[float] = dataclasses.field(default_factory=list)
    llm_call_types: list[str] = dataclasses.field(default_factory=list)
    # Telemetry
    telemetry: dict = dataclasses.field(default_factory=dict)
    # Provider failure
    provider_failed: bool = False
    error: str = ""


async def run_single_query(
    query_item: dict,
    router,
    retriever,
    reranker,
    settings,
) -> QueryResult:
    """Run one query through the full pipeline with instrumentation."""
    from app.orchestration.graph import run_query
    from app.orchestration.models import ResearchPlan

    qid = query_item["id"]
    query = query_item["query"]
    query_class = query_item["class"]

    _instrument.reset()

    # Instrument all components
    orig_complete = _patch_llm_router(router)
    orig_embed = _patch_embedder(retriever.embedder)
    orig_bm25 = _patch_bm25(retriever.bm25)
    orig_vector = _patch_vector(retriever.vector)
    orig_rerank = _patch_reranker(reranker)

    # Patch evidence selector (created inside nodes, so we need to patch the class)
    from app.retrieval.evidence_selector import EvidenceSelector
    orig_select = EvidenceSelector.select
    EvidenceSelector.select = lambda self, evidence, *a, **kw: _instrument_evidence_selector(self, evidence, *a, **kw, _original=orig_select)

    # Patch search_async for duplicate detection
    orig_search_async = _patch_hybrid_search_async(retriever)

    t0 = time.perf_counter()
    try:
        result = await run_query(
            query=query,
            request_id=f"bench:{qid}",
            router=router,
            retriever=retriever,
            reranker=reranker,
            settings=settings,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        # Extract telemetry if available
        telemetry = result.telemetry or {}

        qr = QueryResult(
            query_id=qid,
            query_class=query_class,
            query=query,
            total_latency_ms=round(total_ms, 2),
            answer=result.answer[:200],
            stop_reason=result.stop_reason.value if hasattr(result.stop_reason, 'value') else str(result.stop_reason),
            iterations_used=result.iterations_used,
            outcome=result.outcome.value if hasattr(result.outcome, 'value') else str(result.outcome),
            llm_calls=_instrument.llm_calls,
            embedding_calls=_instrument.embedding_calls,
            bm25_calls=_instrument.bm25_calls,
            vector_calls=_instrument.vector_calls,
            rerank_calls=_instrument.rerank_calls,
            evidence_selector_calls=_instrument.evidence_selector_calls,
            retrieval_calls=_instrument.retrieval_calls,
            duplicate_embeddings=_instrument.duplicate_embeddings,
            duplicate_retrievals=_instrument.duplicate_retrievals,
            evidence_selector_input_sizes=list(_instrument.evidence_selector_input_sizes),
            evidence_selector_output_sizes=list(_instrument.evidence_selector_output_sizes),
            evidence_selector_latencies_ms=list(_instrument.evidence_selector_latencies_ms),
            llm_latencies_ms=list(_instrument.llm_latencies_ms),
            llm_call_types=list(_instrument.llm_call_types),
            telemetry=telemetry,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000
        qr = QueryResult(
            query_id=qid,
            query_class=query_class,
            query=query,
            total_latency_ms=round(total_ms, 2),
            answer="",
            stop_reason="error",
            iterations_used=0,
            outcome="error",
            llm_calls=_instrument.llm_calls,
            embedding_calls=_instrument.embedding_calls,
            bm25_calls=_instrument.bm25_calls,
            vector_calls=_instrument.vector_calls,
            rerank_calls=_instrument.rerank_calls,
            evidence_selector_calls=_instrument.evidence_selector_calls,
            retrieval_calls=_instrument.retrieval_calls,
            error=str(exc),
            provider_failed=True,
        )

    # Restore original methods
    router.complete = orig_complete
    retriever.embedder.embed_texts = orig_embed
    retriever.bm25.search = orig_bm25
    retriever.vector.search = orig_vector
    retriever.rerank = orig_rerank
    retriever.search_async = orig_search_async
    EvidenceSelector.select = orig_select

    return qr


def _instrument_evidence_selector(self, evidence, *args, _original=None, **kwargs):
    """Wrapper for evidence selector that tracks timing."""
    input_size = len(evidence)
    t0 = time.perf_counter()
    result = _original(self, evidence, *args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000
    output_size = len(result)
    with _instrument._lock:
        _instrument.evidence_selector_calls += 1
        _instrument.evidence_selector_latencies_ms.append(latency_ms)
        _instrument.evidence_selector_input_sizes.append(input_size)
        _instrument.evidence_selector_output_sizes.append(output_size)
    return result


# ── Statistics helpers ───────────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    import numpy as np
    return float(np.percentile(data, pct)) if data else 0.0


def _mean(data: list[float]) -> float:
    import numpy as np
    return float(np.mean(data)) if data else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 90)
    print("ARGUS Phase 33: System-Level Performance Profiler")
    print("=" * 90)

    # 1. Build benchmark store
    print("\n[1/4] Building benchmark corpus & store...")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # 2. Build retriever + indexes
    print("\n[2/4] Building retrieval indexes...")
    t0 = time.monotonic()
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import Reranker, NoOpReranker
    from app.config import get_settings
    from app.llm_gateway import get_router

    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # 3. Get router, settings, reranker
    settings = get_settings()
    router = get_router()
    try:
        reranker = Reranker()
    except Exception:
        reranker = NoOpReranker()

    # 4. Warm-up (1 query to initialize models)
    print("\n[3/4] Warming up...")
    _ = retriever.embedder.embed_texts(["warm up"])
    _ = retriever.bm25.search("warm up", top_k=5)
    print("  Done.")

    # 5. Run benchmark
    print(f"\n[4/4] Profiling {len(BENCHMARK_QUERIES)} queries...")
    results: list[QueryResult] = []

    for i, qi in enumerate(BENCHMARK_QUERIES, 1):
        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qi['id']:<10} {qi['class']:<24} ... ", end="", flush=True)
        qr = await run_single_query(qi, router, retriever, reranker, settings)
        results.append(qr)
        status = "OK" if not qr.provider_failed else "FAIL"
        print(f"{qr.total_latency_ms:>7.0f}ms  LLM={qr.llm_calls}  Emb={qr.embedding_calls}  Ret={qr.retrieval_calls}  [{status}]")

    # ══════════════════════════════════════════════════════════════════════
    # AGGREGATE REPORT
    # ══════════════════════════════════════════════════════════════════════

    successful = [r for r in results if not r.provider_failed]
    failed = [r for r in results if r.provider_failed]

    print("\n" + "=" * 90)
    print(f"RESULTS: {len(successful)} succeeded, {len(failed)} failed (provider errors)")
    print("=" * 90)

    if successful:
        # Aggregate latency
        total_latencies = [r.total_latency_ms for r in successful]
        llm_call_counts = [r.llm_calls for r in successful]
        emb_call_counts = [r.embedding_calls for r in successful]
        bm25_call_counts = [r.bm25_calls for r in successful]
        vec_call_counts = [r.vector_calls for r in successful]
        rerank_call_counts = [r.rerank_calls for r in successful]
        esel_call_counts = [r.evidence_selector_calls for r in successful]
        ret_call_counts = [r.retrieval_calls for r in successful]
        dup_emb = [r.duplicate_embeddings for r in successful]
        dup_ret = [r.duplicate_retrievals for r in successful]

        print(f"\n{'Metric':<40} {'Avg':>8} {'P50':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
        print("-" * 80)
        for name, vals in [
            ("Total latency (ms)", total_latencies),
            ("LLM calls", llm_call_counts),
            ("Embedding calls", emb_call_counts),
            ("BM25 calls", bm25_call_counts),
            ("Vector calls", vec_call_counts),
            ("Rerank calls", rerank_call_counts),
            ("Evidence selector calls", esel_call_counts),
            ("Retrieval calls", ret_call_counts),
            ("Duplicate embeddings", dup_emb),
            ("Duplicate retrievals", dup_ret),
        ]:
            print(f"{name:<40} {_mean(vals):>8.1f} {_percentile(vals, 50):>8.1f} {_percentile(vals, 95):>8.1f} {min(vals) if vals else 0:>8.1f} {max(vals) if vals else 0:>8.1f}")

        # LLM latency breakdown
        all_llm_latencies = []
        for r in successful:
            all_llm_latencies.extend(r.llm_latencies_ms)
        if all_llm_latencies:
            print(f"\n{'LLM latency (ms)':<40} {_mean(all_llm_latencies):>8.1f} {_percentile(all_llm_latencies, 50):>8.1f} {_percentile(all_llm_latencies, 95):>8.1f} {min(all_llm_latencies):>8.1f} {max(all_llm_latencies):>8.1f}")

        # LLM call types
        all_call_types = []
        for r in successful:
            all_call_types.extend(r.llm_call_types)
        if all_call_types:
            type_counts = defaultdict(int)
            for ct in all_call_types:
                type_counts[ct] += 1
            print(f"\nLLM call type distribution:")
            for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"  {ct:<30} {count:>5}")

        # Evidence selector analysis
        all_esel_latencies = []
        all_esel_inputs = []
        all_esel_outputs = []
        for r in successful:
            all_esel_latencies.extend(r.evidence_selector_latencies_ms)
            all_esel_inputs.extend(r.evidence_selector_input_sizes)
            all_esel_outputs.extend(r.evidence_selector_output_sizes)
        if all_esel_latencies:
            print(f"\nEvidence selector analysis:")
            print(f"  Total calls: {sum(esel_call_counts)}")
            print(f"  Avg latency: {_mean(all_esel_latencies):.2f}ms")
            print(f"  Avg input size: {_mean(all_esel_inputs):.1f} chunks")
            print(f"  Avg output size: {_mean(all_esel_outputs):.1f} chunks")
            print(f"  Avg reduction: {1 - _mean(all_esel_outputs) / max(1, _mean(all_esel_inputs)):.1%}")

        # Duplicate work summary
        total_dup_emb = sum(dup_emb)
        total_dup_ret = sum(dup_ret)
        total_emb = sum(emb_call_counts)
        total_ret = sum(ret_call_counts)
        print(f"\nDuplicate work summary:")
        print(f"  Duplicate embeddings: {total_dup_emb}/{total_emb} ({total_dup_emb/max(1,total_emb):.1%})")
        print(f"  Duplicate retrievals: {total_dup_ret}/{total_ret} ({total_dup_ret/max(1,total_ret):.1%})")

        # Per-query detail
        print(f"\n{'ID':<10} {'Class':<24} {'Total':>7} {'LLM':>4} {'Emb':>4} {'BM25':>4} {'Vec':>4} {'Rerank':>6} {'ESel':>5} {'DupE':>5} {'DupR':>5} {'Stop':<20}")
        print("-" * 110)
        for r in successful:
            print(f"{r.query_id:<10} {r.query_class:<24} {r.total_latency_ms:>7.0f} {r.llm_calls:>4} {r.embedding_calls:>4} {r.bm25_calls:>4} {r.vector_calls:>4} {r.rerank_calls:>6} {r.evidence_selector_calls:>5} {r.duplicate_embeddings:>5} {r.duplicate_retrievals:>5} {r.stop_reason:<20}")

    if failed:
        print(f"\nFailed queries:")
        for r in failed:
            print(f"  {r.query_id}: {r.error[:80]}")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE JSON
    # ══════════════════════════════════════════════════════════════════════

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase33_performance",
        "query_count": len(BENCHMARK_QUERIES),
        "successful": len(successful),
        "failed": len(failed),
        "aggregate": {},
        "per_query": [],
    }

    if successful:
        total_latencies = [r.total_latency_ms for r in successful]
        report["aggregate"] = {
            "total_latency_ms": {
                "avg": round(_mean(total_latencies), 2),
                "p50": round(_percentile(total_latencies, 50), 2),
                "p95": round(_percentile(total_latencies, 95), 2),
            },
            "llm_calls": {
                "avg": round(_mean([r.llm_calls for r in successful]), 2),
                "total": sum(r.llm_calls for r in successful),
            },
            "embedding_calls": {
                "avg": round(_mean([r.embedding_calls for r in successful]), 2),
                "total": sum(r.embedding_calls for r in successful),
            },
            "retrieval_calls": {
                "avg": round(_mean([r.retrieval_calls for r in successful]), 2),
                "total": sum(r.retrieval_calls for r in successful),
            },
            "evidence_selector_calls": {
                "avg": round(_mean([r.evidence_selector_calls for r in successful]), 2),
                "total": sum(r.evidence_selector_calls for r in successful),
            },
            "duplicate_embeddings": sum(r.duplicate_embeddings for r in successful),
            "duplicate_retrievals": sum(r.duplicate_retrievals for r in successful),
        }

    for r in results:
        report["per_query"].append({
            "query_id": r.query_id,
            "query_class": r.query_class,
            "query": r.query,
            "total_latency_ms": r.total_latency_ms,
            "answer": r.answer,
            "stop_reason": r.stop_reason,
            "iterations_used": r.iterations_used,
            "outcome": r.outcome,
            "llm_calls": r.llm_calls,
            "embedding_calls": r.embedding_calls,
            "bm25_calls": r.bm25_calls,
            "vector_calls": r.vector_calls,
            "rerank_calls": r.rerank_calls,
            "evidence_selector_calls": r.evidence_selector_calls,
            "retrieval_calls": r.retrieval_calls,
            "duplicate_embeddings": r.duplicate_embeddings,
            "duplicate_retrievals": r.duplicate_retrievals,
            "evidence_selector_latencies_ms": r.evidence_selector_latencies_ms,
            "evidence_selector_input_sizes": r.evidence_selector_input_sizes,
            "evidence_selector_output_sizes": r.evidence_selector_output_sizes,
            "llm_latencies_ms": r.llm_latencies_ms,
            "llm_call_types": r.llm_call_types,
            "provider_failed": r.provider_failed,
            "error": r.error,
        })

    out_path = output_dir / "phase33_performance.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
