#!/usr/bin/env python3
"""Phase 33: Retrieval-Only Diagnostic Profiler.

Measures the retrieval pipeline WITHOUT LLM calls to avoid provider
rate limits. Focuses on:
- Embedding call count and timing
- BM25 call count and timing
- Vector search count and timing
- Evidence selector call count and timing
- Duplicate work detection
- Cache hit/miss rates
- search_async() cache bypass measurement
"""
from __future__ import annotations

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


# ── Instrumentation ──────────────────────────────────────────────────────────

class RetrievalInstrument:
    """Thread-safe instrumentation for retrieval pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.bm25_calls = 0
            self.bm25_latencies_ms: list[float] = []
            self.vector_calls = 0
            self.vector_latencies_ms: list[float] = []
            self.embedding_calls = 0
            self.embedding_latencies_ms: list[float] = []
            self.rerank_calls = 0
            self.rerank_latencies_ms: list[float] = []
            self.fusion_calls = 0
            self.fusion_latencies_ms: list[float] = []
            self.esel_calls = 0
            self.esel_latencies_ms: list[float] = []
            self.esel_input_sizes: list[int] = []
            self.esel_output_sizes: list[int] = []
            self.search_async_calls = 0
            self.search_sync_calls = 0
            self.cache_check_count = 0
            self.cache_hit_count = 0
            self.duplicate_embeddings: list[str] = []
            self._seen_embeddings: set[str] = set()
            self.total_retrieval_ms = 0.0
            self.total_rerank_ms = 0.0
            self.total_esel_ms = 0.0

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "bm25_calls": self.bm25_calls,
                "bm25_latencies_ms": list(self.bm25_latencies_ms),
                "vector_calls": self.vector_calls,
                "vector_latencies_ms": list(self.vector_latencies_ms),
                "embedding_calls": self.embedding_calls,
                "embedding_latencies_ms": list(self.embedding_latencies_ms),
                "rerank_calls": self.rerank_calls,
                "rerank_latencies_ms": list(self.rerank_latencies_ms),
                "fusion_calls": self.fusion_calls,
                "fusion_latencies_ms": list(self.fusion_latencies_ms),
                "esel_calls": self.esel_calls,
                "esel_latencies_ms": list(self.esel_latencies_ms),
                "esel_input_sizes": list(self.esel_input_sizes),
                "esel_output_sizes": list(self.esel_output_sizes),
                "search_async_calls": self.search_async_calls,
                "search_sync_calls": self.search_sync_calls,
                "cache_check_count": self.cache_check_count,
                "cache_hit_count": self.cache_hit_count,
                "duplicate_embeddings": list(self.duplicate_embeddings),
                "total_retrieval_ms": self.total_retrieval_ms,
                "total_rerank_ms": self.total_rerank_ms,
                "total_esel_ms": self.total_esel_ms,
            }


_inst = RetrievalInstrument()


def patch_retriever(retriever):
    """Instrument retriever methods. Returns unpatter function."""
    from app.retrieval.hybrid import HybridRetriever

    # Patch BM25
    orig_bm25_search = retriever.bm25.search
    def patched_bm25(query, **kwargs):
        t0 = time.perf_counter()
        result = orig_bm25_search(query, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.bm25_calls += 1
            _inst.bm25_latencies_ms.append(ms)
        return result
    retriever.bm25.search = patched_bm25

    # Patch vector
    orig_vector_search = retriever.vector.search
    def patched_vector(embedding, **kwargs):
        t0 = time.perf_counter()
        result = orig_vector_search(embedding, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.vector_calls += 1
            _inst.vector_latencies_ms.append(ms)
        return result
    retriever.vector.search = patched_vector

    # Patch embedder
    orig_embed = retriever.embedder.embed_texts
    def patched_embed(texts, **kwargs):
        t0 = time.perf_counter()
        result = orig_embed(texts, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.embedding_calls += 1
            _inst.embedding_latencies_ms.append(ms)
            for text in texts:
                key = text.strip().lower()
                if key in _inst._seen_embeddings:
                    _inst.duplicate_embeddings.append(text[:60])
                else:
                    _inst._seen_embeddings.add(key)
        return result
    retriever.embedder.embed_texts = patched_embed

    # Patch search_async
    orig_search_async = retriever.search_async
    async def patched_search_async(query, **kwargs):
        with _inst._lock:
            _inst.search_async_calls += 1
        t0 = time.perf_counter()
        result = await orig_search_async(query, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return result
    retriever.search_async = patched_search_async

    # Patch search (sync)
    orig_search = retriever.search
    def patched_search(query, **kwargs):
        cache_key = f"{query}:{kwargs.get('top_k', retriever.settings.retrieval_top_k)}"
        with _inst._lock:
            _inst.search_sync_calls += 1
            _inst.cache_check_count += 1
            if cache_key in retriever._result_cache:
                _inst.cache_hit_count += 1
        result = orig_search(query, **kwargs)
        return result
    retriever.search = patched_search

    # Patch fusion
    orig_fuse = HybridRetriever._fuse
    @staticmethod
    def patched_fuse(store, query, top_k, bm25_weight, vector_weight, bm25_scores, vector_scores, **kwargs):
        t0 = time.perf_counter()
        result = orig_fuse(store, query, top_k, bm25_weight, vector_weight, bm25_scores, vector_scores, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.fusion_calls += 1
            _inst.fusion_latencies_ms.append(ms)
        return result
    HybridRetriever._fuse = patched_fuse

    def restore():
        retriever.bm25.search = orig_bm25_search
        retriever.vector.search = orig_vector_search
        retriever.embedder.embed_texts = orig_embed
        retriever.search_async = orig_search_async
        retriever.search = orig_search
        HybridRetriever._fuse = orig_fuse

    return restore


def patch_evidence_selector():
    """Instrument EvidenceSelector.select(). Returns unpatch function."""
    from app.retrieval.evidence_selector import EvidenceSelector
    orig_select = EvidenceSelector.select

    def patched_select(self, evidence, *args, **kwargs):
        input_size = len(evidence)
        t0 = time.perf_counter()
        result = orig_select(self, evidence, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        output_size = len(result)
        with _inst._lock:
            _inst.esel_calls += 1
            _inst.esel_latencies_ms.append(ms)
            _inst.esel_input_sizes.append(input_size)
            _inst.esel_output_sizes.append(output_size)
        return result

    EvidenceSelector.select = patched_select

    def restore():
        EvidenceSelector.select = orig_select

    return restore


# ── Benchmark ────────────────────────────────────────────────────────────────

def _percentile(data, pct):
    import numpy as np
    return float(np.percentile(data, pct)) if data else 0.0

def _mean(data):
    import numpy as np
    return float(np.mean(data)) if data else 0.0


@dataclasses.dataclass
class RetrievalResult:
    query_id: str
    query_class: str
    query: str
    total_retrieval_ms: float
    bm25_ms: float
    embedding_ms: float
    vector_ms: float
    fusion_ms: float
    rerank_ms: float
    esel_ms: float
    bm25_calls: int
    embedding_calls: int
    vector_calls: int
    rerank_calls: int
    esel_calls: int
    fusion_calls: int
    duplicate_embeddings: int
    search_async_calls: int
    search_sync_calls: int
    cache_hits: int
    cache_checks: int
    evidence_count: int
    selected_count: int


def run_retrieval_benchmark():
    print("=" * 90)
    print("ARGUS Phase 33: Retrieval-Only Diagnostic Profiler")
    print("=" * 90)

    # 1. Build store
    print("\n[1/3] Building benchmark corpus & store...")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # 2. Build retriever
    print("\n[2/3] Building retrieval indexes...")
    t0 = time.monotonic()
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.evidence_selector import EvidenceSelector
    from app.reranking.reranker import Reranker, NoOpReranker

    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    try:
        reranker = Reranker()
    except Exception:
        reranker = NoOpReranker()

    # 3. Instrument
    restore_retriever = patch_retriever(retriever)
    restore_esel = patch_evidence_selector()

    # 4. Warm-up
    print("\n[3/3] Running retrieval benchmark...")
    _ = retriever.search("warm up", top_k=5)
    _ = retriever.embedder.embed_texts(["warm up"])

    results = []
    for i, qi in enumerate(BENCHMARK_QUERIES, 1):
        _inst.reset()
        query = qi["query"]
        qid = qi["id"]

        t0 = time.perf_counter()
        refs = retriever.search(query, top_k=20)
        total_ret_ms = (time.perf_counter() - t0) * 1000

        # Rerank
        t0 = time.perf_counter()
        reranked = reranker.rerank(query, refs, top_k=10) if refs else []
        rerank_ms = (time.perf_counter() - t0) * 1000

        # Evidence selection
        t0 = time.perf_counter()
        selected = evidence_selector_select(reranked) if reranked else []
        esel_ms = (time.perf_counter() - t0) * 1000

        bm25_ms = sum(_inst.bm25_latencies_ms) if _inst.bm25_latencies_ms else 0
        emb_ms = sum(_inst.embedding_latencies_ms) if _inst.embedding_latencies_ms else 0
        vec_ms = sum(_inst.vector_latencies_ms) if _inst.vector_latencies_ms else 0
        fusion_ms = sum(_inst.fusion_latencies_ms) if _inst.fusion_latencies_ms else 0

        qr = RetrievalResult(
            query_id=qid,
            query_class=qi["class"],
            query=query,
            total_retrieval_ms=round(total_ret_ms, 2),
            bm25_ms=round(bm25_ms, 2),
            embedding_ms=round(emb_ms, 2),
            vector_ms=round(vec_ms, 2),
            fusion_ms=round(fusion_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            esel_ms=round(esel_ms, 2),
            bm25_calls=_inst.bm25_calls,
            embedding_calls=_inst.embedding_calls,
            vector_calls=_inst.vector_calls,
            rerank_calls=_inst.rerank_calls,
            esel_calls=_inst.esel_calls,
            fusion_calls=_inst.fusion_calls,
            duplicate_embeddings=len(_inst.duplicate_embeddings),
            search_async_calls=_inst.search_async_calls,
            search_sync_calls=_inst.search_sync_calls,
            cache_hits=_inst.cache_hit_count,
            cache_checks=_inst.cache_check_count,
            evidence_count=len(refs),
            selected_count=len(selected),
        )
        results.append(qr)
        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} {qi['class']:<24} "
              f"ret={total_ret_ms:>6.1f}ms  bm25={bm25_ms:>5.1f}ms  emb={emb_ms:>5.1f}ms  "
              f"vec={vec_ms:>5.1f}ms  fuse={fusion_ms:>5.1f}ms  rerank={rerank_ms:>5.1f}ms  "
              f"esel={esel_ms:>5.1f}ms  chunks={len(refs):>2}->{len(selected):>2}")

    # Restore
    restore_retriever()
    restore_esel()

    # ══════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("AGGREGATE RETRIEVAL LATENCY")
    print("=" * 90)

    stages = [
        ("BM25", [r.bm25_ms for r in results]),
        ("Embedding", [r.embedding_ms for r in results]),
        ("Vector", [r.vector_ms for r in results]),
        ("Fusion", [r.fusion_ms for r in results]),
        ("Rerank", [r.rerank_ms for r in results]),
        ("EvidenceSelector", [r.esel_ms for r in results]),
        ("Total retrieval", [r.total_retrieval_ms for r in results]),
    ]

    print(f"\n{'Stage':<22} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 72)
    for name, vals in stages:
        print(f"{name:<22} {_mean(vals):>10.2f} {_percentile(vals, 50):>10.2f} "
              f"{_percentile(vals, 95):>10.2f} {min(vals) if vals else 0:>10.2f} "
              f"{max(vals) if vals else 0:>10.2f}")

    # Call counts
    print(f"\n{'Call type':<22} {'Total calls':>12} {'Avg/call':>10}")
    print("-" * 44)
    for name, vals in [
        ("BM25", [r.bm25_calls for r in results]),
        ("Embedding", [r.embedding_calls for r in results]),
        ("Vector", [r.vector_calls for r in results]),
        ("Fusion", [r.fusion_calls for r in results]),
        ("Rerank", [r.rerank_calls for r in results]),
        ("EvidenceSelector", [r.esel_calls for r in results]),
    ]:
        print(f"{name:<22} {sum(vals):>12} {_mean(vals):>10.1f}")

    # Duplicate work
    total_dup_emb = sum(r.duplicate_embeddings for r in results)
    total_emb = sum(r.embedding_calls for r in results)
    print(f"\nDuplicate work:")
    print(f"  Duplicate embeddings: {total_dup_emb}/{total_emb} ({total_dup_emb/max(1,total_emb):.1%})")
    print(f"  search_async calls: {sum(r.search_async_calls for r in results)}")
    print(f"  search_sync calls: {sum(r.search_sync_calls for r in results)}")
    print(f"  Cache checks: {sum(r.cache_checks for r in results)}")
    print(f"  Cache hits: {sum(r.cache_hits for r in results)}")

    # Evidence selector analysis
    all_inputs = []
    all_outputs = []
    for r in results:
        # These come from the instrument, not per-query
        pass
    print(f"\nEvidence selector:")
    print(f"  Total calls: {sum(r.esel_calls for r in results)}")
    print(f"  Avg latency: {_mean([r.esel_ms for r in results]):.2f}ms")

    # Per-query detail
    print(f"\n{'ID':<10} {'Class':<24} {'Total':>7} {'BM25':>6} {'Emb':>6} {'Vec':>6} {'Fuse':>6} {'Rerank':>7} {'ESel':>6} {'In->Out':>8}")
    print("-" * 95)
    for r in results:
        print(f"{r.query_id:<10} {r.query_class:<24} {r.total_retrieval_ms:>7.1f} {r.bm25_ms:>6.1f} "
              f"{r.embedding_ms:>6.1f} {r.vector_ms:>6.1f} {r.fusion_ms:>6.1f} {r.rerank_ms:>7.1f} "
              f"{r.esel_ms:>6.1f} {r.evidence_count:>3}->{r.selected_count:<3}")

    # Dominant stage
    core_stages = {"BM25": _mean([r.bm25_ms for r in results]),
                   "Embedding": _mean([r.embedding_ms for r in results]),
                   "Vector": _mean([r.vector_ms for r in results]),
                   "Fusion": _mean([r.fusion_ms for r in results]),
                   "Rerank": _mean([r.rerank_ms for r in results]),
                   "EvidenceSelector": _mean([r.esel_ms for r in results])}
    dominant = max(core_stages, key=core_stages.get)
    total = sum(core_stages.values())
    print(f"\nDominant retrieval stage: {dominant} ({core_stages[dominant]:.1f}ms avg, {core_stages[dominant]/max(1,total)*100:.1f}% of retrieval)")
    print(f"Total avg retrieval: {total:.1f}ms")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE JSON
    # ══════════════════════════════════════════════════════════════════════

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase33_retrieval_only",
        "query_count": len(results),
        "aggregate": {
            name: {
                "avg_ms": round(_mean(vals), 2),
                "p50_ms": round(_percentile(vals, 50), 2),
                "p95_ms": round(_percentile(vals, 95), 2),
            }
            for name, vals in stages
        },
        "call_counts": {
            name: {"total": sum(vals), "avg": round(_mean(vals), 2)}
            for name, vals in [
                ("bm25", [r.bm25_calls for r in results]),
                ("embedding", [r.embedding_calls for r in results]),
                ("vector", [r.vector_calls for r in results]),
                ("fusion", [r.fusion_calls for r in results]),
                ("rerank", [r.rerank_calls for r in results]),
                ("evidence_selector", [r.esel_calls for r in results]),
            ]
        },
        "duplicate_work": {
            "duplicate_embeddings": total_dup_emb,
            "total_embeddings": total_emb,
            "search_async_calls": sum(r.search_async_calls for r in results),
            "search_sync_calls": sum(r.search_sync_calls for r in results),
            "cache_checks": sum(r.cache_checks for r in results),
            "cache_hits": sum(r.cache_hits for r in results),
        },
        "dominant_stage": dominant,
        "per_query": [
            {
                "query_id": r.query_id,
                "query_class": r.query_class,
                "query": r.query,
                "total_retrieval_ms": r.total_retrieval_ms,
                "bm25_ms": r.bm25_ms,
                "embedding_ms": r.embedding_ms,
                "vector_ms": r.vector_ms,
                "fusion_ms": r.fusion_ms,
                "rerank_ms": r.rerank_ms,
                "esel_ms": r.esel_ms,
                "bm25_calls": r.bm25_calls,
                "embedding_calls": r.embedding_calls,
                "vector_calls": r.vector_calls,
                "rerank_calls": r.rerank_calls,
                "esel_calls": r.esel_calls,
                "fusion_calls": r.fusion_calls,
                "duplicate_embeddings": r.duplicate_embeddings,
                "evidence_count": r.evidence_count,
                "selected_count": r.selected_count,
            }
            for r in results
        ],
    }

    out_path = output_dir / "phase33_retrieval_only.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


def evidence_selector_select(evidence):
    """Run evidence selector without instrumentation (standalone)."""
    from app.retrieval.evidence_selector import EvidenceSelector
    selector = EvidenceSelector()
    return selector.select(evidence)


if __name__ == "__main__":
    run_retrieval_benchmark()
