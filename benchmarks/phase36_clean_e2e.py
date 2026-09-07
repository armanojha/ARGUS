#!/usr/bin/env python3
"""Phase 36: Clean Provider E2E Benchmark.

Runs the full production pipeline and uses telemetry to identify
CLEAN (no fallback) vs CONTAMINATED (fallback-used) queries.

Only CLEAN queries are used for the trustworthy E2E baseline.

Usage:
    python -m benchmarks.phase36_clean_e2e
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


# ── Benchmark queries ────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {"id": "P36-Q01", "pattern": "simple_lookup", "query": "Where is Acme Corporation headquartered?",
     "gold_facts": ["New York City"], "supporting_docs": ["doc-a"]},
    {"id": "P36-Q02", "pattern": "normal_qa", "query": "How does Acme Corporation generate revenue?",
     "gold_facts": ["revenue", "products", "services"], "supporting_docs": ["doc-a"]},
    {"id": "P36-Q03", "pattern": "technical_explanation", "query": "How does the Atlas database achieve low latency?",
     "gold_facts": ["columnar", "storage", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "P36-Q04", "pattern": "multi_hop", "query": "Who manages the team that developed Atlas?",
     "gold_facts": ["team", "Atlas", "manager"], "supporting_docs": ["doc-b", "doc-d"]},
    {"id": "P36-Q05", "pattern": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?",
     "gold_facts": ["revenue", "2022", "2023", "growth"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P36-Q06", "pattern": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?",
     "gold_facts": ["revenue", "2023", "conflicting"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P36-Q07", "pattern": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?",
     "gold_facts": ["competitive", "advantage", "analytics"], "supporting_docs": ["doc-a", "doc-b"]},
    {"id": "P36-Q08", "pattern": "absent_info", "query": "What is Acme's market share in the European robotics market?",
     "gold_facts": [], "supporting_docs": []},
]


# ── Retrieval instrumentation ────────────────────────────────────────────────

class RetrievalInstrument:
    """Thread-safe instrumentation for retrieval pipeline components."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.bm25_latencies_ms: list[float] = []
            self.vector_latencies_ms: list[float] = []
            self.embedding_latencies_ms: list[float] = []
            self.fusion_latencies_ms: list[float] = []
            self.rerank_latencies_ms: list[float] = []
            self.esel_latencies_ms: list[float] = []
            self.bm25_calls = 0
            self.vector_calls = 0
            self.embedding_calls = 0
            self.fusion_calls = 0
            self.rerank_calls = 0
            self.esel_calls = 0

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "bm25_calls": self.bm25_calls,
                "bm25_latencies_ms": list(self.bm25_latencies_ms),
                "vector_calls": self.vector_calls,
                "vector_latencies_ms": list(self.vector_latencies_ms),
                "embedding_calls": self.embedding_calls,
                "embedding_latencies_ms": list(self.embedding_latencies_ms),
                "fusion_calls": self.fusion_calls,
                "fusion_latencies_ms": list(self.fusion_latencies_ms),
                "rerank_calls": self.rerank_calls,
                "rerank_latencies_ms": list(self.rerank_latencies_ms),
                "esel_calls": self.esel_calls,
                "esel_latencies_ms": list(self.esel_latencies_ms),
            }


_inst = RetrievalInstrument()


def patch_retriever(retriever):
    """Instrument retriever methods. Returns restore function."""
    from app.retrieval.hybrid import HybridRetriever

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

    orig_embed = retriever.embedder.embed_texts
    def patched_embed(texts, **kwargs):
        t0 = time.perf_counter()
        result = orig_embed(texts, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.embedding_calls += 1
            _inst.embedding_latencies_ms.append(ms)
        return result
    retriever.embedder.embed_texts = patched_embed

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
        HybridRetriever._fuse = orig_fuse

    return restore


def patch_evidence_selector():
    """Instrument EvidenceSelector.select(). Returns restore function."""
    from app.retrieval.evidence_selector import EvidenceSelector
    orig_select = EvidenceSelector.select

    def patched_select(self, evidence, *args, **kwargs):
        t0 = time.perf_counter()
        result = orig_select(self, evidence, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.esel_calls += 1
            _inst.esel_latencies_ms.append(ms)
        return result

    EvidenceSelector.select = patched_select

    def restore():
        EvidenceSelector.select = orig_select

    return restore


def patch_reranker(reranker):
    """Instrument reranker.rerank(). Returns restore function."""
    orig_rerank = reranker.rerank
    def patched_rerank(query, chunks, top_k=10, **kwargs):
        t0 = time.perf_counter()
        result = orig_rerank(query, chunks, top_k=top_k, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.rerank_calls += 1
            _inst.rerank_latencies_ms.append(ms)
        return result
    reranker.rerank = patched_rerank

    def restore():
        reranker.rerank = orig_rerank

    return restore


# ── Helpers ──────────────────────────────────────────────────────────────────

def _percentile(data, pct):
    import numpy as np
    return float(np.percentile(data, pct)) if data else 0.0

def _mean(data):
    import numpy as np
    return float(np.mean(data)) if data else 0.0


def extract_telemetry_data(telemetry_summary: dict | None) -> dict:
    """Extract LLM call data from telemetry summary."""
    if not telemetry_summary:
        return {"calls": 0, "total_ms": 0, "by_type": {}, "providers": [], "models": [],
                "fallbacks": 0, "rate_limits": 0, "is_clean": True, "fallback_details": []}

    decisions = telemetry_summary.get("routing_decisions", [])
    by_type = defaultdict(lambda: {"count": 0, "total_ms": 0, "latencies_ms": []})
    providers = set()
    models = set()
    fallbacks = 0
    rate_limits = 0
    total_ms = 0
    fallback_details = []

    for d in decisions:
        ct = d.get("call_type", "unknown")
        lat = d.get("latency_ms", 0) or 0
        by_type[ct]["count"] += 1
        by_type[ct]["total_ms"] += lat
        by_type[ct]["latencies_ms"].append(lat)
        total_ms += lat
        providers.add(d.get("provider", "unknown"))
        models.add(d.get("model", "unknown"))
        if d.get("is_fallback"):
            fallbacks += 1
            fallback_details.append({
                "call_type": ct,
                "provider": d.get("provider"),
                "model": d.get("model"),
                "reason": d.get("fallback_reason"),
            })
        if d.get("error_class") and "rate" in str(d.get("error_class", "")).lower():
            rate_limits += 1

    return {
        "calls": len(decisions),
        "total_ms": total_ms,
        "by_type": dict(by_type),
        "providers": sorted(providers),
        "models": sorted(models),
        "fallbacks": fallbacks,
        "rate_limits": rate_limits,
        "is_clean": fallbacks == 0,
        "fallback_details": fallback_details,
    }


# ── Main benchmark ───────────────────────────────────────────────────────────

async def _run_all_queries(retriever, reranker, settings):
    """Run all benchmark queries in a single event loop."""
    from app.orchestration.graph import run_query
    from app.llm_gateway.telemetry import start_run_telemetry, end_run_telemetry

    results = []
    clean_results = []
    contaminated_results = []

    # Warm-up (2 calls to stabilize providers)
    print("\n[1/3] Warming up...")
    for i in range(2):
        try:
            start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=f"warmup_{i}")
            await run_query("warm up", settings=settings, retriever=retriever, reranker=reranker)
            end_run_telemetry()
        except Exception:
            try:
                end_run_telemetry()
            except Exception:
                pass
    print("  Warm-up complete")

    # Cold start measurement
    print("\n[2/3] Cold start measurement...")
    cold_start_t0 = time.perf_counter()
    try:
        start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id="cold_start")
        await run_query("cold start test", settings=settings, retriever=retriever, reranker=reranker)
        end_run_telemetry()
    except Exception:
        try:
            end_run_telemetry()
        except Exception:
            pass
    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000
    print(f"  Cold start: {cold_start_ms:.0f}ms")

    # Benchmark
    print(f"\n[3/3] Running {len(BENCHMARK_QUERIES)} benchmark queries...")
    print("  (Queries with provider fallbacks will be marked CONTAMINATED)")

    for i, qi in enumerate(BENCHMARK_QUERIES, 1):
        _inst.reset()
        query = qi["query"]
        qid = qi["id"]

        t0 = time.perf_counter()
        try:
            start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=qid)
            result = await run_query(query, settings=settings, retriever=retriever, reranker=reranker)
            telemetry_summary = end_run_telemetry()
        except Exception as e:
            print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} FAILED: {e}")
            try:
                end_run_telemetry()
            except Exception:
                pass
            continue
        total_ms = (time.perf_counter() - t0) * 1000

        telemetry = extract_telemetry_data(telemetry_summary)
        is_clean = telemetry["is_clean"]

        # Build result record
        record = {
            "query_id": qid,
            "pattern": qi["pattern"],
            "query": query,
            "total_latency_ms": round(total_ms, 2),
            "is_clean": is_clean,
            # Retrieval
            "bm25_ms": round(sum(_inst.bm25_latencies_ms), 2),
            "embedding_ms": round(sum(_inst.embedding_latencies_ms), 2),
            "vector_ms": round(sum(_inst.vector_latencies_ms), 2),
            "fusion_ms": round(sum(_inst.fusion_latencies_ms), 2),
            "rerank_ms": round(sum(_inst.rerank_latencies_ms), 2),
            "esel_ms": round(sum(_inst.esel_latencies_ms), 2),
            # LLM
            "llm_calls": telemetry["calls"],
            "llm_total_ms": telemetry["total_ms"],
            "llm_by_type": telemetry["by_type"],
            "providers_used": telemetry["providers"],
            "models_used": telemetry["models"],
            "fallback_count": telemetry["fallbacks"],
            "rate_limit_events": telemetry["rate_limits"],
            "fallback_details": telemetry["fallback_details"],
            # State
            "iterations": getattr(result, 'iterations', 0) or 0,
            "stop_reason": getattr(result, 'stop_reason', None),
            "evidence_count": len(getattr(result, 'citations', []) or []),
            "answer_length": len(getattr(result, 'answer', '') or ''),
            "outcome": getattr(result, 'outcome', 'unknown'),
            "warnings": getattr(result, 'warnings', []) or [],
            "answer": getattr(result, 'answer', '') or '',
        }

        results.append(record)
        if is_clean:
            clean_results.append(record)
        else:
            contaminated_results.append(record)

        status = "CLEAN" if is_clean else f"CONTAMINATED ({telemetry['fallbacks']} fallbacks)"
        ret_ms = record["bm25_ms"] + record["embedding_ms"] + record["vector_ms"] + record["fusion_ms"] + record["rerank_ms"]
        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} {qi['pattern']:<22} "
              f"ret={ret_ms:>6.0f}ms  llm={telemetry['total_ms']:>6.0f}ms({telemetry['calls']}c)  "
              f"total={total_ms:>6.0f}ms  {status}")

    return results, clean_results, contaminated_results, cold_start_ms


def run_clean_e2e():
    print("=" * 90)
    print("ARGUS Phase 36: Clean Provider E2E Benchmark")
    print("=" * 90)

    # Build store and retriever
    print("\n[Setup] Building benchmark corpus & store...")
    t0 = time.monotonic()
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    print("\n[Setup] Building retrieval indexes...")
    t0 = time.monotonic()
    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    from app.reranking.reranker import Reranker, NoOpReranker
    try:
        reranker = Reranker()
    except Exception:
        reranker = NoOpReranker()

    # Instrument
    restore_retriever = patch_retriever(retriever)
    restore_esel = patch_evidence_selector()
    restore_reranker = patch_reranker(reranker)

    from app.config import Settings
    settings = Settings()

    import asyncio
    results, clean_results, contaminated_results, cold_start_ms = asyncio.run(
        _run_all_queries(retriever, reranker, settings)
    )

    # Restore
    restore_retriever()
    restore_esel()
    restore_reranker()

    if not results:
        print("\nNo successful results. Exiting.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════

    total_avg = _mean([r["total_latency_ms"] for r in results])

    print("\n" + "=" * 90)
    print("PROVIDER STABILITY REPORT")
    print("=" * 90)
    print(f"\n  Total queries:      {len(results)}")
    print(f"  Clean (no fallback): {len(clean_results)}")
    print(f"  Contaminated:       {len(contaminated_results)}")
    print(f"  Cold start:         {cold_start_ms:.0f}ms")

    if contaminated_results:
        print(f"\n  Contaminated queries:")
        for r in contaminated_results:
            print(f"    {r['query_id']}: {r['fallback_count']} fallbacks — {r['fallback_details']}")

    # CLEAN results only
    if clean_results:
        print("\n" + "=" * 90)
        print("CLEAN E2E LATENCY BREAKDOWN (no fallbacks)")
        print("=" * 90)

        clean_total_avg = _mean([r["total_latency_ms"] for r in clean_results])

        stages = [
            ("BM25", [r["bm25_ms"] for r in clean_results]),
            ("Embedding", [r["embedding_ms"] for r in clean_results]),
            ("Vector", [r["vector_ms"] for r in clean_results]),
            ("Fusion", [r["fusion_ms"] for r in clean_results]),
            ("Rerank", [r["rerank_ms"] for r in clean_results]),
            ("EvidenceSelector", [r["esel_ms"] for r in clean_results]),
        ]

        # Add LLM stages from clean results
        all_llm_types = set()
        for r in clean_results:
            all_llm_types.update(r["llm_by_type"].keys())
        for ct in sorted(all_llm_types):
            vals = [r["llm_by_type"].get(ct, {}).get("total_ms", 0) for r in clean_results]
            stages.append((ct, vals))

        stages.append(("Total E2E", [r["total_latency_ms"] for r in clean_results]))

        print(f"\n{'Component':<25} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'% of E2E':>10}")
        print("-" * 65)
        for name, vals in stages:
            avg = _mean(vals)
            pct = (avg / clean_total_avg * 100) if clean_total_avg > 0 else 0
            print(f"{name:<25} {avg:>10.1f} {_percentile(vals, 50):>10.1f} {_percentile(vals, 95):>10.1f} {pct:>9.1f}%")

        # LLM call audit (clean only)
        print("\n" + "=" * 90)
        print("LLM CALL AUDIT (CLEAN ONLY)")
        print("=" * 90)

        total_llm_calls = sum(r["llm_calls"] for r in clean_results)
        total_llm_ms = sum(r["llm_total_ms"] for r in clean_results)
        avg_llm_calls = total_llm_calls / len(clean_results) if clean_results else 0

        print(f"\n  Total LLM calls: {total_llm_calls}")
        print(f"  Avg calls/query: {avg_llm_calls:.1f}")
        print(f"  Total LLM latency: {total_llm_ms:.0f}ms")
        print(f"  LLM % of E2E: {total_llm_ms/sum(r['total_latency_ms'] for r in clean_results)*100:.1f}%")

        all_ct = defaultdict(lambda: {"count": 0, "total_ms": 0})
        for r in clean_results:
            for ct, d in r["llm_by_type"].items():
                all_ct[ct]["count"] += d["count"]
                all_ct[ct]["total_ms"] += d["total_ms"]

        print(f"\n{'Call Type':<25} {'Count':>8} {'Total ms':>12} {'Avg ms':>10} {'% of LLM':>10}")
        print("-" * 65)
        for ct in sorted(all_ct.keys()):
            c = all_ct[ct]
            avg = c["total_ms"] / c["count"] if c["count"] else 0
            pct = c["total_ms"] / total_llm_ms * 100 if total_llm_ms else 0
            print(f"{ct:<25} {c['count']:>8} {c['total_ms']:>12.0f} {avg:>10.0f} {pct:>9.1f}%")

        # Per-pattern breakdown (clean only)
        print("\n" + "=" * 90)
        print("LATENCY BY QUERY PATTERN (CLEAN ONLY)")
        print("=" * 90)

        patterns = defaultdict(list)
        for r in clean_results:
            patterns[r["pattern"]].append(r)

        print(f"\n{'Pattern':<22} {'Count':>5} {'Avg E2E':>10} {'Retrieval':>10} {'LLM':>10} {'LLM calls':>10}")
        print("-" * 75)
        for pattern in sorted(patterns.keys()):
            prs = patterns[pattern]
            avg_e2e = _mean([r["total_latency_ms"] for r in prs])
            avg_ret = _mean([r["bm25_ms"] + r["embedding_ms"] + r["vector_ms"] + r["fusion_ms"] + r["rerank_ms"] for r in prs])
            avg_llm = _mean([r["llm_total_ms"] for r in prs])
            avg_llm_calls = _mean([r["llm_calls"] for r in prs])
            print(f"{pattern:<22} {len(prs):>5} {avg_e2e:>10.0f} {avg_ret:>10.0f} {avg_llm:>10.0f} {avg_llm_calls:>10.1f}")

    # ALL results (for comparison)
    print("\n" + "=" * 90)
    print("ALL RESULTS (CLEAN + CONTAMINATED)")
    print("=" * 90)

    all_stages = [
        ("BM25", [r["bm25_ms"] for r in results]),
        ("Embedding", [r["embedding_ms"] for r in results]),
        ("Vector", [r["vector_ms"] for r in results]),
        ("Fusion", [r["fusion_ms"] for r in results]),
        ("Rerank", [r["rerank_ms"] for r in results]),
        ("EvidenceSelector", [r["esel_ms"] for r in results]),
        ("Total E2E", [r["total_latency_ms"] for r in results]),
    ]

    print(f"\n{'Component':<25} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'% of E2E':>10}")
    print("-" * 65)
    for name, vals in all_stages:
        avg = _mean(vals)
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"{name:<25} {avg:>10.1f} {_percentile(vals, 50):>10.1f} {_percentile(vals, 95):>10.1f} {pct:>9.1f}%")

    # Save JSON
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase36_clean_e2e",
        "cold_start_ms": round(cold_start_ms, 2),
        "total_queries": len(results),
        "clean_queries": len(clean_results),
        "contaminated_queries": len(contaminated_results),
        "clean_avg_e2e_ms": round(_mean([r["total_latency_ms"] for r in clean_results]), 2) if clean_results else 0,
        "all_avg_e2e_ms": round(total_avg, 2),
        "contaminated_details": [
            {"id": r["query_id"], "fallbacks": r["fallback_count"], "details": r["fallback_details"]}
            for r in contaminated_results
        ],
        "per_query": results,
    }

    out_path = output_dir / "phase36_clean_e2e.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    run_clean_e2e()
