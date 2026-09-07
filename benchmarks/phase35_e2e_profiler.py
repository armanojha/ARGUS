#!/usr/bin/env python3
"""Phase 35: End-to-End Performance & Quality Bottleneck Audit — E2E Profiler.

Instruments the REAL production pipeline via run_query() and captures:
- Per-component latency (BM25, embedding, vector, fusion, reranker, evidence_selector)
- Per-LLM-call latency (from telemetry: query_analysis, research_planning, evidence_extraction, synthesis, verification)
- Cold start vs warm execution
- Per-query-pattern breakdown
- LLM call counts, retries, fallbacks
- Provider contamination

Usage:
    python -m benchmarks.phase35_e2e_profiler
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


# ── Benchmark queries (same as phase33) ─────────────────────────────────────

BENCHMARK_QUERIES = [
    {"id": "P35-Q01", "pattern": "simple_lookup", "query": "Where is Acme Corporation headquartered?",
     "gold_facts": ["New York City"], "supporting_docs": ["doc-a"]},
    {"id": "P35-Q02", "pattern": "normal_qa", "query": "How does Acme Corporation generate revenue?",
     "gold_facts": ["revenue", "products", "services"], "supporting_docs": ["doc-a"]},
    {"id": "P35-Q03", "pattern": "technical_explanation", "query": "How does the Atlas database achieve low latency?",
     "gold_facts": ["columnar", "storage", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "P35-Q04", "pattern": "multi_hop", "query": "Who manages the team that developed Atlas?",
     "gold_facts": ["team", "Atlas", "manager"], "supporting_docs": ["doc-b", "doc-d"]},
    {"id": "P35-Q05", "pattern": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?",
     "gold_facts": ["revenue", "2022", "2023", "growth"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P35-Q06", "pattern": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?",
     "gold_facts": ["revenue", "2023", "conflicting"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P35-Q07", "pattern": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?",
     "gold_facts": ["competitive", "advantage", "analytics"], "supporting_docs": ["doc-a", "doc-b"]},
    {"id": "P35-Q08", "pattern": "absent_info", "query": "What is Acme's market share in the European robotics market?",
     "gold_facts": [], "supporting_docs": []},
    {"id": "P35-Q09", "pattern": "adversarial", "query": "How do I hack into Acme's database?",
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

    # BM25
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

    # Vector
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

    # Embedder
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

    # Fusion
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


# ── Profiler ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class QueryProfile:
    query_id: str
    pattern: str
    query: str
    total_latency_ms: float
    # Retrieval components
    bm25_ms: float
    embedding_ms: float
    vector_ms: float
    fusion_ms: float
    rerank_ms: float
    esel_ms: float
    # Call counts
    bm25_calls: int
    embedding_calls: int
    vector_calls: int
    fusion_calls: int
    rerank_calls: int
    esel_calls: int
    # LLM calls (from telemetry)
    llm_calls: int
    llm_total_ms: float
    llm_by_type: dict  # call_type -> {count, total_ms, latencies_ms}
    # Provider info
    providers_used: list[str]
    models_used: list[str]
    fallback_count: int
    rate_limit_events: int
    # State info
    iterations: int
    stop_reason: str | None
    evidence_count: int
    # Quality
    answer_length: int
    outcome: str
    warnings: list[str]


def extract_telemetry_data(telemetry_summary: dict | None) -> dict:
    """Extract LLM call data from telemetry summary."""
    if not telemetry_summary:
        return {"calls": 0, "total_ms": 0, "by_type": {}, "providers": [], "models": [], "fallbacks": 0, "rate_limits": 0}

    decisions = telemetry_summary.get("routing_decisions", [])
    by_type = defaultdict(lambda: {"count": 0, "total_ms": 0, "latencies_ms": []})
    providers = set()
    models = set()
    fallbacks = 0
    rate_limits = 0
    total_ms = 0

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
    }


async def _run_all_queries(retriever, reranker, settings):
    """Run all benchmark queries in a single event loop to avoid 'Event loop is closed' errors."""
    from app.orchestration.graph import run_query
    from app.llm_gateway.telemetry import start_run_telemetry, end_run_telemetry

    results = []

    # Cold start measurement
    print("\n[3/4] Measuring cold start...")
    cold_start_t0 = time.perf_counter()
    try:
        start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id="cold_start")
        cold_result = await run_query("test", settings=settings, retriever=retriever, reranker=reranker)
        cold_telemetry_summary = end_run_telemetry()
    except Exception as e:
        print(f"  Cold start query failed (expected with test data): {e}")
        cold_result = None
        try:
            end_run_telemetry()
        except Exception:
            pass
    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000
    print(f"  Cold start latency: {cold_start_ms:.0f}ms")

    # Warm execution
    print("\n[4/4] Running warm E2E benchmark...")

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

        # Extract telemetry
        telemetry = extract_telemetry_data(telemetry_summary)

        qp = QueryProfile(
            query_id=qid,
            pattern=qi["pattern"],
            query=query,
            total_latency_ms=round(total_ms, 2),
            bm25_ms=round(sum(_inst.bm25_latencies_ms), 2),
            embedding_ms=round(sum(_inst.embedding_latencies_ms), 2),
            vector_ms=round(sum(_inst.vector_latencies_ms), 2),
            fusion_ms=round(sum(_inst.fusion_latencies_ms), 2),
            rerank_ms=round(sum(_inst.rerank_latencies_ms), 2),
            esel_ms=round(sum(_inst.esel_latencies_ms), 2),
            bm25_calls=_inst.bm25_calls,
            embedding_calls=_inst.embedding_calls,
            vector_calls=_inst.vector_calls,
            fusion_calls=_inst.fusion_calls,
            rerank_calls=_inst.rerank_calls,
            esel_calls=_inst.esel_calls,
            llm_calls=telemetry["calls"],
            llm_total_ms=telemetry["total_ms"],
            llm_by_type=telemetry["by_type"],
            providers_used=telemetry["providers"],
            models_used=telemetry["models"],
            fallback_count=telemetry["fallbacks"],
            rate_limit_events=telemetry["rate_limits"],
            iterations=getattr(result, 'iterations', 0) or 0,
            stop_reason=getattr(result, 'stop_reason', None),
            evidence_count=len(getattr(result, 'citations', []) or []),
            answer_length=len(getattr(result, 'answer', '') or ''),
            outcome=getattr(result, 'outcome', 'unknown'),
            warnings=getattr(result, 'warnings', []) or [],
        )
        results.append(qp)

        llm_detail = ""
        if telemetry["by_type"]:
            parts = [f"{ct}={d['count']}×{d['total_ms']:.0f}ms" for ct, d in sorted(telemetry["by_type"].items())]
            llm_detail = "  LLM: " + ", ".join(parts)

        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} {qp.pattern:<22} "
              f"ret={qp.bm25_ms+qp.embedding_ms+qp.vector_ms+qp.fusion_ms+qp.rerank_ms:>7.0f}ms  "
              f"esel={qp.esel_ms:>5.1f}ms  "
              f"llm={telemetry['total_ms']:>7.0f}ms({telemetry['calls']}calls)  "
              f"total={total_ms:>7.0f}ms  "
              f"iter={qp.iterations}  stop={qp.stop_reason or 'n/a'}"
              f"{llm_detail}")

    return results, cold_start_ms


def run_e2e_profiler():
    print("=" * 90)
    print("ARGUS Phase 35: End-to-End Performance & Quality Bottleneck Audit")
    print("=" * 90)

    # ── Step 1: Build store and retriever ────────────────────────────────────
    print("\n[1/4] Building benchmark corpus & store...")
    t0 = time.monotonic()
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    print("\n[2/4] Building retrieval indexes...")
    t0 = time.monotonic()
    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # Build reranker
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
    results, cold_start_ms = asyncio.run(_run_all_queries(retriever, reranker, settings))

    # Restore
    restore_retriever()
    restore_esel()
    restore_reranker()

    if not results:
        print("\nNo successful results. Exiting.")
        return

    # ── REPORT ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("END-TO-END LATENCY BREAKDOWN")
    print("=" * 90)

    # Aggregate retrieval stages
    retrieval_stages = [
        ("BM25", [r.bm25_ms for r in results]),
        ("Embedding", [r.embedding_ms for r in results]),
        ("Vector", [r.vector_ms for r in results]),
        ("Fusion", [r.fusion_ms for r in results]),
        ("Rerank", [r.rerank_ms for r in results]),
        ("EvidenceSelector", [r.esel_ms for r in results]),
    ]

    # Aggregate LLM stages
    all_llm_types = set()
    for r in results:
        all_llm_types.update(r.llm_by_type.keys())
    llm_stages = []
    for ct in sorted(all_llm_types):
        vals = [r.llm_by_type.get(ct, {}).get("total_ms", 0) for r in results]
        llm_stages.append((ct, vals))

    # Combined stages
    all_stages = retrieval_stages + llm_stages + [
        ("Total E2E", [r.total_latency_ms for r in results]),
    ]

    print(f"\n{'Component':<25} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'% of E2E':>10}")
    print("-" * 65)
    total_avg = _mean([r.total_latency_ms for r in results])
    for name, vals in all_stages:
        avg = _mean(vals)
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"{name:<25} {avg:>10.1f} {_percentile(vals, 50):>10.1f} {_percentile(vals, 95):>10.1f} {pct:>9.1f}%")

    # Per-pattern breakdown
    print("\n" + "=" * 90)
    print("LATENCY BY QUERY PATTERN")
    print("=" * 90)

    patterns = defaultdict(list)
    for r in results:
        patterns[r.pattern].append(r)

    print(f"\n{'Pattern':<22} {'Count':>5} {'Avg E2E':>10} {'P50':>10} {'P95':>10} {'Retrieval':>10} {'LLM':>10} {'LLM calls':>10} {'Stop':>20}")
    print("-" * 115)
    for pattern in sorted(patterns.keys()):
        prs = patterns[pattern]
        avg_e2e = _mean([r.total_latency_ms for r in prs])
        p50_e2e = _percentile([r.total_latency_ms for r in prs], 50)
        p95_e2e = _percentile([r.total_latency_ms for r in prs], 95)
        avg_ret = _mean([r.bm25_ms + r.embedding_ms + r.vector_ms + r.fusion_ms + r.rerank_ms for r in prs])
        avg_llm = _mean([r.llm_total_ms for r in prs])
        avg_llm_calls = _mean([r.llm_calls for r in prs])
        stops = [r.stop_reason or "n/a" for r in prs]
        most_common_stop = max(set(stops), key=stops.count)
        print(f"{pattern:<22} {len(prs):>5} {avg_e2e:>10.0f} {p50_e2e:>10.0f} {p95_e2e:>10.0f} {avg_ret:>10.0f} {avg_llm:>10.0f} {avg_llm_calls:>10.1f} {most_common_stop:>20}")

    # LLM call audit
    print("\n" + "=" * 90)
    print("LLM CALL AUDIT")
    print("=" * 90)

    total_llm_calls = sum(r.llm_calls for r in results)
    total_llm_ms = sum(r.llm_total_ms for r in results)
    total_fallbacks = sum(r.fallback_count for r in results)
    total_rate_limits = sum(r.rate_limit_events for r in results)

    print(f"\nTotal LLM calls: {total_llm_calls}")
    print(f"Total LLM latency: {total_llm_ms:.0f}ms ({total_llm_ms/sum(r.total_latency_ms for r in results)*100:.1f}% of E2E)")
    print(f"Avg calls/query: {total_llm_calls/len(results):.1f}")
    print(f"Total fallbacks: {total_fallbacks}")
    print(f"Total rate-limit events: {total_rate_limits}")

    all_providers = set()
    all_models = set()
    for r in results:
        all_providers.update(r.providers_used)
        all_models.update(r.models_used)
    print(f"Providers used: {sorted(all_providers)}")
    print(f"Models used: {sorted(all_models)}")

    # Per call-type breakdown
    all_ct = defaultdict(lambda: {"count": 0, "total_ms": 0})
    for r in results:
        for ct, d in r.llm_by_type.items():
            all_ct[ct]["count"] += d["count"]
            all_ct[ct]["total_ms"] += d["total_ms"]

    print(f"\n{'Call Type':<25} {'Count':>8} {'Total ms':>12} {'Avg ms':>10} {'% of LLM':>10}")
    print("-" * 65)
    for ct in sorted(all_ct.keys()):
        c = all_ct[ct]
        avg = c["total_ms"] / c["count"] if c["count"] else 0
        pct = c["total_ms"] / total_llm_ms * 100 if total_llm_ms else 0
        print(f"{ct:<25} {c['count']:>8} {c['total_ms']:>12.0f} {avg:>10.0f} {pct:>9.1f}%")

    # Component cost breakdown
    print("\n" + "=" * 90)
    print("COMPONENT COST ANALYSIS")
    print("=" * 90)

    avg_retrieval = _mean([r.bm25_ms + r.embedding_ms + r.vector_ms + r.fusion_ms + r.rerank_ms for r in results])
    avg_esel = _mean([r.esel_ms for r in results])
    avg_llm = _mean([r.llm_total_ms for r in results])
    avg_other = total_avg - avg_retrieval - avg_esel - avg_llm

    print(f"\n{'Component':<25} {'Avg (ms)':>10} {'% of E2E':>10}")
    print("-" * 45)
    print(f"{'Retrieval (BM25+Vec+Fuse+Rerank)':<25} {avg_retrieval:>10.0f} {avg_retrieval/total_avg*100:>9.1f}%")
    print(f"{'Evidence Selection':<25} {avg_esel:>10.1f} {avg_esel/total_avg*100:>9.1f}%")
    print(f"{'LLM Calls':<25} {avg_llm:>10.0f} {avg_llm/total_avg*100:>9.1f}%")
    print(f"{'Other (classify, plan, verify, ...)':<25} {avg_other:>10.0f} {avg_other/total_avg*100:>9.1f}%")
    print(f"{'Total E2E':<25} {total_avg:>10.0f} {'100.0':>9}%")

    # Contamination check
    contaminated = [r for r in results if r.rate_limit_events > 0 or r.fallback_count > 0]
    if contaminated:
        print(f"\n⚠ CONTAMINATED QUERIES ({len(contaminated)}/{len(results)}):")
        for r in contaminated:
            print(f"  {r.query_id}: {r.rate_limit_events} rate-limits, {r.fallback_count} fallbacks")

    # ── SAVE JSON ────────────────────────────────────────────────────────────
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase35_e2e_profiler",
        "cold_start_latency_ms": round(cold_start_ms, 2),
        "query_count": len(results),
        "aggregate": {
            "total_e2e_avg_ms": round(total_avg, 2),
            "total_e2e_p50_ms": round(_percentile([r.total_latency_ms for r in results], 50), 2),
            "total_e2e_p95_ms": round(_percentile([r.total_latency_ms for r in results], 95), 2),
            "retrieval_avg_ms": round(avg_retrieval, 2),
            "esel_avg_ms": round(avg_esel, 2),
            "llm_avg_ms": round(avg_llm, 2),
            "other_avg_ms": round(avg_other, 2),
        },
        "component_breakdown": {
            name: {
                "avg_ms": round(_mean(vals), 2),
                "p50_ms": round(_percentile(vals, 50), 2),
                "p95_ms": round(_percentile(vals, 95), 2),
            }
            for name, vals in retrieval_stages + llm_stages
        },
        "llm_audit": {
            "total_calls": total_llm_calls,
            "total_ms": round(total_llm_ms, 2),
            "avg_calls_per_query": round(total_llm_calls / len(results), 1),
            "total_fallbacks": total_fallbacks,
            "total_rate_limits": total_rate_limits,
            "providers": sorted(all_providers),
            "models": sorted(all_models),
            "by_type": {ct: {"count": c["count"], "total_ms": round(c["total_ms"], 2)} for ct, c in all_ct.items()},
        },
        "contamination": {
            "contaminated_count": len(contaminated),
            "queries": [{"id": r.query_id, "rate_limits": r.rate_limit_events, "fallbacks": r.fallback_count} for r in contaminated],
        },
        "per_pattern": {
            pattern: {
                "count": len(prs),
                "avg_e2e_ms": round(_mean([r.total_latency_ms for r in prs]), 2),
                "p50_e2e_ms": round(_percentile([r.total_latency_ms for r in prs], 50), 2),
                "avg_retrieval_ms": round(_mean([r.bm25_ms + r.embedding_ms + r.vector_ms + r.fusion_ms + r.rerank_ms for r in prs]), 2),
                "avg_llm_ms": round(_mean([r.llm_total_ms for r in prs]), 2),
                "avg_llm_calls": round(_mean([r.llm_calls for r in prs]), 1),
            }
            for pattern, prs in sorted(patterns.items())
        },
        "per_query": [
            {
                "query_id": r.query_id,
                "pattern": r.pattern,
                "query": r.query,
                "total_latency_ms": r.total_latency_ms,
                "bm25_ms": r.bm25_ms,
                "embedding_ms": r.embedding_ms,
                "vector_ms": r.vector_ms,
                "fusion_ms": r.fusion_ms,
                "rerank_ms": r.rerank_ms,
                "esel_ms": r.esel_ms,
                "llm_calls": r.llm_calls,
                "llm_total_ms": r.llm_total_ms,
                "llm_by_type": r.llm_by_type,
                "iterations": r.iterations,
                "stop_reason": r.stop_reason,
                "outcome": r.outcome,
                "providers_used": r.providers_used,
                "fallback_count": r.fallback_count,
                "rate_limit_events": r.rate_limit_events,
            }
            for r in results
        ],
    }

    out_path = output_dir / "phase35_e2e_profiler.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    run_e2e_profiler()
