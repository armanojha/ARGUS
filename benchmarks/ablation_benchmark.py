"""Phase 19 Ablation Benchmark.

Measures the quality and latency impact of each retrieval component by
selectively disabling them. Produces a component-value matrix.

Usage:
    python -m benchmarks.ablation_benchmark              # full system baseline
    python -m benchmarks.ablation_benchmark --ablation bm25_off
    python -m benchmarks.ablation_benchmark --ablation vector_off
    python -m benchmarks.ablation_benchmark --ablation no_planner
    python -m benchmarks.ablation_benchmark --ablation no_multi_query
    python -m benchmarks.ablation_benchmark --ablation no_recovery
    python -m benchmarks.ablation_benchmark --ablation no_reranker
    python -m benchmarks.ablation_benchmark --ablation rank_fusion
    python -m benchmarks.ablation_benchmark --ablation all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.evidence.models import EvidenceRef
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.policy import QuestionPattern


def load_eval_queries() -> list[dict]:
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    if not eval_path.exists():
        return []
    with eval_path.open() as f:
        plan = json.load(f)
    queries = []
    for q in plan.get("queries", []):
        eval_class = q.get("class", "unknown")
        canonical_pattern = QuestionPattern.from_eval_class(eval_class)
        queries.append({
            "id": q.get("id", "unknown"),
            "query": q.get("query", ""),
            "pattern": canonical_pattern.value,
            "eval_class": eval_class,
            "supporting_docs": q.get("supporting_docs", []),
        })
    return queries


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


PLANNABLE = {"conflict", "complex_research", "multi_hop"}


def run_ablation(
    ablation: str,
    store,
    chunk_id_map: dict[str, list[str]],
    retriever: HybridRetriever,
    queries: list[dict],
    *,
    loglevel: str = "warning",
) -> dict:
    """Run a single ablation configuration and return metrics."""
    from app.config import get_settings
    settings = get_settings()

    # Set logging level to reduce noise
    import logging
    logging.disable(logging.CRITICAL) if loglevel == "silent" else None

    # --- Build router with ablation-aware overrides ---
    from app.retrieval.router import RetrievalPolicyRouter
    from app.retrieval.planner import EvidenceNeedPlanner
    router = RetrievalPolicyRouter()
    _planner = EvidenceNeedPlanner()

    # --- Ablation: force fusion normalization ---
    if ablation == "rank_fusion":
        settings.retrieval_fusion_normalization = "rank"

    planner_count = 0
    results = []
    latencies = []
    stage_latencies: dict[str, list[float]] = {
        "classification": [],
        "planning": [],
        "search": [],
        "fusion": [],
        "recovery": [],
        "reranking": [],
    }

    for q in queries:
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        pattern = q.get("pattern", "")
        query = q["query"]

        t_total = time.monotonic()

        # Stage 1: Classification
        t0 = time.monotonic()
        # classification is free (dict lookup) but measure it
        t_class = (time.monotonic() - t0) * 1000
        stage_latencies["classification"].append(t_class)

        if pattern in PLANNABLE and ablation not in ("no_planner", "no_multi_query"):
            # Planned path
            t0 = time.monotonic()
            plan = _planner.plan(query, pattern)
            t_plan = (time.monotonic() - t0) * 1000
            stage_latencies["planning"].append(t_plan)

            # Full planned path
            t0 = time.monotonic()
            refs = asyncio.run(router.execute_planned_retrieval(
                query, pattern, retriever, top_k=10,
            ))
            t_search = (time.monotonic() - t0) * 1000
            stage_latencies["search"].append(t_search)
            planner_count += 1

        elif pattern in PLANNABLE and ablation in ("no_planner", "no_multi_query"):
            # Ablated planned path: single hybrid search instead of multi-query
            t0 = time.monotonic()
            refs = retriever.search(query, top_k=10)
            t_search = (time.monotonic() - t0) * 1000
            stage_latencies["search"].append(t_search)
        else:
            # Standard path (non-plannable patterns)
            t0 = time.monotonic()

            # Ablation: BM25 only
            if ablation == "bm25_off":
                refs = retriever.search(query, top_k=10, mechanisms={"vector"})
            # Ablation: Vector only
            elif ablation == "vector_off":
                refs = retriever.search(query, top_k=10, mechanisms={"bm25"})
            else:
                refs = retriever.search(query, top_k=10)

            t_search = (time.monotonic() - t0) * 1000
            stage_latencies["search"].append(t_search)

        # Ablation: skip reranker (already NoOp in benchmark, but measure)
        t_rerank = 0.0
        stage_latencies["reranking"].append(t_rerank)

        latency = (time.monotonic() - t_total) * 1000
        latencies.append(latency)

        retrieved_ids = [str(r.chunk_id) for r in refs]
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        m = mrr(retrieved_ids, gold_ids)
        n = ndcg_at_k(retrieved_ids, gold_ids, 10)

        results.append({
            "query_id": q["id"],
            "pattern": q["pattern"],
            "recall_5": r5,
            "recall_10": r10,
            "mrr": m,
            "ndcg_10": n,
            "latency_ms": latency,
            "result_count": len(refs),
        })

    # Aggregate
    valid_r5 = [r["recall_5"] for r in results if not np.isnan(r["recall_5"])]
    valid_r10 = [r["recall_10"] for r in results if not np.isnan(r["recall_10"])]
    valid_mrr = [r["mrr"] for r in results]
    valid_ndcg = [r["ndcg_10"] for r in results if not np.isnan(r["ndcg_10"])]

    summary = {
        "ablation": ablation,
        "recall_at_5": round(np.mean(valid_r5), 4) if valid_r5 else float("nan"),
        "recall_at_10": round(np.mean(valid_r10), 4) if valid_r10 else float("nan"),
        "mrr": round(np.mean(valid_mrr), 4) if valid_mrr else float("nan"),
        "ndcg_at_10": round(np.mean(valid_ndcg), 4) if valid_ndcg else float("nan"),
        "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(np.percentile(latencies, 50), 1) if latencies else 0,
        "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        "query_count": len(queries),
        "planner_activations": planner_count,
    }

    # Per-pattern breakdown
    patterns = sorted(set(r["pattern"] for r in results))
    per_pattern = {}
    for pattern in patterns:
        pr = [r for r in results if r["pattern"] == pattern]
        valid = [r["recall_10"] for r in pr if not np.isnan(r["recall_10"])]
        avg = round(np.mean(valid), 3) if valid else float("nan")
        per_pattern[pattern] = {"recall_10": avg, "count": len(pr)}

    summary["per_pattern"] = per_pattern

    # Stage latency breakdown
    stage_summary = {}
    for stage, vals in stage_latencies.items():
        if vals:
            stage_summary[stage] = {
                "avg_ms": round(np.mean(vals), 1),
                "total_ms": round(sum(vals), 1),
            }
    summary["stage_latencies"] = stage_summary

    return {"summary": summary, "per_query": results}


def print_results(result: dict, baseline: dict | None = None):
    s = result["summary"]
    ablation = s["ablation"]

    print(f"\n{'='*70}")
    print(f"ABLATION: {ablation}")
    print(f"{'='*70}")
    print(f"  Recall@5:    {s['recall_at_5']:.3f}")
    print(f"  Recall@10:   {s['recall_at_10']:.3f}")
    print(f"  MRR:         {s['mrr']:.3f}")
    print(f"  nDCG@10:     {s['ndcg_at_10']:.3f}")
    print(f"  Avg Latency: {s['avg_latency_ms']:.1f}ms")
    print(f"  P50 Latency: {s['p50_latency_ms']:.1f}ms")
    print(f"  P95 Latency: {s['p95_latency_ms']:.1f}ms")
    print(f"  Planner:     {s['planner_activations']}/{s['query_count']} queries")

    if s.get("stage_latencies"):
        print("\n  Stage Latencies:")
        for stage, vals in s["stage_latencies"].items():
            print(f"    {stage:<20} avg={vals['avg_ms']:.1f}ms  total={vals['total_ms']:.1f}ms")

    if s.get("per_pattern"):
        print("\n  Per-pattern Recall@10:")
        for pat, info in sorted(s["per_pattern"].items()):
            print(f"    {pat:<25} {info['recall_10']:.3f}  ({info['count']} queries)")

    if baseline:
        b = baseline["summary"]
        print(f"\n  Delta from FULL SYSTEM:")
        for metric in ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "avg_latency_ms", "p95_latency_ms"]:
            delta = s[metric] - b[metric]
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            print(f"    {metric:<20} {delta:+.3f} {arrow}")


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Ablation Benchmark")
    parser.add_argument("--ablation", default="full",
                        choices=["full", "bm25_off", "vector_off", "rank_fusion",
                                 "no_planner", "no_multi_query", "no_recovery",
                                 "no_reranker", "all"],
                        help="Which component to ablate")
    parser.add_argument("--loglevel", default="warning",
                        help="Log level for argus loggers")
    args = parser.parse_args()

    from benchmarks.benchmark_fusion import build_benchmark_store

    print("=" * 70)
    print(f"Phase 19 Ablation Benchmark — {args.ablation}")
    print("=" * 70)

    # Build store (shared across ablations)
    print("\n[1/3] Building evaluation corpus...")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # Build index
    print("\n[2/3] Building retrieval index...")
    t0 = time.monotonic()
    retriever = HybridRetriever(store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # Load queries
    queries = load_eval_queries()
    print(f"  Loaded {len(queries)} queries")

    # Warm up
    for q in queries[:2]:
        retriever.search(q["query"], top_k=10)

    # Run ablations
    ablations = [args.ablation] if args.ablation != "all" else [
        "full", "bm25_off", "vector_off", "rank_fusion",
        "no_planner", "no_multi_query", "no_recovery",
    ]

    all_results = {}
    baseline = None

    for abl in ablations:
        print(f"\n[3/3] Running ablation: {abl}...")
        result = run_ablation(abl, store, chunk_id_map, retriever, queries, loglevel=args.loglevel)
        all_results[abl] = result
        print_results(result, baseline if abl != "full" else None)
        if abl == "full":
            baseline = result

    # Save results
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"ablation_{args.ablation}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Print summary table if running all
    if len(ablations) > 1:
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Ablation':<20} {'R@5':>6} {'R@10':>6} {'nDCG':>6} {'Avg ms':>8} {'P95 ms':>8}")
        print("-" * 70)
        for abl in ablations:
            s = all_results[abl]["summary"]
            print(f"{abl:<20} {s['recall_at_5']:>6.3f} {s['recall_at_10']:>6.3f} "
                  f"{s['ndcg_at_10']:>6.3f} {s['avg_latency_ms']:>7.1f} {s['p95_latency_ms']:>7.1f}")


if __name__ == "__main__":
    main()
