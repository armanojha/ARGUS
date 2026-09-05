"""Retrieval quality evaluation suite.

Runs the full retrieval pipeline against the eval query set and reports
R@5, R@10, MRR, nDCG@10, and latency. Designed for regression testing.
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.config import get_settings
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.policy import QuestionPattern


def load_eval_queries() -> list[dict]:
    """Load eval queries from the benchmark eval plan.

    Uses QuestionPattern.from_eval_class() to convert eval plan classes
    to canonical patterns. This ensures consistency between evaluation
    and production classification.
    """
    from app.retrieval.policy import QuestionPattern

    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    if not eval_path.exists():
        return []
    with eval_path.open() as f:
        plan = json.load(f)
    queries = []
    for q in plan.get("queries", []):
        eval_class = q.get("class", "unknown")
        # Convert to canonical pattern
        canonical_pattern = QuestionPattern.from_eval_class(eval_class)
        queries.append({
            "id": q.get("id", "unknown"),
            "query": q.get("query", ""),
            "pattern": canonical_pattern.value,  # Use canonical value
            "eval_class": eval_class,  # Preserve original eval class for reporting
            "supporting_docs": q.get("supporting_docs", []),
        })
    return queries


def build_eval_store() -> tuple[EvidenceStore, dict[str, list[str]]]:
    """Build the evaluation store from corpus documents."""
    from benchmarks.benchmark_fusion import build_benchmark_store
    return build_benchmark_store()


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


def run_evaluation() -> dict:
    """Run full retrieval evaluation."""
    print("=" * 70)
    print("Retrieval Quality Evaluation")
    print("=" * 70)

    # Build store
    print("\n[1/3] Building evaluation corpus...")
    t0 = time.monotonic()
    store, chunk_id_map = build_eval_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # Build index
    print("\n[2/3] Building retrieval index...")
    t0 = time.monotonic()
    retriever = HybridRetriever(store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # Build router for planned retrieval (Phase 15)
    from app.retrieval.router import RetrievalPolicyRouter
    router = RetrievalPolicyRouter()
    planner_activation_count = 0

    # Load queries
    queries = load_eval_queries()
    if not queries:
        print("  No eval queries found, using benchmark queries")
        from benchmarks.benchmark_fusion import BENCHMARK_QUERIES
        queries = [{"id": bq["id"], "query": bq["query"], "pattern": bq["pattern"],
                     "gold_chunks": bq.get("supporting_docs", [])} for bq in BENCHMARK_QUERIES]

    # Warm up
    for q in queries[:2]:
        retriever.search(q["query"], top_k=10)

    # Run evaluation
    print(f"\n[3/3] Evaluating {len(queries)} queries...")
    results = []
    latencies = []

    for q in queries:
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        # Use planned retrieval for complex patterns, direct search for simple
        pattern = q.get("pattern", "")
        t0 = time.monotonic()
        if pattern in ("conflict", "complex_research", "multi_hop"):
            import asyncio
            # Use eval plan class (not router classification) for planned retrieval
            refs = asyncio.run(router.execute_planned_retrieval(
                q["query"], pattern, retriever, top_k=10,
            ))
            planner_activation_count += 1
        else:
            refs = retriever.search(q["query"], top_k=10)
        latency = (time.monotonic() - t0) * 1000
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
        "recall_at_5": round(np.mean(valid_r5), 4) if valid_r5 else float("nan"),
        "recall_at_10": round(np.mean(valid_r10), 4) if valid_r10 else float("nan"),
        "mrr": round(np.mean(valid_mrr), 4) if valid_mrr else float("nan"),
        "ndcg_at_10": round(np.mean(valid_ndcg), 4) if valid_ndcg else float("nan"),
        "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(np.percentile(latencies, 95), 1) if latencies else 0,
        "query_count": len(queries),
        "planner_activations": planner_activation_count,
    }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Recall@5:    {summary['recall_at_5']:.3f}")
    print(f"  Recall@10:   {summary['recall_at_10']:.3f}")
    print(f"  MRR:         {summary['mrr']:.3f}")
    print(f"  nDCG@10:     {summary['ndcg_at_10']:.3f}")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"  P95 Latency: {summary['p95_latency_ms']:.1f}ms")
    print(f"  Planner:     {planner_activation_count}/{len(queries)} queries")

    # Per-pattern breakdown
    patterns = sorted(set(r["pattern"] for r in results))
    print("\nPer-pattern Recall@10:")
    for pattern in patterns:
        pr = [r for r in results if r["pattern"] == pattern]
        valid = [r["recall_10"] for r in pr if not np.isnan(r["recall_10"])]
        avg = np.mean(valid) if valid else float("nan")
        print(f"  {pattern:<25} {avg:.3f}  ({len(pr)} queries)")

    # Save report
    output = {
        "benchmark": "retrieval_quality_evaluation",
        "summary": summary,
        "per_query": results,
    }
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "retrieval_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/retrieval_evaluation.json")

    return output


if __name__ == "__main__":
    run_evaluation()
