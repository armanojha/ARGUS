#!/usr/bin/env python3
"""Phase 34: Reranker Ablation Study.

Compares:
A. Retrieval WITHOUT reranking
B. Retrieval WITH current reranker

At different candidate counts: 5, 10, 20, 30, 50

Measures:
- R@5, R@10, MRR, nDCG@10
- Latency (P50, P95)
- Per-pattern impact
- Rank changes
"""
from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from benchmarks.benchmark_fusion import build_benchmark_store


# ── Queries with gold facts for quality evaluation ──────────────────────────

BENCHMARK_QUERIES = [
    {"id": "P34-Q01", "class": "simple_lookup", "query": "Where is Acme Corporation headquartered?",
     "gold_facts": ["New York City"], "supporting_docs": ["doc-a"]},
    {"id": "P34-Q02", "class": "multi_doc_synthesis", "query": "What are the key differences between Atlas v1 and Atlas v2?",
     "gold_facts": ["Atlas v1", "Atlas v2"], "supporting_docs": ["doc-b", "doc-c"]},
    {"id": "P34-Q03", "class": "multi_hop", "query": "Who manages the team that developed Atlas?",
     "gold_facts": ["manager", "team", "Atlas"], "supporting_docs": ["doc-b", "doc-d"]},
    {"id": "P34-Q04", "class": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?",
     "gold_facts": ["revenue", "2023"], "supporting_docs": ["doc-a", "doc-e"]},
    {"id": "P34-Q05", "class": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?",
     "gold_facts": ["revenue", "growth", "rate"], "supporting_docs": ["doc-a"]},
    {"id": "P34-Q06", "class": "technical_explanation", "query": "How does the Atlas database achieve low latency?",
     "gold_facts": ["Atlas", "latency", "architecture"], "supporting_docs": ["doc-c"]},
    {"id": "P34-Q07", "class": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?",
     "gold_facts": ["competitive", "advantage", "analytics"], "supporting_docs": ["doc-a", "doc-f"]},
    {"id": "P34-Q08", "class": "absent_info", "query": "What is Acme's market share in the European robotics market?",
     "gold_facts": [], "supporting_docs": []},
    {"id": "P34-Q09", "class": "absent_info", "query": "What is the salary range for Acme's software engineers?",
     "gold_facts": [], "supporting_docs": []},
    {"id": "P34-Q10", "class": "adversarial", "query": "What undisclosed legal issues has Acme faced?",
     "gold_facts": [], "supporting_docs": []},
    {"id": "P34-Q11", "class": "adversarial", "query": "What internal documents reveal Acme's planned layoffs?",
     "gold_facts": [], "supporting_docs": []},
]


def _is_relevant(chunk, gold_facts: list[str]) -> bool:
    """Check if a chunk contains any gold fact."""
    if not gold_facts:
        return False
    text_lower = chunk.text.lower()
    return any(fact.lower() in text_lower for fact in gold_facts)


def _compute_retrieval_metrics(
    retrieved_chunks: list,
    gold_facts: list[str],
    top_k: int,
) -> dict[str, float]:
    """Compute R@k, MRR, nDCG@k."""
    if not gold_facts:
        return {"r_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    # Determine relevance
    relevant = [_is_relevant(c, gold_facts) for c in retrieved_chunks[:top_k]]
    total_relevant_in_pool = sum(1 for c in retrieved_chunks if _is_relevant(c, gold_facts))

    if total_relevant_in_pool == 0:
        return {"r_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    # R@k
    r_at_k = sum(relevant) / total_relevant_in_pool

    # MRR
    mrr = 0.0
    for i, rel in enumerate(relevant):
        if rel:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@k
    import math
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevant) if rel)
    ideal_rels = min(total_relevant_in_pool, top_k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"r_at_k": r_at_k, "mrr": mrr, "ndcg_at_k": ndcg}


def run_ablation():
    print("=" * 90)
    print("Phase 34: Reranker Ablation Study")
    print("=" * 90)

    # Build store + retriever
    print("\n[1/3] Building store & retriever...")
    store, chunk_id_map = build_benchmark_store()
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import Reranker

    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    # Load reranker
    print("\n[2/3] Loading reranker...")
    reranker = Reranker()
    model = reranker._get_model()
    print(f"  Model: {reranker.model_name}, Device: cpu")

    # Candidate counts to test
    candidate_counts = [5, 10, 20, 30, 50]

    # Run ablation
    print(f"\n[3/3] Running ablation ({len(BENCHMARK_QUERIES)} queries x {len(candidate_counts)} candidate counts x 2 modes)...")
    print(f"  Modes: NO_RERANK vs RERANK")

    results = []

    for qi in BENCHMARK_QUERIES:
        query = qi["query"]
        qid = qi["id"]
        gold_facts = qi.get("gold_facts", [])

        # Get ALL candidates (max needed)
        all_candidates = retriever.search(query, top_k=50)
        if not all_candidates:
            continue

        for cand_count in candidate_counts:
            candidates = all_candidates[:cand_count]

            # Mode A: NO RERANK (keep original order)
            t0 = time.perf_counter()
            no_rerank = candidates[:10]  # Take top 10 as-is
            no_rerank_ms = (time.perf_counter() - t0) * 1000

            # Mode B: RERANK
            t0 = time.perf_counter()
            reranked = reranker.rerank(query, candidates, top_k=10)
            rerank_ms = (time.perf_counter() - t0) * 1000

            # Compute metrics
            metrics_no_rerank = _compute_retrieval_metrics(no_rerank, gold_facts, 10)
            metrics_rerank = _compute_retrieval_metrics(reranked, gold_facts, 10)

            # Rank changes
            rank_changes = 0
            if reranked:
                for j, r in enumerate(reranked):
                    orig_rank = r.metadata.get("original_rank", j + 1)
                    if orig_rank != r.rank:
                        rank_changes += 1

            results.append({
                "query_id": qid,
                "query_class": qi["class"],
                "candidate_count": cand_count,
                "no_rerank": {
                    "latency_ms": round(no_rerank_ms, 2),
                    "r_at_10": round(metrics_no_rerank["r_at_k"], 4),
                    "mrr": round(metrics_no_rerank["mrr"], 4),
                    "ndcg_at_10": round(metrics_no_rerank["ndcg_at_k"], 4),
                },
                "rerank": {
                    "latency_ms": round(rerank_ms, 2),
                    "r_at_10": round(metrics_rerank["r_at_k"], 4),
                    "mrr": round(metrics_rerank["mrr"], 4),
                    "ndcg_at_10": round(metrics_rerank["ndcg_at_k"], 4),
                    "rank_changes": rank_changes,
                },
                "quality_delta": {
                    "r_at_10": round(metrics_rerank["r_at_k"] - metrics_no_rerank["r_at_k"], 4),
                    "mrr": round(metrics_rerank["mrr"] - metrics_no_rerank["mrr"], 4),
                    "ndcg_at_10": round(metrics_rerank["ndcg_at_k"] - metrics_no_rerank["ndcg_at_k"], 4),
                },
            })

        print(f"  {qid:<10} {qi['class']:<24} done")

    # ══════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════

    import numpy as np

    def _mean(data):
        return float(np.mean(data)) if data else 0.0

    def _pct(data, p):
        return float(np.percentile(data, p)) if data else 0.0

    print("\n" + "=" * 90)
    print("ABLATION RESULTS BY CANDIDATE COUNT")
    print("=" * 90)

    print(f"\n{'Cand':>5} │ {'NO RERANK':>30} │ {'RERANK':>30} │ {'DELTA':>30}")
    print(f"{'Count':>5} │ {'R@10':>8} {'MRR':>8} {'nDCG':>8} {'Lat(ms)':>8} │ {'R@10':>8} {'MRR':>8} {'nDCG':>8} {'Lat(ms)':>8} │ {'R@10':>8} {'MRR':>8} {'nDCG':>8}")
    print("-" * 120)

    for cand_count in candidate_counts:
        nr_r10 = [r["no_rerank"]["r_at_10"] for r in results if r["candidate_count"] == cand_count]
        nr_mrr = [r["no_rerank"]["mrr"] for r in results if r["candidate_count"] == cand_count]
        nr_ndcg = [r["no_rerank"]["ndcg_at_10"] for r in results if r["candidate_count"] == cand_count]
        nr_lat = [r["no_rerank"]["latency_ms"] for r in results if r["candidate_count"] == cand_count]

        rr_r10 = [r["rerank"]["r_at_10"] for r in results if r["candidate_count"] == cand_count]
        rr_mrr = [r["rerank"]["mrr"] for r in results if r["candidate_count"] == cand_count]
        rr_ndcg = [r["rerank"]["ndcg_at_10"] for r in results if r["candidate_count"] == cand_count]
        rr_lat = [r["rerank"]["latency_ms"] for r in results if r["candidate_count"] == cand_count]

        d_r10 = [r["quality_delta"]["r_at_10"] for r in results if r["candidate_count"] == cand_count]
        d_mrr = [r["quality_delta"]["mrr"] for r in results if r["candidate_count"] == cand_count]
        d_ndcg = [r["quality_delta"]["ndcg_at_10"] for r in results if r["candidate_count"] == cand_count]

        print(f"{cand_count:>5} │ "
              f"{_mean(nr_r10):>8.4f} {_mean(nr_mrr):>8.4f} {_mean(nr_ndcg):>8.4f} {_mean(nr_lat):>8.1f} │ "
              f"{_mean(rr_r10):>8.4f} {_mean(rr_mrr):>8.4f} {_mean(rr_ndcg):>8.4f} {_mean(rr_lat):>8.1f} │ "
              f"{_mean(d_r10):>+8.4f} {_mean(d_mrr):>+8.4f} {_mean(d_ndcg):>+8.4f}")

    # Per-pattern analysis
    print("\n" + "=" * 90)
    print("PER-PATTERN IMPACT (20 candidates)")
    print("=" * 90)

    patterns = sorted(set(r["query_class"] for r in results))
    print(f"\n{'Pattern':<24} │ {'NO RERANK':>20} │ {'RERANK':>20} │ {'DELTA':>20}")
    print(f"{'':24} │ {'R@10':>8} {'nDCG':>8} {'Lat':>8} │ {'R@10':>8} {'nDCG':>8} {'Lat':>8} │ {'R@10':>8} {'nDCG':>8}")
    print("-" * 100)

    for pattern in patterns:
        pr = [r for r in results if r["query_class"] == pattern and r["candidate_count"] == 20]
        if not pr:
            continue
        nr_r10 = _mean([r["no_rerank"]["r_at_10"] for r in pr])
        nr_ndcg = _mean([r["no_rerank"]["ndcg_at_10"] for r in pr])
        nr_lat = _mean([r["no_rerank"]["latency_ms"] for r in pr])
        rr_r10 = _mean([r["rerank"]["r_at_10"] for r in pr])
        rr_ndcg = _mean([r["rerank"]["ndcg_at_10"] for r in pr])
        rr_lat = _mean([r["rerank"]["latency_ms"] for r in pr])
        d_r10 = _mean([r["quality_delta"]["r_at_10"] for r in pr])
        d_ndcg = _mean([r["quality_delta"]["ndcg_at_10"] for r in pr])
        print(f"{pattern:<24} │ {nr_r10:>8.4f} {nr_ndcg:>8.4f} {nr_lat:>8.1f} │ {rr_r10:>8.4f} {rr_ndcg:>8.4f} {rr_lat:>8.1f} │ {d_r10:>+8.4f} {d_ndcg:>+8.4f}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Overall with 20 candidates (production-like)
    pr20 = [r for r in results if r["candidate_count"] == 20]
    if pr20:
        avg_r10_delta = _mean([r["quality_delta"]["r_at_10"] for r in pr20])
        avg_mrr_delta = _mean([r["quality_delta"]["mrr"] for r in pr20])
        avg_ndcg_delta = _mean([r["quality_delta"]["ndcg_at_10"] for r in pr20])
        avg_rerank_lat = _mean([r["rerank"]["latency_ms"] for r in pr20])
        queries_improved = sum(1 for r in pr20 if r["quality_delta"]["r_at_10"] > 0)
        queries_degraded = sum(1 for r in pr20 if r["quality_delta"]["r_at_10"] < 0)
        queries_unchanged = sum(1 for r in pr20 if r["quality_delta"]["r_at_10"] == 0)

        print(f"\nWith 20 candidates (production-like):")
        print(f"  R@10 improvement:   {avg_r10_delta:>+.4f}")
        print(f"  MRR improvement:    {avg_mrr_delta:>+.4f}")
        print(f"  nDCG@10 improvement: {avg_ndcg_delta:>+.4f}")
        print(f"  Rerank latency:     {avg_rerank_lat:.1f}ms")
        print(f"  Queries improved:   {queries_improved}/{len(pr20)}")
        print(f"  Queries degraded:   {queries_degraded}/{len(pr20)}")
        print(f"  Queries unchanged:  {queries_unchanged}/{len(pr20)}")

        if avg_rerank_lat > 0:
            quality_per_second = (avg_r10_delta + avg_mrr_delta + avg_ndcg_delta) / 3 / (avg_rerank_lat / 1000)
            print(f"  Quality gain/second: {quality_per_second:.4f}")

    # Save JSON
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase34_ablation",
        "query_count": len(BENCHMARK_QUERIES),
        "candidate_counts_tested": candidate_counts,
        "per_query": results,
    }

    out_path = output_dir / "phase34_ablation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    run_ablation()
