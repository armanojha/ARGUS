#!/usr/bin/env python3
"""Phase 34: Reranker Profiler.

Measures the exact latency breakdown of the reranking operation:
- Candidate generation (retrieval before reranking)
- Pair preparation (creating query-passage pairs)
- CrossEncoder predict (tokenization + inference)
- Postprocessing (sorting, building results)
- Total reranking latency

Also profiles model loading, device characteristics, and batch behavior.
"""
from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from benchmarks.benchmark_fusion import build_benchmark_store


BENCHMARK_QUERIES = [
    {"id": "P34-Q01", "class": "simple_lookup", "query": "Where is Acme Corporation headquartered?"},
    {"id": "P34-Q02", "class": "multi_doc_synthesis", "query": "What are the key differences between Atlas v1 and Atlas v2?"},
    {"id": "P34-Q03", "class": "multi_hop", "query": "Who manages the team that developed Atlas?"},
    {"id": "P34-Q04", "class": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?"},
    {"id": "P34-Q05", "class": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?"},
    {"id": "P34-Q06", "class": "technical_explanation", "query": "How does the Atlas database achieve low latency?"},
    {"id": "P34-Q07", "class": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?"},
    {"id": "P34-Q08", "class": "absent_info", "query": "What is Acme's market share in the European robotics market?"},
    {"id": "P34-Q09", "class": "absent_info", "query": "What is the salary range for Acme's software engineers?"},
    {"id": "P34-Q10", "class": "adversarial", "query": "What undisclosed legal issues has Acme faced?"},
    {"id": "P34-Q11", "class": "adversarial", "query": "What internal documents reveal Acme's planned layoffs?"},
]


@dataclasses.dataclass
class RerankProfile:
    query_id: str
    query_class: str
    candidate_count: int
    reranked_count: int
    # Latency breakdown
    pair_prep_ms: float = 0.0
    predict_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_rerank_ms: float = 0.0
    # Retrieval latency (before reranking)
    retrieval_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    # Model info
    reranker_model: str = ""
    device: str = ""
    # Quality
    top1_changed: bool = False
    rank_changes: int = 0


def _detect_device(model) -> str:
    """Detect which device the model is on."""
    try:
        import torch
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            params = next(model.model.parameters())
            return str(params.device)
        return "unknown"
    except Exception:
        return "unknown"


def profile_reranker():
    print("=" * 90)
    print("Phase 34: Reranker Profiler")
    print("=" * 90)

    # Build store + retriever
    print("\n[1/3] Building store & retriever...")
    store, chunk_id_map = build_benchmark_store()
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import Reranker

    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    # Create reranker and load model
    print("\n[2/3] Loading reranker model...")
    reranker = Reranker()
    t0 = time.perf_counter()
    model = reranker._get_model()
    model_load_ms = (time.perf_counter() - t0) * 1000
    device = _detect_device(model)
    print(f"  Model: {reranker.model_name}")
    print(f"  Device: {device}")
    print(f"  Load time: {model_load_ms:.1f}ms (if already cached: ~0ms)")

    # Warm up
    print("\n[3/3] Running reranker profile...")
    warmup_candidates = retriever.search("warm up", top_k=10)
    if warmup_candidates:
        reranker.rerank("warm up", warmup_candidates, top_k=5)

    results = []
    for i, qi in enumerate(BENCHMARK_QUERIES, 1):
        query = qi["query"]
        qid = qi["id"]

        # Step 1: Retrieval (candidate generation)
        t0 = time.perf_counter()
        candidates = retriever.search(query, top_k=20)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not candidates:
            continue

        # Step 2: Pair preparation
        t0 = time.perf_counter()
        pairs = [(query, cand.text) for cand in candidates]
        pair_prep_ms = (time.perf_counter() - t0) * 1000

        # Step 3: CrossEncoder predict (tokenization + inference)
        t0 = time.perf_counter()
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
        predict_ms = (time.perf_counter() - t0) * 1000

        # Step 4: Postprocessing (sorting + building results)
        t0 = time.perf_counter()
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = 10
        reranked = []
        for rank, (cand, score) in enumerate(scored[:top_k], 1):
            from app.evidence.models import EvidenceRef
            reranked.append(EvidenceRef(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                source_id=cand.source_id,
                source_path=cand.source_path,
                source_type=cand.source_type,
                text=cand.text,
                page_start=cand.page_start,
                page_end=cand.page_end,
                section_path=cand.section_path,
                score=float(score),
                rank=rank,
                metadata={**cand.metadata, "rerank_score": float(score), "original_score": cand.score, "original_rank": cand.rank},
            ))
        postprocess_ms = (time.perf_counter() - t0) * 1000

        total_rerank_ms = pair_prep_ms + predict_ms + postprocess_ms

        # Check if top-1 changed
        top1_changed = candidates[0].chunk_id != reranked[0].chunk_id if reranked else False
        rank_changes = sum(1 for j, r in enumerate(reranked) if r.metadata.get("original_rank", j+1) != r.rank)

        qr = RerankProfile(
            query_id=qid,
            query_class=qi["class"],
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            pair_prep_ms=round(pair_prep_ms, 2),
            predict_ms=round(predict_ms, 2),
            postprocess_ms=round(postprocess_ms, 2),
            total_rerank_ms=round(total_rerank_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            total_retrieval_ms=round(retrieval_ms + total_rerank_ms, 2),
            reranker_model=reranker.model_name,
            device=device,
            top1_changed=top1_changed,
            rank_changes=rank_changes,
        )
        results.append(qr)

        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} {qi['class']:<24} "
              f"ret={retrieval_ms:>6.1f}ms  prep={pair_prep_ms:>5.2f}ms  "
              f"predict={predict_ms:>7.1f}ms  post={postprocess_ms:>5.2f}ms  "
              f"total={total_rerank_ms:>7.1f}ms  changes={rank_changes:>2}  top1={'CHANGED' if top1_changed else 'same'}")

    # ══════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════

    import numpy as np

    def _pct(data, p):
        return float(np.percentile(data, p)) if data else 0.0

    def _mean(data):
        return float(np.mean(data)) if data else 0.0

    print("\n" + "=" * 90)
    print("RERANKER LATENCY BREAKDOWN")
    print("=" * 90)

    stages = [
        ("Pair preparation", [r.pair_prep_ms for r in results]),
        ("Predict (tok+inference)", [r.predict_ms for r in results]),
        ("Postprocessing", [r.postprocess_ms for r in results]),
        ("Total reranking", [r.total_rerank_ms for r in results]),
        ("Retrieval (pre-rerank)", [r.retrieval_ms for r in results]),
    ]

    print(f"\n{'Stage':<28} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 78)
    for name, vals in stages:
        print(f"{name:<28} {_mean(vals):>10.2f} {_pct(vals, 50):>10.2f} "
              f"{_pct(vals, 95):>10.2f} {min(vals) if vals else 0:>10.2f} "
              f"{max(vals) if vals else 0:>10.2f}")

    # Percentage breakdown
    avg_predict = _mean([r.predict_ms for r in results])
    avg_prep = _mean([r.pair_prep_ms for r in results])
    avg_post = _mean([r.postprocess_ms for r in results])
    avg_total = _mean([r.total_rerank_ms for r in results])

    print(f"\nReranking percentage breakdown:")
    if avg_total > 0:
        print(f"  Pair preparation:  {avg_prep:>7.2f}ms  {avg_prep/avg_total*100:>5.1f}%")
        print(f"  Predict (tok+inf): {avg_predict:>7.2f}ms  {avg_predict/avg_total*100:>5.1f}%")
        print(f"  Postprocessing:    {avg_post:>7.2f}ms  {avg_post/avg_total*100:>5.1f}%")
        print(f"  ─────────────────────────────────────")
        print(f"  Total:             {avg_total:>7.2f}ms  100.0%")

    # Reranking impact
    print(f"\nReranking impact:")
    print(f"  Top-1 changed: {sum(1 for r in results if r.top1_changed)}/{len(results)} queries ({sum(1 for r in results if r.top1_changed)/max(1,len(results))*100:.0f}%)")
    print(f"  Avg rank changes: {_mean([r.rank_changes for r in results]):.1f} per query")
    print(f"  Candidates: {_mean([r.candidate_count for r in results]):.0f} avg -> {_mean([r.reranked_count for r in results]):.0f} output")

    # Device info
    print(f"\nModel info:")
    print(f"  Model: {results[0].reranker_model if results else 'N/A'}")
    print(f"  Device: {results[0].device if results else 'N/A'}")

    # Per-query detail
    print(f"\n{'ID':<10} {'Class':<24} {'Cand':>5} {'Prep':>7} {'Predict':>9} {'Post':>7} {'Total':>8} {'Changes':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r.query_id:<10} {r.query_class:<24} {r.candidate_count:>5} {r.pair_prep_ms:>7.2f} "
              f"{r.predict_ms:>9.1f} {r.postprocess_ms:>7.2f} {r.total_rerank_ms:>8.1f} {r.rank_changes:>8}")

    # Save JSON
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(exist_ok=True)

    report = {
        "benchmark": "phase34_reranker_profile",
        "model": results[0].reranker_model if results else "",
        "device": results[0].device if results else "",
        "query_count": len(results),
        "aggregate": {
            name: {
                "avg_ms": round(_mean(vals), 2),
                "p50_ms": round(_pct(vals, 50), 2),
                "p95_ms": round(_pct(vals, 95), 2),
            }
            for name, vals in stages
        },
        "impact": {
            "top1_changed_count": sum(1 for r in results if r.top1_changed),
            "avg_rank_changes": round(_mean([r.rank_changes for r in results]), 2),
            "avg_candidates": round(_mean([r.candidate_count for r in results]), 0),
            "avg_output": round(_mean([r.reranked_count for r in results]), 0),
        },
        "per_query": [
            {
                "query_id": r.query_id,
                "query_class": r.query_class,
                "candidate_count": r.candidate_count,
                "reranked_count": r.reranked_count,
                "pair_prep_ms": r.pair_prep_ms,
                "predict_ms": r.predict_ms,
                "postprocess_ms": r.postprocess_ms,
                "total_rerank_ms": r.total_rerank_ms,
                "retrieval_ms": r.retrieval_ms,
                "top1_changed": r.top1_changed,
                "rank_changes": r.rank_changes,
            }
            for r in results
        ],
    }

    out_path = output_dir / "phase34_reranker_profile.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    profile_reranker()
