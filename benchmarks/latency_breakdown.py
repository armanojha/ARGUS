"""Per-Stage Latency Breakdown for ARGUS Retrieval Pipeline.

Measures classification, planning, BM25, embedding, vector search,
fusion, multi-query, and recovery latencies for each eval query.
Reports aggregate stats (avg, P50, P95) and identifies the dominant stage.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import EvidenceNeedPlanner
from app.retrieval.policy import QuestionPattern
from app.retrieval.router import RetrievalPolicyRouter
from benchmarks.benchmark_fusion import build_benchmark_store


def load_eval_queries() -> list[dict]:
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        plan = json.load(f)
    queries = []
    for q in plan.get("queries", []):
        eval_class = q.get("class", "unknown")
        canonical_pattern = QuestionPattern.from_eval_class(eval_class)
        queries.append({
            "id": q.get("id"),
            "query": q.get("query"),
            "pattern": canonical_pattern.value,
            "eval_class": eval_class,
            "supporting_docs": q.get("supporting_docs", []),
        })
    return queries


def _percentile(data: list[float], pct: float) -> float:
    return float(np.percentile(data, pct)) if data else 0.0


def _mean(data: list[float]) -> float:
    return float(np.mean(data)) if data else 0.0


def main() -> None:
    print("=" * 80)
    print("ARGUS Retrieval Pipeline — Per-Stage Latency Breakdown")
    print("=" * 80)

    # ── 1. Build benchmark store ──────────────────────────────────────────
    print("\n[1/4] Building benchmark corpus & store …")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # ── 2. Build hybrid retriever (indexes) ──────────────────────────────
    print("\n[2/4] Building retrieval indexes …")
    t0 = time.monotonic()
    retriever = HybridRetriever(store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # ── 3. Router + planner ──────────────────────────────────────────────
    router = RetrievalPolicyRouter()
    planner = EvidenceNeedPlanner()

    # ── 4. Warm-up (2 queries) ────────────────────────────────────────────
    queries = load_eval_queries()
    print(f"\n[3/4] Warming up with {min(2, len(queries))} queries …")
    for q in queries[:2]:
        retriever.search(q["query"], top_k=20)
    print("  Done.")

    # ── 5. Measure per-stage latency for every query ─────────────────────
    PLANNABLE = EvidenceNeedPlanner.PLANNABLE_PATTERNS
    print(f"\n[4/4] Measuring {len(queries)} queries …")

    stage_latencies: dict[str, list[float]] = {
        "classification": [],
        "planning": [],
        "bm25": [],
        "embedding": [],
        "vector": [],
        "fusion": [],
        "multi_query_total": [],
        "recovery": [],
    }
    query_results: list[dict] = []

    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        qid = q["id"]
        pattern = q["pattern"]

        # --- Classification ---
        t = time.perf_counter()
        classified = router.classify_question(query_text)
        cls_ms = (time.perf_counter() - t) * 1000
        stage_latencies["classification"].append(cls_ms)

        # --- Planning (only for plannable patterns) ---
        plan = None
        plan_ms = 0.0
        if pattern in PLANNABLE:
            t = time.perf_counter()
            plan = planner.plan(query_text, pattern)
            plan_ms = (time.perf_counter() - t) * 1000
        stage_latencies["planning"].append(plan_ms)

        # --- BM25 ---
        t = time.perf_counter()
        bm25_results = retriever.bm25.search(query_text, top_k=20)
        bm25_ms = (time.perf_counter() - t) * 1000
        stage_latencies["bm25"].append(bm25_ms)

        # --- Embedding ---
        t = time.perf_counter()
        embedding = retriever.embedder.embed_texts([query_text])[0]
        emb_ms = (time.perf_counter() - t) * 1000
        stage_latencies["embedding"].append(emb_ms)

        # --- Vector search ---
        t = time.perf_counter()
        vector_results = retriever.vector.search(embedding, top_k=20)
        vec_ms = (time.perf_counter() - t) * 1000
        stage_latencies["vector"].append(vec_ms)

        # --- Fusion ---
        bm25_scores = {str(cid): sc for cid, sc in bm25_results}
        vec_scores = {str(cid): sc for cid, sc in vector_results}
        t = time.perf_counter()
        fused_refs = HybridRetriever._fuse(
            store, query_text, 20, 0.5, 0.5, bm25_scores, vec_scores,
        )
        fusion_ms = (time.perf_counter() - t) * 1000
        stage_latencies["fusion"].append(fusion_ms)

        # --- Multi-query total (plannable patterns) ---
        mq_ms = 0.0
        recovery_ms = 0.0
        if pattern in PLANNABLE:
            t = time.perf_counter()
            refs = asyncio.run(
                router.execute_planned_retrieval(
                    query_text, classified, retriever, top_k=20,
                )
            )
            mq_ms = (time.perf_counter() - t) * 1000

            # Detect recovery activation from result metadata
            recovery_detected = any(
                r.metadata.get("recovery_source") for r in refs
            )
            if recovery_detected:
                # Estimate recovery cost: multi_query_total - (bm25 + emb + vec) × sub_queries
                sub_q = plan.query_count if plan else 1
                base_cost = (bm25_ms + emb_ms + vec_ms) * sub_q
                recovery_ms = max(0.0, mq_ms - base_cost - plan_ms)
        stage_latencies["multi_query_total"].append(mq_ms)
        stage_latencies["recovery"].append(recovery_ms)

        query_results.append({
            "id": qid,
            "query": query_text[:60],
            "pattern": pattern,
            "eval_class": q["eval_class"],
            "cls_ms": round(cls_ms, 2),
            "plan_ms": round(plan_ms, 2),
            "bm25_ms": round(bm25_ms, 2),
            "emb_ms": round(emb_ms, 2),
            "vec_ms": round(vec_ms, 2),
            "fusion_ms": round(fusion_ms, 2),
            "mq_ms": round(mq_ms, 2),
            "recovery_ms": round(recovery_ms, 2),
        })

        status = " [PLANNABLE]" if pattern in PLANNABLE else ""
        print(f"  [{i:>2}/{len(queries)}] {qid:<4} {cls_ms:>6.1f}ms cls | "
              f"{bm25_ms:>6.1f}ms bm25 | {emb_ms:>6.1f}ms emb | "
              f"{vec_ms:>6.1f}ms vec | {fusion_ms:>5.1f}ms fuse"
              + (f" | {mq_ms:>7.1f}ms MQ" if pattern in PLANNABLE else "")
              + status)

    # ══════════════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════════════

    stages_for_table = [
        "classification", "planning", "bm25", "embedding",
        "vector", "fusion", "multi_query_total", "recovery",
    ]

    print("\n" + "=" * 92)
    print("AGGREGATE LATENCY (all 38 queries)")
    print("=" * 92)
    header = f"{'Stage':<22} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}"
    print(header)
    print("-" * len(header))
    for stg in stages_for_table:
        vals = stage_latencies[stg]
        print(f"{stg:<22} {_mean(vals):>10.2f} {_percentile(vals, 50):>10.2f} "
              f"{_percentile(vals, 95):>10.2f} {min(vals) if vals else 0:>10.2f} "
              f"{max(vals) if vals else 0:>10.2f}")

    # ── Per-class breakdown ──────────────────────────────────────────────
    classes = sorted(set(q["eval_class"] for q in queries))
    print("\n" + "=" * 92)
    print("AVERAGE LATENCY BY EVAL CLASS")
    print("=" * 92)
    cls_header = f"{'Class':<26} {'N':>3} {'Cls':>7} {'Plan':>7} {'BM25':>7} {'Emb':>7} {'Vec':>7} {'Fuse':>7} {'MQ-Tot':>8}"
    print(cls_header)
    print("-" * len(cls_header))
    for cls_name in classes:
        cls_queries = [r for r in query_results if r["eval_class"] == cls_name]
        n = len(cls_queries)
        if n == 0:
            continue
        print(
            f"{cls_name:<26} {n:>3} "
            f"{_mean([r['cls_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['plan_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['bm25_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['emb_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['vec_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['fusion_ms'] for r in cls_queries]):>7.1f} "
            f"{_mean([r['mq_ms'] for r in cls_queries]):>8.1f}"
        )

    # ── Dominant stage identification ────────────────────────────────────
    core_stages = ["classification", "bm25", "embedding", "vector", "fusion"]
    core_avgs = {s: _mean(stage_latencies[s]) for s in core_stages}
    dominant_core = max(core_avgs, key=core_avgs.get)

    all_stages_with_data = [s for s in stages_for_table if _mean(stage_latencies[s]) > 0]
    all_avgs = {s: _mean(stage_latencies[s]) for s in all_stages_with_data}
    dominant_all = max(all_avgs, key=all_avgs.get) if all_avgs else "N/A"

    print("\n" + "=" * 92)
    print("LATENCY DOMINANCE")
    print("=" * 92)
    print(f"  Core pipeline (cls+bm25+emb+vec+fuse) dominant stage: {dominant_core}  ({core_avgs[dominant_core]:.1f} ms avg)")
    print(f"  Overall dominant stage (incl. multi-query):            {dominant_all}  ({all_avgs.get(dominant_all, 0):.1f} ms avg)")

    # Percentage breakdown of core pipeline
    total_core = sum(core_avgs.values())
    if total_core > 0:
        print(f"\n  Core pipeline percentage breakdown:")
        for s in core_stages:
            pct = core_avgs[s] / total_core * 100
            bar = "#" * int(pct / 2)
            print(f"    {s:<16} {core_avgs[s]:>7.1f} ms  {pct:>5.1f}%  {bar}")

    # ── Per-query detail table ───────────────────────────────────────────
    print("\n" + "=" * 110)
    print("PER-QUERY DETAIL")
    print("=" * 110)
    det_header = f"{'ID':<5} {'Pattern':<22} {'Cls':>6} {'Plan':>6} {'BM25':>6} {'Emb':>6} {'Vec':>6} {'Fuse':>6} {'MQ':>8} {'Rec':>6}"
    print(det_header)
    print("-" * len(det_header))
    for r in query_results:
        print(
            f"{r['id']:<5} {r['pattern']:<22} {r['cls_ms']:>6.1f} {r['plan_ms']:>6.1f} "
            f"{r['bm25_ms']:>6.1f} {r['emb_ms']:>6.1f} {r['vec_ms']:>6.1f} "
            f"{r['fusion_ms']:>6.1f} {r['mq_ms']:>8.1f} {r['recovery_ms']:>6.1f}"
        )

    # ── Save JSON report ─────────────────────────────────────────────────
    report = {
        "benchmark": "latency_breakdown",
        "query_count": len(queries),
        "aggregate": {
            stg: {
                "avg_ms": round(_mean(stage_latencies[stg]), 2),
                "p50_ms": round(_percentile(stage_latencies[stg], 50), 2),
                "p95_ms": round(_percentile(stage_latencies[stg], 95), 2),
            }
            for stg in stages_for_table
        },
        "per_class": {
            cls_name: {
                "count": len([r for r in query_results if r["eval_class"] == cls_name]),
                "avg_cls_ms": round(_mean([r["cls_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
                "avg_bm25_ms": round(_mean([r["bm25_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
                "avg_emb_ms": round(_mean([r["emb_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
                "avg_vec_ms": round(_mean([r["vec_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
                "avg_fusion_ms": round(_mean([r["fusion_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
                "avg_mq_ms": round(_mean([r["mq_ms"] for r in query_results if r["eval_class"] == cls_name]), 2),
            }
            for cls_name in classes
        },
        "dominant_stage": {
            "core_pipeline": dominant_core,
            "overall": dominant_all,
        },
        "per_query": query_results,
    }
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "latency_breakdown.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
