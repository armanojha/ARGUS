#!/usr/bin/env python3
"""Phase 40: Run only Strategy A (baseline) on conflict queries."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.phase40_conflict_synthesis import (
    CONFLICT_QUERIES,
    evaluate_conflict_answer,
    ConflictMetrics,
)


async def run_baseline():
    from app.config import Settings
    from app.orchestration.graph import run_query
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import NoOpReranker

    settings = Settings()

    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()

    results = []

    for qd in CONFLICT_QUERIES:
        query = qd["query"]
        print(f"\n  [{qd['id']}] {query}")

        try:
            t0 = time.time()
            result = await run_query(
                query=query,
                settings=settings,
                retriever=retriever,
                reranker=reranker,
            )
            total_ms = (time.time() - t0) * 1000

            contradiction_signals = getattr(result, 'contradiction_signals', []) or []
            evidence_texts = [c.text for c in (result.citations or [])]

            metrics = evaluate_conflict_answer(
                answer=result.answer or "",
                query_data=qd,
                evidence_texts=evidence_texts,
                contradiction_signals=contradiction_signals,
            )

            record = {
                "query_id": qd["id"],
                "query": query,
                "strategy": "A",
                "strategy_name": "Baseline",
                "answer": result.answer,
                "answer_length": len(result.answer or ""),
                "evidence_count": len(result.citations or []),
                "contradiction_signals_count": len(contradiction_signals),
                "conflict_detected": metrics.conflict_detected,
                "conflict_acknowledged": metrics.conflict_acknowledged,
                "source_attribution_correct": metrics.source_attribution_correct,
                "false_resolution": metrics.false_resolution,
                "required_elements_present": metrics.required_elements_present,
                "gold_fact_coverage": metrics.gold_fact_coverage,
                "citation_precision": metrics.citation_precision,
                "total_e2e_ms": round(total_ms, 2),
                "outcome": str(getattr(result, "outcome", "unknown")),
                "warnings": getattr(result, "warnings", []) or [],
                "providers": getattr(result, "providers_used", []) or [],
            }
            results.append(record)

            ack = "ACK" if metrics.conflict_acknowledged else "NO-ACK"
            fr = "FALSE-RES" if metrics.false_resolution else "OK"
            det = "DET" if metrics.conflict_detected else "NO-DET"
            print(f"    {det} | {ack} | {fr} | GFC={metrics.gold_fact_coverage:.0%} | CP={metrics.citation_precision:.0%}")
            print(f"    Answer preview: {(result.answer or '')[:120]}...")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query_id": qd["id"],
                "query": query,
                "strategy": "A",
                "error": str(e),
            })

    return results


def main():
    import asyncio
    print("=" * 70)
    print("PHASE 40: STRATEGY A (BASELINE) - CONFLICT QUERIES")
    print("=" * 70)

    results = asyncio.run(run_baseline())

    output_path = Path("benchmarks/results/phase40_strategy_a.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary
    valid = [r for r in results if "error" not in r]
    if valid:
        n = len(valid)
        print(f"\n{'='*70}")
        print(f"SUMMARY (n={n})")
        print(f"{'='*70}")
        print(f"  Conflict detected:    {sum(1 for r in valid if r.get('conflict_detected'))}/{n}")
        print(f"  Conflict acknowledged: {sum(1 for r in valid if r.get('conflict_acknowledged'))}/{n}")
        print(f"  Source attribution:    {sum(1 for r in valid if r.get('source_attribution_correct'))}/{n}")
        print(f"  False resolution:     {sum(1 for r in valid if r.get('false_resolution'))}/{n}")
        avg_gfc = sum(r.get('gold_fact_coverage', 0) for r in valid) / n
        avg_cp = sum(r.get('citation_precision', 0) for r in valid) / n
        print(f"  Gold fact coverage:   {avg_gfc:.0%}")
        print(f"  Citation precision:   {avg_cp:.0%}")


if __name__ == "__main__":
    main()
