#!/usr/bin/env python3
"""Conflict benchmark with filtering ENABLED."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.orchestration.graph import run_query
from app.retrieval.hybrid import HybridRetriever
from app.reranking.reranker import NoOpReranker
from benchmarks.phase40_conflict_synthesis import (
    CONFLICT_QUERIES,
    evaluate_conflict_answer,
)
from benchmarks.benchmark_fusion import build_benchmark_store

async def test():
    settings = Settings()
    settings.conflict_filtering_enabled = True
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()
    results = []
    for qd in CONFLICT_QUERIES:
        query = qd["query"]
        print(f"  [{qd['id']}] {query[:50]}...", end=" ")
        result = await run_query(query=query, settings=settings, retriever=retriever, reranker=reranker)
        cs = getattr(result, "contradiction_signals", []) or []
        evidence_texts = [c.text for c in (result.citations or [])]
        metrics = evaluate_conflict_answer(
            answer=result.answer or "",
            query_data=qd,
            evidence_texts=evidence_texts,
            contradiction_signals=cs,
        )
        ack = "ACK" if metrics.conflict_acknowledged else "NO-ACK"
        det = "DET" if metrics.conflict_detected else "NO-DET"
        fr = "FALSE-RES" if metrics.false_resolution else "OK"
        print(f"{det} | {ack} | {fr} | signals={len(cs)}")
        results.append({
            "query_id": qd["id"],
            "detected": metrics.conflict_detected,
            "acknowledged": metrics.conflict_acknowledged,
            "false_resolution": metrics.false_resolution,
            "signal_count": len(cs),
        })
    print()
    n = len(results)
    print(f"SUMMARY (n={n})")
    print(f"  Conflict detected:    {sum(1 for r in results if r['detected'])}/{n}")
    print(f"  Conflict acknowledged: {sum(1 for r in results if r['acknowledged'])}/{n}")
    print(f"  False resolution:     {sum(1 for r in results if r['false_resolution'])}/{n}")

asyncio.run(test())
