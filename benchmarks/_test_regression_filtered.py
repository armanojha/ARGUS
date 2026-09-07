#!/usr/bin/env python3
"""Regression test with conflict filtering ENABLED."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.orchestration.graph import run_query
from app.retrieval.hybrid import HybridRetriever
from app.reranking.reranker import NoOpReranker
from benchmarks.benchmark_fusion import build_benchmark_store

QUERIES = [
    "What is Acme Corporation?",
    "What products does Acme make?",
    "Where are Acme plants located?",
    "How many employees does Acme have?",
    "What industry is Acme in?",
]

async def test():
    settings = Settings()
    settings.conflict_filtering_enabled = True
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()
    results = []
    for q in QUERIES:
        result = await run_query(query=q, settings=settings, retriever=retriever, reranker=reranker)
        cs = getattr(result, "contradiction_signals", []) or []
        ack = "conflict" in (result.answer or "").lower() or "contradict" in (result.answer or "").lower()
        has_det = len(cs) > 0
        print(f"  Q: {q}")
        print(f"    detected={has_det} ({len(cs)} signals)  acknowledged={ack}")
        if has_det:
            for s in cs[:3]:
                print(f"      -> {s.get('conflict_type', '?')} | {s.get('confidence', '?')} | {s.get('description', '?')[:80]}")
        results.append({"query": q, "detected": has_det, "signal_count": len(cs), "acknowledged": ack})
    print()
    false_pos = sum(1 for r in results if r["detected"])
    print(f"FALSE POSITIVE RATE: {false_pos}/{len(results)} ({false_pos/len(results):.0%})")
    return results

results = asyncio.run(test())
output = Path("benchmarks/results/phase41_regression_filtered.json")
output.parent.mkdir(parents=True, exist_ok=True)
with open(output, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {output}")
