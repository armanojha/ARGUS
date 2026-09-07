#!/usr/bin/env python3
"""Debug: see what contradictions are detected for a non-conflict query."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.orchestration.graph import run_query
from app.retrieval.hybrid import HybridRetriever
from app.reranking.reranker import NoOpReranker
from app.orchestration.nodes import _detect_contradictions
from benchmarks.benchmark_fusion import build_benchmark_store

async def debug():
    settings = Settings()
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()

    q = "What products does Acme make?"
    print(f"Query: {q}\n")

    # Get evidence directly
    from app.retrieval.model import RetrievalQuery
    rq = RetrievalQuery(query=q, top_k=8)
    evidence = retriever.retrieve(rq)
    print(f"Evidence chunks: {len(evidence)}")
    for idx, e in enumerate(evidence):
        preview = e.text[:120].replace("\n", " ")
        print(f"  [{idx+1}] score={e.score:.3f} doc={e.document_id} | {preview}...")

    print(f"\n--- Detecting contradictions ---")
    contradictions = _detect_contradictions(evidence, query=q)
    print(f"Contradictions found: {len(contradictions)}")
    for c in contradictions:
        print(f"  {c['description'][:120]}")

asyncio.run(debug())
