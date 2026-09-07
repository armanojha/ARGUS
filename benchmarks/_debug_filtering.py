#!/usr/bin/env python3
"""Debug: check why C1 (revenue) is filtered."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.orchestration.graph import run_query
from app.retrieval.hybrid import HybridRetriever
from app.reranking.reranker import NoOpReranker
from app.orchestration.nodes import _detect_contradictions, filter_contradictions_by_query
from benchmarks.benchmark_fusion import build_benchmark_store

async def debug():
    settings = Settings()
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()

    q = "What was Acme's annual revenue?"
    print(f"Query: {q}\n")

    result = await run_query(query=q, settings=settings, retriever=retriever, reranker=reranker)
    evidence = result.citations or []
    print(f"Evidence chunks: {len(evidence)}")
    for idx, e in enumerate(evidence):
        preview = e.text[:120].replace("\n", " ")
        print(f"  [{idx+1}] score={e.score:.3f} | {preview}...")

    # Convert citations to EvidenceRef-like objects for detection
    from app.retrieval.model import EvidenceRef, SourceType
    evidence_refs = []
    for c in evidence:
        evidence_refs.append(EvidenceRef(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source_id=c.source_id,
            source_path=c.source_path,
            source_type=SourceType(c.source_type),
            text=c.text,
            score=c.score,
        ))

    detected = _detect_contradictions(evidence_refs, query=q)
    print(f"\nDetected contradictions: {len(detected)}")
    for d in detected:
        print(f"  type={d.get('conflict_type')} confidence={d.get('confidence')} metrics={d.get('metric_overlap')}")

    filtered = filter_contradictions_by_query(detected, evidence_refs, q)
    print(f"\nFiltered contradictions: {len(filtered)}")

asyncio.run(debug())
