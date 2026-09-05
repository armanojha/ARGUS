"""Diagnostic trace of all failed retrieval queries.

For each query, shows:
- Query text, pattern, gold docs
- Retrieved chunk IDs, source paths, scores
- Which gold chunks were missed and why
- BM25 vs vector contribution per result
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.config import get_settings
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import RetrievalPolicyRouter, _query_terms


def load_eval_data() -> tuple[list[dict], dict[str, list[str]], dict, EvidenceStore]:
    """Load eval queries, chunk_id_map, eval plan, and the evidence store."""
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        plan = json.load(f)
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    queries = []
    for q in plan.get("queries", []):
        queries.append({
            "id": q.get("id", "unknown"),
            "query": q.get("query", ""),
            "pattern": q.get("class", "unknown"),
            "supporting_docs": q.get("supporting_docs", []),
            "gold_facts": q.get("gold_facts", []),
        })
    return queries, chunk_id_map, plan, store


def trace_query(retriever, router, chunk_id_map, q, top_k=10):
    """Trace a single query with full detail."""
    gold_ids = set()
    for doc_id in q.get("supporting_docs", []):
        if doc_id in chunk_id_map:
            gold_ids.update(chunk_id_map[doc_id])

    pattern = router.classify_question(q["query"])
    mix = router.get_retrieval_mix(pattern)

    # Get raw BM25 and vector scores
    bm25_results = retriever.bm25.search(q["query"], top_k=top_k * 3)
    bm25_scores = {str(cid): score for cid, score in bm25_results}

    query_emb = retriever.embedder.embed_texts([q["query"]])[0]
    vector_results = retriever.vector.search(query_emb, top_k=top_k * 3)
    vector_scores = {str(cid): score for cid, score in vector_results}

    # Get fused results
    refs = retriever.search(q["query"], top_k=top_k)
    retrieved_ids = [str(r.chunk_id) for r in refs]

    # Find hits and misses
    hits = set(retrieved_ids) & gold_ids
    misses = gold_ids - set(retrieved_ids)

    # Get source info for retrieved chunks
    chunk_sources = {}
    for r in refs:
        cid = str(r.chunk_id)
        chunk_sources[cid] = {
            "source_path": r.source_path,
            "text_preview": r.text[:100],
            "bm25_score": bm25_scores.get(cid, 0),
            "vector_score": vector_scores.get(cid, 0),
            "fused_score": r.score,
        }

    # Get source info for missed chunks
    missed_sources = {}
    for cid in misses:
        # Try to find in BM25 or vector results
        bm25_rank = list(bm25_scores.keys()).index(cid) + 1 if cid in bm25_scores else None
        vector_rank = list(vector_scores.keys()).index(cid) + 1 if cid in vector_scores else None
        missed_sources[cid] = {
            "bm25_rank": bm25_rank,
            "vector_rank": vector_rank,
            "bm25_score": bm25_scores.get(cid, 0),
            "vector_score": vector_scores.get(cid, 0),
        }

    return {
        "query_id": q["id"],
        "query": q["query"],
        "pattern": q["pattern"],
        "classified_pattern": pattern.value,
        "gold_docs": q["supporting_docs"],
        "gold_facts": q.get("gold_facts", []),
        "gold_count": len(gold_ids),
        "retrieved_count": len(refs),
        "hit_count": len(hits),
        "miss_count": len(misses),
        "recall": len(hits) / len(gold_ids) if gold_ids else 0,
        "hits": list(hits),
        "misses": list(misses),
        "retrieved_sources": chunk_sources,
        "missed_sources": missed_sources,
        "policy_methods": [m.value for m in mix.methods],
        "bm25_weight": mix.bm25_weight,
        "vector_weight": mix.vector_weight,
        "query_terms": sorted(_query_terms(q["query"])),
    }


def main():
    print("=" * 90)
    print("RETRIEVAL DIAGNOSTIC TRACE")
    print("=" * 90)

    queries, chunk_id_map, plan, store = load_eval_data()
    print(f"Loaded {len(queries)} queries, {sum(len(v) for v in chunk_id_map.values())} chunks")

    settings = get_settings()
    bm25 = BM25Retriever(store)
    vector = FAISSVectorStore(store)
    embedder = EmbeddingGenerator()
    retriever = HybridRetriever(store=store, bm25=bm25, vector=vector, embedder=embedder)
    retriever.ensure_indexes()

    router = RetrievalPolicyRouter()

    # Warm up
    for q in queries[:2]:
        retriever.search(q["query"], top_k=10)

    # Trace all queries
    results = []
    for q in queries:
        r = trace_query(retriever, router, chunk_id_map, q)
        results.append(r)

    # Separate by pattern
    by_pattern = defaultdict(list)
    for r in results:
        by_pattern[r["pattern"]].append(r)

    # Print failures for each pattern
    for pattern in ["complex_research", "conflict", "multi_hop"]:
        pr = by_pattern.get(pattern, [])
        if not pr:
            continue
        print(f"\n{'=' * 90}")
        print(f"PATTERN: {pattern} ({len(pr)} queries)")
        print(f"{'=' * 90}")

        for r in pr:
            status = "PASS" if r["recall"] >= 0.5 else "FAIL"
            print(f"\n--- [{status}] {r['query_id']}: Recall={r['recall']:.3f} ---")
            print(f"  Query: {r['query'][:100]}")
            print(f"  Classified as: {r['classified_pattern']} (expected: {r['pattern']})")
            print(f"  Gold docs: {r['gold_docs']}")
            print(f"  Gold facts: {r['gold_facts'][:3]}")
            print(f"  Policy: {r['policy_methods']} (bm25={r['bm25_weight']}, vec={r['vector_weight']})")
            print(f"  Gold chunks: {r['gold_count']}, Retrieved: {r['retrieved_count']}, Hits: {r['hit_count']}, Misses: {r['miss_count']}")
            print(f"  Query terms: {r['query_terms']}")

            if r["hits"]:
                print(f"  HIT chunks:")
                for cid in r["hits"][:5]:
                    src = r["retrieved_sources"].get(cid, {})
                    print(f"    {cid}: bm25={src.get('bm25_score', 0):.3f} vec={src.get('vector_score', 0):.3f} fused={src.get('fused_score', 0):.3f}")
                    print(f"           source={src.get('source_path', '?')}")
                    print(f"           text={src.get('text_preview', '?')}")

            if r["misses"]:
                print(f"  MISS chunks:")
                for cid in list(r["misses"])[:5]:
                    src = r["missed_sources"].get(cid, {})
                    bm25_r = src.get("bm25_rank")
                    vec_r = src.get("vector_rank")
                    print(f"    {cid}: bm25_rank={bm25_r} vec_rank={vec_r} bm25_score={src.get('bm25_score', 0):.3f} vec_score={src.get('vector_score', 0):.3f}")

    # Summary stats
    print(f"\n{'=' * 90}")
    print("SUMMARY BY PATTERN")
    print(f"{'=' * 90}")
    for pattern in sorted(by_pattern.keys()):
        pr = by_pattern[pattern]
        avg_recall = np.mean([r["recall"] for r in pr])
        fail_count = sum(1 for r in pr if r["recall"] < 0.5)
        print(f"  {pattern:<25} avg_recall={avg_recall:.3f}  queries={len(pr)}  failed={fail_count}")

    # Save trace
    report_path = Path("data/benchmark_reports/retrieval_diagnostic.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull trace saved to {report_path}")


if __name__ == "__main__":
    main()
