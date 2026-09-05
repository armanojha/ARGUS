"""Quick benchmark: rank-normalized fusion vs max-normalized fusion."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.config import get_settings
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from benchmarks.benchmark_fusion import (
    BENCHMARK_QUERIES, build_benchmark_store, recall_at_k, mrr, ndcg_at_k,
)


def main():
    settings = get_settings()
    store, chunk_id_map = build_benchmark_store()
    bm25 = BM25Retriever(store)
    vector = FAISSVectorStore(store)
    embedder = EmbeddingGenerator()
    retriever = HybridRetriever(store=store, bm25=bm25, vector=vector, embedder=embedder)
    retriever.ensure_indexes()

    # Warm up
    for bq in BENCHMARK_QUERIES[:2]:
        retriever.search(bq["query"], top_k=10)

    # Collect raw scores
    raw_scores = []
    for bq in BENCHMARK_QUERIES:
        query = bq["query"]
        bm25_results = bm25.search(query, top_k=30)
        bm25_scores = {str(cid): score for cid, score in bm25_results}
        query_emb = embedder.embed_texts([query])[0]
        vector_results = vector.search(query_emb, top_k=30)
        vector_scores = {str(cid): score for cid, score in vector_results}
        gold_ids = set()
        for doc_id in bq.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])
        raw_scores.append({
            "query": query, "pattern": bq["pattern"],
            "bm25_scores": bm25_scores, "vector_scores": vector_scores,
            "gold_ids": gold_ids,
        })

    print("Rank vs Max normalization (using actual HybridRetriever._fuse):")
    for norm in ["max", "rank"]:
        r5s, r10s, mrrs, ndcgs = [], [], [], []
        for rs in raw_scores:
            # Use the actual _fuse method with different normalization
            refs = HybridRetriever._fuse(
                store, rs["query"], 10, 0.5, 0.5,
                rs["bm25_scores"], rs["vector_scores"], normalization=norm,
            )
            chunk_ids = [str(r.chunk_id) for r in refs]
            r5s.append(recall_at_k(chunk_ids, rs["gold_ids"], 5))
            r10s.append(recall_at_k(chunk_ids, rs["gold_ids"], 10))
            mrrs.append(mrr(chunk_ids, rs["gold_ids"]))
            ndcgs.append(ndcg_at_k(chunk_ids, rs["gold_ids"], 10))
        print(f"  {norm:4s}: R@5={np.mean(r5s):.3f}  R@10={np.mean(r10s):.3f}  MRR={np.mean(mrrs):.3f}  nDCG={np.mean(ndcgs):.3f}")


if __name__ == "__main__":
    main()
