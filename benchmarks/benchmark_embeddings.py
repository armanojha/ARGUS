"""Phase 21-A: Embedding Model Benchmark.

Compares candidate embedding models on the same ARGUS evaluation dataset.
Only the embedding backend changes — everything else (BM25, fusion, planner,
recovery, evaluation) stays identical.

Usage:
    python benchmarks/benchmark_embeddings.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from app.config import get_settings
from app.evidence.models import Chunk
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.policy import QuestionPattern

# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
CANDIDATES = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "short": "all-MiniLM-L6-v2", "dim": 384},
    {"name": "BAAI/bge-small-en-v1.5", "short": "bge-small-en-v1.5", "dim": 384},
    {"name": "BAAI/bge-base-en-v1.5", "short": "bge-base-en-v1.5", "dim": 768},
    {"name": "nomic-ai/nomic-embed-text-v1.5", "short": "nomic-embed-v1.5", "dim": 768},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_queries() -> list[dict]:
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        plan = json.load(f)
    queries = []
    for q in plan.get("queries", []):
        eval_class = q.get("class", "unknown")
        canonical = QuestionPattern.from_eval_class(eval_class)
        queries.append({
            "id": q["id"],
            "query": q["query"],
            "pattern": canonical.value,
            "eval_class": eval_class,
            "supporting_docs": q.get("supporting_docs", []),
        })
    return queries


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_cfg: dict,
    store: EvidenceStore,
    chunk_id_map: dict[str, list[str]],
    queries: list[dict],
    corpus_chunks: list[Chunk],
) -> dict:
    """Run full benchmark for a single embedding model."""
    from sentence_transformers import SentenceTransformer

    model_name = model_cfg["name"]
    short_name = model_cfg["short"]
    dim = model_cfg["dim"]

    print(f"\n{'='*60}")
    print(f"Benchmarking: {short_name} (dim={dim})")
    print(f"{'='*60}")

    # 1) Load model + measure load time
    t0 = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # 2) Generate embeddings for all chunks
    texts = [c.text for c in corpus_chunks]
    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    embed_time = time.time() - t0
    embed_latency_ms = (embed_time / len(texts)) * 1000
    print(f"  Embedded {len(texts)} chunks in {embed_time:.1f}s ({embed_latency_ms:.1f}ms/chunk)")

    # 3) Build FAISS index (isolated, non-destructive)
    index_dir = Path(f"data/indexes/embedding_bench_{short_name}")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"

    norm_emb = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(norm_emb)
    index = faiss.IndexFlatIP(dim)
    index.add(norm_emb)
    faiss.write_index(index, str(index_path))

    chunk_ids = [str(c.id) for c in corpus_chunks]
    ids_path = index_path.with_suffix(".ids.pkl")
    import pickle
    with ids_path.open("wb") as f:
        pickle.dump(chunk_ids, f)

    index_size = index_path.stat().st_size
    print(f"  FAISS index: {index_size / 1024:.0f} KB, {index.ntotal} vectors")

    # 4) Build a FAISSVectorStore wrapper for search
    # We'll search directly to avoid loading the production index
    vec_store = FAISSVectorStore(store=store, index_path=index_path, embedding_dim=dim)
    vec_store._index = index
    vec_store._chunk_ids = chunk_ids

    # 5) Benchmark queries
    results = []
    latencies = []
    embed_latencies = []

    for q in queries:
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        # Embed query
        t0 = time.time()
        q_emb = model.encode([q["query"]], convert_to_numpy=True, normalize_embeddings=False)
        q_embed_ms = (time.time() - t0) * 1000
        embed_latencies.append(q_embed_ms)

        # Search
        t0 = time.time()
        search_results = vec_store.search(q_emb[0], top_k=10)
        search_ms = (time.time() - t0) * 1000
        latencies.append(search_ms)

        retrieved_ids = [str(rid) for rid, _ in search_results]

        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        m = mrr(retrieved_ids, gold_ids)
        n = ndcg_at_k(retrieved_ids, gold_ids, 10)

        results.append({
            "query_id": q["id"],
            "pattern": q["pattern"],
            "eval_class": q["eval_class"],
            "recall_5": r5,
            "recall_10": r10,
            "mrr": m,
            "ndcg_10": n,
            "embed_ms": q_embed_ms,
            "search_ms": search_ms,
            "total_ms": q_embed_ms + search_ms,
        })

    # 6) Aggregate
    valid_r5 = [r["recall_5"] for r in results if not np.isnan(r["recall_5"])]
    valid_r10 = [r["recall_10"] for r in results if not np.isnan(r["recall_10"])]
    valid_mrr = [r["mrr"] for r in results]
    valid_ndcg = [r["ndcg_10"] for r in results if not np.isnan(r["ndcg_10"])]

    summary = {
        "model": model_name,
        "short_name": short_name,
        "dim": dim,
        "recall_at_5": round(np.mean(valid_r5), 4) if valid_r5 else float("nan"),
        "recall_at_10": round(np.mean(valid_r10), 4) if valid_r10 else float("nan"),
        "mrr": round(np.mean(valid_mrr), 4) if valid_mrr else float("nan"),
        "ndcg_at_10": round(np.mean(valid_ndcg), 4) if valid_ndcg else float("nan"),
        "avg_embed_ms": round(np.mean(embed_latencies), 2),
        "avg_search_ms": round(np.mean(latencies), 2),
        "avg_total_ms": round(np.mean(embed_latencies) + np.mean(latencies), 2),
        "p95_total_ms": round(np.percentile(
            [e + s for e, s in zip(embed_latencies, latencies)], 95
        ), 2),
        "model_load_s": round(load_time, 2),
        "index_size_kb": round(index_size / 1024, 1),
        "query_count": len(queries),
    }

    # Per-pattern breakdown
    patterns = sorted(set(r["pattern"] for r in results))
    per_pattern = {}
    for p in patterns:
        pr = [r for r in results if r["pattern"] == p]
        valid = [r["recall_10"] for r in pr if not np.isnan(r["recall_10"])]
        per_pattern[p] = {
            "recall_10": round(np.mean(valid), 4) if valid else float("nan"),
            "count": len(pr),
        }
    summary["per_pattern"] = per_pattern

    # Print
    print(f"\n  Results:")
    print(f"    R@5:  {summary['recall_at_5']:.4f}")
    print(f"    R@10: {summary['recall_at_10']:.4f}")
    print(f"    MRR:  {summary['mrr']:.4f}")
    print(f"    nDCG: {summary['ndcg_at_10']:.4f}")
    print(f"    Latency: {summary['avg_total_ms']:.1f}ms avg, {summary['p95_total_ms']:.1f}ms p95")
    print(f"  Per-pattern R@10:")
    for p, v in per_pattern.items():
        print(f"    {p:<25} {v['recall_10']:.4f}  ({v['count']} queries)")

    # Cleanup model from memory
    del model
    import gc
    gc.collect()

    return {"summary": summary, "per_query": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase 21-A: Embedding Model Benchmark")
    print("=" * 70)

    # Build shared corpus + BM25 index (once)
    from benchmarks.benchmark_fusion import build_benchmark_store
    print("\n[1/4] Building evaluation corpus...")
    t0 = time.time()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.time()-t0:.1f}s)")

    # Load queries
    queries = load_eval_queries()
    print(f"  Queries: {len(queries)}")

    # Get corpus chunks for embedding
    all_chunk_ids = []
    for doc_chunks in chunk_id_map.values():
        all_chunk_ids.extend(doc_chunks)
    from uuid import UUID
    corpus_chunks = store.get_chunks_by_ids([UUID(cid) for cid in all_chunk_ids])

    # Run benchmark for each candidate
    all_results = []
    for cfg in CANDIDATES:
        try:
            result = benchmark_model(cfg, store, chunk_id_map, queries, corpus_chunks)
            all_results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {cfg['short']} — {e}")
            import traceback
            traceback.print_exc()

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Model':<30} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG':>6} {'E2E ms':>8} {'P95 ms':>8} {'Dim':>5} {'Idx KB':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        s = r["summary"]
        print(f"{s['short_name']:<30} {s['recall_at_5']:>6.4f} {s['recall_at_10']:>6.4f} "
              f"{s['mrr']:>6.4f} {s['ndcg_at_10']:>6.4f} {s['avg_total_ms']:>8.1f} "
              f"{s['p95_total_ms']:>8.1f} {s['dim']:>5} {s['index_size_kb']:>8.1f}")

    # Baseline comparison
    if all_results:
        baseline = all_results[0]["summary"]
        print(f"\nBaseline: {baseline['short_name']}")
        for r in all_results[1:]:
            s = r["summary"]
            r5_delta = s["recall_at_5"] - baseline["recall_at_5"]
            r10_delta = s["recall_at_10"] - baseline["recall_at_10"]
            mrr_delta = s["mrr"] - baseline["mrr"]
            ndcg_delta = s["ndcg_at_10"] - baseline["ndcg_at_10"]
            lat_delta = s["avg_total_ms"] - baseline["avg_total_ms"]
            print(f"  vs {s['short_name']}: R@5={r5_delta:+.4f} R@10={r10_delta:+.4f} "
                  f"MRR={mrr_delta:+.4f} nDCG={ndcg_delta:+.4f} Lat={lat_delta:+.1f}ms")

    # Save report
    report = {
        "benchmark": "phase21_embedding_comparison",
        "candidates": [r["summary"] for r in all_results],
        "per_query": {r["summary"]["short_name"]: r["per_query"] for r in all_results},
    }
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase21_embedding_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase21_embedding_benchmark.json")


if __name__ == "__main__":
    main()
