"""Fusion Strategy Benchmark.

Compares different score fusion methods on the existing ARGUS evaluation
query set. Takes raw BM25 and vector results from the baseline and
applies different fusion strategies to measure which produces the best
ranking quality.

Usage:
    python -m benchmarks.benchmark_fusion
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices


# ---------------------------------------------------------------------------
# Benchmark queries (same as bge_m3 benchmark)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {"id": "A1", "query": "Where is Acme Corporation headquartered?", "pattern": "exact_term",
     "gold_facts": ["New York City"], "supporting_docs": ["doc-a"]},
    {"id": "A2", "query": "Who is the CEO of Acme Corporation?", "pattern": "exact_term",
     "gold_facts": ["Diana Reyes"], "supporting_docs": ["doc-a"]},
    {"id": "A3", "query": "What year was Acme Corporation founded?", "pattern": "exact_term",
     "gold_facts": ["1987"], "supporting_docs": ["doc-a"]},
    {"id": "A4", "query": "What is the default storage engine of Atlas?", "pattern": "exact_term",
     "gold_facts": ["columnar"], "supporting_docs": ["doc-b"]},
    {"id": "A5", "query": "At what depth is the Polaris probe deployed?", "pattern": "exact_term",
     "gold_facts": ["1400", "meters"], "supporting_docs": ["doc-j"]},
    {"id": "B1", "query": "How does Acme Corporation generate revenue?", "pattern": "conceptual",
     "gold_facts": ["revenue", "products", "services"], "supporting_docs": ["doc-a"]},
    {"id": "B2", "query": "What are the main features of Atlas database?", "pattern": "conceptual",
     "gold_facts": ["columnar", "storage", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "B3", "query": "Explain the supply chain process at Acme.", "pattern": "conceptual",
     "gold_facts": ["supply chain", "logistics"], "supporting_docs": ["doc-c"]},
    {"id": "C1", "query": "Why does Atlas use a columnar storage engine?", "pattern": "causal",
     "gold_facts": ["columnar", "compression", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "C2", "query": "How does the Polaris probe sensor array work?", "pattern": "conceptual",
     "gold_facts": ["sensor", "array", "depth"], "supporting_docs": ["doc-j"]},
    {"id": "D1", "query": "Compare Acme's 2023 and 2025 revenue figures.", "pattern": "comparative",
     "gold_facts": ["revenue", "2023", "2025"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "D2", "query": "Give me a comprehensive report on Frontier Fusion's technology.", "pattern": "long_report",
     "gold_facts": ["Frontier Fusion", "technology"], "supporting_docs": ["doc-g"]},
    {"id": "E1", "query": "What is the relationship between Acme and its suppliers?", "pattern": "entity_relationship",
     "gold_facts": ["Acme", "suppliers", "relationship"], "supporting_docs": ["doc-a", "doc-c"]},
    {"id": "H1", "query": "What was the Q3 metrics report for Acme?", "pattern": "exact_term",
     "gold_facts": ["Q3", "metrics"], "supporting_docs": ["doc-h"]},
    {"id": "I1", "query": "What is Acme Corporation's product roadmap for the next year?", "pattern": "long_report",
     "gold_facts": ["product", "roadmap", "planned"], "supporting_docs": ["doc-i"]},
    {"id": "J1", "query": "How do I hack into Acme's database?", "pattern": "procedural",
     "gold_facts": [], "supporting_docs": []},
]


# ---------------------------------------------------------------------------
# Corpus builder (same as bge_m3 benchmark)
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent / "eval_data"
_CORPUS_DIR = _EVAL_DIR / "corpus_v1"


def _load_corpus_chunks() -> dict[str, list[str]]:
    chunks_by_doc: dict[str, list[str]] = {}
    if not _CORPUS_DIR.exists():
        return chunks_by_doc
    for path in sorted(_CORPUS_DIR.glob("*.md")):
        doc_id = path.stem
        short_id = doc_id.split("-")[0] + "-" + doc_id.split("-")[1] if "-" in doc_id else doc_id
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        doc_chunks = []
        for para in paragraphs:
            if len(current_chunk) + len(para) > 512 and current_chunk:
                doc_chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()
        if current_chunk:
            doc_chunks.append(current_chunk)
        chunks_by_doc[short_id] = doc_chunks
    return chunks_by_doc


def build_benchmark_store() -> tuple[EvidenceStore, dict[str, list[str]]]:
    chunks_by_doc = _load_corpus_chunks()
    if not chunks_by_doc:
        raise RuntimeError(f"No corpus found at {_CORPUS_DIR}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_fusion_bench_"))
    store = EvidenceStore(
        db_path=tmp_dir / "evidence.db",
        bm25_index_path=tmp_dir / "bm25.pkl",
        faiss_index_path=tmp_dir / "faiss.index",
    )
    source = Source(type=SourceType.TEXT, path="/benchmark/corpus", checksum="benchmark_corpus_v1")
    store.upsert_source(source)
    chunk_id_map: dict[str, list[str]] = {}
    all_chunks: list[Chunk] = []
    doc_version = 0
    for doc_id, texts in chunks_by_doc.items():
        doc_version += 1
        doc = Document(source_id=source.id, version=doc_version, checksum=f"bench_{doc_id}", chunking_strategy="benchmark_v1")
        store.insert_document(doc)
        doc_chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(document_id=doc.id, ordinal=i, text=text, token_count=len(text.split()))
            doc_chunks.append(chunk)
        store.insert_chunks(doc_chunks)
        chunk_id_map[doc_id] = [str(c.id) for c in doc_chunks]
        all_chunks.extend(doc_chunks)
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)
    return store, chunk_id_map


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & gold) / len(gold)

def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, cid in enumerate(retrieved[:k]) if cid in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

@dataclass
class ScoredResult:
    chunk_id: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int = 0
    vector_rank: int = 0


def fuse_current(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[ScoredResult]:
    """Current ARGUS fusion: max-normalized BM25 + raw cosine vector."""
    if not bm25_scores and not vector_scores:
        return []
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    results = []
    for cid in all_ids:
        bm25_s = bm25_scores.get(cid, 0.0)
        vec_s = vector_scores.get(cid, 0.0)
        norm_bm25 = bm25_s / max_bm25 if max_bm25 > 0 else 0.0
        fused = bm25_weight * norm_bm25 + vector_weight * vec_s
        results.append(ScoredResult(chunk_id=cid, score=fused, bm25_score=bm25_s, vector_score=vec_s))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuse_rrf(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
    k: int = 60,
) -> list[ScoredResult]:
    """Reciprocal Rank Fusion: 1/(k + rank) for each channel."""
    bm25_sorted = sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)
    vector_sorted = sorted(vector_scores.keys(), key=lambda c: vector_scores[c], reverse=True)
    bm25_ranks = {cid: i + 1 for i, cid in enumerate(bm25_sorted)}
    vector_ranks = {cid: i + 1 for i, cid in enumerate(vector_sorted)}
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    results = []
    for cid in all_ids:
        rrf_bm25 = 1.0 / (k + bm25_ranks.get(cid, len(bm25_scores) + 1))
        rrf_vector = 1.0 / (k + vector_ranks.get(cid, len(vector_scores) + 1))
        fused = rrf_bm25 + rrf_vector
        results.append(ScoredResult(chunk_id=cid, score=fused, bm25_score=bm25_scores.get(cid, 0.0), vector_score=vector_scores.get(cid, 0.0)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuse_sum_normalized(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[ScoredResult]:
    """Sum-normalized: each channel's scores sum to 1.0."""
    bm25_sum = sum(bm25_scores.values()) or 1.0
    vector_sum = sum(vector_scores.values()) or 1.0
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    results = []
    for cid in all_ids:
        norm_bm25 = bm25_scores.get(cid, 0.0) / bm25_sum
        norm_vec = vector_scores.get(cid, 0.0) / vector_sum
        fused = bm25_weight * norm_bm25 + vector_weight * norm_vec
        results.append(ScoredResult(chunk_id=cid, score=fused, bm25_score=bm25_scores.get(cid, 0.0), vector_score=vector_scores.get(cid, 0.0)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuse_borda(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[ScoredResult]:
    """Borda count: rank-based voting. Each rank position = n - rank."""
    bm25_sorted = sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)
    vector_sorted = sorted(vector_scores.keys(), key=lambda c: vector_scores[c], reverse=True)
    n_bm25 = len(bm25_sorted)
    n_vec = len(vector_sorted)
    borda_scores: dict[str, float] = defaultdict(float)
    for i, cid in enumerate(bm25_sorted):
        borda_scores[cid] += bm25_weight * (n_bm25 - i) / max(n_bm25, 1)
    for i, cid in enumerate(vector_sorted):
        borda_scores[cid] += vector_weight * (n_vec - i) / max(n_vec, 1)
    results = []
    for cid, score in borda_scores.items():
        results.append(ScoredResult(chunk_id=cid, score=score, bm25_score=bm25_scores.get(cid, 0.0), vector_score=vector_scores.get(cid, 0.0)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuse_query_adaptive(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
    query: str,
    pattern: str,
) -> list[ScoredResult]:
    """Query-dependent weighting: different weights based on query pattern."""
    if pattern == "exact_term":
        bm25_w, vector_w = 0.7, 0.3
    elif pattern == "conceptual":
        bm25_w, vector_w = 0.3, 0.7
    elif pattern == "long_report":
        bm25_w, vector_w = 0.4, 0.6
    elif pattern == "entity_relationship":
        bm25_w, vector_w = 0.5, 0.5
    elif pattern == "comparative":
        bm25_w, vector_w = 0.4, 0.6
    elif pattern == "causal":
        bm25_w, vector_w = 0.3, 0.7
    else:
        bm25_w, vector_w = 0.5, 0.5
    return fuse_current(bm25_scores, vector_scores, bm25_w, vector_w)


def fuse_max_of_both(
    bm25_scores: dict[str, float],
    vector_scores: dict[str, float],
) -> list[ScoredResult]:
    """Take the max score from either channel (no weighting)."""
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    bm25_sorted = sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)
    vector_sorted = sorted(vector_scores.keys(), key=lambda c: vector_scores[c], reverse=True)
    bm25_ranks = {cid: i + 1 for i, cid in enumerate(bm25_sorted)}
    vector_ranks = {cid: i + 1 for i, cid in enumerate(vector_sorted)}
    results = []
    for cid in all_ids:
        # Use rank-based score: 1/rank (higher = better)
        bm25_r = 1.0 / bm25_ranks.get(cid, 100)
        vec_r = 1.0 / vector_ranks.get(cid, 100)
        fused = max(bm25_r, vec_r)
        results.append(ScoredResult(chunk_id=cid, score=fused, bm25_score=bm25_scores.get(cid, 0.0), vector_score=vector_scores.get(cid, 0.0)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

FUSION_STRATEGIES = {
    "current": ("Current (max-norm BM25 + cosine)", fuse_current),
    "rrf": ("RRF (k=60)", lambda bm, vs, **kw: fuse_rrf(bm, vs)),
    "sum_norm": ("Sum-Normalized", fuse_sum_normalized),
    "borda": ("Borda Count", fuse_borda),
    "query_adaptive": ("Query-Adaptive Weights", fuse_query_adaptive),
    "max_of_both": ("Max-of-Both", fuse_max_of_both),
}


def run_benchmark(top_k: int = 10) -> dict[str, Any]:
    print("=" * 80)
    print("Fusion Strategy Benchmark")
    print("=" * 80)

    # Build corpus
    print("\n[1/4] Building benchmark corpus...")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} documents ({time.monotonic()-t0:.1f}s)")

    # Build baseline index
    print("\n[2/4] Building baseline index...")
    t0 = time.monotonic()
    baseline = HybridRetriever(store)
    baseline.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # Warmup
    print("\n[3/4] Warming up...")
    for q in BENCHMARK_QUERIES[:2]:
        baseline.search(q["query"], top_k=top_k)

    # Run benchmark
    print(f"\n[4/4] Running benchmark ({len(BENCHMARK_QUERIES)} queries)...")

    # Collect raw BM25 and vector scores per query
    raw_scores: list[dict[str, Any]] = []
    for bq in BENCHMARK_QUERIES:
        query = bq["query"]
        # Get raw BM25 scores
        bm25_results = baseline.bm25.search(query, top_k=top_k * 3)
        bm25_scores = {str(cid): score for cid, score in bm25_results}
        # Get raw vector scores
        query_emb = baseline.embedder.embed_texts([query])[0]
        vector_results = baseline.vector.search(query_emb, top_k=top_k * 3)
        vector_scores = {str(cid): score for cid, score in vector_results}
        # Get gold IDs
        gold_ids = set()
        for doc_id in bq.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])
        raw_scores.append({
            "query_id": bq["id"],
            "pattern": bq["pattern"],
            "query": query,
            "bm25_scores": bm25_scores,
            "vector_scores": vector_scores,
            "gold_ids": gold_ids,
        })

    # Apply each fusion strategy
    all_results: dict[str, list[dict[str, Any]]] = {name: [] for name in FUSION_STRATEGIES}

    for rs in raw_scores:
        for strategy_name, (_, strategy_fn) in FUSION_STRATEGIES.items():
            if strategy_name == "query_adaptive":
                fused = strategy_fn(rs["bm25_scores"], rs["vector_scores"], query=rs["query"], pattern=rs["pattern"])
            else:
                fused = strategy_fn(rs["bm25_scores"], rs["vector_scores"])
            chunk_ids = [r.chunk_id for r in fused[:top_k]]
            r5 = recall_at_k(chunk_ids, rs["gold_ids"], 5)
            r10 = recall_at_k(chunk_ids, rs["gold_ids"], 10)
            m = mrr(chunk_ids, rs["gold_ids"])
            n = ndcg_at_k(chunk_ids, rs["gold_ids"], 10)
            all_results[strategy_name].append({
                "query_id": rs["query_id"],
                "pattern": rs["pattern"],
                "recall_5": r5,
                "recall_10": r10,
                "mrr": m,
                "ndcg_10": n,
            })

    # Aggregate
    summary: dict[str, dict[str, Any]] = {}
    for strategy_name, results in all_results.items():
        valid_r5 = [r["recall_5"] for r in results if not np.isnan(r["recall_5"])]
        valid_r10 = [r["recall_10"] for r in results if not np.isnan(r["recall_10"])]
        valid_n = [r["ndcg_10"] for r in results if not np.isnan(r["ndcg_10"])]
        summary[strategy_name] = {
            "display_name": FUSION_STRATEGIES[strategy_name][0],
            "recall_5": round(np.mean(valid_r5), 4) if valid_r5 else float("nan"),
            "recall_10": round(np.mean(valid_r10), 4) if valid_r10 else float("nan"),
            "mrr": round(np.mean([r["mrr"] for r in results]), 4),
            "ndcg_10": round(np.mean(valid_n), 4) if valid_n else float("nan"),
        }

    # Print table
    print("\n" + "=" * 90)
    print("FUSION STRATEGY COMPARISON")
    print("=" * 90)
    header = f"{'Strategy':<30} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8}"
    print(header)
    print("-" * len(header))
    best_r5 = max(s["recall_5"] for s in summary.values() if not np.isnan(s["recall_5"]))
    for name, s in summary.items():
        marker = " *" if s["recall_5"] == best_r5 else ""
        print(f"{s['display_name']:<30} {s['recall_5']:>6.3f} {s['recall_10']:>6.3f} {s['mrr']:>6.3f} {s['ndcg_10']:>8.3f}{marker}")

    # Per-pattern breakdown
    print("\n" + "=" * 90)
    print("PER-PATTERN Recall@10")
    print("=" * 90)
    patterns = sorted(set(rs["pattern"] for rs in raw_scores))
    strategy_names = list(FUSION_STRATEGIES.keys())
    ph = f"{'Pattern':<20}" + "".join(f" {FUSION_STRATEGIES[s][0][:12]:>12}" for s in strategy_names)
    print(ph)
    print("-" * len(ph))
    for pattern in patterns:
        row = f"{pattern:<20}"
        for sname in strategy_names:
            pr = [r for r in all_results[sname] if r["pattern"] == pattern]
            if pr:
                valid = [r["recall_10"] for r in pr if not np.isnan(r["recall_10"])]
                avg = np.mean(valid) if valid else float("nan")
                row += f" {avg:>12.3f}" if not np.isnan(avg) else f" {'N/A':>12}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Save
    output = {
        "benchmark": "fusion_strategy_comparison",
        "query_count": len(BENCHMARK_QUERIES),
        "top_k": top_k,
        "summary": summary,
        "per_query": {name: results for name, results in all_results.items()},
    }
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "fusion_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/fusion_benchmark.json")
    return output


if __name__ == "__main__":
    run_benchmark()
