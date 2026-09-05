"""BGE-M3 vs Baseline Retrieval Benchmark.

Compares the baseline BM25 + FAISS (nomic-embed) pipeline against
BGE-M3 dense, BGE-M3 sparse, and BGE-M3 hybrid on the existing
ARGUS evaluation query set.

Usage:
    python -m benchmarks.benchmark_bge_m3

Produces a comparison table printed to stdout and saved to
data/benchmark_reports/bge_m3_benchmark.json.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.evidence.models import Chunk, Document, EvidenceRef, Source, SourceType
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices


# ---------------------------------------------------------------------------
# Benchmark query set (representative of eval_plan_v1 query classes)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    # Class A: simple_lookup (EXACT_TERM pattern)
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

    # Class B: normal_qa (CONCEPTUAL pattern)
    {"id": "B1", "query": "How does Acme Corporation generate revenue?", "pattern": "conceptual",
     "gold_facts": ["revenue", "products", "services"], "supporting_docs": ["doc-a"]},
    {"id": "B2", "query": "What are the main features of Atlas database?", "pattern": "conceptual",
     "gold_facts": ["columnar", "storage", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "B3", "query": "Explain the supply chain process at Acme.", "pattern": "conceptual",
     "gold_facts": ["supply chain", "logistics"], "supporting_docs": ["doc-c"]},

    # Class C: technical_explanation (CONCEPTUAL / CAUSAL)
    {"id": "C1", "query": "Why does Atlas use a columnar storage engine?", "pattern": "causal",
     "gold_facts": ["columnar", "compression", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "C2", "query": "How does the Polaris probe sensor array work?", "pattern": "conceptual",
     "gold_facts": ["sensor", "array", "depth"], "supporting_docs": ["doc-j"]},

    # Class D: multi_doc_synthesis (LONG_REPORT / COMPARATIVE)
    {"id": "D1", "query": "Compare Acme's 2023 and 2025 revenue figures.", "pattern": "comparative",
     "gold_facts": ["revenue", "2023", "2025"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "D2", "query": "Give me a comprehensive report on Frontier Fusion's technology.", "pattern": "long_report",
     "gold_facts": ["Frontier Fusion", "technology"], "supporting_docs": ["doc-g"]},

    # Class E: multi_hop (ENTITY_RELATIONSHIP)
    {"id": "E1", "query": "What is the relationship between Acme and its suppliers?", "pattern": "entity_relationship",
     "gold_facts": ["Acme", "suppliers", "relationship"], "supporting_docs": ["doc-a", "doc-c"]},

    # Class H: numerical (EXACT_TERM)
    {"id": "H1", "query": "What was the Q3 metrics report for Acme?", "pattern": "exact_term",
     "gold_facts": ["Q3", "metrics"], "supporting_docs": ["doc-h"]},

    # Class I: complex_research (LONG_REPORT)
    {"id": "I1", "query": "What is Acme Corporation's product roadmap for the next year?", "pattern": "long_report",
     "gold_facts": ["product", "roadmap", "planned"], "supporting_docs": ["doc-i"]},

    # Class J: adversarial (PROCEDURAL — trick questions)
    {"id": "J1", "query": "How do I hack into Acme's database?", "pattern": "procedural",
     "gold_facts": [], "supporting_docs": []},
]


# ---------------------------------------------------------------------------
# Corpus builder (from eval_plan_v1.json documents)
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent / "eval_data"
_CORPUS_DIR = _EVAL_DIR / "corpus_v1"


def _load_corpus_chunks() -> dict[str, list[str]]:
    """Load document texts from the eval corpus directory.

    Returns dict mapping doc_id -> list of text chunks.
    """
    chunks_by_doc: dict[str, list[str]] = {}
    if not _CORPUS_DIR.exists():
        return chunks_by_doc

    for path in sorted(_CORPUS_DIR.glob("*.md")):
        doc_id = path.stem  # e.g. "doc-a-acme-corp"
        # Map back to short doc_id used in eval_plan
        short_id = doc_id.split("-")[0] + "-" + doc_id.split("-")[1] if "-" in doc_id else doc_id
        text = path.read_text(encoding="utf-8")
        # Simple chunking: split on double newlines, ~512 char chunks
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
    """Build a temporary EvidenceStore with the eval corpus.

    Returns (store, chunk_id_map) where chunk_id_map maps
    doc_id -> list of chunk_id strings.
    """
    chunks_by_doc = _load_corpus_chunks()
    if not chunks_by_doc:
        raise RuntimeError(f"No corpus found at {_CORPUS_DIR}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_bge_bench_"))
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
        doc = Document(
            source_id=source.id,
            version=doc_version,
            checksum=f"bench_{doc_id}",
            chunking_strategy="benchmark_v1",
        )
        store.insert_document(doc)
        doc_chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                document_id=doc.id,
                ordinal=i,
                text=text,
                token_count=len(text.split()),
            )
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

@dataclass
class RetrievalResult:
    chunk_ids: list[str]
    scores: list[float]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    top_k = set(retrieved[:k])
    return len(top_k & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    if not gold:
        return float("nan")
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    # Ideal DCG
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Baseline retriever
# ---------------------------------------------------------------------------

def run_baseline(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> RetrievalResult:
    """Run baseline BM25 + FAISS hybrid search."""
    t0 = time.monotonic()
    results = retriever.search(query, top_k=top_k, bm25_weight=bm25_weight, vector_weight=vector_weight)
    latency = (time.monotonic() - t0) * 1000

    return RetrievalResult(
        chunk_ids=[str(r.chunk_id) for r in results],
        scores=[r.score for r in results],
        latency_ms=round(latency, 1),
        metadata={"method": "baseline_hybrid"},
    )


def run_baseline_bm25_only(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 10,
) -> RetrievalResult:
    """Run BM25-only search (baseline lexical)."""
    t0 = time.monotonic()
    results = retriever.search_bm25_only(query, top_k=top_k)
    latency = (time.monotonic() - t0) * 1000

    return RetrievalResult(
        chunk_ids=[str(r.chunk_id) for r in results],
        scores=[r.score for r in results],
        latency_ms=round(latency, 1),
        metadata={"method": "baseline_bm25_only"},
    )


def run_baseline_vector_only(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 10,
) -> RetrievalResult:
    """Run vector-only search (baseline dense)."""
    t0 = time.monotonic()
    results = retriever.search_vector_only(query, top_k=top_k)
    latency = (time.monotonic() - t0) * 1000

    return RetrievalResult(
        chunk_ids=[str(r.chunk_id) for r in results],
        scores=[r.score for r in results],
        latency_ms=round(latency, 1),
        metadata={"method": "baseline_vector_only"},
    )


# ---------------------------------------------------------------------------
# BGE-M3 retriever
# ---------------------------------------------------------------------------

def run_bge_m3(
    bge: Any,
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
) -> RetrievalResult:
    """Run BGE-M3 search."""
    t0 = time.monotonic()
    results = bge.search_as_refs(query, top_k=top_k, mode=mode)
    latency = (time.monotonic() - t0) * 1000

    return RetrievalResult(
        chunk_ids=[str(r.chunk_id) for r in results],
        scores=[r.score for r in results],
        latency_ms=round(latency, 1),
        metadata={
            "method": f"bge_m3_{mode}",
            "dense_scores": [r.metadata.get("bge_m3_dense_score", 0) for r in results],
            "sparse_scores": [r.metadata.get("bge_m3_sparse_score", 0) for r in results],
        },
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    top_k: int = 10,
    warmup_queries: int = 2,
) -> dict[str, Any]:
    """Run the full BGE-M3 vs baseline benchmark.

    Returns a dict with per-method metrics and a comparison table.
    """
    print("=" * 80)
    print("BGE-M3 vs Baseline Retrieval Benchmark")
    print("=" * 80)

    # Build corpus
    print("\n[1/5] Building benchmark corpus...")
    t0 = time.monotonic()
    store, chunk_id_map = build_benchmark_store()
    corpus_time = time.monotonic() - t0
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} documents ({corpus_time:.1f}s)")

    # Build baseline index
    print("\n[2/5] Building baseline index (BM25 + FAISS)...")
    t0 = time.monotonic()
    baseline = HybridRetriever(store)
    baseline.ensure_indexes()
    baseline_time = time.monotonic() - t0
    print(f"  Baseline index built in {baseline_time:.1f}s")

    # Build BGE-M3 index
    print("\n[3/5] Building BGE-M3 index (dense + sparse)...")
    from app.retrieval.bge_m3 import BGEM3Retriever
    bge = BGEM3Retriever(store)
    t0 = time.monotonic()
    bge.build_index()
    bge_time = time.monotonic() - t0
    print(f"  BGE-M3 index built in {bge_time:.1f}s")
    bge_stats = bge.get_stats()
    print(f"  Dense dim: {bge_stats['dense_dim']}, Sparse nnz: {bge_stats['sparse_nnz']}")

    # Warmup
    print(f"\n[4/5] Warming up ({warmup_queries} queries)...")
    for q in BENCHMARK_QUERIES[:warmup_queries]:
        run_baseline(baseline, q["query"], top_k=top_k)
        run_bge_m3(bge, q["query"], top_k=top_k, mode="hybrid")

    # Run benchmark
    print(f"\n[5/5] Running benchmark ({len(BENCHMARK_QUERIES)} queries, top_k={top_k})...")
    methods = {
        "baseline_hybrid": lambda q, k: run_baseline(baseline, q, top_k=k),
        "baseline_bm25": lambda q, k: run_baseline_bm25_only(baseline, q, top_k=k),
        "baseline_vector": lambda q, k: run_baseline_vector_only(baseline, q, top_k=k),
        "bge_m3_dense": lambda q, k: run_bge_m3(bge, q, top_k=k, mode="dense"),
        "bge_m3_sparse": lambda q, k: run_bge_m3(bge, q, top_k=k, mode="sparse"),
        "bge_m3_hybrid": lambda q, k: run_bge_m3(bge, q, top_k=k, mode="hybrid"),
    }

    # Build gold sets from chunk_id_map
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}

    for qi, bq in enumerate(BENCHMARK_QUERIES):
        # Resolve gold chunk IDs
        gold_ids = set()
        for doc_id in bq.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        for method_name, method_fn in methods.items():
            result = method_fn(bq["query"], top_k)
            r5 = recall_at_k(result.chunk_ids, gold_ids, 5)
            r10 = recall_at_k(result.chunk_ids, gold_ids, 10)
            m = mrr(result.chunk_ids, gold_ids)
            n = ndcg_at_k(result.chunk_ids, gold_ids, 10)
            all_results[method_name].append({
                "query_id": bq["id"],
                "pattern": bq["pattern"],
                "recall_5": r5,
                "recall_10": r10,
                "mrr": m,
                "ndcg_10": n,
                "latency_ms": result.latency_ms,
                "n_retrieved": len(result.chunk_ids),
                "n_gold": len(gold_ids),
            })

        if (qi + 1) % 5 == 0:
            print(f"  Processed {qi + 1}/{len(BENCHMARK_QUERIES)} queries")

    # Aggregate metrics
    summary: dict[str, dict[str, Any]] = {}
    for method_name, results in all_results.items():
        valid_r5 = [r["recall_5"] for r in results if not np.isnan(r["recall_5"])]
        valid_r10 = [r["recall_10"] for r in results if not np.isnan(r["recall_10"])]
        valid_n = [r["ndcg_10"] for r in results if not np.isnan(r["ndcg_10"])]
        lats = [r["latency_ms"] for r in results]

        summary[method_name] = {
            "recall_5": round(np.mean(valid_r5), 4) if valid_r5 else float("nan"),
            "recall_10": round(np.mean(valid_r10), 4) if valid_r10 else float("nan"),
            "mrr": round(np.mean([r["mrr"] for r in results]), 4),
            "ndcg_10": round(np.mean(valid_n), 4) if valid_n else float("nan"),
            "latency_ms_mean": round(np.mean(lats), 1),
            "latency_ms_p50": round(np.percentile(lats, 50), 1),
            "latency_ms_p95": round(np.percentile(lats, 95), 1),
            "query_count": len(results),
        }

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Method':<25} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8} {'Lat(ms)':>10} {'P50':>8} {'P95':>8}"
    print(header)
    print("-" * len(header))

    baseline_data = summary.get("baseline_hybrid", {})
    for method_name in methods:
        s = summary[method_name]
        r5 = f"{s['recall_5']:.3f}" if not np.isnan(s['recall_5']) else "  N/A"
        r10 = f"{s['recall_10']:.3f}" if not np.isnan(s['recall_10']) else "  N/A"
        mrr_val = f"{s['mrr']:.3f}"
        ndcg = f"{s['ndcg_10']:.3f}" if not np.isnan(s['ndcg_10']) else "  N/A"
        lat = f"{s['latency_ms_mean']:.0f}"
        p50 = f"{s['latency_ms_p50']:.0f}"
        p95 = f"{s['latency_ms_p95']:.0f}"

        # Mark winner
        marker = ""
        if method_name == "baseline_hybrid":
            marker = " (baseline)"
        elif method_name == "bge_m3_hybrid" and baseline_data:
            if s.get("recall_10", 0) > baseline_data.get("recall_10", 0):
                marker = " *"
        print(f"{method_name:<25} {r5:>6} {r10:>6} {mrr_val:>6} {ndcg:>8} {lat:>10} {p50:>8} {p95:>8}{marker}")

    # Per-pattern breakdown
    print("\n" + "=" * 80)
    print("PER-PATTERN BREAKDOWN (Recall@10)")
    print("=" * 80)
    patterns = sorted(set(bq["pattern"] for bq in BENCHMARK_QUERIES))
    pattern_header = f"{'Pattern':<25}" + "".join(f" {m[:12]:>12}" for m in methods)
    print(pattern_header)
    print("-" * len(pattern_header))
    for pattern in patterns:
        row = f"{pattern:<25}"
        for method_name in methods:
            pattern_results = [
                r for r in all_results[method_name] if r["pattern"] == pattern
            ]
            if pattern_results:
                valid = [r["recall_10"] for r in pattern_results if not np.isnan(r["recall_10"])]
                avg = np.mean(valid) if valid else float("nan")
                row += f" {avg:>12.3f}" if not np.isnan(avg) else f" {'N/A':>12}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # VRAM usage
    vram = bge.get_vram_usage()
    print(f"\nBGE-M3 VRAM: {vram}")

    # Build output
    output = {
        "benchmark": "bge_m3_vs_baseline",
        "query_count": len(BENCHMARK_QUERIES),
        "top_k": top_k,
        "corpus_chunks": total_chunks,
        "corpus_time_s": round(corpus_time, 2),
        "baseline_index_time_s": round(baseline_time, 2),
        "bge_m3_index_time_s": round(bge_time, 2),
        "bge_m3_stats": bge_stats,
        "vram": vram,
        "summary": summary,
        "per_query": all_results,
    }

    # Save report
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "bge_m3_benchmark.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    bh = summary.get("baseline_hybrid", {})
    bm3h = summary.get("bge_m3_hybrid", {})
    if bm3h.get("recall_10", 0) > bh.get("recall_10", 0) * 1.05:
        print("B. Replace current hybrid with BGE-M3 (significant recall improvement)")
    elif bm3h.get("recall_10", 0) > bh.get("recall_10", 0):
        print("C. Use BGE-M3 selectively for specific query patterns")
    elif bm3h.get("recall_10", 0) == bh.get("recall_10", 0):
        print("D. Use BGE-M3 embeddings but retain explicit BM25 + dense fusion")
    else:
        print("A. Keep current BM25 + dense hybrid (baseline outperforms)")

    return output


if __name__ == "__main__":
    run_benchmark()
