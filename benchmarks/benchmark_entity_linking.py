"""Phase 20 Benchmark: Entity Linking Impact Measurement (v3).

Uses proper gold chunk ID evaluation matching the real benchmark approach.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmarks.benchmark_fusion import build_benchmark_store
from app.retrieval.entity_linking import EntityLinker, build_entity_index
from app.retrieval.router import get_retrieval_policy_router
from app.retrieval.hybrid import HybridRetriever


def _load_corpus_chunks() -> list[dict]:
    corpus_dir = _REPO / "benchmarks" / "eval_data" / "corpus_v1"
    chunks = []
    for doc_path in sorted(corpus_dir.glob("*.md")):
        doc_id = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        chunk_idx = 0
        for para in paragraphs:
            if len(current_chunk) + len(para) > 512 and current_chunk:
                chunks.append({"text": current_chunk, "chunk_id": f"{doc_id}_c{chunk_idx}", "document_id": doc_id})
                chunk_idx += 1
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if current_chunk:
            chunks.append({"text": current_chunk, "chunk_id": f"{doc_id}_c{chunk_idx}", "document_id": doc_id})
    return chunks


def _load_eval_plan() -> list[dict]:
    plan_path = _REPO / "benchmarks" / "eval_data" / "eval_plan_v1.json"
    with open(plan_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    import numpy as np
    if not gold:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 1) for i, cid in enumerate(retrieved[:k], 1) if cid in gold)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def run_benchmark(entity_linking_enabled: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"Benchmark: entity_linking={'ON' if entity_linking_enabled else 'OFF'}")
    print(f"{'='*60}")

    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    entity_linker = None
    if entity_linking_enabled:
        corpus_chunks = _load_corpus_chunks()
        entity_linker = EntityLinker(max_expansions=3, min_expansion_score=0.1)
        entity_linker.build_from_chunks(corpus_chunks)
        print(f"  Entities: {len(entity_linker.index.entities)}, Relationships: {len(entity_linker.index.relationships)}")

    router = get_retrieval_policy_router()
    eval_plan = _load_eval_plan()

    results = []
    latencies = []
    pattern_results: dict[str, list[float]] = {}

    for item in eval_plan:
        query = item["query"]
        eval_class = item.get("class", "unknown")
        supporting_docs = item.get("supporting_docs", [])

        gold_ids = set()
        for doc_id in supporting_docs:
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        pattern = router.classify_question(query)
        pattern_value = pattern.value if hasattr(pattern, 'value') else pattern

        t0 = time.perf_counter()
        try:
            import asyncio
            if pattern_value in ("conflict", "complex_research", "multi_hop"):
                # Expand with entity linking if enabled
                if entity_linker and entity_linker.index.entities:
                    plan = router._planner.plan(query, pattern_value)
                    if plan.is_planned:
                        entity_linker.expand_evidence_needs(plan.evidence_needs, pattern_value)
                        from app.retrieval.multi_query import MultiQueryRetriever
                        if router._multi_query_retriever is None:
                            router._multi_query_retriever = MultiQueryRetriever(
                                router=router, retriever=retriever, top_k=10,
                            )
                        result = asyncio.run(router._multi_query_retriever.retrieve(plan))
                        refs = result.selected
                    else:
                        refs = asyncio.run(router.execute_planned_retrieval(query, pattern, retriever))
                else:
                    refs = asyncio.run(router.execute_planned_retrieval(query, pattern, retriever))
            else:
                refs = retriever.search(query, top_k=10)
        except Exception as e:
            print(f"  Error on {query[:40]}: {e}")
            refs = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = [str(r.chunk_id) for r in refs]
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        m = mrr(retrieved_ids, gold_ids)
        n = ndcg_at_k(retrieved_ids, gold_ids, 10)

        pattern_results.setdefault(eval_class, []).append(r10)
        results.append({"query_id": item.get("id", ""), "pattern": eval_class, "recall_5": r5, "recall_10": r10, "mrr": m, "ndcg_10": n, "latency_ms": elapsed_ms})

    n = max(1, len(results))
    metrics = {
        "recall_at_5": round(sum(r["recall_5"] for r in results) / n, 3),
        "recall_at_10": round(sum(r["recall_10"] for r in results) / n, 3),
        "mrr": round(sum(r["mrr"] for r in results) / n, 3),
        "ndcg_at_10": round(sum(r["ndcg_10"] for r in results) / n, 3),
        "pattern_results": {k: round(sum(v) / len(v), 3) for k, v in pattern_results.items()},
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
        "n_queries": n,
    }

    print(f"\nResults:")
    print(f"  Recall@5:  {metrics['recall_at_5']}")
    print(f"  Recall@10: {metrics['recall_at_10']}")
    print(f"  MRR:       {metrics['mrr']}")
    print(f"  nDCG@10:   {metrics['ndcg_at_10']}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']}ms")
    print(f"\nPer-pattern Recall@10:")
    for p, s in sorted(metrics["pattern_results"].items()):
        print(f"  {p}: {s}")

    return metrics


if __name__ == "__main__":
    metrics_off = run_benchmark(entity_linking_enabled=False)
    metrics_on = run_benchmark(entity_linking_enabled=True)

    print(f"\n{'='*60}")
    print(f"COMPARISON: Entity Linking Impact")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'No EL':>10} {'With EL':>10} {'Delta':>10}")
    print(f"{'-'*50}")
    for metric in ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]:
        v_off = metrics_off[metric]
        v_on = metrics_on[metric]
        print(f"{metric:<20} {v_off:>10.3f} {v_on:>10.3f} {v_on - v_off:>+10.3f}")

    print(f"\n{'Pattern':<25} {'No EL':>10} {'With EL':>10} {'Delta':>10}")
    print(f"{'-'*55}")
    all_patterns = set(metrics_off["pattern_results"]) | set(metrics_on["pattern_results"])
    for p in sorted(all_patterns):
        v_off = metrics_off["pattern_results"].get(p, 0)
        v_on = metrics_on["pattern_results"].get(p, 0)
        print(f"{p:<25} {v_off:>10.3f} {v_on:>10.3f} {v_on - v_off:>+10.3f}")

    output = {"baseline": metrics_off, "entity_linking": metrics_on}
    output_path = _REPO / "data" / "benchmark_reports" / "phase20_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")
