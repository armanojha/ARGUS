"""Phase 24 Baseline: Research Orchestration Diagnostics.

Traces per-query execution through the full ARGUS pipeline, capturing
planner activation, retrieval calls, subqueries, chunks, coverage,
verification, recovery, synthesis, and latency. Establishes the baseline
for adaptive research orchestration.

Usage:
    python benchmarks/baseline_orchestration.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evidence.models import EvidenceRef
from app.evidence.store import EvidenceStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import RetrievalPolicyRouter
from app.retrieval.planner import EvidenceNeedPlanner, QueryPlan
from app.retrieval.policy import QuestionPattern
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.recovery import TargetedRecovery
from app.retrieval.evidence_selector import EvidenceSelector


def load_eval_queries() -> list[dict]:
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        plan = json.load(f)
    return plan.get("queries", [])


def build_corpus() -> tuple[EvidenceStore, dict[str, list[str]]]:
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    return store, chunk_id_map


def compute_need_coverage(selected: list[EvidenceRef], plan: QueryPlan) -> dict[str, float]:
    if not plan.is_planned or not plan.evidence_needs:
        return {"_unplanned": 1.0}

    coverage = {}
    for need in plan.evidence_needs:
        need_lower = need.topic.lower()
        entities = [e.lower() for e in need.entities]

        covered = False
        for ref in selected:
            text_lower = ref.text.lower()
            if need_lower in text_lower or all(e in text_lower for e in entities if len(e) > 2):
                covered = True
                break

        coverage[need.id] = 1.0 if covered else 0.0

    return coverage


def run_diagnostics():
    print("=" * 70)
    print("Phase 24 Baseline: Research Orchestration Diagnostics")
    print("=" * 70)

    settings = get_settings()
    router = RetrievalPolicyRouter()
    planner = EvidenceNeedPlanner()

    print("\n[1/3] Building evaluation corpus...")
    t0 = time.time()
    store, chunk_id_map = build_corpus()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.time()-t0:.1f}s)")
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    queries = load_eval_queries()
    print(f"  Queries: {len(queries)}")

    evidence_selector = EvidenceSelector(
        similarity_threshold=settings.evidence_coverage_similarity_threshold,
        max_chunks=settings.evidence_selection_max_chunks,
        max_tokens=settings.evidence_selection_max_tokens,
        min_chunks=settings.evidence_selection_min_chunks,
        min_sources=settings.evidence_selection_min_sources,
    )

    print("\n[2/3] Running diagnostics for each query...")
    diagnostics = []
    total_latency = 0

    for q in queries:
        query_text = q["query"]
        eval_class = q.get("class", "unknown")
        expected_pattern = QuestionPattern.from_eval_class(eval_class)
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        t0 = time.time()

        pattern = router.classify_question(query_text)
        plan = planner.plan(query_text, pattern)

        trace = {
            "planner_activated": plan.is_planned,
            "need_count": len(plan.evidence_needs) if plan.is_planned else 0,
            "subqueries_planned": plan.search_variants if plan.is_planned else [query_text],
        }

        if plan.is_planned:
            multi_retriever = MultiQueryRetriever(router, retriever)
            import asyncio
            result = asyncio.run(multi_retriever.retrieve(plan))
            selected = result.selected
            need_coverage = result.need_coverage
            total_retrieval_calls = result.total_retrieval_calls
        else:
            refs = retriever.search(query_text, top_k=20)
            selected = refs[:10]
            need_coverage = {"_unplanned": 1.0}
            total_retrieval_calls = 1

        elapsed = time.time() - t0
        total_latency += elapsed

        selected_for_synth = evidence_selector.select(selected)

        retrieved_ids = [str(r.chunk_id) for r in selected]
        r5 = len(set(retrieved_ids[:5]) & gold_ids) / len(gold_ids) if gold_ids else float("nan")
        r10 = len(set(retrieved_ids[:10]) & gold_ids) / len(gold_ids) if gold_ids else float("nan")
        avg_coverage = sum(need_coverage.values()) / len(need_coverage) if need_coverage else 0

        avg_top3_score = sum(r.score for r in selected[:3]) / min(3, len(selected)) if selected else 0

        source_diversity = len({r.document_id for r in selected})

        from app.verification.deterministic import TextContradictionDetector
        detector = TextContradictionDetector()
        text_contradictions = detector.detect_contradictions(selected)

        missed_gold = gold_ids - set(retrieved_ids)

        diag = {
            "id": q["id"],
            "query": query_text,
            "eval_class": eval_class,
            "expected_pattern": expected_pattern.value,
            "classified_pattern": pattern.value,
            "pattern_match": pattern == expected_pattern,
            "trace": trace,
            "retrieved_count": len(selected),
            "selected_for_synth_count": len(selected_for_synth),
            "gold_count": len(gold_ids),
            "retrieved_ids": retrieved_ids,
            "gold_ids": list(gold_ids),
            "missed_gold": list(missed_gold),
            "recall_5": round(r5, 4) if not (r5 != r5) else None,
            "recall_10": round(r10, 4) if not (r10 != r10) else None,
            "need_coverage": need_coverage,
            "avg_need_coverage": round(avg_coverage, 4),
            "avg_top3_score": round(avg_top3_score, 4),
            "source_diversity": source_diversity,
            "text_contradictions": [
                {
                    "type": c.contradiction_type,
                    "chunk_a": c.chunk_a_id,
                    "chunk_b": c.chunk_b_id,
                    "description": c.description,
                    "severity": c.severity,
                }
                for c in text_contradictions
            ],
            "contradiction_count": len(text_contradictions),
            "latency_ms": round(elapsed * 1000, 1),
            "retrieval_calls": total_retrieval_calls,
        }
        diagnostics.append(diag)

        status = "OK" if diag["pattern_match"] else "MISMATCH"
        print(f"  [{q['id']}] {eval_class:<25} {pattern.value:<20} {status:<8} "
              f"R@10={r5:.2f} cov={avg_coverage:.2f} "
              f"score3={avg_top3_score:.3f} src={source_diversity} "
              f"contr={len(text_contradictions)} {elapsed*1000:.0f}ms")

    print(f"\n[3/3] Aggregating results...")

    total = len(diagnostics)
    pattern_matches = sum(1 for d in diagnostics if d["pattern_match"])
    avg_r10 = sum(d["recall_10"] for d in diagnostics if d["recall_10"] is not None) / total
    avg_coverage = sum(d["avg_need_coverage"] for d in diagnostics) / total
    avg_top3 = sum(d["avg_top3_score"] for d in diagnostics) / total
    avg_diversity = sum(d["source_diversity"] for d in diagnostics) / total
    queries_with_contradictions = sum(1 for d in diagnostics if d["contradiction_count"] > 0)
    avg_latency = total_latency / total * 1000

    print(f"\n{'='*70}")
    print(f"ORCHESTRATION BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total queries: {total}")
    print(f"  Pattern match: {pattern_matches}/{total} ({pattern_matches/total:.1%})")
    print(f"  Avg Recall@10: {avg_r10:.4f}")
    print(f"  Avg need coverage: {avg_coverage:.4f}")
    print(f"  Avg top-3 score: {avg_top3:.4f}")
    print(f"  Avg source diversity: {avg_diversity:.1f}")
    print(f"  Queries with text contradictions: {queries_with_contradictions}")
    print(f"  Avg latency: {avg_latency:.0f}ms")

    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<25} {'Count':>5} {'R@10':>6} {'Cov':>6} {'S3':>6} {'Src':>5} {'Contr':>6}")
    print(f"  {'-'*59}")
    classes = sorted(set(d["eval_class"] for d in diagnostics))
    for cls in classes:
        cls_diags = [d for d in diagnostics if d["eval_class"] == cls]
        cls_r10 = [d["recall_10"] for d in cls_diags if d["recall_10"] is not None]
        cls_cov = [d["avg_need_coverage"] for d in cls_diags]
        cls_s3 = [d["avg_top3_score"] for d in cls_diags]
        cls_src = [d["source_diversity"] for d in cls_diags]
        cls_contr = sum(d["contradiction_count"] for d in cls_diags)
        r10_str = f"{sum(cls_r10)/len(cls_r10):>6.3f}" if cls_r10 else "   N/A"
        cov_str = f"{sum(cls_cov)/len(cls_cov):>6.3f}" if cls_cov else "   N/A"
        s3_str = f"{sum(cls_s3)/len(cls_s3):>6.3f}" if cls_s3 else "   N/A"
        src_str = f"{sum(cls_src)/len(cls_src):>5.1f}" if cls_src else "  N/A"
        print(f"  {cls:<25} {len(cls_diags):>5} {r10_str} {cov_str} {s3_str} {src_str} {cls_contr:>6}")

    report = {
        "baseline": "phase24_orchestration_baseline",
        "summary": {
            "total_queries": total,
            "pattern_match_rate": pattern_matches / total,
            "avg_recall_10": avg_r10,
            "avg_need_coverage": avg_coverage,
            "avg_top3_score": avg_top3,
            "avg_source_diversity": avg_diversity,
            "queries_with_text_contradictions": queries_with_contradictions,
            "avg_latency_ms": avg_latency,
        },
        "diagnostics": diagnostics,
    }
    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase24_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase24_baseline.json")

    return diagnostics


if __name__ == "__main__":
    run_diagnostics()
