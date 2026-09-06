"""Phase 24 Ablation: Adaptive Research Orchestration.

Compares baseline (fixed-budget) vs adaptive research policies on the
38-query evaluation set. Measures latency, evidence quality, and
sufficiency accuracy.

Usage:
    python benchmarks/benchmark_adaptive_research.py
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
from app.retrieval.evidence_selector import EvidenceSelector
from app.orchestration.adaptive_research import (
    AdaptiveResearchPolicy,
    ResearchSufficiency,
    SufficiencyLevel,
    MarginalGainCalculator,
    PatternSpecificPolicies,
    SynthesisGate,
)


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


def run_ablation():
    print("=" * 70)
    print("Phase 24 Ablation: Adaptive Research Orchestration")
    print("=" * 70)

    settings = get_settings()
    router = RetrievalPolicyRouter()
    planner = EvidenceNeedPlanner()

    print("\n[1/4] Building evaluation corpus...")
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

    # ------------------------------------------------------------------
    # Baseline: existing fixed-budget behavior (disabled adaptive)
    # ------------------------------------------------------------------
    print("\n[2/4] Running baseline (fixed-budget)...")
    adaptive_policy_disabled = AdaptiveResearchPolicy(enabled=False)
    baseline_results = _run_all(
        queries, router, planner, retriever, evidence_selector,
        chunk_id_map, adaptive_policy_disabled, settings,
    )

    # ------------------------------------------------------------------
    # Adaptive: new adaptive policy (enabled)
    # ------------------------------------------------------------------
    print("\n[3/4] Running adaptive policy...")
    adaptive_policy_enabled = AdaptiveResearchPolicy(
        enabled=True,
        marginal_gain_threshold=settings.stopping_evidence_gain_threshold,
    )
    adaptive_results = _run_all(
        queries, router, planner, retriever, evidence_selector,
        chunk_id_map, adaptive_policy_enabled, settings,
    )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("\n[4/4] Comparing results...")
    report = _compare(baseline_results, adaptive_results, len(queries))

    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase24_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase24_ablation.json")

    return report


def _run_all(
    queries: list[dict],
    router: RetrievalPolicyRouter,
    planner: EvidenceNeedPlanner,
    retriever: HybridRetriever,
    evidence_selector: EvidenceSelector,
    chunk_id_map: dict[str, list[str]],
    adaptive_policy: AdaptiveResearchPolicy,
    settings: Settings,
) -> list[dict]:
    results = []
    total_latency = 0

    for q in queries:
        query_text = q["query"]
        eval_class = q.get("class", "unknown")
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        t0 = time.time()

        pattern = router.classify_question(query_text)
        plan = planner.plan(query_text, pattern)
        pattern_value = pattern.value

        if plan.is_planned:
            multi_retriever = MultiQueryRetriever(router, retriever)
            import asyncio
            result = asyncio.run(multi_retriever.retrieve(plan))
            selected = result.selected
            need_coverage = result.need_coverage
        else:
            refs = retriever.search(query_text, top_k=20)
            selected = refs[:10]
            need_coverage = {"_unplanned": 1.0}

        elapsed = time.time() - t0
        total_latency += elapsed

        # Simulate adaptive policy decision
        gain_history = [0.5]  # simulate first iteration gain
        decision = adaptive_policy.should_continue_retrieval(
            evidence=selected,
            need_coverage=need_coverage,
            gain_history=gain_history,
            iteration=1,
            max_iterations=settings.orchestration_max_iterations,
            pattern=pattern_value,
        )

        # Compute sufficiency
        sufficiency = ResearchSufficiency.from_state(
            selected, need_coverage, 0, pattern_value
        )
        level = sufficiency.assess()

        retrieved_ids = [str(r.chunk_id) for r in selected]
        r5 = len(set(retrieved_ids[:5]) & gold_ids) / len(gold_ids) if gold_ids else float("nan")
        r10 = len(set(retrieved_ids[:10]) & gold_ids) / len(gold_ids) if gold_ids else float("nan")
        avg_coverage = sum(need_coverage.values()) / len(need_coverage) if need_coverage else 0

        avg_top3 = 0
        if selected:
            top = sorted(selected, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

        results.append({
            "id": q["id"],
            "eval_class": eval_class,
            "pattern": pattern_value,
            "decision_action": decision.action,
            "decision_reason": decision.reason,
            "sufficiency_level": level.value,
            "retrieved_count": len(selected),
            "gold_count": len(gold_ids),
            "recall_5": round(r5, 4) if not (r5 != r5) else None,
            "recall_10": round(r10, 4) if not (r10 != r10) else None,
            "avg_need_coverage": round(avg_coverage, 4),
            "avg_top3_score": round(avg_top3, 4),
            "latency_ms": round(elapsed * 1000, 1),
        })

        print(f"  [{q['id']}] {eval_class:<25} {decision.action:<20} "
              f"{level.value:<12} R@10={r5:.2f} {elapsed*1000:.0f}ms")

    return results


def _compare(baseline: list[dict], adaptive: list[dict], total: int) -> dict:
    b_latency = sum(r["latency_ms"] for r in baseline) / total
    a_latency = sum(r["latency_ms"] for r in adaptive) / total

    b_r5 = [r["recall_5"] for r in baseline if r["recall_5"] is not None]
    a_r5 = [r["recall_5"] for r in adaptive if r["recall_5"] is not None]
    b_r10 = [r["recall_10"] for r in baseline if r["recall_10"] is not None]
    a_r10 = [r["recall_10"] for r in adaptive if r["recall_10"] is not None]

    b_cov = sum(r["avg_need_coverage"] for r in baseline) / total
    a_cov = sum(r["avg_need_coverage"] for r in adaptive) / total

    b_s3 = sum(r["avg_top3_score"] for r in baseline) / total
    a_s3 = sum(r["avg_top3_score"] for r in adaptive) / total

    b_retrieved = sum(r["retrieved_count"] for r in baseline) / total
    a_retrieved = sum(r["retrieved_count"] for r in adaptive) / total

    decisions = {}
    for r in adaptive:
        action = r["decision_action"]
        decisions[action] = decisions.get(action, 0) + 1

    levels = {}
    for r in adaptive:
        level = r["sufficiency_level"]
        levels[level] = levels.get(level, 0) + 1

    print(f"\n{'='*70}")
    print(f"ABLATION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Baseline':>12} {'Adaptive':>12} {'Delta':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Avg Latency (ms)':<30} {b_latency:>12.1f} {a_latency:>12.1f} {a_latency-b_latency:>+12.1f}")
    print(f"  {'Avg Recall@5':<30} {sum(b_r5)/len(b_r5):>12.4f} {sum(a_r5)/len(a_r5):>12.4f} {sum(a_r5)/len(a_r5)-sum(b_r5)/len(b_r5):>+12.4f}")
    print(f"  {'Avg Recall@10':<30} {sum(b_r10)/len(b_r10):>12.4f} {sum(a_r10)/len(a_r10):>12.4f} {sum(a_r10)/len(a_r10)-sum(b_r10)/len(b_r10):>+12.4f}")
    print(f"  {'Avg Need Coverage':<30} {b_cov:>12.4f} {a_cov:>12.4f} {a_cov-b_cov:>+12.4f}")
    print(f"  {'Avg Top-3 Score':<30} {b_s3:>12.4f} {a_s3:>12.4f} {a_s3-b_s3:>+12.4f}")
    print(f"  {'Avg Retrieved Chunks':<30} {b_retrieved:>12.1f} {a_retrieved:>12.1f} {a_retrieved-b_retrieved:>+12.1f}")
    print(f"\n  Adaptive decisions: {decisions}")
    print(f"  Sufficiency levels: {levels}")

    return {
        "ablation": "phase24_adaptive_research",
        "summary": {
            "total_queries": total,
            "baseline": {
                "avg_latency_ms": b_latency,
                "avg_recall_5": sum(b_r5) / len(b_r5) if b_r5 else 0,
                "avg_recall_10": sum(b_r10) / len(b_r10) if b_r10 else 0,
                "avg_need_coverage": b_cov,
                "avg_top3_score": b_s3,
                "avg_retrieved_chunks": b_retrieved,
            },
            "adaptive": {
                "avg_latency_ms": a_latency,
                "avg_recall_5": sum(a_r5) / len(a_r5) if a_r5 else 0,
                "avg_recall_10": sum(a_r10) / len(a_r10) if a_r10 else 0,
                "avg_need_coverage": a_cov,
                "avg_top3_score": a_s3,
                "avg_retrieved_chunks": a_retrieved,
                "decisions": decisions,
                "sufficiency_levels": levels,
            },
            "delta": {
                "latency_ms": a_latency - b_latency,
                "recall_5": (sum(a_r5) / len(a_r5) if a_r5 else 0) - (sum(b_r5) / len(b_r5) if b_r5 else 0),
                "recall_10": (sum(a_r10) / len(a_r10) if a_r10 else 0) - (sum(b_r10) / len(b_r10) if b_r10 else 0),
                "need_coverage": a_cov - b_cov,
                "top3_score": a_s3 - b_s3,
                "retrieved_chunks": a_retrieved - b_retrieved,
            },
        },
        "baseline": baseline,
        "adaptive": adaptive,
    }


if __name__ == "__main__":
    run_ablation()
