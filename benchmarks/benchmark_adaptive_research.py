"""Phase 24 Proper Ablation: Adaptive Research Orchestration.

Simulates multi-iteration retrieval to measure what the adaptive policy
would actually do if controlling the research loop. Unlike the previous
flawed benchmark, this measures real iteration-by-iteration behavior.

Usage:
    python benchmarks/benchmark_adaptive_research.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, get_settings
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


def _merge_evidence(existing: list[EvidenceRef], new: list[EvidenceRef]) -> tuple[list[EvidenceRef], int]:
    by_id = {ref.chunk_id: ref for ref in existing}
    new_count = 0
    for ref in new:
        prior = by_id.get(ref.chunk_id)
        if prior is None:
            new_count += 1
            by_id[ref.chunk_id] = ref
        elif ref.score > prior.score:
            by_id[ref.chunk_id] = ref
    merged = sorted(by_id.values(), key=lambda r: r.score, reverse=True)
    return merged, new_count


def simulate_multi_iteration(
    query_text: str,
    plan: QueryPlan,
    pattern_value: str,
    retriever: HybridRetriever,
    router: RetrievalPolicyRouter,
    settings: Settings,
    gold_ids: set,
    max_iterations: int = 3,
) -> dict:
    """Simulate the retrieval loop with adaptive policy decisions."""
    all_evidence: list[EvidenceRef] = []
    gain_history: list[float] = []
    iteration_trace: list[dict] = []
    pending = list(plan.search_variants) if plan.is_planned else [query_text]
    issued = set()
    contradictions_detected = 0

    for iteration in range(1, max_iterations + 1):
        if not pending:
            break

        subquery = pending.pop(0)
        if subquery.lower() in issued:
            if not pending:
                break
            subquery = pending.pop(0)
        issued.add(subquery.lower())

        t0 = time.time()
        if plan.is_planned and plan.evidence_needs:
            single_plan = QueryPlan(
                original_query=subquery,
                pattern=pattern_value,
                evidence_needs=plan.evidence_needs[:2] if len(plan.evidence_needs) > 2 else plan.evidence_needs,
                search_variants=[subquery],
                is_planned=False,
            )
            refs = retriever.search(subquery, top_k=settings.orchestration_retrieval_top_k)
        else:
            refs = retriever.search(subquery, top_k=settings.orchestration_retrieval_top_k)

        elapsed = time.time() - t0

        merged, new_count = _merge_evidence(all_evidence, refs)
        prev_total = len(all_evidence)
        gain = (new_count / prev_total) if prev_total else (1.0 if new_count else 0.0)
        gain_history.append(round(gain, 4))
        all_evidence = merged

        from app.verification.deterministic import TextContradictionDetector
        detector = TextContradictionDetector()
        contradictions = detector.detect_contradictions(all_evidence)
        contradictions_detected = len(contradictions)

        need_coverage = {}
        if plan.is_planned and plan.evidence_needs:
            for need in plan.evidence_needs:
                need_lower = need.topic.lower()
                entities = [e.lower() for e in need.entities]
                covered = any(
                    need_lower in r.text.lower() or
                    all(e in r.text.lower() for e in entities if len(e) > 2)
                    for r in all_evidence
                )
                need_coverage[need.id] = 1.0 if covered else 0.0
        else:
            need_coverage = {"_unplanned": 1.0}

        retrieved_ids = [str(r.chunk_id) for r in all_evidence]
        r5 = len(set(retrieved_ids[:5]) & gold_ids) / len(gold_ids) if gold_ids else 0
        r10 = len(set(retrieved_ids[:10]) & gold_ids) / len(gold_ids) if gold_ids else 0
        avg_coverage = sum(need_coverage.values()) / len(need_coverage) if need_coverage else 0

        avg_top3 = 0
        if all_evidence:
            top = sorted(all_evidence, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

        iteration_trace.append({
            "iteration": iteration,
            "subquery": subquery[:100],
            "new_evidence": new_count,
            "total_evidence": len(all_evidence),
            "gain": round(gain, 4),
            "recall_5": round(r5, 4),
            "recall_10": round(r10, 4),
            "avg_need_coverage": round(avg_coverage, 4),
            "avg_top3_score": round(avg_top3, 4),
            "source_diversity": len({r.document_id for r in all_evidence}),
            "contradictions": contradictions_detected,
            "latency_ms": round(elapsed * 1000, 1),
        })

    # Final assessment
    sufficiency = ResearchSufficiency.from_state(
        all_evidence, need_coverage, contradictions_detected, pattern_value
    )
    level = sufficiency.assess()

    gate = SynthesisGate()
    gate_result = gate.check(all_evidence, need_coverage)

    gain_calc = MarginalGainCalculator(threshold=settings.stopping_evidence_gain_threshold)
    gain_result = gain_calc.evaluate(gain_history) if gain_history else None

    policy = PatternSpecificPolicies.get_policy(pattern_value)

    return {
        "iterations_used": len(iteration_trace),
        "total_evidence": len(all_evidence),
        "sufficiency_level": level.value,
        "synthesis_gate_passes": gate_result.should_synthesize,
        "synthesis_gate_reason": gate_result.reason,
        "gain_stop": gain_result.should_stop if gain_result else False,
        "gain_reason": gain_result.reason if gain_result else "no history",
        "pattern_min_iterations": policy.min_iterations,
        "pattern_max_iterations": policy.max_iterations,
        "contradictions_detected": contradictions_detected,
        "trace": iteration_trace,
        "final_r5": iteration_trace[-1]["recall_5"] if iteration_trace else 0,
        "final_r10": iteration_trace[-1]["recall_10"] if iteration_trace else 0,
        "final_coverage": iteration_trace[-1]["avg_need_coverage"] if iteration_trace else 0,
        "final_top3": iteration_trace[-1]["avg_top3_score"] if iteration_trace else 0,
        "final_diversity": iteration_trace[-1]["source_diversity"] if iteration_trace else 0,
        "total_latency_ms": sum(t["latency_ms"] for t in iteration_trace),
    }


def run_ablation():
    print("=" * 70)
    print("Phase 24 Proper Ablation: Adaptive Research Orchestration")
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

    max_iter = settings.orchestration_max_iterations

    print(f"\n[2/3] Simulating {max_iter}-iteration retrieval for each query...")
    results = []
    total_latency = 0

    for q in queries:
        query_text = q["query"]
        eval_class = q.get("class", "unknown")
        gold_ids = set()
        for doc_id in q.get("supporting_docs", []):
            if doc_id in chunk_id_map:
                gold_ids.update(chunk_id_map[doc_id])

        pattern = router.classify_question(query_text)
        plan = planner.plan(query_text, pattern)
        pattern_value = pattern.value

        sim = simulate_multi_iteration(
            query_text, plan, pattern_value, retriever, router, settings,
            gold_ids, max_iterations=max_iter,
        )
        total_latency += sim["total_latency_ms"]

        results.append({
            "id": q["id"],
            "query": query_text,
            "eval_class": eval_class,
            "pattern": pattern_value,
            **sim,
        })

        print(f"  [{q['id']}] {eval_class:<25} iters={sim['iterations_used']} "
              f"ev={sim['total_evidence']} R@5={sim['final_r5']:.2f} "
              f"R@10={sim['final_r10']:.2f} cov={sim['final_coverage']:.2f} "
              f"lvl={sim['sufficiency_level']:<12} "
              f"gate={'PASS' if sim['synthesis_gate_passes'] else 'BLOCK'} "
              f"{sim['total_latency_ms']:.0f}ms")

    print(f"\n[3/3] Aggregating results...")

    total = len(results)
    avg_latency = total_latency / total
    avg_iters = mean(r["iterations_used"] for r in results)
    avg_evidence = mean(r["total_evidence"] for r in results)
    avg_r5 = mean(r["final_r5"] for r in results)
    avg_r10 = mean(r["final_r10"] for r in results)
    avg_cov = mean(r["final_coverage"] for r in results)
    avg_top3 = mean(r["final_top3"] for r in results)
    avg_diversity = mean(r["final_diversity"] for r in results)

    # Decision distribution
    decisions = {}
    for r in results:
        lvl = r["sufficiency_level"]
        decisions[lvl] = decisions.get(lvl, 0) + 1

    # Gate analysis
    gate_passes = sum(1 for r in results if r["synthesis_gate_passes"])
    gate_blocks = total - gate_passes

    # Gain stop analysis
    gain_stops = sum(1 for r in results if r["gain_stop"])

    # Iteration distribution
    iter_dist = {}
    for r in results:
        iters = r["iterations_used"]
        iter_dist[iters] = iter_dist.get(iters, 0) + 1

    # Per-class breakdown
    classes = sorted(set(r["eval_class"] for r in results))

    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS ({max_iter}-iteration simulation)")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Avg iterations:     {avg_iters:.2f}")
    print(f"  Avg evidence:       {avg_evidence:.1f}")
    print(f"  Avg Recall@5:       {avg_r5:.4f}")
    print(f"  Avg Recall@10:      {avg_r10:.4f}")
    print(f"  Avg need coverage:  {avg_cov:.4f}")
    print(f"  Avg top-3 score:    {avg_top3:.4f}")
    print(f"  Avg source diversity: {avg_diversity:.1f}")
    print(f"  Avg latency:        {avg_latency:.0f}ms")

    print(f"\nSufficiency Distribution:")
    for lvl, count in sorted(decisions.items()):
        print(f"  {lvl:<20} {count:>3} ({count/total:.1%})")

    print(f"\nSynthesis Gate:")
    print(f"  Pass: {gate_passes} ({gate_passes/total:.1%})")
    print(f"  Block: {gate_blocks} ({gate_blocks/total:.1%})")

    print(f"\nGain Stop:")
    print(f"  Stopped: {gain_stops} ({gain_stops/total:.1%})")

    print(f"\nIteration Distribution:")
    for iters, count in sorted(iter_dist.items()):
        print(f"  {iters} iterations: {count} queries ({count/total:.1%})")

    print(f"\nPer-Class Breakdown:")
    print(f"  {'Class':<25} {'N':>3} {'Iters':>5} {'R@5':>6} {'R@10':>6} {'Cov':>6} {'S3':>6} {'Lvl':<15} {'Gate':>6}")
    print(f"  {'-'*82}")
    for cls in classes:
        cls_results = [r for r in results if r["eval_class"] == cls]
        n = len(cls_results)
        iters = mean(r["iterations_used"] for r in cls_results)
        r5 = mean(r["final_r5"] for r in cls_results)
        r10 = mean(r["final_r10"] for r in cls_results)
        cov = mean(r["final_coverage"] for r in cls_results)
        s3 = mean(r["final_top3"] for r in cls_results)
        # Most common level
        lvl_counts = {}
        for r in cls_results:
            lvl_counts[r["sufficiency_level"]] = lvl_counts.get(r["sufficiency_level"], 0) + 1
        top_lvl = max(lvl_counts, key=lvl_counts.get)
        gate_p = sum(1 for r in cls_results if r["synthesis_gate_passes"])
        print(f"  {cls:<25} {n:>3} {iters:>5.1f} {r5:>6.3f} {r10:>6.3f} {cov:>6.3f} {s3:>6.3f} {top_lvl:<15} {gate_p:>3}/{n}")

    # Marginal gain analysis
    print(f"\nMarginal Gain Analysis:")
    for iter_num in range(1, max_iter):
        gains_at_iter = []
        for r in results:
            for t in r["trace"]:
                if t["iteration"] == iter_num:
                    gains_at_iter.append(t["gain"])
        if gains_at_iter:
            print(f"  Iteration {iter_num}: avg_gain={mean(gains_at_iter):.4f}, "
                  f"min={min(gains_at_iter):.4f}, max={max(gains_at_iter):.4f}, "
                  f"n={len(gains_at_iter)}")

    # Build report
    report = {
        "ablation": "phase24_proper_ablation",
        "max_iterations": max_iter,
        "summary": {
            "total_queries": total,
            "avg_iterations": avg_iters,
            "avg_evidence": avg_evidence,
            "avg_recall_5": avg_r5,
            "avg_recall_10": avg_r10,
            "avg_need_coverage": avg_cov,
            "avg_top3_score": avg_top3,
            "avg_source_diversity": avg_diversity,
            "avg_latency_ms": avg_latency,
            "sufficiency_distribution": decisions,
            "gate_passes": gate_passes,
            "gate_blocks": gate_blocks,
            "gain_stops": gain_stops,
            "iteration_distribution": iter_dist,
        },
        "per_class": {},
        "results": results,
    }
    for cls in classes:
        cls_results = [r for r in results if r["eval_class"] == cls]
        report["per_class"][cls] = {
            "count": len(cls_results),
            "avg_iterations": mean(r["iterations_used"] for r in cls_results),
            "avg_recall_5": mean(r["final_r5"] for r in cls_results),
            "avg_recall_10": mean(r["final_r10"] for r in cls_results),
            "avg_need_coverage": mean(r["final_coverage"] for r in cls_results),
            "avg_top3_score": mean(r["final_top3"] for r in cls_results),
        }

    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase24_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase24_ablation.json")

    return report


if __name__ == "__main__":
    run_ablation()
