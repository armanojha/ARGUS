"""Phase 24.1 Proper Ablation: Adaptive Research Orchestration.

Compares two genuinely different code paths:
  - BASELINE: Always runs max iterations (no adaptive stopping)
  - ADAPTIVE: AdaptiveResearchPolicy decides per-iteration whether to stop

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


def _compute_need_coverage(
    plan: QueryPlan, all_evidence: list[EvidenceRef]
) -> dict[str, float]:
    if not plan.is_planned or not plan.evidence_needs:
        return {"_unplanned": 1.0}
    need_coverage = {}
    for need in plan.evidence_needs:
        need_lower = need.topic.lower()
        entities = [e.lower() for e in need.entities]
        covered = any(
            need_lower in r.text.lower() or
            all(e in r.text.lower() for e in entities if len(e) > 2)
            for r in all_evidence
        )
        need_coverage[need.id] = 1.0 if covered else 0.0
    return need_coverage


def run_baseline(
    query_text: str,
    plan: QueryPlan,
    pattern_value: str,
    retriever: HybridRetriever,
    settings: Settings,
    gold_ids: set,
    max_iterations: int,
) -> dict:
    """Baseline: always iterate to max iterations (no adaptive stopping)."""
    all_evidence: list[EvidenceRef] = []
    gain_history: list[float] = []
    iteration_trace: list[dict] = []
    contradictions_detected = 0

    pending = list(plan.search_variants) if plan.is_planned else [query_text]
    issued = set()

    for iteration in range(1, max_iterations + 1):
        if not pending:
            # Baseline: even with no pending queries, still counts as used iteration
            break

        subquery = pending.pop(0)
        if subquery.lower() in issued:
            if not pending:
                break
            subquery = pending.pop(0)
        issued.add(subquery.lower())

        t0 = time.time()
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

        need_coverage = _compute_need_coverage(plan, all_evidence)

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

    sufficiency = ResearchSufficiency.from_state(
        all_evidence, need_coverage, contradictions_detected, pattern_value
    )
    result = sufficiency.assess()

    gate = SynthesisGate()
    gate_result = gate.check(all_evidence, need_coverage)

    return {
        "iterations_used": len(iteration_trace),
        "total_evidence": len(all_evidence),
        "sufficiency_level": result.level.value,
        "synthesis_gate_passes": gate_result.should_synthesize,
        "synthesis_gate_reason": gate_result.reason,
        "contradictions_detected": contradictions_detected,
        "trace": iteration_trace,
        "final_r5": iteration_trace[-1]["recall_5"] if iteration_trace else 0,
        "final_r10": iteration_trace[-1]["recall_10"] if iteration_trace else 0,
        "final_coverage": iteration_trace[-1]["avg_need_coverage"] if iteration_trace else 0,
        "final_top3": iteration_trace[-1]["avg_top3_score"] if iteration_trace else 0,
        "final_diversity": iteration_trace[-1]["source_diversity"] if iteration_trace else 0,
        "total_latency_ms": sum(t["latency_ms"] for t in iteration_trace),
    }


def run_adaptive(
    query_text: str,
    plan: QueryPlan,
    pattern_value: str,
    retriever: HybridRetriever,
    settings: Settings,
    gold_ids: set,
    max_iterations: int,
) -> dict:
    """Adaptive: policy decides per-iteration whether to continue or stop."""
    policy = AdaptiveResearchPolicy(enabled=True)
    all_evidence: list[EvidenceRef] = []
    gain_history: list[float] = []
    iteration_trace: list[dict] = []
    contradictions_detected = 0
    stopped_early = False
    stop_reason = None

    pending = list(plan.search_variants) if plan.is_planned else [query_text]
    issued = set()

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

        need_coverage = _compute_need_coverage(plan, all_evidence)

        retrieved_ids = [str(r.chunk_id) for r in all_evidence]
        r5 = len(set(retrieved_ids[:5]) & gold_ids) / len(gold_ids) if gold_ids else 0
        r10 = len(set(retrieved_ids[:10]) & gold_ids) / len(gold_ids) if gold_ids else 0
        avg_coverage = sum(need_coverage.values()) / len(need_coverage) if need_coverage else 0

        avg_top3 = 0
        if all_evidence:
            top = sorted(all_evidence, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

        # *** THIS IS THE KEY DIFFERENCE: adaptive policy decides ***
        decision = policy.should_continue_retrieval(
            evidence=all_evidence,
            need_coverage=need_coverage,
            gain_history=gain_history,
            iteration=iteration,
            max_iterations=max_iterations,
            pattern=pattern_value,
            contradictions_detected=contradictions_detected,
            pending_subquestions=pending,
        )

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
            "policy_action": decision.action,
            "policy_reason": decision.reason,
            "policy_sufficiency": decision.sufficiency_level,
        })

        if decision.action in ("synthesize", "investigate"):
            stopped_early = True
            stop_reason = decision.action
            break

    sufficiency = ResearchSufficiency.from_state(
        all_evidence, need_coverage, contradictions_detected, pattern_value
    )
    result = sufficiency.assess()

    gate = SynthesisGate()
    gate_result = gate.check(all_evidence, need_coverage)

    return {
        "iterations_used": len(iteration_trace),
        "total_evidence": len(all_evidence),
        "sufficiency_level": result.level.value,
        "synthesis_gate_passes": gate_result.should_synthesize,
        "synthesis_gate_reason": gate_result.reason,
        "contradictions_detected": contradictions_detected,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
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
    print("Phase 24.1 Proper Ablation: Baseline vs Adaptive Orchestration")
    print("=" * 70)

    settings = get_settings()
    planner = EvidenceNeedPlanner()
    router = RetrievalPolicyRouter()

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

    print(f"\n[2/3] Running {max_iter}-iteration ablation (baseline vs adaptive)...")
    baseline_results = []
    adaptive_results = []

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

        # Baseline: always runs max iterations
        base = run_baseline(
            query_text, plan, pattern_value, retriever, settings,
            gold_ids, max_iterations=max_iter,
        )
        base["id"] = q["id"]
        base["query"] = query_text
        base["eval_class"] = eval_class
        base["pattern"] = pattern_value
        baseline_results.append(base)

        # Adaptive: policy controls stopping
        adapt = run_adaptive(
            query_text, plan, pattern_value, retriever, settings,
            gold_ids, max_iterations=max_iter,
        )
        adapt["id"] = q["id"]
        adapt["query"] = query_text
        adapt["eval_class"] = eval_class
        adapt["pattern"] = pattern_value
        adaptive_results.append(adapt)

        iters_saved = base["iterations_used"] - adapt["iterations_used"]
        print(f"  [{q['id']}] {eval_class:<25} "
              f"base={base['iterations_used']} adapt={adapt['iterations_used']} "
              f"saved={iters_saved} "
              f"R@10 base={base['final_r10']:.2f} adapt={adapt['final_r10']:.2f} "
              f"{'EARLY' if adapt['stopped_early'] else 'FULL'}")

    print(f"\n[3/3] Aggregating results...")

    total = len(queries)

    # Baseline stats
    b_avg_iters = mean(r["iterations_used"] for r in baseline_results)
    b_avg_r10 = mean(r["final_r10"] for r in baseline_results)
    b_avg_cov = mean(r["final_coverage"] for r in baseline_results)
    b_avg_top3 = mean(r["final_top3"] for r in baseline_results)
    b_avg_evidence = mean(r["total_evidence"] for r in baseline_results)
    b_avg_latency = mean(r["total_latency_ms"] for r in baseline_results)

    # Adaptive stats
    a_avg_iters = mean(r["iterations_used"] for r in adaptive_results)
    a_avg_r10 = mean(r["final_r10"] for r in adaptive_results)
    a_avg_cov = mean(r["final_coverage"] for r in adaptive_results)
    a_avg_top3 = mean(r["final_top3"] for r in adaptive_results)
    a_avg_evidence = mean(r["total_evidence"] for r in adaptive_results)
    a_avg_latency = mean(r["total_latency_ms"] for r in adaptive_results)
    a_early_stops = sum(1 for r in adaptive_results if r["stopped_early"])

    # Iteration distribution
    b_iter_dist = {}
    for r in baseline_results:
        iters = r["iterations_used"]
        b_iter_dist[iters] = b_iter_dist.get(iters, 0) + 1

    a_iter_dist = {}
    for r in adaptive_results:
        iters = r["iterations_used"]
        a_iter_dist[iters] = a_iter_dist.get(iters, 0) + 1

    # Sufficiency distribution
    a_suff_dist = {}
    for r in adaptive_results:
        lvl = r["sufficiency_level"]
        a_suff_dist[lvl] = a_suff_dist.get(lvl, 0) + 1

    # Adaptive policy stop reasons
    a_stop_reasons = {}
    for r in adaptive_results:
        if r["stopped_early"]:
            reason = r["stop_reason"]
            a_stop_reasons[reason] = a_stop_reasons.get(reason, 0) + 1

    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS ({max_iter}-iteration max)")
    print(f"{'='*70}")

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Adaptive':>12} {'Delta':>10}")
    print(f"{'-'*64}")
    print(f"{'Avg iterations':<30} {b_avg_iters:>12.2f} {a_avg_iters:>12.2f} {a_avg_iters-b_avg_iters:>+10.2f}")
    print(f"{'Avg evidence':<30} {b_avg_evidence:>12.1f} {a_avg_evidence:>12.1f} {a_avg_evidence-b_avg_evidence:>+10.1f}")
    print(f"{'Avg Recall@10':<30} {b_avg_r10:>12.4f} {a_avg_r10:>12.4f} {a_avg_r10-b_avg_r10:>+10.4f}")
    print(f"{'Avg need coverage':<30} {b_avg_cov:>12.4f} {a_avg_cov:>12.4f} {a_avg_cov-b_avg_cov:>+10.4f}")
    print(f"{'Avg top-3 score':<30} {b_avg_top3:>12.4f} {a_avg_top3:>12.4f} {a_avg_top3-b_avg_top3:>+10.4f}")
    print(f"{'Avg latency (ms)':<30} {b_avg_latency:>12.0f} {a_avg_latency:>12.0f} {a_avg_latency-b_avg_latency:>+10.0f}")

    print(f"\nAdaptive Early Stops: {a_early_stops}/{total} ({a_early_stops/total:.1%})")

    print(f"\nIteration Distribution:")
    print(f"  {'Iters':<8} {'Baseline':>12} {'Adaptive':>12}")
    for iters in range(1, max_iter + 1):
        bc = b_iter_dist.get(iters, 0)
        ac = a_iter_dist.get(iters, 0)
        print(f"  {iters:<8} {bc:>12} {ac:>12}")

    print(f"\nAdaptive Sufficiency Distribution:")
    for lvl, count in sorted(a_suff_dist.items()):
        print(f"  {lvl:<20} {count:>3} ({count/total:.1%})")

    print(f"\nAdaptive Stop Reasons:")
    for reason, count in sorted(a_stop_reasons.items()):
        print(f"  {reason:<20} {count:>3} ({count/total:.1%})")

    # Per-class breakdown
    classes = sorted(set(r["eval_class"] for r in baseline_results))
    print(f"\nPer-Class Breakdown:")
    print(f"  {'Class':<25} {'N':>3} {'B-It':>5} {'A-It':>5} {'B-R@10':>7} {'A-R@10':>7} {'A-Early':>8}")
    print(f"  {'-'*65}")
    for cls in classes:
        b_cls = [r for r in baseline_results if r["eval_class"] == cls]
        a_cls = [r for r in adaptive_results if r["eval_class"] == cls]
        n = len(b_cls)
        b_it = mean(r["iterations_used"] for r in b_cls)
        a_it = mean(r["iterations_used"] for r in a_cls)
        b_r10 = mean(r["final_r10"] for r in b_cls)
        a_r10 = mean(r["final_r10"] for r in a_cls)
        a_early = sum(1 for r in a_cls if r["stopped_early"])
        print(f"  {cls:<25} {n:>3} {b_it:>5.1f} {a_it:>5.1f} {b_r10:>7.3f} {a_r10:>7.3f} {a_early:>5}/{n}")

    # Build report
    report = {
        "ablation": "phase24_1_proper_ablation",
        "max_iterations": max_iter,
        "summary": {
            "total_queries": total,
            "baseline": {
                "avg_iterations": b_avg_iters,
                "avg_evidence": b_avg_evidence,
                "avg_recall_10": b_avg_r10,
                "avg_need_coverage": b_avg_cov,
                "avg_top3_score": b_avg_top3,
                "avg_latency_ms": b_avg_latency,
                "iteration_distribution": b_iter_dist,
            },
            "adaptive": {
                "avg_iterations": a_avg_iters,
                "avg_evidence": a_avg_evidence,
                "avg_recall_10": a_avg_r10,
                "avg_need_coverage": a_avg_cov,
                "avg_top3_score": a_avg_top3,
                "avg_latency_ms": a_avg_latency,
                "early_stops": a_early_stops,
                "iteration_distribution": a_iter_dist,
                "sufficiency_distribution": a_suff_dist,
                "stop_reasons": a_stop_reasons,
            },
            "delta_iterations": a_avg_iters - b_avg_iters,
            "delta_latency_ms": a_avg_latency - b_avg_latency,
        },
        "per_class": {},
        "baseline_results": baseline_results,
        "adaptive_results": adaptive_results,
    }
    for cls in classes:
        b_cls = [r for r in baseline_results if r["eval_class"] == cls]
        a_cls = [r for r in adaptive_results if r["eval_class"] == cls]
        report["per_class"][cls] = {
            "count": len(b_cls),
            "baseline_avg_iterations": mean(r["iterations_used"] for r in b_cls),
            "adaptive_avg_iterations": mean(r["iterations_used"] for r in a_cls),
            "baseline_avg_recall_10": mean(r["final_r10"] for r in b_cls),
            "adaptive_avg_recall_10": mean(r["final_r10"] for r in a_cls),
            "adaptive_early_stops": sum(1 for r in a_cls if r["stopped_early"]),
        }

    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase24_1_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase24_1_ablation.json")

    return report


if __name__ == "__main__":
    run_ablation()
