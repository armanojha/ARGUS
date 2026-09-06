"""Phase 24.2 Stress Test: Adaptive Research Orchestration.

Compares baseline (fixed iterations) vs adaptive (policy-controlled) on a
harder corpus designed to create genuine retrieval difficulty.

Usage:
    python benchmarks/benchmark_stress_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, get_settings
from app.evidence.models import EvidenceRef, SourceType
from app.evidence.store import EvidenceStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.router import RetrievalPolicyRouter
from app.retrieval.planner import EvidenceNeedPlanner, QueryPlan
from app.orchestration.adaptive_research import (
    AdaptiveResearchPolicy,
    ResearchSufficiency,
    SufficiencyLevel,
    MarginalGainCalculator,
    PatternSpecificPolicies,
    SynthesisGate,
)
from app.verification.deterministic import TextContradictionDetector


# ---------------------------------------------------------------------------
# Corpus & Query Loading
# ---------------------------------------------------------------------------

_STRESS_CORPUS_DIR = Path("benchmarks/eval_data/corpus_v2_stress")
_STRESS_EVAL_PATH = Path("benchmarks/eval_data/stress_test_plan_v1.json")


def load_stress_queries() -> list[dict]:
    with _STRESS_EVAL_PATH.open(encoding="utf-8") as f:
        plan = json.load(f)
    return plan.get("queries", [])


def build_stress_corpus() -> tuple[EvidenceStore, dict[str, list[str]]]:
    """Build EvidenceStore from stress-test corpus (corpus_v2_stress)."""
    from app.evidence.models import Chunk, Document, Source, SourceType
    from app.retrieval.bm25 import assign_bm25_doc_ids
    from app.retrieval.vector import assign_embedding_indices
    import tempfile

    chunks_by_doc: dict[str, list[str]] = {}
    for path in sorted(_STRESS_CORPUS_DIR.glob("*.md")):
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

    if not chunks_by_doc:
        raise RuntimeError(f"No corpus found at {_STRESS_CORPUS_DIR}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_stress_bench_"))
    store = EvidenceStore(
        db_path=tmp_dir / "evidence.db",
        bm25_index_path=tmp_dir / "bm25.pkl",
        faiss_index_path=tmp_dir / "faiss.index",
    )
    source = Source(type=SourceType.TEXT, path="/benchmark/corpus_v2_stress", checksum="stress_v1")
    store.upsert_source(source)
    chunk_id_map: dict[str, list[str]] = {}
    all_chunks: list[Chunk] = []
    doc_version = 0
    for doc_id, texts in chunks_by_doc.items():
        doc_version += 1
        doc = Document(source_id=source.id, version=doc_version, checksum=f"stress_{doc_id}",
                       chunking_strategy="stress_v1")
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
# Evidence Helpers
# ---------------------------------------------------------------------------

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


def _compute_gold_ids(query: dict, chunk_id_map: dict[str, list[str]]) -> set[str]:
    gold_ids: set[str] = set()
    for doc_id in query.get("supporting_docs", []):
        if doc_id in chunk_id_map:
            gold_ids.update(chunk_id_map[doc_id])
    return gold_ids


# ---------------------------------------------------------------------------
# Baseline: always runs to max iterations
# ---------------------------------------------------------------------------

def run_baseline(
    query_text: str,
    plan: QueryPlan,
    pattern_value: str,
    retriever: HybridRetriever,
    settings: Settings,
    gold_ids: set[str],
    max_iterations: int,
) -> dict:
    all_evidence: list[EvidenceRef] = []
    gain_history: list[float] = []
    iteration_trace: list[dict] = []
    contradictions_detected = 0
    retrieval_calls = 0

    pending = list(plan.search_variants) if plan.is_planned else [query_text]
    issued: set[str] = set()

    t_total = time.time()

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
        retrieval_calls += 1
        elapsed = time.time() - t0

        merged, new_count = _merge_evidence(all_evidence, refs)
        prev_total = len(all_evidence)
        gain = (new_count / prev_total) if prev_total else (1.0 if new_count else 0.0)
        gain_history.append(round(gain, 4))
        all_evidence = merged

        detector = TextContradictionDetector()
        contradictions = detector.detect_contradictions(all_evidence)
        contradictions_detected = len(contradictions)

        need_coverage = _compute_need_coverage(plan, all_evidence)
        avg_coverage = sum(v for k, v in need_coverage.items() if not k.startswith("_")) / max(1, sum(1 for k in need_coverage if not k.startswith("_")))

        retrieved_ids = [str(r.chunk_id) for r in all_evidence]
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        mrr_val = mrr(retrieved_ids, gold_ids)
        ndcg10 = ndcg_at_k(retrieved_ids, gold_ids, 10)

        avg_top3 = 0.0
        if all_evidence:
            top = sorted(all_evidence, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

        iteration_trace.append({
            "iteration": iteration,
            "subquery": subquery[:100],
            "new_evidence": new_count,
            "total_evidence": len(all_evidence),
            "gain": round(gain, 4),
            "recall_5": round(r5, 4) if not np.isnan(r5) else 0.0,
            "recall_10": round(r10, 4) if not np.isnan(r10) else 0.0,
            "mrr": round(mrr_val, 4),
            "ndcg_10": round(ndcg10, 4) if not np.isnan(ndcg10) else 0.0,
            "avg_need_coverage": round(avg_coverage, 4),
            "avg_top3_score": round(avg_top3, 4),
            "source_diversity": len({r.document_id for r in all_evidence}),
            "contradictions": contradictions_detected,
            "latency_ms": round(elapsed * 1000, 1),
        })

    total_latency = (time.time() - t_total) * 1000

    sufficiency = ResearchSufficiency.from_state(
        all_evidence, need_coverage, contradictions_detected, pattern_value
    )
    result = sufficiency.assess()

    gate = SynthesisGate()
    gate_result = gate.check(all_evidence, need_coverage)

    final_r5 = iteration_trace[-1]["recall_5"] if iteration_trace else 0.0
    final_r10 = iteration_trace[-1]["recall_10"] if iteration_trace else 0.0
    final_mrr = iteration_trace[-1]["mrr"] if iteration_trace else 0.0
    final_ndcg = iteration_trace[-1]["ndcg_10"] if iteration_trace else 0.0

    return {
        "iterations_used": len(iteration_trace),
        "total_evidence": len(all_evidence),
        "retrieval_calls": retrieval_calls,
        "sufficiency_level": result.level.value,
        "synthesis_gate_passes": gate_result.should_synthesize,
        "synthesis_gate_reason": gate_result.reason,
        "contradictions_detected": contradictions_detected,
        "stopped_early": False,
        "stop_reason": None,
        "trace": iteration_trace,
        "final_r5": final_r5,
        "final_r10": final_r10,
        "final_mrr": final_mrr,
        "final_ndcg_10": final_ndcg,
        "final_coverage": iteration_trace[-1]["avg_need_coverage"] if iteration_trace else 0.0,
        "final_top3": iteration_trace[-1]["avg_top3_score"] if iteration_trace else 0.0,
        "final_diversity": iteration_trace[-1]["source_diversity"] if iteration_trace else 0,
        "total_latency_ms": total_latency,
    }


# ---------------------------------------------------------------------------
# Adaptive: policy controls stopping
# ---------------------------------------------------------------------------

def run_adaptive(
    query_text: str,
    plan: QueryPlan,
    pattern_value: str,
    retriever: HybridRetriever,
    settings: Settings,
    gold_ids: set[str],
    max_iterations: int,
) -> dict:
    policy = AdaptiveResearchPolicy(enabled=True)
    all_evidence: list[EvidenceRef] = []
    gain_history: list[float] = []
    iteration_trace: list[dict] = []
    contradictions_detected = 0
    retrieval_calls = 0
    stopped_early = False
    stop_reason = None
    additional_investigations = 0

    pending = list(plan.search_variants) if plan.is_planned else [query_text]
    issued: set[str] = set()

    t_total = time.time()

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
        retrieval_calls += 1
        elapsed = time.time() - t0

        merged, new_count = _merge_evidence(all_evidence, refs)
        prev_total = len(all_evidence)
        gain = (new_count / prev_total) if prev_total else (1.0 if new_count else 0.0)
        gain_history.append(round(gain, 4))
        all_evidence = merged

        detector = TextContradictionDetector()
        contradictions = detector.detect_contradictions(all_evidence)
        contradictions_detected = len(contradictions)

        need_coverage = _compute_need_coverage(plan, all_evidence)
        avg_coverage = sum(v for k, v in need_coverage.items() if not k.startswith("_")) / max(1, sum(1 for k in need_coverage if not k.startswith("_")))

        retrieved_ids = [str(r.chunk_id) for r in all_evidence]
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        mrr_val = mrr(retrieved_ids, gold_ids)
        ndcg10 = ndcg_at_k(retrieved_ids, gold_ids, 10)

        avg_top3 = 0.0
        if all_evidence:
            top = sorted(all_evidence, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

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
            "recall_5": round(r5, 4) if not np.isnan(r5) else 0.0,
            "recall_10": round(r10, 4) if not np.isnan(r10) else 0.0,
            "mrr": round(mrr_val, 4),
            "ndcg_10": round(ndcg10, 4) if not np.isnan(ndcg10) else 0.0,
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
            if decision.action == "investigate":
                additional_investigations += 1
            break

    total_latency = (time.time() - t_total) * 1000

    sufficiency = ResearchSufficiency.from_state(
        all_evidence, need_coverage, contradictions_detected, pattern_value
    )
    result = sufficiency.assess()

    gate = SynthesisGate()
    gate_result = gate.check(all_evidence, need_coverage)

    final_r5 = iteration_trace[-1]["recall_5"] if iteration_trace else 0.0
    final_r10 = iteration_trace[-1]["recall_10"] if iteration_trace else 0.0
    final_mrr = iteration_trace[-1]["mrr"] if iteration_trace else 0.0
    final_ndcg = iteration_trace[-1]["ndcg_10"] if iteration_trace else 0.0

    return {
        "iterations_used": len(iteration_trace),
        "total_evidence": len(all_evidence),
        "retrieval_calls": retrieval_calls,
        "sufficiency_level": result.level.value,
        "synthesis_gate_passes": gate_result.should_synthesize,
        "synthesis_gate_reason": gate_result.reason,
        "contradictions_detected": contradictions_detected,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "additional_investigations": additional_investigations,
        "trace": iteration_trace,
        "final_r5": final_r5,
        "final_r10": final_r10,
        "final_mrr": final_mrr,
        "final_ndcg_10": final_ndcg,
        "final_coverage": iteration_trace[-1]["avg_need_coverage"] if iteration_trace else 0.0,
        "final_top3": iteration_trace[-1]["avg_top3_score"] if iteration_trace else 0.0,
        "final_diversity": iteration_trace[-1]["source_diversity"] if iteration_trace else 0,
        "total_latency_ms": total_latency,
    }


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def run_stress_test():
    print("=" * 70)
    print("Phase 24.2 Stress Test: Baseline vs Adaptive Orchestration")
    print("=" * 70)

    settings = get_settings()
    planner = EvidenceNeedPlanner()
    router = RetrievalPolicyRouter()

    print("\n[1/3] Building stress-test corpus...")
    t0 = time.time()
    store, chunk_id_map = build_stress_corpus()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.time()-t0:.1f}s)")
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    queries = load_stress_queries()
    print(f"  Queries: {len(queries)}")

    max_iter = settings.orchestration_max_iterations

    print(f"\n[2/3] Running {max_iter}-iteration stress test (baseline vs adaptive)...")
    baseline_results = []
    adaptive_results = []

    for q in queries:
        query_text = q["query"]
        eval_class = q.get("class", "unknown")
        gold_ids = _compute_gold_ids(q, chunk_id_map)

        pattern = router.classify_question(query_text)
        plan = planner.plan(query_text, pattern)
        pattern_value = pattern.value

        base = run_baseline(
            query_text, plan, pattern_value, retriever, settings,
            gold_ids, max_iterations=max_iter,
        )
        base["id"] = q["id"]
        base["query"] = query_text
        base["eval_class"] = eval_class
        base["pattern"] = pattern_value
        base["difficulty"] = q.get("difficulty", "medium")
        base["gold_count"] = len(gold_ids)
        baseline_results.append(base)

        adapt = run_adaptive(
            query_text, plan, pattern_value, retriever, settings,
            gold_ids, max_iterations=max_iter,
        )
        adapt["id"] = q["id"]
        adapt["query"] = query_text
        adapt["eval_class"] = eval_class
        adapt["pattern"] = pattern_value
        adapt["difficulty"] = q.get("difficulty", "medium")
        adapt["gold_count"] = len(gold_ids)
        adaptive_results.append(adapt)

        delta_r10 = adapt["final_r10"] - base["final_r10"]
        delta_mrr = adapt["final_mrr"] - base["final_mrr"]
        iters_saved = base["iterations_used"] - adapt["iterations_used"]

        print(f"  [{q['id']}] {eval_class:<25} "
              f"base_iters={base['iterations_used']} adapt_iters={adapt['iterations_used']} "
              f"saved={iters_saved} "
              f"R@10 base={base['final_r10']:.3f} adapt={adapt['final_r10']:.3f} "
              f"delta={delta_r10:+.3f} "
              f"{'EARLY' if adapt['stopped_early'] else 'FULL'} "
              f"{adapt.get('stop_reason', '')}")

    print(f"\n[3/3] Aggregating results...")

    total = len(queries)

    # Baseline stats
    b_avg_iters = mean(r["iterations_used"] for r in baseline_results)
    b_avg_r5 = mean(r["final_r5"] for r in baseline_results)
    b_avg_r10 = mean(r["final_r10"] for r in baseline_results)
    b_avg_mrr = mean(r["final_mrr"] for r in baseline_results)
    b_avg_ndcg = mean(r["final_ndcg_10"] for r in baseline_results)
    b_avg_cov = mean(r["final_coverage"] for r in baseline_results)
    b_avg_top3 = mean(r["final_top3"] for r in baseline_results)
    b_avg_evidence = mean(r["total_evidence"] for r in baseline_results)
    b_avg_latency = mean(r["total_latency_ms"] for r in baseline_results)
    b_p95_latency = sorted(r["total_latency_ms"] for r in baseline_results)[int(0.95 * total)]

    # Adaptive stats
    a_avg_iters = mean(r["iterations_used"] for r in adaptive_results)
    a_avg_r5 = mean(r["final_r5"] for r in adaptive_results)
    a_avg_r10 = mean(r["final_r10"] for r in adaptive_results)
    a_avg_mrr = mean(r["final_mrr"] for r in adaptive_results)
    a_avg_ndcg = mean(r["final_ndcg_10"] for r in adaptive_results)
    a_avg_cov = mean(r["final_coverage"] for r in adaptive_results)
    a_avg_top3 = mean(r["final_top3"] for r in adaptive_results)
    a_avg_evidence = mean(r["total_evidence"] for r in adaptive_results)
    a_avg_latency = mean(r["total_latency_ms"] for r in adaptive_results)
    a_p95_latency = sorted(r["total_latency_ms"] for r in adaptive_results)[int(0.95 * total)]
    a_early_stops = sum(1 for r in adaptive_results if r["stopped_early"])
    a_investigations = sum(1 for r in adaptive_results if r.get("additional_investigations", 0) > 0)

    # Iteration distribution
    b_iter_dist: dict[int, int] = {}
    for r in baseline_results:
        iters = r["iterations_used"]
        b_iter_dist[iters] = b_iter_dist.get(iters, 0) + 1
    a_iter_dist: dict[int, int] = {}
    for r in adaptive_results:
        iters = r["iterations_used"]
        a_iter_dist[iters] = a_iter_dist.get(iters, 0) + 1

    # Sufficiency distribution
    a_suff_dist: dict[str, int] = {}
    for r in adaptive_results:
        lvl = r["sufficiency_level"]
        a_suff_dist[lvl] = a_suff_dist.get(lvl, 0) + 1

    # Adaptive stop reasons
    a_stop_reasons: dict[str, int] = {}
    for r in adaptive_results:
        if r["stopped_early"]:
            reason = r["stop_reason"]
            a_stop_reasons[reason] = a_stop_reasons.get(reason, 0) + 1

    # Per-category breakdown
    categories = sorted(set(r["eval_class"] for r in baseline_results))

    print(f"\n{'='*70}")
    print(f"STRESS TEST RESULTS ({max_iter}-iteration max)")
    print(f"{'='*70}")

    print(f"\n{'Metric':<35} {'Baseline':>12} {'Adaptive':>12} {'Delta':>10}")
    print(f"{'-'*69}")
    print(f"{'Avg iterations':<35} {b_avg_iters:>12.2f} {a_avg_iters:>12.2f} {a_avg_iters-b_avg_iters:>+10.2f}")
    print(f"{'Avg evidence':<35} {b_avg_evidence:>12.1f} {a_avg_evidence:>12.1f} {a_avg_evidence-b_avg_evidence:>+10.1f}")
    print(f"{'Avg Recall@5':<35} {b_avg_r5:>12.4f} {a_avg_r5:>12.4f} {a_avg_r5-b_avg_r5:>+10.4f}")
    print(f"{'Avg Recall@10':<35} {b_avg_r10:>12.4f} {a_avg_r10:>12.4f} {a_avg_r10-b_avg_r10:>+10.4f}")
    print(f"{'Avg MRR':<35} {b_avg_mrr:>12.4f} {a_avg_mrr:>12.4f} {a_avg_mrr-b_avg_mrr:>+10.4f}")
    print(f"{'Avg nDCG@10':<35} {b_avg_ndcg:>12.4f} {a_avg_ndcg:>12.4f} {a_avg_ndcg-b_avg_ndcg:>+10.4f}")
    print(f"{'Avg need coverage':<35} {b_avg_cov:>12.4f} {a_avg_cov:>12.4f} {a_avg_cov-b_avg_cov:>+10.4f}")
    print(f"{'Avg top-3 score':<35} {b_avg_top3:>12.4f} {a_avg_top3:>12.4f} {a_avg_top3-b_avg_top3:>+10.4f}")
    print(f"{'Avg latency (ms)':<35} {b_avg_latency:>12.0f} {a_avg_latency:>12.0f} {a_avg_latency-b_avg_latency:>+10.0f}")
    print(f"{'P95 latency (ms)':<35} {b_p95_latency:>12.0f} {a_p95_latency:>12.0f} {a_p95_latency-b_p95_latency:>+10.0f}")

    print(f"\nAdaptive Early Stops: {a_early_stops}/{total} ({a_early_stops/total:.1%})")
    print(f"Adaptive Investigations: {a_investigations}/{total} ({a_investigations/total:.1%})")

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
        print(f"  {reason:<40} {count:>3} ({count/total:.1%})")

    # Per-category breakdown
    print(f"\nPer-Category Breakdown:")
    print(f"  {'Category':<25} {'N':>3} {'B-R@10':>7} {'A-R@10':>7} {'dR@10':>7} {'B-R@5':>7} {'A-R@5':>7} {'B-MRR':>7} {'A-MRR':>7} {'B-It':>5} {'A-It':>5} {'A-Early':>8}")
    print(f"  {'-'*110}")
    for cat in categories:
        b_cls = [r for r in baseline_results if r["eval_class"] == cat]
        a_cls = [r for r in adaptive_results if r["eval_class"] == cat]
        n = len(b_cls)
        b_r10 = mean(r["final_r10"] for r in b_cls)
        a_r10 = mean(r["final_r10"] for r in a_cls)
        b_r5 = mean(r["final_r5"] for r in b_cls)
        a_r5 = mean(r["final_r5"] for r in a_cls)
        b_mrr = mean(r["final_mrr"] for r in b_cls)
        a_mrr = mean(r["final_mrr"] for r in a_cls)
        b_it = mean(r["iterations_used"] for r in b_cls)
        a_it = mean(r["iterations_used"] for r in a_cls)
        a_early = sum(1 for r in a_cls if r["stopped_early"])
        delta_r10 = a_r10 - b_r10
        print(f"  {cat:<25} {n:>3} {b_r10:>7.3f} {a_r10:>7.3f} {delta_r10:>+7.3f} {b_r5:>7.3f} {a_r5:>7.3f} {b_mrr:>7.3f} {a_mrr:>7.3f} {b_it:>5.1f} {a_it:>5.1f} {a_early:>5}/{n}")

    # Delta analysis: where adaptive improved vs regressed
    print(f"\nDelta Analysis (Adaptive - Baseline):")
    improved = []
    regressed = []
    same = []
    for b, a in zip(baseline_results, adaptive_results):
        delta_r10 = a["final_r10"] - b["final_r10"]
        if delta_r10 > 0.01:
            improved.append((b["id"], b["eval_class"], delta_r10))
        elif delta_r10 < -0.01:
            regressed.append((b["id"], b["eval_class"], delta_r10))
        else:
            same.append(b["id"])

    print(f"  Improved (dR@10 > 0.01): {len(improved)}/{total}")
    for qid, cls, delta in sorted(improved, key=lambda x: -x[2]):
        print(f"    {qid} ({cls}): dR@10 = {delta:+.3f}")
    print(f"  Regressed (dR@10 < -0.01): {len(regressed)}/{total}")
    for qid, cls, delta in sorted(regressed, key=lambda x: x[2]):
        print(f"    {qid} ({cls}): dR@10 = {delta:+.3f}")
    print(f"  No change: {len(same)}/{total}")

    # Build report
    report = {
        "stress_test": "phase24_2_stress_test",
        "max_iterations": max_iter,
        "corpus": {
            "total_chunks": total_chunks,
            "total_documents": len(chunk_id_map),
        },
        "summary": {
            "total_queries": total,
            "baseline": {
                "avg_iterations": b_avg_iters,
                "avg_evidence": b_avg_evidence,
                "avg_recall_5": b_avg_r5,
                "avg_recall_10": b_avg_r10,
                "avg_mrr": b_avg_mrr,
                "avg_ndcg_10": b_avg_ndcg,
                "avg_need_coverage": b_avg_cov,
                "avg_top3_score": b_avg_top3,
                "avg_latency_ms": b_avg_latency,
                "p95_latency_ms": b_p95_latency,
                "iteration_distribution": b_iter_dist,
            },
            "adaptive": {
                "avg_iterations": a_avg_iters,
                "avg_evidence": a_avg_evidence,
                "avg_recall_5": a_avg_r5,
                "avg_recall_10": a_avg_r10,
                "avg_mrr": a_avg_mrr,
                "avg_ndcg_10": a_avg_ndcg,
                "avg_need_coverage": a_avg_cov,
                "avg_top3_score": a_avg_top3,
                "avg_latency_ms": a_avg_latency,
                "p95_latency_ms": a_p95_latency,
                "early_stops": a_early_stops,
                "investigations": a_investigations,
                "iteration_distribution": a_iter_dist,
                "sufficiency_distribution": a_suff_dist,
                "stop_reasons": a_stop_reasons,
            },
            "delta": {
                "avg_iterations": a_avg_iters - b_avg_iters,
                "avg_latency_ms": a_avg_latency - b_avg_latency,
                "avg_recall_10": a_avg_r10 - b_avg_r10,
                "avg_mrr": a_avg_mrr - b_avg_mrr,
                "improved": len(improved),
                "regressed": len(regressed),
                "same": len(same),
            },
        },
        "per_category": {},
        "baseline_results": baseline_results,
        "adaptive_results": adaptive_results,
    }
    for cat in categories:
        b_cls = [r for r in baseline_results if r["eval_class"] == cat]
        a_cls = [r for r in adaptive_results if r["eval_class"] == cat]
        report["per_category"][cat] = {
            "count": len(b_cls),
            "baseline_avg_recall_10": mean(r["final_r10"] for r in b_cls),
            "adaptive_avg_recall_10": mean(r["final_r10"] for r in a_cls),
            "baseline_avg_mrr": mean(r["final_mrr"] for r in b_cls),
            "adaptive_avg_mrr": mean(r["final_mrr"] for r in a_cls),
            "baseline_avg_iterations": mean(r["iterations_used"] for r in b_cls),
            "adaptive_avg_iterations": mean(r["iterations_used"] for r in a_cls),
            "adaptive_early_stops": sum(1 for r in a_cls if r["stopped_early"]),
        }

    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase24_2_stress_test.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase24_2_stress_test.json")

    return report


if __name__ == "__main__":
    run_stress_test()
