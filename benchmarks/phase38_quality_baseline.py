#!/usr/bin/env python3
"""Phase 38: Quality Baseline & Gold-Standard Evaluation.

Runs the current ARGUS pipeline on the eval_plan_v1 gold benchmark (38 queries)
and evaluates answer quality using the existing evaluator plus new metrics
for abstention, hallucination, and conflict handling.

Usage:
    python -m benchmarks.phase38_quality_baseline
"""
from __future__ import annotations

import io
import json
import sys
import time
import threading
from pathlib import Path
from typing import Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Load gold-standard dataset ───────────────────────────────────────────────

def load_gold_standard() -> list[dict]:
    """Load eval_plan_v1.json and enrich with structured evaluation metadata."""
    plan_path = Path(__file__).resolve().parent / "eval_data" / "eval_plan_v1.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    queries = []
    for q in plan["queries"]:
        gold_facts = q.get("gold_facts", [])
        is_absent = q.get("absent", False)
        is_conflict = q.get("conflict", False)
        supporting_docs = q.get("supporting_docs", [])

        # Determine expected behavior
        if is_absent:
            expected_behavior = "abstain"
        elif is_conflict:
            expected_behavior = "acknowledge_conflict"
        else:
            expected_behavior = "answer_with_citations"

        # Determine forbidden claims
        forbidden_claims = []
        if is_absent:
            forbidden_claims.append("Any specific numerical answer or factual claim about the topic")

        queries.append({
            "query_id": q["id"],
            "question": q["query"],
            "category": q["class"],
            "difficulty": q.get("note", ""),
            "gold_facts": gold_facts,
            "required_facts": gold_facts,  # All gold facts are required
            "supporting_docs": supporting_docs,
            "is_absent": is_absent,
            "is_conflict": is_conflict,
            "expected_behavior": expected_behavior,
            "forbidden_claims": forbidden_claims,
            "verification_expect": q.get("verification_expect", "skip"),
        })

    return queries


# ── Enhanced evaluation metrics ──────────────────────────────────────────────

def evaluate_abstention(answer: str, is_absent: bool) -> dict:
    """Evaluate whether the answer correctly abstains for absent-info queries."""
    if not is_absent:
        return {"abstention_correct": True, "abstention_type": "not_applicable"}

    # Check if answer indicates insufficient evidence
    abstention_phrases = [
        "no supporting evidence", "no evidence", "not available",
        "don't have enough", "cannot determine", "not found",
        "no data", "insufficient information", "unable to find",
        "no information", "not mentioned", "cannot answer",
        "no record", "not enough information", "doesn't provide",
    ]
    answer_lower = answer.lower()
    detected = any(phrase in answer_lower for phrase in abstention_phrases)

    # Check if answer fabricates a specific response
    has_numbers = bool(__import__("re").search(r"\d+", answer))
    has_specific_claims = len(answer.split()) > 30 and has_numbers

    if detected:
        return {"abstention_correct": True, "abstention_type": "correct_abstention"}
    elif has_specific_claims:
        return {"abstention_correct": False, "abstention_type": "fabrication"}
    else:
        return {"abstention_correct": False, "abstention_type": "vague_no_abstention"}


def evaluate_conflict_handling(answer: str, is_conflict: bool) -> dict:
    """Evaluate whether the answer correctly handles conflicting evidence."""
    if not is_conflict:
        return {"conflict_correct": True, "conflict_type": "not_applicable"}

    answer_lower = answer.lower()

    # Check if conflict is acknowledged
    conflict_phrases = [
        "conflict", "contradict", "discrepan", "different figure",
        "varying", "inconsistent", "disagree", "varies",
        "different source", "different report", "old report",
    ]
    acknowledged = any(phrase in answer_lower for phrase in conflict_phrases)

    # Check if sources are attributed
    source_phrases = [
        "according to", "reported", "stated", "document",
        "legacy", "current", "2023", "2025",
    ]
    attributed = any(phrase in answer_lower for phrase in source_phrases)

    return {
        "conflict_correct": acknowledged,
        "conflict_acknowledged": acknowledged,
        "conflict_attributed": attributed,
    }


def evaluate_hallucination(answer: str, evidence_texts: list[str], gold_facts: list[str]) -> dict:
    """Detect claims in the answer not grounded in any evidence.

    This is a lexical check — claims that share no key terms with any evidence
    or gold fact are flagged as potential hallucinations.
    """
    from app.evaluation.answer_quality import _extract_key_terms, _SENTENCE_SPLIT

    # Extract key terms from all evidence + gold facts
    evidence_terms = set()
    for text in evidence_texts:
        evidence_terms.update(_extract_key_terms(text))
    for fact in gold_facts:
        evidence_terms.update(_extract_key_terms(fact))

    # Split answer into claims
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]

    hallucinated_claims = []
    grounded_claims = []

    for sentence in sentences:
        claim_terms = _extract_key_terms(sentence)
        if not claim_terms:
            continue
        overlap = claim_terms & evidence_terms
        coverage = len(overlap) / len(claim_terms) if claim_terms else 0

        if coverage < 0.2 and len(claim_terms) > 3:
            hallucinated_claims.append(sentence[:120])
        else:
            grounded_claims.append(sentence[:120])

    total = len(hallucinated_claims) + len(grounded_claims)
    hallucination_rate = len(hallucinated_claims) / total if total > 0 else 0.0

    return {
        "hallucination_rate": hallucination_rate,
        "hallucinated_count": len(hallucinated_claims),
        "grounded_count": len(grounded_claims),
        "hallucinated_claims": hallucinated_claims[:5],  # First 5 for debugging
    }


# ── Retrieval instrumentation ───────────────────────────────────────────────

class RetrievalInstrument:
    """Thread-safe instrumentation for retrieval pipeline components."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.bm25_latencies_ms: list[float] = []
            self.vector_latencies_ms: list[float] = []
            self.embedding_latencies_ms: list[float] = []
            self.fusion_latencies_ms: list[float] = []
            self.rerank_latencies_ms: list[float] = []
            self.esel_latencies_ms: list[float] = []

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "bm25_ms": sum(self.bm25_latencies_ms),
                "vector_ms": sum(self.vector_latencies_ms),
                "embedding_ms": sum(self.embedding_latencies_ms),
                "fusion_ms": sum(self.fusion_latencies_ms),
                "rerank_ms": sum(self.rerank_latencies_ms),
                "esel_ms": sum(self.esel_latencies_ms),
            }


_inst = RetrievalInstrument()


def patch_retriever(retriever):
    """Instrument retriever methods. Returns restore function."""
    from app.retrieval.hybrid import HybridRetriever

    orig_bm25 = retriever.bm25.search
    def patched_bm25(query, **kwargs):
        t0 = time.perf_counter()
        result = orig_bm25(query, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.bm25_latencies_ms.append(ms)
        return result
    retriever.bm25.search = patched_bm25

    orig_vector = retriever.vector.search
    def patched_vector(embedding, **kwargs):
        t0 = time.perf_counter()
        result = orig_vector(embedding, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.vector_latencies_ms.append(ms)
        return result
    retriever.vector.search = patched_vector

    orig_embed = retriever.embedder.embed_texts
    def patched_embed(texts, **kwargs):
        t0 = time.perf_counter()
        result = orig_embed(texts, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.embedding_latencies_ms.append(ms)
        return result
    retriever.embedder.embed_texts = patched_embed

    orig_fuse = HybridRetriever._fuse
    @staticmethod
    def patched_fuse(store, query, top_k, bm25_weight, vector_weight, bm25_scores, vector_scores, **kwargs):
        t0 = time.perf_counter()
        result = orig_fuse(store, query, top_k, bm25_weight, vector_weight, bm25_scores, vector_scores, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.fusion_latencies_ms.append(ms)
        return result
    HybridRetriever._fuse = patched_fuse

    def restore():
        retriever.bm25.search = orig_bm25
        retriever.vector.search = orig_vector
        retriever.embedder.embed_texts = orig_embed
        HybridRetriever._fuse = orig_fuse

    return restore


def patch_reranker(reranker):
    """Instrument reranker.rerank(). Returns restore function."""
    orig_rerank = reranker.rerank
    def patched_rerank(query, chunks, top_k=10, **kwargs):
        t0 = time.perf_counter()
        result = orig_rerank(query, chunks, top_k=top_k, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.rerank_latencies_ms.append(ms)
        return result
    reranker.rerank = patched_rerank

    def restore():
        reranker.rerank = orig_rerank

    return restore


def patch_evidence_selector():
    """Instrument EvidenceSelector.select(). Returns restore function."""
    from app.retrieval.evidence_selector import EvidenceSelector
    orig_select = EvidenceSelector.select

    def patched_select(self, evidence, *args, **kwargs):
        t0 = time.perf_counter()
        result = orig_select(self, evidence, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        with _inst._lock:
            _inst.esel_latencies_ms.append(ms)
        return result

    EvidenceSelector.select = patched_select

    def restore():
        EvidenceSelector.select = orig_select

    return restore


# ── Telemetry extraction ─────────────────────────────────────────────────────

def extract_telemetry(telemetry_summary: dict | None) -> dict:
    if not telemetry_summary:
        return {"calls": 0, "total_ms": 0, "total_tokens": 0, "by_type": {},
                "fallbacks": 0, "rate_limits": 0, "is_clean": True,
                "providers": [], "models": [], "fallback_details": []}

    decisions = telemetry_summary.get("routing_decisions", [])
    by_type = defaultdict(lambda: {"count": 0, "total_ms": 0, "total_tokens": 0})
    providers = set()
    models = set()
    total_ms = 0
    total_tokens = 0
    fallbacks = 0
    rate_limits = 0
    fallback_details = []

    for d in decisions:
        ct = d.get("call_type", "unknown")
        lat = d.get("latency_ms", 0) or 0
        tt = d.get("total_tokens", 0) or 0
        by_type[ct]["count"] += 1
        by_type[ct]["total_ms"] += lat
        by_type[ct]["total_tokens"] += tt
        total_ms += lat
        total_tokens += tt
        providers.add(d.get("provider", "unknown"))
        models.add(d.get("model", "unknown"))
        if d.get("is_fallback"):
            fallbacks += 1
            fallback_details.append({"call_type": ct, "provider": d.get("provider"),
                                     "reason": d.get("fallback_reason")})
        if d.get("error_class") and "rate" in str(d.get("error_class", "")).lower():
            rate_limits += 1

    return {
        "calls": len(decisions),
        "total_ms": total_ms,
        "total_tokens": total_tokens,
        "by_type": dict(by_type),
        "fallbacks": fallbacks,
        "rate_limits": rate_limits,
        "is_clean": fallbacks == 0,
        "providers": sorted(providers),
        "models": sorted(models),
        "fallback_details": fallback_details,
    }


# ── Main benchmark ───────────────────────────────────────────────────────────

async def _run_benchmark(queries, retriever, reranker, settings):
    """Run all gold-standard queries through the production pipeline."""
    from app.orchestration.graph import run_query
    from app.llm_gateway.telemetry import start_run_telemetry, end_run_telemetry
    from app.evaluation.answer_quality import evaluate_answer

    results = []

    # Warm-up
    print("\n[1/3] Warming up...")
    for i in range(2):
        try:
            start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=f"warmup_{i}")
            await run_query("warm up", settings=settings, retriever=retriever, reranker=reranker)
            end_run_telemetry()
        except Exception:
            try:
                end_run_telemetry()
            except Exception:
                pass
    print("  Warm-up complete")

    # Cold start
    print("\n[2/3] Cold start measurement...")
    cold_t0 = time.perf_counter()
    try:
        start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id="cold")
        await run_query("cold start", settings=settings, retriever=retriever, reranker=reranker)
        end_run_telemetry()
    except Exception:
        try:
            end_run_telemetry()
        except Exception:
            pass
    cold_ms = (time.perf_counter() - cold_t0) * 1000
    print(f"  Cold start: {cold_ms:.0f}ms")

    # Benchmark
    print(f"\n[3/3] Running {len(queries)} gold-standard queries...")

    for i, q in enumerate(queries, 1):
        query = q["question"]
        qid = q["query_id"]
        category = q["category"]

        _inst.reset()
        t0 = time.perf_counter()
        try:
            start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=qid)
            result = await run_query(query, settings=settings, retriever=retriever, reranker=reranker)
            telemetry_summary = end_run_telemetry()
        except Exception as e:
            print(f"  [{i:>2}/{len(queries)}] {qid:<8} {category:<25} FAILED: {e}")
            try:
                end_run_telemetry()
            except Exception:
                pass
            continue
        total_ms = (time.perf_counter() - t0) * 1000

        telemetry = extract_telemetry(telemetry_summary)
        answer = getattr(result, "answer", "") or ""
        citations = getattr(result, "citations", []) or []
        evidence_texts = [c.text for c in citations] if citations else []

        # Run existing evaluator
        quality = evaluate_answer(
            answer, citations,
            gold_facts=q["gold_facts"],
            query=query,
            check_numerical=True,
        )

        # Run new evaluators
        abstention = evaluate_abstention(answer, q["is_absent"])
        conflict = evaluate_conflict_handling(answer, q["is_conflict"])
        hallucination = evaluate_hallucination(answer, evidence_texts, q["gold_facts"])

        # Build record
        record = {
            "query_id": qid,
            "category": category,
            "question": query,
            "gold_facts": q["gold_facts"],
            "expected_behavior": q["expected_behavior"],
            "is_absent": q["is_absent"],
            "is_conflict": q["is_conflict"],
            # Answer
            "answer": answer,
            "answer_length": len(answer),
            "evidence_count": len(citations),
            "cited_indices": [c.ref_id for c in citations],
            # Quality metrics
            "claim_support_rate": quality.claim_support_rate,
            "unsupported_claim_rate": quality.unsupported_claim_rate,
            "partially_supported_rate": quality.partially_supported_rate,
            "contradicted_claim_rate": quality.contradicted_claim_rate,
            "citation_presence_rate": quality.citation_presence_rate,
            "citation_precision": quality.citation_precision,
            "gold_fact_coverage": quality.gold_fact_coverage,
            "gold_facts_found": quality.gold_facts_found,
            "gold_facts_missing": quality.gold_facts_missing,
            "numerical_consistency_rate": quality.numerical_consistency_rate,
            "query_relevance": quality.query_relevance,
            # New metrics
            "abstention_correct": abstention["abstention_correct"],
            "abstention_type": abstention["abstention_type"],
            "conflict_correct": conflict["conflict_correct"],
            "conflict_acknowledged": conflict.get("conflict_acknowledged", False),
            "hallucination_rate": hallucination["hallucination_rate"],
            "hallucinated_count": hallucination["hallucinated_count"],
            "hallucinated_claims": hallucination["hallucinated_claims"],
            # Pipeline
            "total_e2e_ms": round(total_ms, 2),
            "llm_calls": telemetry["calls"],
            "llm_ms": telemetry["total_ms"],
            "total_tokens": telemetry["total_tokens"],
            "llm_by_type": telemetry["by_type"],
            "fallback_count": telemetry["fallbacks"],
            "is_clean": telemetry["is_clean"],
            "providers": telemetry["providers"],
            "iterations": getattr(result, "iterations_used", 0) or 0,
            "stop_reason": str(getattr(result, "stop_reason", "")) if getattr(result, "stop_reason", None) else None,
            "outcome": str(getattr(result, "outcome", "unknown")),
            "warnings": getattr(result, "warnings", []) or [],
            # Retrieval
            "retrieval": _inst.to_dict(),
        }
        results.append(record)

        # Print status
        cs = quality.claim_support_rate
        cp = quality.citation_precision
        gfc = quality.gold_fact_coverage or 0
        hr = hallucination["hallucination_rate"]
        abst = "✓" if abstention["abstention_correct"] else "✗"
        conf = "✓" if conflict["conflict_correct"] else "✗"
        status = "CLEAN" if telemetry["is_clean"] else f"FB({telemetry['fallbacks']})"

        print(f"  [{i:>2}/{len(queries)}] {qid:<8} {category:<25} "
              f"cs={cs:.2f} cp={cp:.2f} gfc={gfc:.2f} hr={hr:.2f} "
              f"abst={abst} conf={conf} "
              f"calls={telemetry['calls']:<3} tok={telemetry['total_tokens']:<5} "
              f"{total_ms:>6.0f}ms  {status}")

    return results, cold_ms


def _mean(data):
    import numpy as np
    return float(np.mean(data)) if data else 0.0


def run_quality_baseline():
    print("=" * 110)
    print("ARGUS Phase 38: Quality Baseline & Gold-Standard Evaluation")
    print("=" * 110)

    # Load gold standard
    print("\n[Setup] Loading gold-standard dataset...")
    queries = load_gold_standard()
    print(f"  {len(queries)} queries across {len(set(q['category'] for q in queries))} categories")

    # Category distribution
    cats = defaultdict(int)
    for q in queries:
        cats[q["category"]] += 1
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    # Build store
    print("\n[Setup] Building benchmark corpus & store...")
    t0 = time.monotonic()
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    from app.reranking.reranker import Reranker, NoOpReranker
    try:
        reranker = Reranker()
    except Exception:
        reranker = NoOpReranker()

    from app.config import Settings
    settings = Settings()

    # Instrument
    restore_retriever = patch_retriever(retriever)
    restore_reranker = patch_reranker(reranker)
    restore_esel = patch_evidence_selector()

    import asyncio
    results, cold_ms = asyncio.run(_run_benchmark(queries, retriever, reranker, settings))

    # Restore
    restore_retriever()
    restore_reranker()
    restore_esel()

    if not results:
        print("\nNo successful results. Exiting.")
        return

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "phase38_quality_baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cold_start_ms": cold_ms, "queries": results}, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════

    clean = [r for r in results if r["is_clean"]]
    contaminated = [r for r in results if not r["is_clean"]]

    print("\n" + "=" * 110)
    print("EXECUTIVE SUMMARY")
    print("=" * 110)
    print(f"\n  Total queries:       {len(results)}")
    print(f"  Clean (no fallback): {len(clean)}")
    print(f"  Contaminated:        {len(contaminated)}")
    print(f"  Cold start:          {cold_ms:.0f}ms")

    # Overall metrics
    print("\n" + "=" * 110)
    print("OVERALL QUALITY METRICS")
    print("=" * 110)

    metrics = [
        ("claim_support_rate", "Claim Support Rate"),
        ("unsupported_claim_rate", "Unsupported Claim Rate"),
        ("partially_supported_rate", "Partially Supported Rate"),
        ("contradicted_claim_rate", "Contradicted Claim Rate"),
        ("citation_presence_rate", "Citation Presence Rate"),
        ("citation_precision", "Citation Precision"),
        ("gold_fact_coverage", "Gold Fact Coverage"),
        ("numerical_consistency_rate", "Numerical Consistency"),
        ("query_relevance", "Query Relevance"),
        ("hallucination_rate", "Hallucination Rate"),
    ]

    for key, label in metrics:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            avg = _mean(vals)
            print(f"  {label:<35} {avg:.3f}  (n={len(vals)})")

    # Abstention and conflict
    absent_queries = [r for r in results if r["is_absent"]]
    conflict_queries = [r for r in results if r["is_conflict"]]

    if absent_queries:
        abst_correct = sum(1 for r in absent_queries if r["abstention_correct"])
        print(f"\n  Abstention Correctness             {abst_correct}/{len(absent_queries)} "
              f"({abst_correct/len(absent_queries)*100:.0f}%)")
        for r in absent_queries:
            print(f"    {r['query_id']}: {r['abstention_type']}")

    if conflict_queries:
        conf_correct = sum(1 for r in conflict_queries if r["conflict_correct"])
        print(f"\n  Conflict Handling Correctness       {conf_correct}/{len(conflict_queries)} "
              f"({conf_correct/len(conflict_queries)*100:.0f}%)")
        for r in conflict_queries:
            print(f"    {r['query_id']}: ack={r['conflict_acknowledged']}")

    # Per-category breakdown
    print("\n" + "=" * 110)
    print("PER-CATEGORY QUALITY BREAKDOWN")
    print("=" * 110)

    categories = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)

    header = f"  {'Category':<25} {'N':<4} {'CS':<6} {'CP':<6} {'GFC':<6} {'HR':<6} {'Abst':<6} {'Conf':<6} {'LLM':<5} {'Tok':<6} {'E2E ms':<8}"
    print(f"\n{header}")
    print("  " + "-" * 95)

    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        n = len(cat_results)
        cs = _mean([r["claim_support_rate"] for r in cat_results])
        cp = _mean([r["citation_precision"] for r in cat_results])
        gfc = _mean([r["gold_fact_coverage"] for r in cat_results if r["gold_fact_coverage"] is not None])
        hr = _mean([r["hallucination_rate"] for r in cat_results])
        calls = _mean([r["llm_calls"] for r in cat_results])
        tok = _mean([r["total_tokens"] for r in cat_results])
        e2e = _mean([r["total_e2e_ms"] for r in cat_results])

        absent_cat = [r for r in cat_results if r["is_absent"]]
        conflict_cat = [r for r in cat_results if r["is_conflict"]]
        abst_str = f"{sum(1 for r in absent_cat if r['abstention_correct'])}/{len(absent_cat)}" if absent_cat else "N/A"
        conf_str = f"{sum(1 for r in conflict_cat if r['conflict_correct'])}/{len(conflict_cat)}" if conflict_cat else "N/A"

        print(f"  {cat:<25} {n:<4} {cs:<6.2f} {cp:<6.2f} {gfc:<6.2f} {hr:<6.2f} "
              f"{abst_str:<6} {conf_str:<6} {calls:<5.1f} {tok:<6.0f} {e2e:<8.0f}")

    # Error taxonomy
    print("\n" + "=" * 110)
    print("ERROR TAXONOMY")
    print("=" * 110)

    errors = defaultdict(int)
    for r in results:
        if r["claim_support_rate"] < 0.5:
            errors["LOW_CLAIM_SUPPORT"] += 1
        if r["unsupported_claim_rate"] > 0.3:
            errors["HIGH_UNSUPPORTED_CLAIMS"] += 1
        if r["citation_precision"] < 0.5 and r["evidence_count"] > 0:
            errors["CITATION_ERROR"] += 1
        if r["hallucination_rate"] > 0.3:
            errors["HALLUCINATION"] += 1
        if r["gold_fact_coverage"] is not None and r["gold_fact_coverage"] < 0.5:
            errors["LOW_GOLD_FACT_COVERAGE"] += 1
        if r["is_absent"] and not r["abstention_correct"]:
            errors["ABSENT_INFO_FABRICATION"] += 1
        if r["is_conflict"] and not r["conflict_correct"]:
            errors["CONFLICT_MISSED"] += 1
        if r["numerical_consistency_rate"] is not None and r["numerical_consistency_rate"] < 0.5:
            errors["NUMERICAL_ERROR"] += 1
        if r["fallback_count"] > 0:
            errors["PROVIDER_FAILURE"] += 1
        if r["outcome"] in ("NO_ANSWER", "ANSWERED_DEGRADED"):
            errors["INCOMPLETE_SYNTHESIS"] += 1

    print(f"\n  {'Error Type':<35} {'Count':<6} {'%':<6}")
    print("  " + "-" * 50)
    for err, count in sorted(errors.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {err:<35} {count:<6} {pct:.0f}%")

    # LLM call analysis
    print("\n" + "=" * 110)
    print("LLM CALL ANALYSIS")
    print("=" * 110)

    agg_by_type = defaultdict(lambda: {"count": 0, "total_ms": 0, "tokens": 0})
    for r in results:
        for ct, data in r["llm_by_type"].items():
            agg_by_type[ct]["count"] += data["count"]
            agg_by_type[ct]["total_ms"] += data["total_ms"]
            agg_by_type[ct]["tokens"] += data["total_tokens"]

    total_calls = sum(v["count"] for v in agg_by_type.values())
    total_ms_llm = sum(v["total_ms"] for v in agg_by_type.values())
    total_tok = sum(v["tokens"] for v in agg_by_type.values())

    print(f"\n  {'Call Type':<25} {'Count':<7} {'%':<6} {'Avg ms':<9} {'Tokens':<8}")
    print("  " + "-" * 55)
    for ct in ["query_analysis", "research_planning", "evidence_extraction", "synthesis", "verification"]:
        if ct in agg_by_type:
            v = agg_by_type[ct]
            pct = v["count"] / total_calls * 100 if total_calls else 0
            avg_ms = v["total_ms"] / v["count"] if v["count"] else 0
            print(f"  {ct:<25} {v['count']:<7} {pct:<5.1f}% {avg_ms:<9.0f} {v['tokens']:<8}")
    print(f"\n  {'TOTAL':<25} {total_calls:<7} {'100%':<6} {total_ms_llm/total_calls if total_calls else 0:<9.0f} {total_tok:<8}")

    print(f"\n  Avg tokens/query: {total_tok/len(results):.0f}")
    print(f"  Avg calls/query: {total_calls/len(results):.1f}")

    # Latency analysis
    print("\n" + "=" * 110)
    print("LATENCY ANALYSIS")
    print("=" * 110)

    avg_e2e = _mean([r["total_e2e_ms"] for r in results])
    avg_llm = _mean([r["llm_ms"] for r in results])
    avg_ret = _mean([r["retrieval"]["bm25_ms"] + r["retrieval"]["fusion_ms"] + r["retrieval"]["rerank_ms"]
                     for r in results])

    print(f"\n  Avg E2E latency:     {avg_e2e:.0f}ms")
    print(f"  Avg LLM latency:     {avg_llm:.0f}ms ({avg_llm/avg_e2e*100:.0f}% of E2E)")
    print(f"  Avg Retrieval latency: {avg_ret:.0f}ms ({avg_ret/avg_e2e*100:.0f}% of E2E)")
    print(f"  Cold start:          {cold_ms:.0f}ms")

    # Provider contamination
    print("\n" + "=" * 110)
    print("PROVIDER CONTAMINATION")
    print("=" * 110)

    print(f"\n  Clean runs:     {len(clean)}/{len(results)} ({len(clean)/len(results)*100:.0f}%)")
    print(f"  Contaminated:   {len(contaminated)}/{len(results)} ({len(contaminated)/len(results)*100:.0f}%)")

    if contaminated:
        fb_by_type = defaultdict(int)
        for r in contaminated:
            for fb in r.get("fallback_details", []):
                fb_by_type[fb["call_type"]] += 1
        print(f"\n  Fallbacks by call type:")
        for ct, count in sorted(fb_by_type.items(), key=lambda x: -x[1]):
            print(f"    {ct}: {count}")


if __name__ == "__main__":
    run_quality_baseline()
