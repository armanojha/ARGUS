#!/usr/bin/env python3
"""Phase 37 Step 5: Controlled Ablation Study.

Runs the same benchmark queries with different pipeline configurations
by monkey-patching node functions to skip specific LLM calls.

Configurations tested:
  A. Full current pipeline (baseline)
  B. Skip query_analysis (deterministic passthrough)
  C. Skip research_planning (deterministic passthrough)
  D. Skip analyze + plan (deterministic passthrough)
  E. Skip verification (deterministic passthrough)
  F. Skip analyze + plan + verify

Usage:
    python -m benchmarks.phase37_ablation
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Benchmark queries ────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {"id": "ABL-Q01", "pattern": "simple_lookup", "query": "Where is Acme Corporation headquartered?"},
    {"id": "ABL-Q02", "pattern": "normal_qa", "query": "How does Acme Corporation generate revenue?"},
    {"id": "ABL-Q03", "pattern": "technical_explanation", "query": "How does the Atlas database achieve low latency?"},
    {"id": "ABL-Q04", "pattern": "multi_hop", "query": "Who manages the team that developed Atlas?"},
    {"id": "ABL-Q05", "pattern": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?"},
    {"id": "ABL-Q06", "pattern": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?"},
    {"id": "ABL-Q07", "pattern": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?"},
    {"id": "ABL-Q08", "pattern": "absent_info", "query": "What is Acme's market share in the European robotics market?"},
]


# ── Telemetry extraction ─────────────────────────────────────────────────────

def extract_call_details(telemetry_summary: dict | None) -> dict:
    if not telemetry_summary:
        return {"total_calls": 0, "total_llm_ms": 0, "total_tokens": 0,
                "by_type": {}, "fallbacks": 0, "is_clean": True}
    decisions = telemetry_summary.get("routing_decisions", [])
    by_type = defaultdict(lambda: {"count": 0, "total_ms": 0, "total_tokens": 0, "fallbacks": 0})
    total_ms = 0
    total_tokens = 0
    fallbacks = 0
    for d in decisions:
        ct = d.get("call_type", "unknown")
        lat = d.get("latency_ms", 0) or 0
        tt = d.get("total_tokens", 0) or 0
        by_type[ct]["count"] += 1
        by_type[ct]["total_ms"] += lat
        by_type[ct]["total_tokens"] += tt
        total_ms += lat
        total_tokens += tt
        if d.get("is_fallback"):
            fallbacks += 1
            by_type[ct]["fallbacks"] += 1
    return {
        "total_calls": len(decisions),
        "total_llm_ms": total_ms,
        "total_tokens": total_tokens,
        "by_type": dict(by_type),
        "fallbacks": fallbacks,
        "is_clean": fallbacks == 0,
    }


# ── Ablation configurations ─────────────────────────────────────────────────

CONFIGS = {
    "A_full": {
        "description": "Full current pipeline (baseline)",
        "skip_analyze": False,
        "skip_plan": False,
        "skip_verify": False,
    },
    "B_no_analyze": {
        "description": "Skip query_analysis",
        "skip_analyze": True,
        "skip_plan": False,
        "skip_verify": False,
    },
    "C_no_plan": {
        "description": "Skip research_planning",
        "skip_analyze": False,
        "skip_plan": True,
        "skip_verify": False,
    },
    "D_no_analyze_plan": {
        "description": "Skip analyze + plan",
        "skip_analyze": True,
        "skip_plan": True,
        "skip_verify": False,
    },
    "E_no_verify": {
        "description": "Skip verification",
        "skip_analyze": False,
        "skip_plan": False,
        "skip_verify": True,
    },
    "F_no_analyze_plan_verify": {
        "description": "Skip analyze + plan + verify",
        "skip_analyze": True,
        "skip_plan": True,
        "skip_verify": True,
    },
}


# ── Monkey-patches ───────────────────────────────────────────────────────────

_patches: list[tuple[Any, str, Any]] = []


def _patch(obj, attr, new_val):
    """Monkey-patch an attribute and save the original."""
    import types
    old_val = getattr(obj, attr)
    if isinstance(old_val, types.FunctionType) or isinstance(old_val, types.MethodType):
        setattr(obj, attr, new_val)
    else:
        setattr(obj, attr, new_val)
    _patches.append((obj, attr, old_val))


def _restore_all():
    """Restore all monkey-patches."""
    for obj, attr, old_val in _patches:
        setattr(obj, attr, old_val)
    _patches.clear()


def apply_ablation_patches(config):
    """Apply monkey-patches for the given ablation configuration."""
    _restore_all()  # Start clean

    skip_analyze = config.get("skip_analyze", False)
    skip_plan = config.get("skip_plan", False)
    skip_verify = config.get("skip_verify", False)

    if skip_analyze:
        from app.orchestration import nodes as nodes_mod
        from app.orchestration.models import QueryAnalysis, ComplexityLevel

        original_make_analyze = nodes_mod.make_analyze_node

        def patched_make_analyze(router, settings):
            original_factory = original_make_analyze(router, settings)
            async def patched_analyze(state):
                return {
                    "query_analysis": QueryAnalysis(
                        complexity=ComplexityLevel.MODERATE,
                        reasoning="Ablation: query analysis skipped (deterministic).",
                        suggested_subquestion_count=3,
                    ),
                }
            return patched_analyze

        _patch(nodes_mod, "make_analyze_node", patched_make_analyze)

    if skip_plan:
        from app.orchestration import nodes as nodes_mod
        from app.orchestration.models import ResearchPlan

        original_make_plan = nodes_mod.make_plan_node

        def patched_make_plan(router, settings):
            original_factory = original_make_plan(router, settings)
            async def patched_plan(state):
                query = state["query"]
                return {
                    "plan": ResearchPlan(
                        objective=query.strip(),
                        entities=[],
                        time_window=None,
                        subquestions=[query.strip()],
                        evidence_type="factual",
                        preferred_retrieval_methods=["hybrid"],
                        required_sources=[],
                        risk_level="low",
                    ),
                    "pending_subquestions": [query.strip()],
                    "max_iterations": 1,
                    "token_budget": 6000,
                }
            return patched_plan

        _patch(nodes_mod, "make_plan_node", patched_make_plan)

    if skip_verify:
        from app.orchestration import graph as graph_mod
        from app.orchestration.models import OrchestrationVerification

        original_verify = graph_mod._run_selective_verification

        async def patched_verify(*args, **kwargs):
            return OrchestrationVerification(
                triggered=False,
                skipped_reason="ablation_skip",
            )

        _patch(graph_mod, "_run_selective_verification", patched_verify)


# ── Run ablation ─────────────────────────────────────────────────────────────

async def _run_all_configs(retriever, reranker, settings):
    """Run all ablation configurations."""
    from app.orchestration.graph import run_query
    from app.llm_gateway.telemetry import start_run_telemetry, end_run_telemetry

    all_results = {}

    for config_name, config in CONFIGS.items():
        print(f"\n{'='*100}")
        print(f"CONFIG: {config_name} — {config['description']}")
        print(f"{'='*100}")

        apply_ablation_patches(config)

        results = []
        for qi in BENCHMARK_QUERIES:
            query = qi["query"]
            qid = qi["id"]

            t0 = time.perf_counter()
            try:
                start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=f"{config_name}_{qid}")
                result = await run_query(query, settings=settings, retriever=retriever, reranker=reranker)
                telemetry_summary = end_run_telemetry()
            except Exception as e:
                print(f"    {qid} FAILED: {e}")
                try:
                    end_run_telemetry()
                except Exception:
                    pass
                continue
            total_ms = (time.perf_counter() - t0) * 1000

            llm_data = extract_call_details(telemetry_summary)

            record = {
                "query_id": qid,
                "pattern": qi["pattern"],
                "total_e2e_ms": round(total_ms, 2),
                "total_llm_calls": llm_data["total_calls"],
                "total_llm_ms": llm_data["total_llm_ms"],
                "total_tokens": llm_data["total_tokens"],
                "llm_by_type": llm_data["by_type"],
                "fallback_count": llm_data["fallbacks"],
                "is_clean": llm_data["is_clean"],
                "iterations": getattr(result, 'iterations_used', 0) or 0,
                "stop_reason": str(getattr(result, 'stop_reason', '')) if getattr(result, 'stop_reason', None) else None,
                "evidence_count": len(getattr(result, 'citations', []) or []),
                "answer_length": len(getattr(result, 'answer', '') or ''),
                "outcome": str(getattr(result, 'outcome', 'unknown')),
                "verification_skipped": (
                    getattr(result, 'verification', None) is not None
                    and not getattr(result.verification, 'triggered', True)
                    if getattr(result, 'verification', None) else False
                ),
            }
            results.append(record)

            calls_str = ", ".join(f"{ct}={d['count']}" for ct, d in llm_data["by_type"].items())
            print(f"  {qid:<10} {qi['pattern']:<22} calls={llm_data['total_calls']:<3} "
                  f"tok={llm_data['total_tokens']:<5} llm={llm_data['total_llm_ms']:>6.0f}ms "
                  f"e2e={total_ms:>6.0f}ms  [{calls_str}]")

        all_results[config_name] = results

        # Aggregate
        avg_calls = _mean([r["total_llm_calls"] for r in results])
        avg_tokens = _mean([r["total_tokens"] for r in results])
        avg_llm = _mean([r["total_llm_ms"] for r in results])
        avg_e2e = _mean([r["total_e2e_ms"] for r in results])
        print(f"\n  AVG: calls={avg_calls:.1f} tokens={avg_tokens:.0f} llm={avg_llm:.0f}ms e2e={avg_e2e:.0f}ms")

    _restore_all()  # Restore all patches
    return all_results


def _mean(data):
    import numpy as np
    return float(np.mean(data)) if data else 0.0


def run_ablation():
    print("=" * 100)
    print("ARGUS Phase 37 Step 5: Controlled Ablation Study")
    print("=" * 100)

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

    import asyncio
    all_results = asyncio.run(_run_all_configs(retriever, reranker, settings))

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "phase37_ablation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON REPORT
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("ABLATION COMPARISON (vs baseline A_full)")
    print("=" * 100)

    baseline = all_results.get("A_full", [])
    b_calls = _mean([r["total_llm_calls"] for r in baseline]) if baseline else 0
    b_tokens = _mean([r["total_tokens"] for r in baseline]) if baseline else 0
    b_llm = _mean([r["total_llm_ms"] for r in baseline]) if baseline else 0
    b_e2e = _mean([r["total_e2e_ms"] for r in baseline]) if baseline else 0

    print(f"\n  {'Config':<35} {'Calls':<8} {'Δ':<6} {'Tokens':<8} {'Δ%':<8} "
          f"{'LLM ms':<9} {'Δ%':<8} {'E2E ms':<9} {'Δ%':<8}")
    print("  " + "-" * 100)

    for config_name, results in all_results.items():
        avg_calls = _mean([r["total_llm_calls"] for r in results])
        avg_tokens = _mean([r["total_tokens"] for r in results])
        avg_llm = _mean([r["total_llm_ms"] for r in results])
        avg_e2e = _mean([r["total_e2e_ms"] for r in results])

        dc = avg_calls - b_calls
        dt = ((avg_tokens - b_tokens) / b_tokens * 100) if b_tokens else 0
        dl = ((avg_llm - b_llm) / b_llm * 100) if b_llm else 0
        de = ((avg_e2e - b_e2e) / b_e2e * 100) if b_e2e else 0

        desc = CONFIGS[config_name]["description"][:35]
        print(f"  {desc:<35} {avg_calls:<8.1f} {dc:>+4.1f}  {avg_tokens:<8.0f} {dt:>+6.1f}% "
              f"{avg_llm:<9.0f} {dl:>+6.1f}%  {avg_e2e:<9.0f} {de:>+6.1f}%")

    # Per-pattern comparison
    print("\n" + "=" * 100)
    print("PER-PATTERN ABLATION")
    print("=" * 100)

    patterns = ["simple_lookup", "normal_qa", "technical_explanation", "multi_hop",
                "numerical", "conflict", "complex_research", "absent_info"]

    for pattern in patterns:
        print(f"\n  {pattern}:")
        print(f"    {'Config':<35} {'Calls':<6} {'Tokens':<8} {'LLM ms':<9} {'E2E ms':<9} {'Answer len'}")
        print("    " + "-" * 80)

        for config_name, results in all_results.items():
            pr = [r for r in results if r["pattern"] == pattern]
            if not pr:
                continue
            ac = _mean([r["total_llm_calls"] for r in pr])
            at = _mean([r["total_tokens"] for r in pr])
            al = _mean([r["total_llm_ms"] for r in pr])
            ae = _mean([r["total_e2e_ms"] for r in pr])
            aa = _mean([r["answer_length"] for r in pr])
            desc = CONFIGS[config_name]["description"][:35]
            print(f"    {desc:<35} {ac:<6.1f} {at:<8.0f} {al:<9.0f} {ae:<9.0f} {aa:.0f}")

    # Verification skip analysis
    print("\n" + "=" * 100)
    print("VERIFICATION SKIP ANALYSIS")
    print("=" * 100)

    for config_name, results in all_results.items():
        verif_skipped = sum(1 for r in results if r.get("verification_skipped", False))
        desc = CONFIGS[config_name]["description"][:35]
        print(f"  {desc:<35} verif_skipped={verif_skipped}/{len(results)}")


if __name__ == "__main__":
    run_ablation()
