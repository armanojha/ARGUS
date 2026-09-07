#!/usr/bin/env python3
"""Phase 37 Step 2: Measure LLM Calls Per Query.

Runs the full production pipeline with telemetry and captures per-query:
- total_llm_calls, calls by type, tokens, latency
- fast_path classification, iterations, stop_reason
- fallback details per call

Usage:
    python -m benchmarks.phase37_llm_call_audit
"""
from __future__ import annotations

import dataclasses
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


# ── Benchmark queries ────────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {"id": "P37-Q01", "pattern": "simple_lookup", "query": "Where is Acme Corporation headquartered?",
     "gold_facts": ["New York City"], "supporting_docs": ["doc-a"]},
    {"id": "P37-Q02", "pattern": "normal_qa", "query": "How does Acme Corporation generate revenue?",
     "gold_facts": ["revenue", "products", "services"], "supporting_docs": ["doc-a"]},
    {"id": "P37-Q03", "pattern": "technical_explanation", "query": "How does the Atlas database achieve low latency?",
     "gold_facts": ["columnar", "storage", "analytics"], "supporting_docs": ["doc-b"]},
    {"id": "P37-Q04", "pattern": "multi_hop", "query": "Who manages the team that developed Atlas?",
     "gold_facts": ["team", "Atlas", "manager"], "supporting_docs": ["doc-b", "doc-d"]},
    {"id": "P37-Q05", "pattern": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?",
     "gold_facts": ["revenue", "2022", "2023", "growth"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P37-Q06", "pattern": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?",
     "gold_facts": ["revenue", "2023", "conflicting"], "supporting_docs": ["doc-e", "doc-f"]},
    {"id": "P37-Q07", "pattern": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?",
     "gold_facts": ["competitive", "advantage", "analytics"], "supporting_docs": ["doc-a", "doc-b"]},
    {"id": "P37-Q08", "pattern": "absent_info", "query": "What is Acme's market share in the European robotics market?",
     "gold_facts": [], "supporting_docs": []},
]


# ── Telemetry extraction ─────────────────────────────────────────────────────

def extract_call_details(telemetry_summary: dict | None) -> dict:
    """Extract per-call-type details from telemetry summary."""
    if not telemetry_summary:
        return {
            "total_calls": 0, "total_llm_ms": 0, "total_tokens": 0,
            "by_type": {}, "fallbacks": 0, "rate_limits": 0,
            "is_clean": True, "fallback_details": [],
            "providers_used": [], "models_used": [],
        }

    decisions = telemetry_summary.get("routing_decisions", [])
    by_type: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_ms": 0, "latencies_ms": [],
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "fallbacks": 0, "providers": set(), "models": set(),
    })
    providers = set()
    models = set()
    fallbacks = 0
    rate_limits = 0
    total_ms = 0
    total_tokens = 0
    fallback_details = []

    for d in decisions:
        ct = d.get("call_type", "unknown")
        lat = d.get("latency_ms", 0) or 0
        pt = d.get("prompt_tokens", 0) or 0
        ct_tokens = d.get("completion_tokens", 0) or 0
        tt = d.get("total_tokens", 0) or (pt + ct_tokens)

        by_type[ct]["count"] += 1
        by_type[ct]["total_ms"] += lat
        by_type[ct]["latencies_ms"].append(lat)
        by_type[ct]["input_tokens"] += pt
        by_type[ct]["output_tokens"] += ct_tokens
        by_type[ct]["total_tokens"] += tt
        by_type[ct]["providers"].add(d.get("provider", "unknown"))
        by_type[ct]["models"].add(d.get("model", "unknown"))

        total_ms += lat
        total_tokens += tt
        providers.add(d.get("provider", "unknown"))
        models.add(d.get("model", "unknown"))

        if d.get("is_fallback"):
            fallbacks += 1
            by_type[ct]["fallbacks"] += 1
            fallback_details.append({
                "call_type": ct,
                "provider": d.get("provider"),
                "model": d.get("model"),
                "reason": d.get("fallback_reason"),
            })
        if d.get("error_class") and "rate" in str(d.get("error_class", "")).lower():
            rate_limits += 1

    # Convert sets to lists for JSON
    by_type_serializable = {}
    for ct, data in by_type.items():
        by_type_serializable[ct] = {
            "count": data["count"],
            "total_ms": data["total_ms"],
            "latencies_ms": data["latencies_ms"],
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "total_tokens": data["total_tokens"],
            "fallbacks": data["fallbacks"],
            "providers": sorted(data["providers"]),
            "models": sorted(data["models"]),
        }

    return {
        "total_calls": len(decisions),
        "total_llm_ms": total_ms,
        "total_tokens": total_tokens,
        "by_type": by_type_serializable,
        "fallbacks": fallbacks,
        "rate_limits": rate_limits,
        "is_clean": fallbacks == 0,
        "fallback_details": fallback_details,
        "providers_used": sorted(providers),
        "models_used": sorted(models),
    }


# ── Main benchmark ───────────────────────────────────────────────────────────

async def _run_all_queries(retriever, reranker, settings):
    """Run all benchmark queries in a single event loop."""
    from app.orchestration.graph import run_query, _is_simple_query
    from app.llm_gateway.telemetry import start_run_telemetry, end_run_telemetry
    from app.llm_gateway.routing.complexity import classify_complexity

    results = []

    # Warm-up (2 calls to stabilize providers)
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

    # Cold start measurement
    print("\n[2/3] Cold start measurement...")
    cold_start_t0 = time.perf_counter()
    try:
        start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id="cold_start")
        await run_query("cold start test", settings=settings, retriever=retriever, reranker=reranker)
        end_run_telemetry()
    except Exception:
        try:
            end_run_telemetry()
        except Exception:
            pass
    cold_start_ms = (time.perf_counter() - cold_start_t0) * 1000
    print(f"  Cold start: {cold_start_ms:.0f}ms")

    # Benchmark
    print(f"\n[3/3] Running {len(BENCHMARK_QUERIES)} benchmark queries...")

    for i, qi in enumerate(BENCHMARK_QUERIES, 1):
        query = qi["query"]
        qid = qi["id"]

        # Classify complexity
        tier = classify_complexity(query).value
        is_simple = _is_simple_query(query)

        t0 = time.perf_counter()
        try:
            start_run_telemetry(call_ceiling=16, call_ceiling_warn=12, run_id=qid)
            result = await run_query(query, settings=settings, retriever=retriever, reranker=reranker)
            telemetry_summary = end_run_telemetry()
        except Exception as e:
            print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} FAILED: {e}")
            try:
                end_run_telemetry()
            except Exception:
                pass
            continue
        total_ms = (time.perf_counter() - t0) * 1000

        llm_data = extract_call_details(telemetry_summary)

        # Build result record
        record = {
            "query_id": qid,
            "pattern": qi["pattern"],
            "query": query,
            "complexity_tier": tier,
            "fast_path": is_simple,
            # E2E timing
            "total_e2e_ms": round(total_ms, 2),
            # LLM breakdown
            "total_llm_calls": llm_data["total_calls"],
            "total_llm_ms": llm_data["total_llm_ms"],
            "total_tokens": llm_data["total_tokens"],
            "llm_by_type": llm_data["by_type"],
            # Fallback info
            "fallback_count": llm_data["fallbacks"],
            "rate_limit_events": llm_data["rate_limits"],
            "is_clean": llm_data["is_clean"],
            "fallback_details": llm_data["fallback_details"],
            "providers_used": llm_data["providers_used"],
            "models_used": llm_data["models_used"],
            # Pipeline state
            "iterations": getattr(result, 'iterations_used', 0) or 0,
            "stop_reason": str(getattr(result, 'stop_reason', '')) if getattr(result, 'stop_reason', None) else None,
            "evidence_count": len(getattr(result, 'citations', []) or []),
            "answer_length": len(getattr(result, 'answer', '') or ''),
            "outcome": str(getattr(result, 'outcome', 'unknown')),
            "warnings": getattr(result, 'warnings', []) or [],
            "verification": {
                "triggered": getattr(result, 'verification', None) is not None and getattr(result.verification, 'triggered', False) if getattr(result, 'verification', None) else False,
                "skipped_reason": getattr(result.verification, 'skipped_reason', None) if getattr(result, 'verification', None) else None,
            },
        }

        results.append(record)

        # Per-query summary
        calls_by_type = ", ".join(f"{ct}={d['count']}" for ct, d in llm_data["by_type"].items())
        status = "CLEAN" if llm_data["is_clean"] else f"CONTAMINATED ({llm_data['fallbacks']}fb)"
        print(f"  [{i:>2}/{len(BENCHMARK_QUERIES)}] {qid:<10} {qi['pattern']:<22} "
              f"tier={tier:<10} calls={llm_data['total_calls']:<3} "
              f"llm={llm_data['total_llm_ms']:>6.0f}ms  "
              f"tok={llm_data['total_tokens']:>5}  "
              f"total={total_ms:>6.0f}ms  {status}")

    return results, cold_start_ms


def run_call_audit():
    print("=" * 100)
    print("ARGUS Phase 37 Step 2: LLM Call Audit Per Query")
    print("=" * 100)

    # Build store and retriever
    print("\n[Setup] Building benchmark corpus & store...")
    t0 = time.monotonic()
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    print("\n[Setup] Building retrieval indexes...")
    t0 = time.monotonic()
    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    from app.reranking.reranker import Reranker, NoOpReranker
    try:
        reranker = Reranker()
    except Exception:
        reranker = NoOpReranker()

    from app.config import Settings
    settings = Settings()

    import asyncio
    results, cold_start_ms = asyncio.run(
        _run_all_queries(retriever, reranker, settings)
    )

    if not results:
        print("\nNo successful results. Exiting.")
        return

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "phase37_llm_call_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cold_start_ms": cold_start_ms, "queries": results}, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════

    clean = [r for r in results if r["is_clean"]]
    contaminated = [r for r in results if not r["is_clean"]]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n  Total queries:      {len(results)}")
    print(f"  Clean (no fallback): {len(clean)}")
    print(f"  Contaminated:       {len(contaminated)}")
    print(f"  Cold start:         {cold_start_ms:.0f}ms")

    # Calls per query
    print("\n" + "=" * 100)
    print("LLM CALLS PER QUERY")
    print("=" * 100)
    print(f"\n  {'Query':<10} {'Pattern':<22} {'Tier':<10} {'Fast?':<6} {'Calls':<6} "
          f"{'Analysis':<9} {'Plan':<6} {'Assess':<7} {'Synth':<6} {'Verif':<6} "
          f"{'Tokens':<7} {'LLM ms':<9} {'E2E ms':<9} {'Clean'}")
    print("  " + "-" * 140)

    for r in results:
        bt = r["llm_by_type"]
        fast = "Y" if r["fast_path"] else "N"
        print(f"  {r['query_id']:<10} {r['pattern']:<22} {r['complexity_tier']:<10} {fast:<6} "
              f"{r['total_llm_calls']:<6} "
              f"{bt.get('query_analysis', {}).get('count', 0):<9} "
              f"{bt.get('research_planning', {}).get('count', 0):<6} "
              f"{bt.get('evidence_extraction', {}).get('count', 0):<7} "
              f"{bt.get('synthesis', {}).get('count', 0):<6} "
              f"{bt.get('verification', {}).get('count', 0):<6} "
              f"{r['total_tokens']:<7} "
              f"{r['total_llm_ms']:<9.0f} "
              f"{r['total_e2e_ms']:<9.0f} "
              f"{'Y' if r['is_clean'] else 'N'}")

    # Aggregate by call type
    print("\n" + "=" * 100)
    print("AGGREGATE CALL TYPE BREAKDOWN (all queries)")
    print("=" * 100)

    agg_by_type: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_ms": 0, "tokens": 0, "fallbacks": 0, "queries": 0,
    })
    for r in results:
        for ct, data in r["llm_by_type"].items():
            agg_by_type[ct]["count"] += data["count"]
            agg_by_type[ct]["total_ms"] += data["total_ms"]
            agg_by_type[ct]["tokens"] += data["total_tokens"]
            agg_by_type[ct]["fallbacks"] += data["fallbacks"]
            agg_by_type[ct]["queries"] += 1

    total_calls_all = sum(v["count"] for v in agg_by_type.values())
    total_ms_all = sum(v["total_ms"] for v in agg_by_type.values())
    total_tok_all = sum(v["tokens"] for v in agg_by_type.values())

    print(f"\n  {'Call Type':<25} {'Count':<7} {'%':<6} {'Avg ms':<9} {'Total ms':<10} "
          f"{'Tokens':<8} {'Fallbacks':<10} {'Queries'}")
    print("  " + "-" * 90)

    for ct in ["query_analysis", "research_planning", "evidence_extraction", "synthesis", "verification"]:
        if ct in agg_by_type:
            v = agg_by_type[ct]
            pct = (v["count"] / total_calls_all * 100) if total_calls_all > 0 else 0
            avg_ms = v["total_ms"] / v["count"] if v["count"] > 0 else 0
            print(f"  {ct:<25} {v['count']:<7} {pct:<5.1f}% {avg_ms:<9.0f} {v['total_ms']:<10.0f} "
                  f"{v['tokens']:<8} {v['fallbacks']:<10} {v['queries']}")

    print(f"\n  {'TOTAL':<25} {total_calls_all:<7} {'100%':<6} "
          f"{total_ms_all/total_calls_all if total_calls_all else 0:<9.0f} {total_ms_all:<10.0f} "
          f"{total_tok_all:<8}")

    # Fast-path analysis
    print("\n" + "=" * 100)
    print("FAST-PATH CLASSIFICATION")
    print("=" * 100)

    fast_queries = [r for r in results if r["fast_path"]]
    non_fast = [r for r in results if not r["fast_path"]]

    print(f"\n  Fast-path queries:  {len(fast_queries)}")
    for r in fast_queries:
        print(f"    {r['query_id']}: {r['query'][:60]}...")

    print(f"\n  Non-fast queries:   {len(non_fast)}")
    for r in non_fast:
        print(f"    {r['query_id']}: {r['complexity_tier']:<10} {r['query'][:60]}...")

    # Per-pattern analysis
    print("\n" + "=" * 100)
    print("PER-PATTERN ANALYSIS")
    print("=" * 100)

    patterns: dict[str, list] = defaultdict(list)
    for r in results:
        patterns[r["pattern"]].append(r)

    for pattern, prs in sorted(patterns.items()):
        avg_calls = sum(r["total_llm_calls"] for r in prs) / len(prs)
        avg_tokens = sum(r["total_tokens"] for r in prs) / len(prs)
        avg_llm_ms = sum(r["total_llm_ms"] for r in prs) / len(prs)
        avg_e2e_ms = sum(r["total_e2e_ms"] for r in prs) / len(prs)
        clean_count = sum(1 for r in prs if r["is_clean"])

        print(f"\n  {pattern} (n={len(prs)}, clean={clean_count}):")
        for r in prs:
            print(f"    {r['query_id']}: calls={r['total_llm_calls']} tokens={r['total_tokens']} "
                  f"llm={r['total_llm_ms']:.0f}ms e2e={r['total_e2e_ms']:.0f}ms")
        print(f"    AVG: calls={avg_calls:.1f} tokens={avg_tokens:.0f} "
              f"llm={avg_llm_ms:.0f}ms e2e={avg_e2e_ms:.0f}ms")

    # Verification skip analysis
    print("\n" + "=" * 100)
    print("VERIFICATION SKIP ANALYSIS")
    print("=" * 100)

    verif_triggered = [r for r in results if r["verification"]["triggered"]]
    verif_skipped = [r for r in results if not r["verification"]["triggered"]]
    verif_skip_reasons = defaultdict(int)
    for r in verif_skipped:
        reason = r["verification"].get("skipped_reason") or "unknown"
        verif_skip_reasons[reason] += 1

    print(f"\n  Verification triggered: {len(verif_triggered)}")
    print(f"  Verification skipped:   {len(verif_skipped)}")
    for reason, count in sorted(verif_skip_reasons.items()):
        print(f"    {reason}: {count}")

    # Fallback analysis
    if contaminated:
        print("\n" + "=" * 100)
        print("FALLBACK ANALYSIS")
        print("=" * 100)

        all_fallbacks = []
        for r in contaminated:
            all_fallbacks.extend(r["fallback_details"])

        fb_by_type: dict[str, int] = defaultdict(int)
        fb_by_provider: dict[str, int] = defaultdict(int)
        for fb in all_fallbacks:
            fb_by_type[fb["call_type"]] += 1
            fb_by_provider[fb["provider"]] += 1

        print(f"\n  Fallbacks by call type:")
        for ct, count in sorted(fb_by_type.items(), key=lambda x: -x[1]):
            print(f"    {ct}: {count}")

        print(f"\n  Fallbacks by provider (target):")
        for prov, count in sorted(fb_by_provider.items(), key=lambda x: -x[1]):
            print(f"    {prov}: {count}")


if __name__ == "__main__":
    run_call_audit()
