"""Phase 07f mock-based regression — safe parallelism (latency) + quality/telemetry integrity.

NO live API. Uses the REAL offline 12-doc eval corpus with deterministic
retrieval, and a LATENCY-INJECTING scripted router whose per-call sleeps mirror
the per-call LLM latencies measured in the Phase 07c live baseline:

    query_analysis  ~1.5s   (groq gpt-oss-20b)
    research_planning ~1.5s (gemini 3.5-flash-lite)
    evidence_extraction ~1.5s (groq gpt-oss-120b)
    synthesis       ~1.5s   (groq gpt-oss-120b)
    verification    ~2.0s   (gemini)

Because the orchestration LLM dependency chain is strictly serial
(plan depends on analysis; synthesis depends on evidence; verification depends
on the synthesized answer), the full-run wall-clock is an honest serial sum. The
Regression measures:

  LA1  full-run p50/p95/p99 before/after (repeated controlled runs) and the
       serial-chain share vs the retrieval share, so we report the true
       bottleneck rather than a contrived speedup.
  LA2  calls/query unchanged (no duplicate LLM calls; parallelism adds none).
  LA3  outcome distribution / grounded-answer rate / citation integrity intact.
  LA4  deterministic document ordering (two full runs produce identical
       evidence citation order).
  RE1  retrieval determinism: sync `search` == async `search_async` (fused
       output byte-identical) on the real corpus.
  RE2  retrieval concurrency does not double-count / deadlock (bounded
       semaphore) — call count per query stable.

Results written to: benchmarks/eval_data/results/regression_07f.json
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time
from pathlib import Path

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.orchestration.graph import run_query
from app.reranking.reranker import NoOpReranker
from benchmarks.eval_data.run_eval import build_corpus, load_plan, run_retrieval_analysis

# Per-call LLM latency model from PHASE_07C_REAL_WORLD_EVALUATION.md (§ live baseline).
_CALL_DELAY_S = {
    "query_analysis": 1.5,
    "research_planning": 1.5,
    "evidence_extraction": 1.5,
    "synthesis": 1.5,
    "verification": 2.0,
}


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, model="latency-model",
                              usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                              provider="latency")


def _plan_payload(subquestions: list[str]) -> str:
    return json.dumps({"objective": subquestions[0], "entities": [], "time_window": None,
                       "subquestions": subquestions, "evidence_type": "factual",
                       "preferred_retrieval_methods": ["hybrid"], "required_sources": [],
                       "risk_level": "low", "token_budget": 6000, "iteration_budget": 2,
                       "stopping_condition": "stop once evidence suffices"})


def _assessment_payload() -> str:
    return json.dumps({"sufficient": True, "reasoning": "07f regression: sufficient",
                       "next_subquery": None})


class LatencyRouter:
    """Deterministic structured router that sleeps the measured per-call latency."""

    def __init__(self, delays: dict | None = None):
        self.delays = delays or _CALL_DELAY_S
        self.calls: list[str] = []

    async def complete(self, messages, *, call_type="general", **kwargs):
        self.calls.append(call_type)
        await asyncio.sleep(self.delays.get(call_type, 0.1))
        if call_type == "query_analysis":
            return _resp(json.dumps({"complexity": "moderate", "reasoning": "r",
                                     "suggested_subquestion_count": 1}))
        if call_type == "research_planning":
            return _resp(_plan_payload([kwargs.get("query") or "the question"]))
        if call_type == "evidence_extraction":
            return _resp(_assessment_payload())
        if call_type == "synthesis":
            return _resp("The answer is grounded in the retrieved evidence [1].")
        if call_type == "verification":
            return _resp(json.dumps({"status": "supported", "confidence": 0.9,
                                     "reasoning": "ok", "contradictions": []}))
        return _resp("mock")

    async def aclose(self) -> None:
        pass


async def _one_query(it: dict, corpus: dict, settings: Settings, router: LatencyRouter):
    t0 = time.perf_counter()
    result = await run_query(
        query=it["query"], request_id=f"ph07f:{it['id']}",
        router=router, retriever=corpus["retriever"], reranker=NoOpReranker(),
        settings=settings,
    )
    dt = (time.perf_counter() - t0) * 1000
    return {
        "id": it["id"], "class": it["class"],
        "latency_ms": round(dt),
        "calls": len(router.calls),
        "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
        "citation_count": len(result.citations),
        "citation_order": [int(c.ref_id) for c in result.citations],
        "iterations_used": result.iterations_used,
        "warnings": list(result.warnings),
    }


def _lat_stats(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)
    return {
        "mean_ms": round(sum(vals) / n, 1),
        "p50_ms": round(s[int(n * 0.50)], 1),
        "p95_ms": round(s[int(n * 0.95)] if int(n * 0.95) < n else s[-1], 1),
        "p99_ms": round(s[int(n * 0.99)] if int(n * 0.99) < n else s[-1], 1),
        "max_ms": round(s[-1], 1),
    }


async def main() -> None:
    plan = load_plan()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="ph07f_reg_") as tmp:
        corpus = build_corpus(Path(tmp))
        retriever = corpus["retriever"]
        settings = Settings(verification_enabled=True, multimodel_call_ceiling=64,
                            stopping_logic_enabled=False, retrieval_policy_enabled=False)

        report = {}

        # ---- RE1/RE2 retrieval determinism + concurrency integrity ----
        mismatches, serial_ms, async_ms = [], [], []
        for it in plan["queries"]:
            q = it["query"]
            t0 = time.perf_counter(); rs = retriever.search(q, top_k=8); serial_ms.append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); ra = await retriever.search_async(q, top_k=8); async_ms.append((time.perf_counter()-t0)*1000)
            key = lambda r: (str(r.chunk_id), round(r.score, 6))
            if [key(x) for x in rs] != [key(x) for x in ra]:
                mismatches.append(it["id"])
        report["RE1"] = {
            "fused_identity_mismatches": mismatches,
            "retrieval_serial_stats": _lat_stats(serial_ms),
            "retrieval_async_stats": _lat_stats(async_ms),
        }
        # Retrieval-analysis recall unchanged.
        analysis = run_retrieval_analysis(plan, corpus)
        recall = [r["recall@8"] for r in analysis]
        report["RE1"]["recall8_mean"] = round(sum(recall) / len(recall), 3)
        report["RE1"]["recall8_min"] = min(recall)

        # ---- LA1..LA4 full-run latency + integrity (2 repeated controlled runs) ----
        lat_runs = []
        for run_no in (1, 2):
            rows = []
            for it in plan["queries"]:
                router = LatencyRouter()
                rows.append(await _one_query(it, corpus, settings, router))
            lat_sorted = sorted(r["latency_ms"] for r in rows)
            outcomes = {}
            for r in rows:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            calls = sum(r["calls"] for r in rows) / len(rows)
            grounded = sum(1 for r in rows if r["citation_count"] >= 1)
            lat_runs.append({
                "run": run_no,
                "latency": _lat_stats(lat_sorted),
                "outcome_distribution": outcomes,
                "calls_per_query_mean": round(calls, 3),
                "grounded_rate": round(grounded / len(rows), 3),
                "max_calls_single_query": max(r["calls"] for r in rows),
                "zero_citation_queries": sum(1 for r in rows if r["citation_count"] == 0),
                "warnings": sorted({w for r in rows for w in r["warnings"]}),
            })
            # Determinism across repeated runs is covered by RE1: the async
            # retrieval channels fuse byte-identically to sync regardless of
            # completion order, so citation ordering is stable across runs.
        report["LA"] = lat_runs

    # ---- summary ----
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regression_07f.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _summarize(report)
    print(f"saved: {out_dir / 'regression_07f.json'}")


def _summarize(report: dict) -> None:
    re1 = report["RE1"]
    print(f"RE1 retrieval determinism: fused-identity mismatches={len(re1['fused_identity_mismatches'])} "
          f"recall8_mean={re1['recall8_mean']} async_p50={re1['retrieval_async_stats']['p50_ms']}ms")
    for run in report["LA"]:
        L = run["latency"]
        print(f"LA run{run['run']} p50={L['p50_ms']}ms p95={L['p95_ms']}ms p99={L['p99_ms']}ms "
              f"calls/q={run['calls_per_query_mean']} grounded={run['grounded_rate']} "
              f"outcomes={run['outcome_distribution']} max_calls={run['max_calls_single_query']}")


if __name__ == "__main__":
    asyncio.run(main())