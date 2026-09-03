"""Phase 07d mock-based control regression over the fixed 38-query set.

NO live API / no quota. Uses fault-injected scripted routers over the real
offline 38-query corpus, and reproduces the concrete paths called out in the
Phase 07c evaluation:

  R1 healthy scan       — every query through a scripted healthy provider with
                          real offline retrieval must land a truthful Outcome,
                          with grounded citations, at zero live cost.
  R2 budget-stop honesty— a healthy run that stops for BUDGET_EXHAUSTED (or any
                          control-flow stop) must be a SUCCESS, not a failure:
                          Outcome is orthogonal to StopReason (07c A1).
  R3 hard-degraded      — a 0-call provider-down run must be truthfully
                          NOT_FOUND / NO_ANSWER, never misread as success
                          (07c G1/I1/J1).
  R4 429 fast-fail      — a rate-limited primary must fail fast (single attempt)
                          and still answer via a healthy fallback model
                          (07c F1: PART D).
  R5 verification-fail  — a failing verification must annotate the grounded
                          answer as error and never downgrade a true success.

Results written to: benchmarks/eval_data/results/regression_07d.json
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile
from pathlib import Path

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.llm_gateway.providers.exceptions import RateLimitError
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.orchestration.graph import run_query
from app.reranking.reranker import NoOpReranker
from benchmarks.eval_data.run_eval import (
    build_corpus,
    load_plan,
    run_retrieval_analysis,
)


def _resp(content: str, model: str = "scripted-model") -> CompletionResponse:
    return CompletionResponse(content=content, model=model,
                              usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                              provider="scripted")


def _plan_payload(subquestions: list[str]) -> str:
    return json.dumps({"objective": subquestions[0], "entities": [], "time_window": None,
                       "subquestions": subquestions, "evidence_type": "factual",
                       "preferred_retrieval_methods": ["hybrid"], "required_sources": [],
                       "risk_level": "low", "token_budget": 6000, "iteration_budget": 2,
                       "stopping_condition": "stop once evidence suffices"})


def _assessment_payload() -> str:
    return json.dumps({"sufficient": True, "reasoning": "regression: sufficient",
                       "next_subquery": None})


def _fresh_handlers(verify_src: str = "ok"):
    """Deterministic scripted handlers. Synthesis cites [1] so citations resolve."""

    async def analysis(messages, **k):
        return _resp(json.dumps({"complexity": "moderate", "reasoning": "r",
                                 "suggested_subquestion_count": 1}))

    async def planning(messages, **k):
        query = k.get("query") or "the question"
        return _resp(_plan_payload([query]))

    async def assess(messages, **k):
        return _resp(_assessment_payload())

    async def synthesis(messages, **k):
        return _resp("The answer is grounded in the retrieved evidence [1].")

    async def verification(messages, **k):
        if verify_src == "fail":
            raise RuntimeError("verifier down")
        return _resp(json.dumps({"status": "supported", "confidence": 0.9,
                                 "reasoning": "ok", "contradictions": []}))

    return {
        "query_analysis": analysis,
        "research_planning": planning,
        "evidence_extraction": assess,
        "synthesis": synthesis,
        "verification": verification,
    }


class ScriptedRouter:
    def __init__(self, handlers: dict):
        self._handlers = handlers
        self.calls: list[str] = []

    async def complete(self, messages, *, call_type="general", **kwargs):
        self.calls.append(call_type)
        handler = self._handlers.get(call_type)
        if handler is None:
            raise RuntimeError(f"no scripted handler for {call_type}")
        return await handler(messages, call_type=call_type, **kwargs)

    async def aclose(self) -> None:
        pass


async def main() -> None:
    plan = load_plan()

    with tempfile.TemporaryDirectory(prefix="ph07d_reg_") as tmp:
        corpus = build_corpus(Path(tmp))
        reranker = NoOpReranker()
        retriever = corpus["retriever"]
        retrieval_quality = run_retrieval_analysis(plan, corpus)
        settings = Settings(verification_enabled=True, multimodel_call_ceiling=64)

        report = {"retrieval_quality": retrieval_quality, "healthy_scan": [],
                  "r2": {}, "r3": {}, "r4": {}, "r5": {}}

        # ---- R1 healthy scan over all 38 ----
        scan = []
        for it in plan["queries"]:
            router = ScriptedRouter(_fresh_handlers(verify_src="ok"))
            try:
                result = await run_query(query=it["query"], request_id=f"ph07d:{it['id']}",
                                         router=router, retriever=retriever, reranker=reranker,
                                         settings=settings)
                scan.append({
                    "id": it["id"], "class": it["class"],
                    "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
                    "stop_reason": result.stop_reason.value,
                    "calls": len(router.calls),
                    "citation_count": len(result.citations),
                    "verification_triggered": bool(result.verification and result.verification.triggered),
                    "verification_status": getattr(result.verification, "status", None),
                })
            except Exception as exc:  # noqa: BLE001
                scan.append({"id": it["id"], "class": it["class"], "error": repr(exc),
                             "outcome": "error", "stop_reason": "?", "calls": len(router.calls)})
        report["healthy_scan"] = scan

        # ---- R2 budget-stop honesty ----
        budget_stopped = [s for s in scan if s["stop_reason"] == "budget_exhausted"]
        report["r2"] = {
            "budget_stopped_count": len(budget_stopped),
            "budget_stopped_all_success": all(s["outcome"] == "answered" for s in budget_stopped),
        }

        # ---- R3 hard-degraded 0-call ----
        from app.llm_gateway.providers.exceptions import LLMProviderError

        def _dead(messages, **kk):
            raise LLMProviderError("PROVIDER_UNAVAILABLE", "provider unreachable",
                                   status_code=503, provider="scripted")

        dead = {k: _dead for k in ("query_analysis", "research_planning", "evidence_extraction", "synthesis", "verification")}
        dead_router = ScriptedRouter(dead)
        degraded = await run_query(query=plan["queries"][0]["query"], request_id="ph07d:degrade",
                                   router=dead_router, retriever=retriever, reranker=reranker,
                                   settings=Settings(verification_enabled=False))
        report["r3"] = {
            "outcome": degraded.outcome.value if hasattr(degraded.outcome, "value") else str(degraded.outcome),
            "calls": len(dead_router.calls),
            # Truthful hardening: a hard-degraded run must NOT present as a bare
            # success. It surfaces as ANSWERED_DEGRADED (raw-evidence fallback) or
            # NOT_FOUND/NO_ANSWER — never plain "answered" without a flag.
            "not_misread_as_bare_success": degraded.outcome.value != "answered",
        }

        # ---- R4 429 fast-fail + fallback ----
        r4 = await _r4_429_fault(corpus)
        report["r4"] = r4

        # ---- R5 verification failure annotates, doesn't downgrade ----
        fail_verif_settings = Settings(verification_enabled=True)
        fail_router = ScriptedRouter(_fresh_handlers(verify_src="fail"))
        first = plan["queries"][0]
        vr = await run_query(query=first["query"], request_id="ph07d:r5",
                             router=fail_router, retriever=retriever, reranker=reranker,
                             settings=fail_verif_settings)
        report["r5"] = {
            "outcome": vr.outcome.value if hasattr(vr.outcome, "value") else str(vr.outcome),
            "verification_status": getattr(vr.verification, "status", None),
            "verification_error_reported": bool(getattr(vr.verification, "error", None)),
            "answer_preserved": bool(vr.answer),
            "citations": len(vr.citations),
        }

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regression_07d.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _summarize(report)
    print(f"saved: {out_dir / 'regression_07d.json'}")


async def _r4_429_fault(corpus):
    """Primary raises 429 (rate limit) on planning; router must fail fast and
    fall back to a healthy model so the query still answers."""
    attempts = {"primary_planning": 0}
    reranker = NoOpReranker()

    class FailingRouter:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, *, call_type="general", **kwargs):
            self.calls.append(call_type)
            if call_type in ("research_planning", "query_analysis"):
                attempts["primary_planning"] += 1
                raise RateLimitError("rate limited", status_code=429, provider="primary")
            if call_type == "evidence_extraction":
                return _resp(_assessment_payload())
            if call_type == "synthesis":
                return _resp("Fallback answer grounded on [1].")
            if call_type == "verification":
                return _resp(json.dumps({"status": "supported", "confidence": 0.8,
                                         "reasoning": "ok", "contradictions": []}))
            return _resp("mock")

        async def aclose(self) -> None:
            pass

    settings = Settings(verification_enabled=True, multimodel_call_ceiling=32)
    router = FailingRouter()
    result = await run_query(query="Return the general answer with a citation.",
                             request_id="ph07d:r4", router=router,
                             retriever=corpus["retriever"], reranker=reranker,
                             settings=settings)
    return {
        "primary_attempts": attempts["primary_planning"],
        "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
        "answered": bool(result.answer),
        "calls": len(router.calls),
        "citation_count": len(result.citations),
    }


def _summarize(report: dict) -> None:
    scan = report["healthy_scan"]
    outcomes = {}
    for s in scan:
        outcomes[s["outcome"]] = outcomes.get(s["outcome"], 0) + 1
    verif_triggered = sum(1 for s in scan if s.get("verification_triggered"))
    grounded = sum(1 for s in scan if s.get("citation_count", 0) >= 1)
    print(f"R1 healthy scan ({len(scan)} queries): outcomes={outcomes} grounded={grounded}/{len(scan)} verification_triggered={verif_triggered}")
    print(f"R2 budget-stopped honesty: {report['r2']}")
    print(f"R3 hard-degraded: {report['r3']}")
    print(f"R4 429 fast-fail+fallback: {report['r4']}")
    print(f"R5 verification-fail: {report['r5']}")


if __name__ == "__main__":
    asyncio.run(main())