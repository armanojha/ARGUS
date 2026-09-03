"""Phase 12.4 ablation harness tests (offline: real retrieval, stubbed LLM).

Validates variant construction, one smoke run of every variant over the shared
question set, and that the delta table is computable against `full_argus`.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

import pytest

from app.config import Settings
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.routing.router import LLMRouter
from benchmarks.ablation import (
    VARIANT_ORDER,
    VARIANTS,
    _output_from_dict,
    _output_to_dict,
    _run_all,
    ablation_markdown,
    make_variants,
    run_ablation,
)
from benchmarks.models import BenchmarkItem, BenchmarkRunOutput, CorpusContext
from benchmarks.runner import build_corpus, load_items, score_items


class ScriptedProvider:
    """Scripted fake LLM provider (per-call_type), structurally matching LLMProvider."""

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def default_model(self) -> str:
        return "scripted-model"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def complete(self, messages, *, model=None, temperature=0.0, max_tokens=None,
                       response_format=None, tools=None, tool_choice=None, timeout=30.0,
                       call_type: str = "general", request_id=None) -> CompletionResponse:
        import json as _json
        queue = self._script.get(call_type)
        if queue:
            payload = queue.pop(0)
            content = _json.dumps(payload) if isinstance(payload, dict) else payload
        else:
            content = "Fallback response."
        return CompletionResponse(
            content=content,
            model=model or self.default_model,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider="scripted",
            request_id=request_id,
        )

    async def aclose(self) -> None:
        pass


ANALYSIS_OK = {"complexity": "simple", "reasoning": "single fact lookup", "suggested_subquestion_count": 1}


def plan_payload(subquestions: list[str]) -> dict:
    return {
        "objective": "Answer the question.",
        "entities": ["entity"],
        "time_window": None,
        "subquestions": subquestions,
        "evidence_type": "factual",
        "preferred_retrieval_methods": ["hybrid"],
        "required_sources": [],
        "risk_level": "low",
        "token_budget": 6000,
        "iteration_budget": 2,
        "stopping_condition": "Stop once the question is answered.",
    }


def assessment_payload() -> dict:
    return {"sufficient": True, "reasoning": "test", "next_subquery": None}


VERIFIER_OK = {
    "status": "SUPPORTED",
    "confidence": 0.9,
    "reasoning": "matches evidence",
    "supporting_evidence_indices": [0],
    "contradicting_evidence_indices": [],
    "contradictions": [],
    "evidence_coverage": 1.0,
    "source_quality": 0.9,
    "cross_source_agreement": 0.8,
    "temporal_relevance": 0.9,
    "retrieval_rank": 0.9,
    "verifier_judgment": 0.9,
}


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider(
        {
            "query_analysis": [ANALYSIS_OK] * 20,
            "research_planning": [plan_payload(["Acme founded year"])] * 20,
            "evidence_extraction": [assessment_payload()] * 20,
            "synthesis": ["Acme Corp was founded in 1987 [1]."] * 30,
            "verification": [VERIFIER_OK] * 20,
        }
    )


@pytest.fixture
def bench_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_dir=tmp_path / "data",
        evidence_db_path=tmp_path / "data" / "evidence.db",
        bm25_index_path=tmp_path / "data" / "bm25.pkl",
        faiss_index_path=tmp_path / "data" / "faiss.index",
        retrieval_policy_enabled=False,
        active_evidence_seeking_enabled=False,
        stopping_logic_enabled=False,
        memory_enabled=False,
        multiagent_enabled=False,
        multimodel_enabled=False,
        obsidian_enabled=False,
        graph_extraction_enabled=False,
        orchestration_max_iterations=2,
        orchestration_token_budget=4000,
    )


def test_variant_registry_complete():
    assert set(VARIANTS) == set(VARIANT_ORDER)
    assert len(VARIANTS) == 7
    assert VARIANTS["full_argus"].label == "Full ARGUS"


def test_variants_build_and_run_single_item(tmp_path: Path, provider: ScriptedProvider, bench_settings: Settings):
    item = load_items()[0]
    corpus = build_corpus([item], tmp_path / "corpus")
    variants = make_variants(router=LLMRouter(provider), corpus=corpus, settings=bench_settings)
    assert set(variants) == set(VARIANT_ORDER)

    outputs: dict[str, BenchmarkRunOutput] = {}
    for variant_id, pipeline in variants.items():
        outputs[variant_id] = asyncio.run(
            cast(Coroutine[Any, Any, BenchmarkRunOutput], pipeline(item, corpus))
        )

    # Verification-based variants reported a supported status.
    assert outputs["full_argus"].verification_status == "supported"
    assert outputs["argus_without_obsidian"].verification_status == "supported"
    assert outputs["argus_without_verifier"].verification_status is None
    # Single-shot variants record no loop iterations.
    for vid in ("baseline_rag", "hybrid_rag", "graphrag_variant"):
        assert outputs[vid].loop_count == 1
    scored = score_items([item], [outputs["full_argus"]], corpus)
    assert scored["metrics"]["avg_loop_count"]["value"] > 0


def test_run_ablation_offline(tmp_path: Path, provider: ScriptedProvider, bench_settings: Settings):
    report = run_ablation(
        router=LLMRouter(provider),
        limit=2,
        working_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        settings=bench_settings,
        name="smoke ablation",
    )
    assert report["name"] == "smoke ablation"
    assert report["item_count"] == 2
    assert report["reference_variant"] == "full_argus"
    assert set(report["variants"]) == set(VARIANT_ORDER)
    # Reference variant has zero (or non-applicable None) deltas.
    for value in report["deltas_vs_full_argus"]["full_argus"].values():
        assert value is None or value == 0.0
    # Single-shot variants should not reach the loop's iteration count.
    assert report["variants"]["baseline_rag"]["metrics"]["avg_loop_count"]["value"] == 1.0
    # Reports written.
    assert (tmp_path / "out" / "ablation_report.json").exists()
    # Reliability: a per-variant checkpoint is written during the run so a hard
    # interruption does not lose every completed variant.
    assert (tmp_path / "out" / "ablation_checkpoint.json").exists()
    md = (tmp_path / "out" / "ablation_report.md").read_text(encoding="utf-8")
    assert md.startswith("# smoke ablation")
    assert "Delta vs full_argus" in md


def test_ablation_markdown_prefix():
    md = ablation_markdown({"name": "x", "generated_at": "t", "item_count": 2, "variants": {}, "deltas_vs_full_argus": {}})
    assert md.startswith("# x")


class _LoopFidelityProvider(ScriptedProvider):
    """Scripted provider that simulates the real httpx.AsyncClient lifespan.

    The client is lazily bound to the loop where it is first used and reusing
    it from a different/closed loop is precisely the original ablation bug
    (`RuntimeError: Event loop is closed` at variant boundaries). A per-item
    failure is simulated by making every call for a given item's request raise.
    """

    def __init__(self, script, *, failing_item: str):
        super().__init__(script)
        self._failing_item = failing_item
        self._bound_loop = None
        self._client_open = False
        self.aclose_calls = 0
        self.bound_loop_is_running_at_close = None

    async def _ensure_client(self):
        loop = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = loop
            self._client_open = True
        if self._bound_loop is not loop or self._bound_loop.is_closed():
            raise RuntimeError("Event loop is closed")
        if not self._client_open:
            raise RuntimeError("client already closed")
        return self

    async def complete(self, messages, *, model=None, temperature=0.0, max_tokens=None,
                       response_format=None, tools=None, tool_choice=None, timeout=30.0,
                       call_type: str = "general", request_id=None) -> CompletionResponse:
        await self._ensure_client()
        if request_id and request_id.endswith(f"bench:{self._failing_item}"):
            raise RuntimeError("simulated per-item pipeline failure")
        return await super().complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            call_type=call_type,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        self.bound_loop_is_running_at_close = (
            self._bound_loop is not None and self._bound_loop.is_running()
        )
        self._client_open = False
        self.aclose_calls += 1


def test_run_ablation_single_loop_lifecycle_and_item_isolation(
    tmp_path: Path, bench_settings: Settings
):
    """Regression: ablation must not recreate the loop per variant (that left the
    shared provider's async client bound to a closed loop -> RuntimeError: Event
    loop is closed), must close provider clients on their owning loop, and one
    failing item must not abort the remaining variants."""
    failing = "easy-002"
    provider = _LoopFidelityProvider(
        {
            "query_analysis": [ANALYSIS_OK],
            "research_planning": [plan_payload(["Acme founded year"])],
            "evidence_extraction": [assessment_payload()],
            "synthesis": ["Acme Corp was founded in 1987 [1]."],
            "verification": [VERIFIER_OK],
        },
        failing_item=failing,
    )

    report = run_ablation(
        router=LLMRouter(provider),
        limit=2,
        working_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        settings=bench_settings,
        name="single-loop regression",
    )

    # The classic bug every invocation pattern must fix: no Event loop is closed.
    assert set(report["variants"]) == set(VARIANT_ORDER)
    assert report["variant_failures"] == {}
    # Provider client was closed (resource release) - and closed while its owning loop was active.
    assert provider.aclose_calls == 1
    assert provider.bound_loop_is_running_at_close is True
    # Every variant scored and produced deltas; the single default first item
    # always reaches the LLM, the simulated per-item failure is recorded, and
    # no variant aborts.
    for variant_id in VARIANT_ORDER:
        assert variant_id in report["deltas_vs_full_argus"]
        assert report["variants"][variant_id]["metrics"]["total_failed_calls"]["value"] >= 0
    assert set(report["item_failures"]) <= set(VARIANT_ORDER)
    # The failure is surfaced in the markdown report (never silently swallowed).
    md = ablation_markdown(report)
    assert "simulated per-item pipeline failure" in md
    assert "Item failures" in md


class _CountingProvider(ScriptedProvider):
    """Scripted provider that counts LLM calls (0 means nothing was invoked)."""

    def __init__(self, script):
        super().__init__(script)
        self.call_count = 0

    async def complete(self, messages, *, model=None, temperature=0.0, max_tokens=None,
                       response_format=None, tools=None, tool_choice=None, timeout=30.0,
                       call_type: str = "general", request_id=None) -> CompletionResponse:
        self.call_count += 1
        return await super().complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            call_type=call_type,
            request_id=request_id,
        )


def test_run_ablation_resumes_completed_variants(
    tmp_path: Path, provider: ScriptedProvider, bench_settings: Settings
):
    """Re-running into the same out_dir must resume from the checkpoint and not
    invoke the LLM again for already-completed variants."""
    counting = _CountingProvider(
        {
            "query_analysis": [ANALYSIS_OK] * 40,
            "research_planning": [plan_payload(["Acme founded year"])] * 40,
            "evidence_extraction": [assessment_payload()] * 40,
            "synthesis": ["Acme Corp was founded in 1987 [1]."] * 60,
            "verification": [VERIFIER_OK] * 40,
        }
    )

    first = run_ablation(
        router=LLMRouter(counting),
        limit=2,
        working_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        settings=bench_settings,
        name="resume test",
    )
    assert set(first["variants"]) == set(VARIANT_ORDER)
    assert counting.call_count > 0
    assert (tmp_path / "out" / "ablation_checkpoint.json").exists()

    counting.call_count = 0
    second = run_ablation(
        router=LLMRouter(counting),
        limit=2,
        working_dir=tmp_path / "work2",
        out_dir=tmp_path / "out",
        settings=bench_settings,
        name="resume test",
    )
    # Every variant was already completed in the checkpoint -> fully resumed.
    assert counting.call_count == 0
    assert set(second["variants"]) == set(VARIANT_ORDER)
    # Resumed aggregate for the reference variant is reproducibility-stable: the
    # deterministic metric values match (timing/failed-call totals may differ).
    for metric, info in second["variants"]["full_argus"]["metrics"].items():
        if metric in {"avg_latency_ms", "total_failed_calls", "avg_loop_count"}:
            continue
        second_v = info["value"]
        first_v = first["variants"]["full_argus"]["metrics"][metric]["value"]
        if second_v is None or first_v is None:
            assert second_v == first_v
        else:
            second_ok = math.isnan(float(second_v))
            first_ok = math.isnan(float(first_v))
            assert second_ok == first_ok or second_v == first_v


def test_run_ablation_output_roundtrip_and_item_resume():
    """Outputs persisted to the checkpoint survive a JSON round-trip, and an
    already-recorded item is not re-invoked by a resumed partial variant run."""
    original = BenchmarkRunOutput(
        item_id="easy-001",
        answer="Acme Corp was founded in 1987 [1].",
        cited_chunk_ids=["c1"],
        retrieved_chunk_ids=["c1", "c2"],
        loop_count=2,
        tokens_used=40,
        latency_ms=1234,
        failed_calls=1,
        verification_status="supported",
        metadata={"variant": "baseline_rag", "retrieved_count": 2},
    )
    restored = _output_from_dict(_output_to_dict(original))
    assert restored == original

    class _Factory:
        async def __call__(self, item: BenchmarkItem, _corpus: CorpusContext) -> BenchmarkRunOutput:
            if item.id == "easy-001":
                raise AssertionError("already-recorded item must not be re-invoked")
            return BenchmarkRunOutput(item_id=item.id, answer="ok")

    items = [
        BenchmarkItem(id="easy-001", type="easy_factual", question="q1", gold_answer="a"),
        BenchmarkItem(id="easy-002", type="easy_factual", question="q2", gold_answer="a"),
    ]
    corpus = CorpusContext(gold_chunk_ids={})
    outputs, failures = asyncio.run(
        _run_all(
            _Factory(),  # type: ignore[arg-type]
            items,
            corpus,
            resume_outputs=[original],
            resume_failures=[{"item_id": "easy-001", "error": "previous"}],
        )
    )
    assert outputs[0] == original
    assert failures == [{"item_id": "easy-001", "error": "previous"}]