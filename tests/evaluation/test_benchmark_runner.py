"""Phase 12.3 benchmark runner tests (offline: real retrieval, stubbed LLM).

Covers dataset integrity, corpus build + retrieval with the **real** Phase 01
stack (BM25 + FAISS, cached embeddings), and one full integration path:
`run_benchmark` through `make_full_argus_pipeline` (orchestration loop +
verification) with a scripted provider, all in a temp directory with no
dependencies on external credentials.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.config import Settings
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.routing.router import LLMRouter
from app.reranking.reranker import NoOpReranker

from benchmarks import runner
from benchmarks.models import ITEM_TYPES
from benchmarks.runner import (
    build_corpus,
    default_sources,
    load_items,
    make_full_argus_pipeline,
    run_benchmark,
    score_items,
)


class ScriptedProvider:
    """Scripted fake LLM provider (per-call_type), structurally matching LLMProvider."""

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

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
        self.calls.append(call_type)
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


def assessment_payload(sufficient: bool) -> dict:
    return {"sufficient": sufficient, "reasoning": "test", "next_subquery": None}


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
def bench_settings(tmp_path: Path) -> Settings:
    return Settings(
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


def test_dataset_has_full_v1_shape():
    items = load_items()
    types = [i.type for i in items]
    assert len(items) == 110  # 100 + 10 adversarial
    adversarial = [i for i in items if i.adversarial_type]
    assert len(adversarial) == 10
    assert sorted(set(types) - {"adversarial"}) == sorted(ITEM_TYPES)
    for item_type in ITEM_TYPES:
        assert types.count(item_type) == 20
    for adv in adversarial:
        assert adv.distractor_evidence, f"{adv.id} must carry a distractor"
    # Temporal items must carry gold years.
    for item in items:
        if item.type == "temporal":
            assert item.gold_years


def test_corpus_build_and_real_retrieval(tmp_path: Path):
    items = [load_items()[0]]  # easy-001: Acme founded 1987
    ctx = build_corpus(items, tmp_path / "corpus")
    assert ctx.gold_chunk_ids[items[0].id]
    sources = default_sources(ctx)
    # Real Phase 01 retrieval: the gold passage must be findable.
    refs = sources["retriever"].search("Acme Corp founded in 1987", top_k=3)
    gold = set(ctx.gold_chunk_ids["easy-001"])
    assert any(str(r.chunk_id) in gold for r in refs)


def test_run_benchmark_with_stub_pipeline(tmp_path: Path):
    async def stub(item, corpus):
        from benchmarks.models import BenchmarkRunOutput
        return BenchmarkRunOutput(
            item_id=item.id,
            answer=item.gold_answer,
            cited_chunk_ids=list(corpus.gold_chunk_ids.get(item.id, []))[:1],
            retrieved_chunk_ids=corpus.gold_chunk_ids.get(item.id, []),
            loop_count=1,
            tokens_used=50,
            latency_ms=5,
        )

    import asyncio

    report = asyncio.run(run_benchmark(
        pipeline=stub,
        limit=4,
        working_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        name="smoke",
        pipeline_label="stub",
    ))
    assert report["item_count"] == 4
    assert report["type_counts"]["easy_factual"] == 4
    assert "recall_at_10" in report["metrics"]
    assert report["v3"]["reindex_cost_ms"] > 0
    assert (tmp_path / "out" / "benchmark_report.json").exists()
    assert (tmp_path / "out" / "benchmark_report.md").exists()
    # The stub answers exactly match gold -> faithfulness must be high.
    assert report["metrics"]["answer_faithfulness"]["value"] >= 0.99


def test_full_argus_pipeline_integration(tmp_path: Path, bench_settings: Settings):
    """One meaningful integration path: real retrieval + orchestration + verification, offline."""
    provider = ScriptedProvider(
        {
            # Two items -> two of each orchestration call plus one verification each.
            "query_analysis": [ANALYSIS_OK, ANALYSIS_OK],
            "research_planning": [plan_payload(["Beta Analytics founded year"]), plan_payload(["Vertex recall month"])],
            "evidence_extraction": [assessment_payload(True), assessment_payload(True)],
            "synthesis": ["Beta Analytics was founded in London [1].", "The Vertex recall was in December 2023 [1]."],
            "verification": [VERIFIER_OK, VERIFIER_OK],
        }
    )
    router = LLMRouter(provider)

    items = [i for i in load_items() if i.id in {"mh-003", "temp-009"}]

    async def _run_all():
        corpus = build_corpus(items, tmp_path / "corpus")
        sources = default_sources(corpus)
        pipeline = make_full_argus_pipeline(
            router=router,
            evidence_store=sources["store"],
            graph_store=sources["graph_store"],
            retriever=sources["retriever"],
            reranker=NoOpReranker(),
            settings=bench_settings,
        )
        outputs = [await pipeline(item, corpus) for item in items]
        return outputs, score_items(items, outputs, corpus)

    import asyncio

    outputs, scored = asyncio.run(_run_all())
    assert scored["metrics"]["avg_loop_count"]["value"] > 0
    assert scored["metrics"]["total_failed_calls"]["value"] == 0
    assert scored["by_type"]["multi_hop"]
    assert scored["by_type"]["temporal"]
    # Verification ran and reported supported.
    assert all(o.verification_status == "supported" for o in outputs)
    # Telemetry summary was captured (run started/ended inside the pipeline).
    assert all(o.failed_calls >= 0 for o in outputs)