"""Lightweight sanity tests for the Agentic RAG loop (Phase 02).

Not exhaustive by design (see vault Phase 02 testing policy: deferred
to a later stabilization pass). Covers the phase's own acceptance
criteria: a plan is produced, the loop can re-retrieve at least once,
the final answer carries citations back to Phase 01 evidence, and
configured budgets bound the loop.

Uses a small scripted fake provider (structurally satisfying
`LLMProvider`) rather than `tests/mocks/mock_provider.py`'s
`MockProvider`, because these tests need different canned responses
per `call_type` (analysis vs plan vs assessment vs synthesis), which
`MockProvider` only keys by model name.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.routing.router import LLMRouter
from app.orchestration.graph import run_query
from app.orchestration.models import OrchestrationResult, StopReason
from app.reranking.reranker import NoOpReranker
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import assign_embedding_indices


class ScriptedProvider:
    """Fake LLMProvider with per-call_type scripted responses (structural Protocol match)."""

    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._script = {k: list(v) for k, v in (script or {}).items()}
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

    async def complete(
        self,
        messages,
        *,
        model=None,
        temperature=0.0,
        max_tokens=None,
        response_format=None,
        tools=None,
        tool_choice=None,
        timeout=30.0,
        call_type: str = "general",
        request_id=None,
    ) -> CompletionResponse:
        self.calls.append(call_type)
        queue = self._script.get(call_type)
        if queue:
            payload = queue.pop(0)
            content = json.dumps(payload) if isinstance(payload, dict) else payload
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


def plan_payload(subquestions: list[str], iteration_budget: int = 5, token_budget: int = 100000) -> dict:
    return {
        "objective": "Explain what the fox does.",
        "entities": ["fox"],
        "time_window": None,
        "subquestions": subquestions,
        "evidence_type": "factual",
        "preferred_retrieval_methods": ["hybrid"],
        "required_sources": [],
        "risk_level": "low",
        "token_budget": token_budget,
        "iteration_budget": iteration_budget,
        "stopping_condition": "Stop once the fox's action is supported by evidence.",
    }


def assessment_payload(sufficient: bool, next_subquery: str | None = None) -> dict:
    return {"sufficient": sufficient, "reasoning": "test", "next_subquery": next_subquery}


@pytest.fixture
def temp_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield tmp


@pytest.fixture
def store(temp_paths) -> EvidenceStore:
    return EvidenceStore(
        db_path=temp_paths / "evidence.db",
        bm25_index_path=temp_paths / "bm25.pkl",
        faiss_index_path=temp_paths / "faiss.index",
    )


@pytest.fixture
def populated_store(store: EvidenceStore) -> EvidenceStore:
    source = Source(type=SourceType.TEXT, path="/test/corpus.txt", checksum="c1")
    store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d1", chunking_strategy="fixed")
    store.insert_document(doc)
    store.insert_chunks(
        [
            Chunk(document_id=doc.id, ordinal=0, text="The quick brown fox jumps over the lazy dog.", token_count=10),
            Chunk(document_id=doc.id, ordinal=1, text="Foxes are members of the Canidae family.", token_count=8),
        ]
    )
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)
    return store


@pytest.fixture
def retriever(populated_store: EvidenceStore) -> HybridRetriever:
    return HybridRetriever(populated_store)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, orchestration_max_iterations=2, orchestration_token_budget=6000)


class TestAgenticLoop:
    async def test_produces_research_plan(self, retriever, settings):
        provider = ScriptedProvider(
            {
                "query_analysis": [ANALYSIS_OK],
                "research_planning": [plan_payload(["fox behavior"])],
                "evidence_extraction": [assessment_payload(True)],
                "synthesis": ["Foxes jump over dogs [1]."],
            }
        )
        router = LLMRouter(provider)
        result = await run_query(
            "What does the fox do?", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )
        assert isinstance(result, OrchestrationResult)
        assert result.plan.objective
        assert result.plan.subquestions == ["fox behavior"]

    async def test_reretrieves_at_least_once_when_evidence_poor(self, retriever, settings):
        provider = ScriptedProvider(
            {
                "query_analysis": [ANALYSIS_OK],
                "research_planning": [plan_payload(["fox family", "fox habitat"], iteration_budget=5)],
                # Never satisfied -> loop should run until the configured
                # hard ceiling (settings.orchestration_max_iterations = 2),
                # not until the plan's proposed budget of 5.
                "evidence_extraction": [
                    assessment_payload(False, "fox habitat"),
                    assessment_payload(False, "fox diet"),
                    assessment_payload(False, "fox lifespan"),
                ],
                "synthesis": ["Foxes are canids [1][2]."],
            }
        )
        router = LLMRouter(provider)
        result = await run_query(
            "Tell me everything about foxes.",
            router=router,
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )
        assert len(result.sub_queries_issued) >= 2
        assert result.iterations_used <= settings.orchestration_max_iterations
        assert result.stop_reason == StopReason.BUDGET_EXHAUSTED

    async def test_answer_includes_citations_to_evidence(self, retriever, settings):
        provider = ScriptedProvider(
            {
                "query_analysis": [ANALYSIS_OK],
                "research_planning": [plan_payload(["fox behavior"])],
                "evidence_extraction": [assessment_payload(True)],
                "synthesis": ["The fox jumps over the dog [1]."],
            }
        )
        router = LLMRouter(provider)
        result = await run_query(
            "What does the fox do?", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )
        assert len(result.citations) > 0
        citation = result.citations[0]
        assert citation.source_path == "/test/corpus.txt"
        assert citation.chunk_id

    async def test_budget_ceiling_overrides_plan_proposal(self, retriever, settings):
        """Plan proposes a huge budget; the orchestrator must still respect settings' ceiling."""
        provider = ScriptedProvider(
            {
                "query_analysis": [ANALYSIS_OK],
                "research_planning": [plan_payload(["a"], iteration_budget=999, token_budget=999_999)],
                "evidence_extraction": [assessment_payload(False, "b"), assessment_payload(False, "c")],
                "synthesis": ["Answer [1]."],
            }
        )
        router = LLMRouter(provider)
        result = await run_query(
            "Runaway budget test", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )
        assert result.plan.iteration_budget == settings.orchestration_max_iterations
        assert result.iterations_used <= settings.orchestration_max_iterations

    async def test_llm_failure_falls_back_gracefully(self, retriever, settings):
        """No scripted responses at all -> every structured call fails validation; loop still completes."""
        provider = ScriptedProvider({})
        router = LLMRouter(provider)
        result = await run_query(
            "What does the fox do?", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )
        assert isinstance(result, OrchestrationResult)
        assert result.plan.objective  # fallback plan still has an objective
        assert result.answer  # synthesis fallback still produces something
