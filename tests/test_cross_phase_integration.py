"""Cross-phase integration tests (Phases 06-08 stabilization pass).

Proves the phases work together end-to-end, using deterministic fake
providers (no live LLM / API keys):

  A. Query flows API -> orchestration -> planning -> Phase 06 policy ->
     retrieval -> evidence -> assessment -> stopping -> synthesis.
  B. Phase 07 MultiModelRouter used by orchestration without breaking
     the Phase 00.3 LLM abstraction (same `complete` interface).
  C. Phase 08 memory participates in planning when enabled (plan is
     enhanced with memory context).
  D. Memory-disabled mode behaves exactly as before (no enhancement
     node, no side effects).
  E. Evidence provenance survives Evidence -> Graph -> Memory/versioning ->
     result citations.
  F. Versioned graph updates (deltas) are recorded and inspectable.
  G. Adaptive stopping works with memory enabled.
  H. Provider failure/fallback never crashes orchestration.
  I. A fresh query does not inherit state from a previous one.

These are deliberately minimal: they exist to stabilize the phase
boundaries, not to re-test each phase's unit behavior.
"""

from __future__ import annotations

import json
import tempfile
import types
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import yaml

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.graph.models import Claim
from app.graph.store import EvidenceGraphStore
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.exceptions import RateLimitError
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.quota import reset_quota_tracker
from app.llm_gateway.routing.router import LLMRouter
from app.memory.interfaces import MemoryLayer, MemoryQuery, MemoryRecord, MemoryScope
from app.memory.store import MemoryStore
from app.memory.versioning import DeltaStatus, DeltaType, GraphDelta, GraphVersionManager
from app.orchestration.graph import run_query
from app.orchestration.models import OrchestrationResult
from app.reranking.reranker import NoOpReranker
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import assign_embedding_indices


class ScriptedProvider:
    """Fake LLMProvider with per-call_type scripted responses (structural Protocol match)."""

    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self.name = "scripted"
        self.default_model = "scripted-model"
        self.calls: list[tuple[str, str]] = []

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
        self.calls.append((call_type, request_id))
        queue = self._script.get(call_type)
        payload = queue.pop(0) if queue else {"fallback": True}
        content = json.dumps(payload) if isinstance(payload, dict) else payload
        return CompletionResponse(
            content=content,
            model=model or self.default_model,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider=self.name,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        pass


class FailingProvider(ScriptedProvider):
    """Provider that always fails with a retryable rate-limit error."""

    async def complete(self, messages, *, call_type: str = "general", **kwargs) -> CompletionResponse:
        self.calls.append((call_type, kwargs.get("request_id")))
        raise RateLimitError(f"simulated rate limit ({call_type})")


ANALYSIS_OK = {"complexity": "simple", "reasoning": "single fact lookup", "suggested_subquestion_count": 1}


def plan_payload(subquestions: list[str]) -> dict:
    return {
        "objective": "Explain what the fox does.",
        "entities": ["fox"],
        "time_window": None,
        "subquestions": subquestions,
        "evidence_type": "factual",
        "preferred_retrieval_methods": ["hybrid"],
        "required_sources": [],
        "risk_level": "low",
        "token_budget": 6000,
        "iteration_budget": 5,
        "stopping_condition": "Stop once the fox's action is supported by evidence.",
    }


def assessment_payload(sufficient: bool, next_subquery: str | None = None) -> dict:
    return {"sufficient": sufficient, "reasoning": "test", "next_subquery": next_subquery}


FULL_SCRIPT = {
    "query_analysis": [ANALYSIS_OK],
    "research_planning": [plan_payload(["fox behavior"])],
    "evidence_extraction": [assessment_payload(True)],
    "synthesis": ["Foxes jump over dogs [1]."],
}


@pytest.fixture
def temp_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings(temp_paths) -> Settings:
    return Settings(
        _env_file=None,
        config_dir=temp_paths / "config",
        memory_db_path=temp_paths / "memory" / "memory.db",
        evidence_db_path=temp_paths / "evidence.db",
        bm25_index_path=temp_paths / "bm25.pkl",
        faiss_index_path=temp_paths / "faiss.index",
    )


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


@pytest.fixture(autouse=True)
def _reset_globals():
    reset_quota_tracker()
    yield
    reset_quota_tracker()


def _write_multimodel_policy(config_dir: Path, *, primary: str = "fake/fake-model", fallbacks: list[str] | None = None) -> None:
    """Write a minimal model_policy.yaml pointing all call types at a fake provider."""
    config_dir.mkdir(parents=True, exist_ok=True)
    call_types = {
        ct: {"primary": primary, "fallbacks": fallbacks or []}
        for ct in ["general", "query_analysis", "research_planning", "evidence_extraction", "reasoning", "synthesis"]
    }
    (config_dir / "model_policy.yaml").write_text(
        yaml.safe_dump(
            {
                "call_types": call_types,
                "provider_fallbacks": [],
                "quota": {},
                "cross_model_verification": {
                    "verifier_must_differ_from_synthesizer": True,
                    "allow_same_provider_different_model": True,
                    "preferred_verifier_providers": [],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class TestFlowAThroughAPI:
    async def test_query_api_to_orchestration_end_to_end(self, retriever, monkeypatch):
        """A: API -> orchestration -> policy -> retrieval -> evidence -> assessment -> stopping -> synthesis."""
        from fastapi.testclient import TestClient

        import app.orchestration.graph as graph_mod
        from app.api.main import create_app

        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        router = LLMRouter(provider)

        monkeypatch.setattr(graph_mod, "get_router", lambda: router)
        monkeypatch.setattr(graph_mod, "get_hybrid_retriever", lambda: retriever)
        monkeypatch.setattr(graph_mod, "get_reranker", lambda: NoOpReranker())

        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/api/v1/query", json={"query": "What does the fox do?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["plan"]["objective"]
        assert body["answer"]
        assert len(body["citations"]) >= 1
        assert body["citations"][0]["source_path"] == "/test/corpus.txt"
        # Phase 06 policy participated (question pattern classified), and the
        # loop stopped early (single iteration) once evidence was sufficient.
        assert body["question_pattern"] is not None
        assert body["iterations_used"] == 1
        assert body["stop_reason"] in {
            "budget_exhausted",
            "sufficient_evidence",
            "claims_supported",
            "no_unresolved_contradiction",
            "negligible_evidence_gain",
        }


class TestFlowBMultiModelRouter:
    async def test_multimodel_router_drives_orchestration(self, retriever, settings):
        """B: the Phase 07 MultiModelRouter satisfies the same complete() interface orchestration relies on."""
        from app.llm_gateway.routing.multi_model_router import MultiModelRouter

        _write_multimodel_policy(settings.config_dir)
        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        router = MultiModelRouter(settings, providers={"fake": provider})

        result = await run_query(
            "What does the fox do?", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )

        assert isinstance(result, OrchestrationResult)
        assert result.plan.subquestions == ["fox behavior"]
        assert result.answer
        assert len(result.citations) >= 1
        assert result.citations[0].source_path == "/test/corpus.txt"
        # Every orchestration call_type was routed through the multi-model fabric.
        routed_call_types = {ct for ct, _ in provider.calls}
        assert {"query_analysis", "research_planning", "evidence_extraction", "synthesis"} <= routed_call_types


class TestFlowCMemoryInPlanning:
    async def test_memory_enhances_plan_when_enabled(self, retriever, settings, temp_paths, monkeypatch):
        """C: with memory enabled, retrieved prior knowledge is injected into the research plan."""
        settings = settings.model_copy(update={"memory_enabled": True})
        memory_store = MemoryStore(db_path=temp_paths / "memory" / "memory.db")
        try:
            await memory_store.store(
                MemoryRecord(
                    id=uuid4(),
                    layer=MemoryLayer.RESEARCH_HISTORY,
                    scope=MemoryScope.GLOBAL,
                    content="Previously I researched fox populations in the Canidae family.",
                    source_query="fox",
                    confidence=0.9,
                )
            )

            factory = types.SimpleNamespace(create_memory_store=lambda: memory_store)

            async def _noop_init() -> None:
                return None

            monkeypatch.setattr("app.memory.initialize_memory_system", _noop_init)
            monkeypatch.setattr("app.memory.get_memory_factory", lambda: factory)

            provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
            result = await run_query(
                "What does the fox do?",
                router=LLMRouter(provider),
                retriever=retriever,
                reranker=NoOpReranker(),
                settings=settings,
            )

            # Memory participated: the objective carries the memory context block.
            assert "[Memory Context]" in result.plan.objective
            assert result.answer
        finally:
            memory_store.close()


class TestFlowDMemoryDisabled:
    async def test_memory_disabled_leaves_plan_unchanged(self, retriever, settings, temp_paths):
        """D: with memory disabled, the loop is byte-for-byte the pre-Phase-08 behavior."""
        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        result = await run_query(
            "What does the fox do?",
            router=LLMRouter(provider),
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )

        assert result.plan.subquestions == ["fox behavior"]
        assert "[Memory Context]" not in result.plan.objective
        # No memory side effects: the memory db was never created.
        assert not (temp_paths / "memory" / "memory.db").exists()


class TestFlowEProvenance:
    async def test_provenance_survives_evidence_to_memory_and_citation(self, retriever, settings, temp_paths):
        """E: citation chunk IDs trace back through the evidence store, graph, and memory store."""
        provider = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        result = await run_query(
            "What does the fox do?",
            router=LLMRouter(provider),
            retriever=retriever,
            reranker=NoOpReranker(),
            settings=settings,
        )

        citation = result.citations[0]

        evidence_chunk_id = UUID(citation.chunk_id)

        # 1) Evidence -> result citation
        ev_chunks = retriever.store.get_chunks_by_ids([evidence_chunk_id])
        assert len(ev_chunks) == 1 and ev_chunks[0].text == citation.text

        # 2) Evidence -> graph: a claim grounded on the same chunk resolves to the same evidence.
        graph = EvidenceGraphStore(graph_path=temp_paths / "graph.pkl", evidence_store=retriever.store)
        claim = Claim(
            id=uuid4(),
            text=citation.text,
            predicate="states",
            confidence=0.8,
            supporting_chunk_ids=[evidence_chunk_id],
        )
        graph.upsert_claim(claim)
        graph.save()
        fetched = graph.get_claim(claim.id)
        assert fetched is not None
        assert evidence_chunk_id in fetched.supporting_chunk_ids

        # 3) Evidence -> memory: a memory record referencing the same chunk keeps the link.
        memory = MemoryStore(db_path=temp_paths / "memory" / "memory.db")
        try:
            await memory.store(
                MemoryRecord(
                    id=uuid4(),
                    layer=MemoryLayer.LONG_TERM_KNOWLEDGE,
                    content=citation.text,
                    supporting_chunk_ids=[citation.chunk_id],
                    source_query=result.query,
                    confidence=0.9,
                )
            )
            rec = await memory.retrieve(MemoryQuery(query_text=citation.text.split()[0], limit=5))
            assert len(rec) == 1
            assert rec[0].supporting_chunk_ids == [citation.chunk_id]
            assert rec[0].content == citation.text
        finally:
            memory.close()


class TestFlowFVersioning:
    def test_versioned_graph_updates_are_inspectable(self, temp_paths, settings):
        """F: deltas against a graph target are versioned, chainable, and inspectable."""
        manager = GraphVersionManager(db_path=settings.memory_db_path)
        target = uuid4()
        chunk = uuid4()

        r1 = manager.record_delta(
            GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_CREATED,
                status=DeltaStatus.PROVISIONAL,
                target_id=target,
                target_type="claim",
                new_state={"claim_text": "v1"},
                supporting_chunk_ids=[chunk],
                source_query="fox question",
                confidence=0.9,  # above the promotion threshold -> auto-promoted
            )
        )
        r2 = manager.record_delta(
            GraphDelta(
                id=uuid4(),
                delta_type=DeltaType.CLAIM_UPDATED,
                status=DeltaStatus.PROVISIONAL,
                target_id=target,
                target_type="claim",
                previous_state={"claim_text": "v1"},
                new_state={"claim_text": "v2"},
                supporting_chunk_ids=[chunk],
                source_query="fox question",
                confidence=0.5,  # below threshold -> stays provisional
            )
        )

        deltas = manager.get_deltas_for_target(target, "claim")
        assert [d.id for d in deltas] == [r1.id, r2.id]
        # High-confidence delta was auto-promoted.
        assert manager.get_delta(r1.id).status == DeltaStatus.PROMOTED
        assert manager.get_delta(r2.id).status == DeltaStatus.PROVISIONAL
        # Claim version history is inspectable.
        history = manager.get_claim_history(target)
        assert len(history) == 2
        by_version = {row["version"]: row for row in history}
        assert by_version[1]["is_current"] == 0 and by_version[2]["is_current"] == 1
        assert manager.promote_delta(r2.id, "manual")


class TestFlowGStoppingWithMemory:
    async def test_adaptive_stopping_with_memory_enabled(self, retriever, settings, temp_paths, monkeypatch):
        """G: the Phase 06 stopping loop still works when Phase 08 memory is enabled."""
        settings = settings.model_copy(update={"memory_enabled": True})
        memory_store = MemoryStore(db_path=temp_paths / "memory" / "memory.db")
        try:
            factory = types.SimpleNamespace(create_memory_store=lambda: memory_store)

            async def _noop_init() -> None:
                return None

            monkeypatch.setattr("app.memory.initialize_memory_system", _noop_init)
            monkeypatch.setattr("app.memory.get_memory_factory", lambda: factory)

            provider = ScriptedProvider(
                {
                    "query_analysis": [ANALYSIS_OK],
                    # loop re-retrieves once (assessor not satisfied) then is satisfied
                    "research_planning": [plan_payload(["fox behavior", "fox habitat"])],
                    "evidence_extraction": [assessment_payload(False, "fox habitat"), assessment_payload(True)],
                    "synthesis": ["Foxes canids [1]."],
                }
            )
            result = await run_query(
                "Tell me about foxes.",
                router=LLMRouter(provider),
                retriever=retriever,
                reranker=NoOpReranker(),
                settings=settings,
            )
            assert result.iterations_used == 2
            assert "fox behavior" in result.sub_queries_issued and "fox habitat" in result.sub_queries_issued
            assert result.answer
            assert len(result.citations) >= 1
        finally:
            memory_store.close()


class TestFlowHProviderFallback:
    async def test_provider_failure_falls_back_without_crashing(self, retriever, settings):
        """H: a failed primary provider falls back to a healthy one; orchestration survives."""
        from app.llm_gateway.routing.multi_model_router import MultiModelRouter

        _write_multimodel_policy(
            settings.config_dir,
            primary="fail/fail-model",
            fallbacks=["ok/ok-model"],
        )
        failing = FailingProvider()
        healthy = ScriptedProvider({k: list(v) for k, v in FULL_SCRIPT.items()})
        router = MultiModelRouter(settings, providers={"fail": failing, "ok": healthy})

        result = await run_query(
            "What does the fox do?", router=router, retriever=retriever, reranker=NoOpReranker(), settings=settings
        )

        assert isinstance(result, OrchestrationResult)
        assert result.answer
        assert len(result.citations) >= 1
        # The primary was attempted, then the fallback took over.
        assert len(failing.calls) >= 1
        assert len(healthy.calls) >= 1


class TestFlowIRequestIsolation:
    async def test_fresh_query_does_not_inherit_previous_state(self, retriever):
        """I: two sequential queries are fully isolated; the second does not reuse the first's state."""
        provider = ScriptedProvider(
            {
                "query_analysis": [ANALYSIS_OK, ANALYSIS_OK],
                "research_planning": [plan_payload(["fox behavior"]), plan_payload(["fox habitat"])],
                "evidence_extraction": [assessment_payload(True), assessment_payload(True)],
                "synthesis": ["First answer [1].", "Second answer [2]."],
            }
        )
        router = LLMRouter(provider)

        r1 = await run_query(
            "What does the fox do?",
            request_id="req_one",
            router=router,
            retriever=retriever,
            reranker=NoOpReranker(),
        )
        r2 = await run_query(
            "Where do foxes live?",
            request_id="req_two",
            router=router,
            retriever=retriever,
            reranker=NoOpReranker(),
        )

        assert r1.request_id == "req_one"
        assert r2.request_id == "req_two"
        # Each query used its own plan and own synthesis (not the previous run's).
        assert r1.plan.subquestions == ["fox behavior"]
        assert r2.plan.subquestions == ["fox habitat"]
        assert "First answer" in r1.answer and "Second answer" in r2.answer
        # No accumulation: each run is a single iteration with fresh evidence lists.
        assert r1.iterations_used == 1
        assert r2.iterations_used == 1