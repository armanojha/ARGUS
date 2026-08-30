"""Phase 12.2 run-trace observability tests (telemetry registry + API).

Covers: run registry + JSONL persistence, additive routing-decision capture
in the single-provider ``LLMRouter``, and the query endpoint embedding the
trace (plus the read-only telemetry endpoints).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.orchestration as api_orchestration
import app.llm_gateway.telemetry as telemetry_mod
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Message, Usage
from app.llm_gateway.routing.router import LLMRouter
from app.orchestration.models import (
    OrchestrationCitation,
    OrchestrationResult,
    ResearchPlan,
    StopReason,
)


class ScriptedProvider:
    def __init__(self, answer: str = "[123] Scripted answer."):
        self._answer = answer

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
        return CompletionResponse(
            content=self._answer,
            model=model or self.default_model,
            usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            provider="scripted",
            request_id=request_id,
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def isolated_telemetry(tmp_path: Path) -> Path:
    telemetry_mod._completed_runs.clear()
    telemetry_mod.set_telemetry_persistence_dir(tmp_path / "telemetry")
    return tmp_path / "telemetry"


@pytest.fixture(autouse=True)
def _teardown_run():
    yield
    telemetry_mod.end_run_telemetry()
    telemetry_mod.set_telemetry_persistence_dir(None)


def _new_result(query: str, request_id: str | None, answer: str) -> OrchestrationResult:
    return OrchestrationResult(
        query=query,
        plan=ResearchPlan(
            objective=query,
            entities=[],
            time_window=None,
            subquestions=[query],
        ),
        answer=answer,
        citations=[OrchestrationCitation(ref_id=1, chunk_id="c1", document_id="d1", source_id="s1", source_path="docs/f.md", source_type="markdown", text="chunk", score=0.9)],
        iterations_used=1,
        sub_queries_issued=[query],
        stop_reason=StopReason.SUFFICIENT_EVIDENCE.value,
        token_usage_estimate=18,
        request_id=request_id,
    )


def test_run_registry_and_jsonl_persistence(isolated_telemetry: Path):
    telemetry_mod.start_run_telemetry(run_id="run-abc")
    telemetry_mod.record_routing_decision(
        call_type="synthesis",
        provider="groq",
        model="llama",
        latency_ms=12,
        prompt_tokens=11,
        completion_tokens=7,
        success=True,
    )
    summary = telemetry_mod.end_run_telemetry()
    assert summary is not None
    assert summary["run_id"] == "run-abc"
    assert summary["total_calls"] == 1

    assert telemetry_mod.list_runs()[0]["run_id"] == "run-abc"
    assert telemetry_mod.get_run("run-abc")["routing_decisions"][0]["call_type"] == "synthesis"

    # Simulate a process restart: the summary survives on disk.
    telemetry_mod._completed_runs.clear()
    assert telemetry_mod.get_run("run-abc") is not None
    assert (isolated_telemetry / "runs.jsonl").exists()


def test_router_records_routing_decisions_additively(isolated_telemetry: Path):
    telemetry_mod.start_run_telemetry()
    router = LLMRouter(ScriptedProvider())
    asyncio.run(
        router.complete([Message(role="user", content="q")], call_type="evidence_extraction")
    )
    summary = telemetry_mod.end_run_telemetry()
    decision = summary["routing_decisions"][0]
    assert decision["call_type"] == "evidence_extraction"
    assert decision["provider"] == "scripted"
    assert decision["success"] is True
    assert decision["total_tokens"] == 18


def test_telemetry_endpoints_query_integration(
    isolated_telemetry: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.api.main import create_app

    router = LLMRouter(ScriptedProvider())

    async def fake_run_query(query, *, request_id=None, user_early_stop=False):
        await router.complete([Message(role="user", content=query)], call_type="synthesis")
        return _new_result(query, request_id, "The answer is 42 [1].")

    monkeypatch.setattr(api_orchestration, "run_query", fake_run_query)
    client = TestClient(create_app(), raise_server_exceptions=False)
    # The app lifespan points persistence at settings.data_dir; keep test writes in tmp.
    telemetry_mod.set_telemetry_persistence_dir(isolated_telemetry)
    assert "/api/v1/telemetry" in {r.path for r in client.app.routes}

    response = client.post("/api/v1/query", json={"query": "How much is 6x7?"})
    assert response.status_code == 200
    body = response.json()
    telemetry = body["telemetry"]
    assert telemetry is not None
    assert telemetry["total_calls"] == 1
    assert telemetry["routing_decisions"][0]["call_type"] == "synthesis"

    run_id = telemetry["run_id"]
    list_response = client.get("/api/v1/telemetry")
    assert list_response.status_code == 200
    assert any(r["run_id"] == run_id for r in list_response.json())

    run_response = client.get(f"/api/v1/telemetry/{run_id}")
    assert run_response.status_code == 200
    assert run_response.json()["total_calls"] == 1

    missing = client.get("/api/v1/telemetry/does-not-exist")
    assert missing.status_code == 404