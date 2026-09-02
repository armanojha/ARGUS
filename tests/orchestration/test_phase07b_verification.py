"""Phase 07b ("Production Default") tests.

Covers the two hardening changes that make the already-built resilience +
verification active by default:

Part 1 — Resilient router is the default:
  * default `get_router()` returns ``MultiModelRouter``
  * explicit ``multimodel_enabled=False`` returns the single-provider
    ``LLMRouter`` escape hatch

Part 2 — Selective verification is wired into the query path:
  * verification disabled => skipped
  * simple/low-risk, high-confidence evidence => skipped (06.5.4 gate)
  * call ceiling exhausted => skipped (bounded budget)
  * verification-eligible query triggers the *existing* verifier once
  * success is represented on the result
  * failure / timeout fail-safe: the grounded, cited answer is preserved
  * verification is never run in a loop (at most one structured call)

All tests use scripted providers / patched verifier call — zero network.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import app.verification.engine as verification_engine
from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.llm_gateway import get_router
from app.orchestration.graph import _run_selective_verification
from app.orchestration.models import (
    OrchestrationResult,
    OrchestrationVerification,
    ResearchPlan,
    StopReason,
)
from app.verification.engine import VerifierOutput


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ref(score: float, text: str = "Some supporting text.") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="/t/corpus.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
    )


def _plan(risk_level: str = "low") -> ResearchPlan:
    return ResearchPlan(
        objective="Explain what the fox does.",
        entities=["fox"],
        time_window=None,
        subquestions=["fox behavior"],
        evidence_type="factual",
        preferred_retrieval_methods=["hybrid"],
        risk_level=risk_level,
        token_budget=6000,
        iteration_budget=2,
        stopping_condition="stop",
    )


def _state(*, evidence: list[EvidenceRef], answer: str = "Foxes jump over dogs [1].", plan: ResearchPlan | None = None) -> dict:
    """A minimal but structurally valid 'final' orchestration state."""
    return {
        "query": "What does the fox do?",
        "request_id": "req-07b",
        "plan": plan or _plan(),
        "evidence": evidence,
        "answer": answer,
        "stop_reason": "sufficient_evidence",
        "iteration": 1,
        "issued_subqueries": ["fox behavior"],
        "tokens_used": 15,
        "warnings": [],
        "question_pattern": None,
        "stop_condition_fired": None,
        "stop_conditions_checked": [],
        "evidence_tasks": [],
        "agent_round": 0,
        "agent_messages": [],
        "disagreement_detected": None,
        "fast_path": False,
    }


class _PassthroughRouter:
    """Minimal structural stand-in for the router used by verify_claim."""

    async def complete(self, messages, *, call_type="general", **kwargs):
        raise AssertionError("router.complete should not be reached; verifier LLM is patched")

    async def aclose(self):
        pass


def _settings(**overrides) -> Settings:
    base = {
        "multimodel_enabled": True,
        "verification_enabled": True,
        "orchestration_llm_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def _no_network(monkeypatch):
    """Replace the verifier LLM call so no provider is touched."""

    async def _patched(router, *, messages, settings, request_id):
        return VerifierOutput(status="SUPPORTED", confidence=0.91, reasoning="ok"), None

    monkeypatch.setattr(verification_engine, "_safe_structured_verification_call", _patched)
    yield


# --------------------------------------------------------------------------- #
# Part 1 — resilient router is the default
# --------------------------------------------------------------------------- #
class TestDefaultIsMultiModelRouter:
    def test_default_settings_use_multimodel(self):
        """A stock Settings object has the resilient router enabled."""
        assert Settings().multimodel_enabled is True

    def test_default_get_router_returns_multimodel(self, monkeypatch):
        """`get_router()` on a stock config builds a MultiModelRouter."""
        import app.llm_gateway as gateway
        from app.llm_gateway.routing.multi_model_router import MultiModelRouter

        gateway._router = None
        monkeypatch.setattr(gateway, "get_settings", lambda: Settings())
        router = get_router()
        assert isinstance(router, MultiModelRouter)

    def test_explicit_single_provider_returns_llm_router(self, monkeypatch):
        """Escape hatch: `multimodel_enabled=False` -> single LLMRouter."""
        import app.llm_gateway as gateway
        from app.llm_gateway.routing.router import LLMRouter
        gateway._router = None
        monkeypatch.setattr(gateway, "get_settings", lambda: _settings(multimodel_enabled=False))
        monkeypatch.setattr(gateway, "create_provider", lambda settings=None: _FakeProvider())
        router = get_router()
        assert isinstance(router, LLMRouter)


class _FakeProvider:
    name = "fake"
    default_model = "fake-model"

    @property
    def capabilities(self):
        from app.llm_gateway.capabilities import ProviderCapabilities
        return ProviderCapabilities()

    async def complete(self, *a, **kw):
        from app.llm_gateway.providers.models import CompletionResponse, Usage

        return CompletionResponse(
            content="mock", model="fake-model",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider="fake",
        )

    async def aclose(self):
        pass


# --------------------------------------------------------------------------- #
# Part 2 — selective verification wiring
# --------------------------------------------------------------------------- #
class TestSelectiveVerificationGate:
    def test_disabled_when_verification_enabled_is_false(self, _no_network):
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.95)]),
                router=_PassthroughRouter(),
                settings=_settings(verification_enabled=False),
                request_id="x",
            )
        )
        assert out.triggered is False
        assert out.skipped_reason == "disabled"

    def test_skipped_for_low_risk_high_confidence(self, _no_network):
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.95), _ref(0.9), _ref(0.93)], plan=_plan("low")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is False
        assert out.skipped_reason == "low_risk"

    def test_triggers_for_high_risk_question(self, _no_network):
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.95), _ref(0.9), _ref(0.93)], plan=_plan("high")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is True
        assert out.status == "supported"
        assert out.confidence == pytest.approx(0.91)

    def test_triggers_for_low_confidence_evidence(self, _no_network):
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.2), _ref(0.3)], plan=_plan("low")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is True
        assert out.status == "supported"

    def test_call_ceiling_skips_verification(self, _no_network, monkeypatch):
        """Never spend the last of the call budget on verification."""
        import app.orchestration.graph as graph_mod

        monkeypatch.setattr(graph_mod, "check_call_ceiling", lambda: True)
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.2)], plan=_plan("high")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is False
        assert out.skipped_reason == "call_budget"

    def test_verifier_called_exactly_once(self, _no_network, monkeypatch):
        """Verification is never repeated in a loop: exactly one structured call."""
        calls = {"n": 0}

        async def _counting(router, *, messages, settings, request_id):
            calls["n"] += 1
            return VerifierOutput(status="SUPPORTED", confidence=0.8, reasoning="ok"), None

        monkeypatch.setattr(verification_engine, "_safe_structured_verification_call", _counting)
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.2)], plan=_plan("high")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is True
        assert calls["n"] == 1


class TestVerificationFailSafe:
    def test_failure_does_not_crash_and_reports_error(self, _no_network, monkeypatch):
        async def _fail(router, *, messages, settings, request_id):
            return None, "verification call failed"

        monkeypatch.setattr(verification_engine, "_safe_structured_verification_call", _fail)
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.2)], plan=_plan("high")),
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        assert out.triggered is True
        assert out.status == "error"
        assert out.error is not None

    def test_timeout_fails_safe(self, _no_network, monkeypatch):
        async def _hang(router, *, messages, settings, request_id):
            await asyncio.sleep(30)  # longer than the stage timeout
            return VerifierOutput(status="SUPPORTED", confidence=0.9, reasoning="ok"), None

        monkeypatch.setattr(verification_engine, "_safe_structured_verification_call", _hang)
        out = asyncio.run(
            _run_selective_verification(
                _state(evidence=[_ref(0.2)], plan=_plan("high")),
                router=_PassthroughRouter(),
                settings=_settings(orchestration_llm_timeout=0.1),
                request_id="x",
            )
        )
        assert out.triggered is True
        assert out.status == "error"

    def test_answer_preserved_on_failure(self, _no_network, monkeypatch):
        """Fail-safe is architectural: a failing verifier never removes the
        grounded answer; it only annotates error metadata."""
        async def _fail(messages, *, router, settings, request_id):
            return None, "verify down"

        monkeypatch.setattr(verification_engine, "_safe_structured_verification_call", _fail)
        state = _state(evidence=[_ref(0.2)])
        out = asyncio.run(
            _run_selective_verification(
                state,
                router=_PassthroughRouter(),
                settings=_settings(),
                request_id="x",
            )
        )
        # The verification stage runs post-synthesis; the original grounded
        # answer in the state is never discarded by a verification failure.
        assert state["answer"] == "Foxes jump over dogs [1]."
        assert out.status == "error"


class TestResultModel:
    def test_orchestration_verification_field_defaults_none(self):
        """The new field is additive and defaulted — existing results are valid."""
        result = OrchestrationResult(
            query="q",
            plan=_plan(),
            answer="a",
            citations=[],
            iterations_used=1,
            sub_queries_issued=[],
            stop_reason=StopReason.SUFFICIENT_EVIDENCE.value,
            token_usage_estimate=1,
            request_id="x",
        )
        assert result.verification is None

    def test_orchestration_verification_serializes(self):
        v = OrchestrationVerification(
            triggered=True, status="supported", confidence=0.9, contradiction_detected=False
        )
        data = v.model_dump()
        assert data["status"] == "supported"
        assert data["confidence"] == 0.9
        assert data["triggered"] is True