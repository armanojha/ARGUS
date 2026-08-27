"""Phase 06 acceptance tests: full stopping logic (06.3).

Covers V2 §5.4: each of the five stop conditions is independently
evaluable and testable, the composed `AdaptiveStoppingLogic` evaluates
them in priority order, positive-completion conditions cannot override
an assessor that reports more evidence is needed, and a queued
targeted re-retrieval action is always run before diminishing-returns
stops the loop.

Also verifies the user-level hook: `run_query(user_early_stop=True)`
terminates with `USER_EARLY_STOP` even when the assessor would have
continued re-retrieving.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, EvidenceRef, Source, SourceType
from app.evidence.store import EvidenceStore
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.llm_gateway.routing.router import LLMRouter
from app.orchestration.contracts import StopCondition
from app.orchestration.graph import run_query
from app.orchestration.models import StopReason
from app.orchestration.state import OrchestrationState
from app.orchestration.stopping import (
    AdaptiveStoppingLogic,
    BudgetExhaustedChecker,
    ClaimsSupportedChecker,
    NegligibleEvidenceGainChecker,
    NoUnresolvedContradictionChecker,
    UserEarlyStopChecker,
    build_stopping_logic,
    stop_condition_to_reason,
)
from app.reranking.reranker import NoOpReranker
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import assign_embedding_indices


def _ref(score: float) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="/test/corpus.txt",
        source_type=SourceType.TEXT,
        text="the fox is a small canine that hunts at night",
        score=score,
        rank=1,
        metadata={},
    )


def _state(**overrides: Any) -> OrchestrationState:
    state: OrchestrationState = {
        "request_id": None,
        "query": "Does the fox hunt at night?",
        "max_iterations": 3,
        "token_budget": 1000,
        "query_analysis": None,
        "plan": None,
        "pending_subquestions": [],
        "issued_subqueries": ["Does the fox hunt at night?"],
        "evidence": [],
        "consecutive_empty_retrievals": 0,
        "iteration": 1,
        "tokens_used": 10,
        "sufficient": False,
        "stop_reason": None,
        "answer": None,
        "warnings": [],
        "question_pattern": "conceptual",
        "retrieval_gain_history": [1.0, 0.5],
        "user_early_stop": False,
        "contradiction_signals": [],
        "stop_conditions_checked": [],
        "stop_condition_fired": None,
        "evidence_tasks": [],
    }
    state.update(overrides)
    return state


# -- each of the five conditions, independently ---------------------------------


async def test_claim_supported_stops_and_low_support_does_not():
    high = _state(sufficient=True, evidence=[_ref(0.9), _ref(0.85), _ref(0.6)])
    checker = ClaimsSupportedChecker(threshold=0.7)
    decision = await checker.check(high)
    assert decision.should_stop and decision.condition == StopCondition.CLAIMS_SUPPORTED

    low = _state(sufficient=True, evidence=[_ref(0.2), _ref(0.3)])
    assert (await checker.check(low)).should_stop is False


async def test_claim_supported_respects_assessor_verdict():
    # The assessor said "more evidence needed": even identical scores must not
    # close the loop via CLAIMS_SUPPORTED (the archetype of the gating rule).
    checker = ClaimsSupportedChecker(threshold=0.7)
    open_state = _state(sufficient=False, evidence=[_ref(0.95), _ref(0.9)])
    assert (await checker.check(open_state)).should_stop is False


async def test_no_unresolved_contradiction_stops_or_continues():
    checker = NoUnresolvedContradictionChecker()
    clear = _state(sufficient=True)
    assert (await checker.check(clear)).should_stop is True

    conflicted = _state(
        sufficient=True,
        contradiction_signals=[{"severity": 0.8, "resolved": False, "critical": True}],
    )
    assert (await checker.check(conflicted)).should_stop is False


async def test_budget_exhausted_at_iteration_and_token_ceiling():
    checker = BudgetExhaustedChecker()
    iter_limit = _state(iteration=3, max_iterations=3)
    assert (await checker.check(iter_limit)).should_stop is True

    token_limit = _state(iteration=1, tokens_used=1000, token_budget=1000)
    assert (await checker.check(token_limit)).should_stop is True

    ok = _state(iteration=1, max_iterations=3, tokens_used=10, token_budget=1000)
    assert (await checker.check(ok)).should_stop is False


async def test_negligible_gain_stops_after_consecutive_flat_retrievals():
    checker = NegligibleEvidenceGainChecker(threshold=0.05, min_window=2)
    flat = _state(retrieval_gain_history=[1.0, 0.0])
    decision = await checker.check(flat)
    assert decision.should_stop and decision.condition == StopCondition.NEGLIGIBLE_EVIDENCE_GAIN

    growing = _state(retrieval_gain_history=[1.0, 0.6])
    assert (await checker.check(growing)).should_stop is False

    single = _state(retrieval_gain_history=[1.0])
    assert (await checker.check(single)).should_stop is False


async def test_negligible_gain_deferred_while_query_pending():
    # A queued targeted re-retrieval action must execute before stopping.
    checker = NegligibleEvidenceGainChecker(threshold=0.05, min_window=2)
    with_pending = _state(
        retrieval_gain_history=[1.0, 0.0],
        pending_subquestions=["fox den construction and supporting evidence"],
    )
    decision = await checker.check(with_pending)
    assert decision.should_stop is False


async def test_user_early_stop_fires_only_when_requested():
    checker = UserEarlyStopChecker()
    assert (await checker.check(_state(user_early_stop=True))).should_stop is True
    assert (await checker.check(_state(user_early_stop=False))).should_stop is False


# -- composed logic -------------------------------------------------------------


async def test_composition_evaluates_in_priority_order():
    settings = Settings(_env_file=None)
    logic = build_stopping_logic(settings)
    # Budget exhausted at the ceiling: the hard budget wins and reports
    # the budget condition even though claims look supported.
    state = _state(iteration=3, max_iterations=3, sufficient=False, evidence=[_ref(0.95)])
    decision = await logic.should_stop(state)
    assert decision.condition == StopCondition.BUDGET_EXHAUSTED
    checked = decision.metadata["checked"]
    assert [c["condition"] for c in checked] == [
        StopCondition.USER_EARLY_STOP.value,
        StopCondition.BUDGET_EXHAUSTED.value,
        StopCondition.NEGLIGIBLE_EVIDENCE_GAIN.value,
        StopCondition.CLAIMS_SUPPORTED.value,
        StopCondition.NO_UNRESOLVED_CONTRADICTION.value,
    ]


async def test_composition_gates_positive_conditions_while_assessor_open():
    logic = AdaptiveStoppingLogic()
    open_state = _state(
        sufficient=False,
        evidence=[_ref(0.95), _ref(0.9)],
        retrieval_gain_history=[1.0, 0.9],
        iteration=1,
        max_iterations=3,
    )
    decision = await logic.should_stop(open_state)
    assert decision.should_stop is False
    checked = decision.metadata["checked"]
    claims = next(c for c in checked if c["condition"] == StopCondition.CLAIMS_SUPPORTED.value)
    contradictions = next(c for c in checked if c["condition"] == StopCondition.NO_UNRESOLVED_CONTRADICTION.value)
    assert claims["evaluated"] is False
    assert contradictions["evaluated"] is False


async def test_composition_confirms_claims_when_assessor_done():
    logic = AdaptiveStoppingLogic()
    done = _state(sufficient=True, evidence=[_ref(0.95), _ref(0.9)], retrieval_gain_history=[1.0, 0.5])
    decision = await logic.should_stop(done)
    assert decision.should_stop is True
    assert decision.condition == StopCondition.CLAIMS_SUPPORTED


async def test_build_stopping_logic_uses_settings_thresholds():
    settings = Settings(_env_file=None, stopping_claim_support_threshold=0.99, stopping_evidence_gain_threshold=0.5)
    logic = build_stopping_logic(settings)
    gains = [c for c in logic.get_checkers() if isinstance(c, NegligibleEvidenceGainChecker)]
    claims = [c for c in logic.get_checkers() if isinstance(c, ClaimsSupportedChecker)]
    assert gains[0].threshold == 0.5
    assert claims[0].threshold == 0.99


def test_stop_condition_to_reason_mapping():
    assert stop_condition_to_reason(StopCondition.CLAIMS_SUPPORTED) == StopReason.CLAIMS_SUPPORTED
    assert stop_condition_to_reason(StopCondition.NO_UNRESOLVED_CONTRADICTION) == StopReason.NO_UNRESOLVED_CONTRADICTION
    assert stop_condition_to_reason(StopCondition.BUDGET_EXHAUSTED) == StopReason.BUDGET_EXHAUSTED
    assert stop_condition_to_reason(StopCondition.NEGLIGIBLE_EVIDENCE_GAIN) == StopReason.NEGLIGIBLE_EVIDENCE_GAIN
    assert stop_condition_to_reason(StopCondition.USER_EARLY_STOP) == StopReason.USER_EARLY_STOP
    assert stop_condition_to_reason(None) is None


# -- user-early-stop integration through the full loop --------------------------


class ScriptedProvider:
    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._script = {k: list(v) for k, v in (script or {}).items()}

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


def _plan_payload(subquestions: list[str], iteration_budget: int = 5) -> dict:
    return {
        "objective": "Explain fox behavior.",
        "entities": ["fox"],
        "time_window": None,
        "subquestions": subquestions,
        "evidence_type": "factual",
        "preferred_retrieval_methods": ["hybrid"],
        "required_sources": [],
        "risk_level": "low",
        "token_budget": 100000,
        "iteration_budget": iteration_budget,
        "stopping_condition": "Stop when supported.",
    }


def _assessment(sufficient: bool, next_subquery: str | None = None) -> dict:
    return {"sufficient": sufficient, "reasoning": "test", "next_subquery": next_subquery}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, orchestration_max_iterations=2, orchestration_token_budget=6000)


@pytest.fixture
def store() -> EvidenceStore:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = EvidenceStore(
            db_path=tmp / "evidence.db",
            bm25_index_path=tmp / "bm25.pkl",
            faiss_index_path=tmp / "faiss.index",
        )
        yield store


@pytest.fixture
def retriever(store: EvidenceStore) -> HybridRetriever:
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
    return HybridRetriever(store)


async def test_user_early_stop_halts_loop_before_second_retrieval(retriever: HybridRetriever, settings: Settings):
    provider = ScriptedProvider(
        {
            "query_analysis": [ANALYSIS_OK],
            "research_planning": [_plan_payload(["fox family", "fox habitat"], iteration_budget=5)],
            # Assessor explicitly wants another subquery — but the caller's
            # early stop must win.
            "evidence_extraction": [_assessment(False, "fox habitat")],
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
        user_early_stop=True,
    )
    assert result.stop_reason == StopReason.USER_EARLY_STOP
    assert result.stop_condition == StopCondition.USER_EARLY_STOP.value
    assert len(result.sub_queries_issued) == 1  # no second retrieval ran
    assert result.iterations_used == 1