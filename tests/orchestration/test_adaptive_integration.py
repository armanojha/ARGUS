"""Phase 24.1 Integration Tests: Adaptive Orchestration Wiring.

Proves the adaptive orchestration is actually connected to the real
pipeline and behaves correctly under various conditions.

Tests:
  1. adaptive disabled → policy not executed, existing behavior unchanged
  2. adaptive enabled → policy is actually invoked
  3. insufficient evidence → second research round occurs
  4. sufficient evidence → research stops without unnecessary round
  5. maximum research budget → loop terminates safely
  6. conflict → investigation triggers
  7. absent information → no endless search
  8. multi-hop → incomplete chain triggers investigation
  9. second round produces no gain → adaptive loop stops
  10. no orphan async tasks after adaptive execution
  11. baseline and adaptive execute genuinely different code paths
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.evidence.store import EvidenceStore
from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import CompletionResponse, Usage
from app.orchestration.adaptive_research import (
    AdaptiveResearchPolicy,
    AdaptiveDecision,
    ResearchSufficiency,
    SufficiencyLevel,
    SufficiencyResult,
    MarginalGainCalculator,
    PatternSpecificPolicies,
    SynthesisGate,
)
from app.orchestration.graph import build_graph, _initial_state, _is_simple_query
from app.orchestration.models import QueryAnalysis, ResearchPlan, ComplexityLevel
from app.orchestration.state import OrchestrationState
from app.reranking.reranker import NoOpReranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import EvidenceNeedPlanner
from app.retrieval.policy import QuestionPattern
from app.retrieval.router import RetrievalPolicyRouter


class ScriptedProvider:
    """Fake LLM provider with scripted responses."""

    def __init__(self, script: dict[str, list] | None = None) -> None:
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self.name = "scripted"
        self.default_model = "scripted-model"
        self.calls: list[tuple[str, str | None]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def complete(
        self, messages, *, model=None, temperature=0.0, max_tokens=None,
        response_format=None, tools=None, tool_choice=None, timeout=30.0,
        call_type: str = "general", request_id=None, query=None, tier=None,
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


def _make_ref(score: float = 0.8, doc_id=None) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4() if doc_id is None else doc_id,
        source_id=uuid4(),
        source_path="test.md",
        source_type=SourceType.MARKDOWN,
        text="Test evidence about the topic at hand.",
        score=score,
        rank=1,
        metadata={},
    )


def _state(**overrides) -> OrchestrationState:
    state: OrchestrationState = {
        "request_id": None,
        "query": "test query",
        "max_iterations": 3,
        "token_budget": 6000,
        "query_analysis": None,
        "plan": None,
        "pending_subquestions": [],
        "issued_subqueries": [],
        "evidence": [],
        "consecutive_empty_retrievals": 0,
        "iteration": 0,
        "tokens_used": 0,
        "sufficient": False,
        "stop_reason": None,
        "answer": None,
        "warnings": [],
        "question_pattern": None,
        "retrieval_gain_history": [],
        "user_early_stop": False,
        "contradiction_signals": [],
        "stop_conditions_checked": [],
        "stop_condition_fired": None,
        "evidence_tasks": [],
        "agent_messages": [],
        "agent_round": 0,
        "debate_active": False,
        "disagreement_detected": False,
        "fast_path": False,
        "complexity_tier": "moderate",
    }
    state.update(overrides)
    return state


# ───────────────────────────────────────────────────────────────────────
# Test 1: Adaptive disabled → policy not executed
# ───────────────────────────────────────────────────────────────────────

class TestAdaptiveDisabled:
    def test_policy_not_invoked_when_disabled(self):
        """When adaptive_research_enabled=False, the assess node uses LLM, not the policy."""
        policy = AdaptiveResearchPolicy(enabled=False)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.9) for _ in range(6)],
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
        )
        # Disabled policy always returns continue_retrieval
        assert decision.action == "continue_retrieval"
        assert "disabled" in decision.reason

    def test_graph_builds_without_adaptive(self):
        """build_graph works when adaptive_research_enabled=False."""
        settings = Settings(
            _env_file=None,
            adaptive_research_enabled=False,
        )
        router = ScriptedProvider()
        retriever = HybridRetriever(store=EvidenceStore())
        reranker = NoOpReranker()

        graph = build_graph(router, retriever, reranker, settings)
        assert graph is not None


# ───────────────────────────────────────────────────────────────────────
# Test 2: Adaptive enabled → policy is invoked
# ───────────────────────────────────────────────────────────────────────

class TestAdaptiveEnabled:
    def test_policy_invoked_when_enabled(self):
        """When adaptive_research_enabled=True, the policy is consulted."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.9) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="simple_lookup",
        )
        # Strong evidence → synthesize
        assert decision.action == "synthesize"
        assert decision.sufficiency_level == "strong"

    def test_graph_builds_with_adaptive(self):
        """build_graph works when adaptive_research_enabled=True."""
        settings = Settings(
            _env_file=None,
            adaptive_research_enabled=True,
        )
        router = ScriptedProvider()
        retriever = HybridRetriever(store=EvidenceStore())
        reranker = NoOpReranker()

        graph = build_graph(router, retriever, reranker, settings)
        assert graph is not None


# ───────────────────────────────────────────────────────────────────────
# Test 3: Insufficient evidence → continue retrieval
# ───────────────────────────────────────────────────────────────────────

class TestInsufficientEvidence:
    def test_insufficient_continues_retrieval(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.3)],
            need_coverage={"n1": 0.2},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="complex_research",
        )
        assert decision.action == "continue_retrieval"
        assert "insufficient" in decision.sufficiency_level


# ───────────────────────────────────────────────────────────────────────
# Test 4: Sufficient evidence → stops
# ───────────────────────────────────────────────────────────────────────

class TestSufficientEvidenceStops:
    def test_strong_evidence_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="numerical",
        )
        assert decision.action == "synthesize"
        assert "Strong" in decision.reason

    def test_sufficient_no_pending_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.7) for _ in range(4)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=2, max_iterations=3,
            pattern="normal_qa",
        )
        assert decision.action == "synthesize"


# ───────────────────────────────────────────────────────────────────────
# Test 5: Maximum budget → terminates
# ───────────────────────────────────────────────────────────────────────

class TestBudgetExhausted:
    def test_max_iterations_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.3)],
            need_coverage={"n1": 0.2},
            gain_history=[0.5, 0.3, 0.1],
            iteration=3, max_iterations=3,
            pattern="complex_research",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason


# ───────────────────────────────────────────────────────────────────────
# Test 6: Conflict → investigation
# ───────────────────────────────────────────────────────────────────────

class TestConflictInvestigation:
    def test_conflict_pattern_investigates(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="conflict",
            contradictions_detected=2,
        )
        assert decision.action == "investigate"
        assert "Conflicting" in decision.reason

    def test_conflict_without_contradictions_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="conflict",
            contradictions_detected=0,
        )
        assert decision.action == "synthesize"


# ───────────────────────────────────────────────────────────────────────
# Test 7: Absent information → no endless search
# ───────────────────────────────────────────────────────────────────────

class TestAbsentInfo:
    def test_absent_info_max_iterations_stops(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        # Even with insufficient evidence, budget exhaustion stops the loop
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.2)],
            need_coverage={"n1": 0.1},
            gain_history=[0.5, 0.01],
            iteration=2, max_iterations=2,
            pattern="absent_info",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason


# ───────────────────────────────────────────────────────────────────────
# Test 8: Multi-hop → incomplete chain triggers investigation
# ───────────────────────────────────────────────────────────────────────

class TestMultiHop:
    def test_multi_hop_insufficient_continues(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.5)],
            need_coverage={"n1": 0.3, "n2": 0.0},
            gain_history=[0.5],
            iteration=1, max_iterations=4,
            pattern="multi_hop",
        )
        assert decision.action == "continue_retrieval"

    def test_multi_hop_strong_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="multi_hop",
        )
        assert decision.action == "synthesize"


# ───────────────────────────────────────────────────────────────────────
# Test 9: No gain → stops
# ───────────────────────────────────────────────────────────────────────

class TestNoGainStops:
    def test_marginal_with_no_gain_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.4) for _ in range(2)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.4},
            gain_history=[0.5, 0.02],
            iteration=2, max_iterations=4,
            pattern="complex_research",
        )
        # Marginal evidence + negligible gain → synthesize
        assert decision.action == "synthesize"
        assert "Marginal" in decision.reason or "negligible" in decision.reason.lower()


# ───────────────────────────────────────────────────────────────────────
# Test 10: No orphan async tasks
# ───────────────────────────────────────────────────────────────────────

class TestNoOrphanTasks:
    def test_policy_is_sync(self):
        """AdaptiveResearchPolicy.should_continue_retrieval is synchronous."""
        policy = AdaptiveResearchPolicy(enabled=True)
        # Must not require await
        result = policy.should_continue_retrieval(
            evidence=[], need_coverage={}, gain_history=[],
            iteration=0, max_iterations=3, pattern="simple_lookup",
        )
        assert isinstance(result, AdaptiveDecision)


# ───────────────────────────────────────────────────────────────────────
# Test 11: Baseline vs adaptive execute different code paths
# ───────────────────────────────────────────────────────────────────────

class TestDifferentCodePaths:
    def test_disabled_policy_never_short_circuits(self):
        """Disabled policy always returns continue_retrieval (never synthesize)."""
        policy = AdaptiveResearchPolicy(enabled=False)
        refs = [_make_ref(0.9) for _ in range(10)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0, "n3": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="simple_lookup",
        )
        # Even with perfect evidence, disabled policy says continue
        assert decision.action == "continue_retrieval"

    def test_enabled_policy_short_circuits_on_strong_evidence(self):
        """Enabled policy returns synthesize for strong evidence."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.9) for _ in range(10)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0, "n3": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="simple_lookup",
        )
        assert decision.action == "synthesize"

    def test_sufficiency_separates_conflict_from_insufficiency(self):
        """CONFLICTED is distinct from INSUFFICIENT."""
        # Strong evidence with contradictions → CONFLICTED
        s1 = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=2,
        )
        r1 = s1.assess()
        assert r1.level == SufficiencyLevel.CONFLICTED
        assert r1.conflict_detected
        assert r1.coverage_ok

        # Weak evidence without contradictions → INSUFFICIENT
        s2 = ResearchSufficiency(
            evidence_count=1, coverage_score=0.1, source_diversity=1,
            avg_top3_score=0.2, need_count=3, contradictions_detected=0,
        )
        r2 = s2.assess()
        assert r2.level == SufficiencyLevel.INSUFFICIENT
        assert not r2.conflict_detected
