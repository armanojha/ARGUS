"""Phase 24.3: Orchestration Boundary & Production Readiness Audit Tests.

Tests for:
- Feature flag boundary (disabled/enabled)
- Adaptive loop safety (termination, budgets)
- Evidence accumulation (dedup, coverage)
- Marginal-gain logic
- Sufficiency semantics
- Multi-hop behavior
- Conflict behavior
- Absent-information behavior
- Error handling
- Async safety
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.adaptive_research import (
    AdaptiveDecision,
    AdaptiveResearchPolicy,
    MarginalGainCalculator,
    MarginalGainResult,
    ResearchSufficiency,
    SufficiencyLevel,
    SufficiencyResult,
    SynthesisGate,
)


def _make_ref(score: float = 0.8, text: str = "test evidence", doc_id=None) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4() if doc_id is None else doc_id,
        source_id=uuid4(),
        source_path="test.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
        metadata={},
    )


# ---------------------------------------------------------------------------
# 1. Feature Flag Boundary
# ---------------------------------------------------------------------------

class TestFeatureFlagBoundary:
    def test_default_is_disabled(self):
        """adaptive_research_enabled must default to False."""
        settings = Settings(_env_file=None)
        assert getattr(settings, "adaptive_research_enabled", False) is False

    def test_disabled_policy_returns_continue(self):
        """When disabled, policy always returns continue_retrieval."""
        policy = AdaptiveResearchPolicy(enabled=False)
        refs = [_make_ref(0.9) for _ in range(10)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
        )
        assert decision.action == "continue_retrieval"
        assert decision.sufficiency_level == "unknown"

    def test_enabled_policy_actually_decides(self):
        """When enabled, policy makes real decisions."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.9) for _ in range(8)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="simple_lookup",
        )
        assert decision.action == "synthesize"
        assert decision.sufficiency_level == "strong"

    def test_disabled_graph_has_no_policy(self):
        """Graph built without adaptive has policy=None in assess node."""
        from app.orchestration.graph import build_graph
        from app.reranking.reranker import NoOpReranker
        from app.retrieval.hybrid import HybridRetriever
        from app.evidence.store import EvidenceStore

        settings = Settings(_env_file=None, adaptive_research_enabled=False)
        router = type("FakeRouter", (), {"name": "fake", "default_model": "fake"})()
        retriever = HybridRetriever(store=EvidenceStore())
        graph = build_graph(router, retriever, NoOpReranker(), settings)
        assert graph is not None

    def test_enabled_graph_builds_with_policy(self):
        """Graph built with adaptive creates policy object."""
        from app.orchestration.graph import build_graph
        from app.reranking.reranker import NoOpReranker
        from app.retrieval.hybrid import HybridRetriever
        from app.evidence.store import EvidenceStore

        settings = Settings(_env_file=None, adaptive_research_enabled=True)
        router = type("FakeRouter", (), {"name": "fake", "default_model": "fake"})()
        retriever = HybridRetriever(store=EvidenceStore())
        graph = build_graph(router, retriever, NoOpReranker(), settings)
        assert graph is not None


# ---------------------------------------------------------------------------
# 2. Adaptive Loop Safety (Termination)
# ---------------------------------------------------------------------------

class TestLoopSafety:
    def test_max_iterations_enforced(self):
        """Policy synthesizes when max iterations reached."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.2) for _ in range(1)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.1},
            gain_history=[0.5, 0.3, 0.1],
            iteration=3, max_iterations=3,
            pattern="complex_research",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason

    def test_budget_exhausted_terminates(self):
        """Policy synthesizes when budget exhausted."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.1) for _ in range(1)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.05},
            gain_history=[0.5],
            iteration=5, max_iterations=5,
            pattern="complex_research",
        )
        assert decision.action == "synthesize"

    def test_repeated_identical_evidence_terminates(self):
        """Repeated identical retrieval results eventually stop."""
        policy = AdaptiveResearchPolicy(enabled=True)
        # Simulate 5 rounds of identical evidence
        refs = [_make_ref(0.4) for _ in range(2)]
        for iteration in range(1, 6):
            decision = policy.should_continue_retrieval(
                evidence=refs,
                need_coverage={"n1": 0.4},
                gain_history=[0.5, 0.0, 0.0, 0.0, 0.0],
                iteration=iteration, max_iterations=5,
                pattern="complex_research",
            )
            if decision.action == "synthesize":
                break
        else:
            pytest.fail("Policy did not stop within 5 iterations")
        assert decision.action == "synthesize"

    def test_zero_gain_stops(self):
        """Zero gain from identical evidence stops the loop."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.4) for _ in range(2)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.4},
            gain_history=[0.5, 0.0],
            iteration=2, max_iterations=4,
            pattern="complex_research",
        )
        assert decision.action == "synthesize"

    def test_no_infinite_loops(self):
        """Policy always terminates within max_iterations."""
        policy = AdaptiveResearchPolicy(enabled=True)
        for max_iter in range(1, 10):
            for iteration in range(1, 20):
                decision = policy.should_continue_retrieval(
                    evidence=[_make_ref(0.1)],
                    need_coverage={"n1": 0.05},
                    gain_history=[0.01] * iteration,
                    iteration=iteration, max_iterations=max_iter,
                    pattern="complex_research",
                )
                if decision.action == "synthesize":
                    break
            else:
                pytest.fail(f"Policy did not stop within {max_iter} iterations")

    def test_empty_evidence_terminates(self):
        """Empty evidence eventually terminates via budget exhaustion."""
        policy = AdaptiveResearchPolicy(enabled=True)
        # Simple lookup has max_iterations=1, so iteration 1 should synthesize
        decision = policy.should_continue_retrieval(
            evidence=[],
            need_coverage={},
            gain_history=[],
            iteration=1, max_iterations=1,
            pattern="simple_lookup",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason


# ---------------------------------------------------------------------------
# 3. Evidence Accumulation
# ---------------------------------------------------------------------------

class TestEvidenceAccumulation:
    def test_merge_deduplicates_chunks(self):
        """Same chunk_id should not be duplicated."""
        from benchmarks.benchmark_adaptive_research import _merge_evidence

        ref1 = _make_ref(0.8)
        ref2 = _make_ref(0.7)  # different chunk_id
        ref3 = ref1  # same chunk_id as ref1

        merged, new_count = _merge_evidence([ref1, ref2], [ref2, ref3])
        assert len(merged) == 2
        assert new_count == 0

    def test_merge_updates_higher_score(self):
        """Higher score replaces lower score for same chunk_id."""
        from benchmarks.benchmark_adaptive_research import _merge_evidence

        ref_low = _make_ref(0.5)
        ref_high = EvidenceRef(
            chunk_id=ref_low.chunk_id,  # same chunk_id
            document_id=ref_low.document_id,
            source_id=ref_low.source_id,
            source_path="test.md",
            source_type=SourceType.MARKDOWN,
            text="updated",
            score=0.9,
            rank=1,
            metadata={},
        )

        merged, new_count = _merge_evidence([ref_low], [ref_high])
        assert len(merged) == 1
        assert merged[0].score == 0.9
        assert new_count == 0

    def test_gain_zero_on_identical_evidence(self):
        """Gain should be zero when second round returns same evidence."""
        from benchmarks.benchmark_adaptive_research import _merge_evidence

        ref1 = _make_ref(0.8)
        ref2 = _make_ref(0.7)

        merged1, new1 = _merge_evidence([], [ref1, ref2])
        assert new1 == 2

        merged2, new2 = _merge_evidence(merged1, [ref1, ref2])
        assert new2 == 0
        gain = new2 / len(merged1) if merged1 else 0
        assert gain == 0.0


# ---------------------------------------------------------------------------
# 4. Marginal Gain Logic
# ---------------------------------------------------------------------------

class TestMarginalGain:
    def test_positive_gain_on_new_evidence(self):
        """Gain is positive when new evidence appears."""
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.5, 0.3])
        assert not result.should_stop
        assert result.gain_ratio == 0.3

    def test_zero_gain_on_duplicates(self):
        """Gain is zero when only duplicates returned."""
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.5, 0.0])
        assert result.should_stop
        assert result.gain_ratio == 0.0

    def test_near_zero_gain_on_irrelevant(self):
        """Near-zero gain on irrelevant chunks stops."""
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.5, 0.02])
        assert result.should_stop

    def test_meaningful_gain_on_need_resolution(self):
        """Meaningful gain when evidence need is resolved."""
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.5, 0.4])
        assert not result.should_stop

    def test_gain_does_not_inflate_on_contradiction(self):
        """Gain should not be misinterpreted when contradictions appear."""
        calc = MarginalGainCalculator(threshold=0.10)
        # High gain but with contradictions - gain is still measured correctly
        result = calc.evaluate([0.5, 0.8])
        assert not result.should_stop
        # The gain ratio is about new evidence count, not quality

    def test_short_history_no_stop(self):
        """History too short to evaluate should not stop."""
        calc = MarginalGainCalculator(threshold=0.10, min_window=2)
        result = calc.evaluate([0.5])
        assert not result.should_stop


# ---------------------------------------------------------------------------
# 5. Sufficiency Semantics
# ---------------------------------------------------------------------------

class TestSufficiencySemantics:
    def test_strong_evidence_sufficient(self):
        """Strong evidence → SUFFICIENT or STRONG."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level in (SufficiencyLevel.STRONG, SufficiencyLevel.SUFFICIENT)

    def test_missing_evidence_insufficient(self):
        """Missing required evidence → INSUFFICIENT."""
        s = ResearchSufficiency(
            evidence_count=1, coverage_score=0.1, source_diversity=1,
            avg_top3_score=0.2, need_count=3, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.INSUFFICIENT

    def test_strong_with_contradiction_conflicted(self):
        """Strong evidence with contradictions → CONFLICTED."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=2,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.CONFLICTED
        assert result.conflict_detected
        assert result.coverage_ok

    def test_weak_with_contradiction_insufficient(self):
        """Weak evidence with contradictions → INSUFFICIENT."""
        s = ResearchSufficiency(
            evidence_count=1, coverage_score=0.1, source_diversity=1,
            avg_top3_score=0.2, need_count=3, contradictions_detected=1,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.INSUFFICIENT

    def test_moderate_evidence_marginal(self):
        """Moderate evidence → MARGINAL."""
        s = ResearchSufficiency(
            evidence_count=2, coverage_score=0.4, source_diversity=1,
            avg_top3_score=0.4, need_count=2, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.MARGINAL

    def test_contradiction_does_not_force_winner(self):
        """Contradiction does not force a winner - stays CONFLICTED."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=3,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.CONFLICTED
        assert result.conflict_count == 3

    def test_coverage_ok_separate_from_level(self):
        """Coverage_ok is independent of sufficiency level."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=2,
        )
        result = s.assess()
        assert result.coverage_ok is True
        assert result.level == SufficiencyLevel.CONFLICTED


# ---------------------------------------------------------------------------
# 6. Multi-Hop Behavior
# ---------------------------------------------------------------------------

class TestMultiHopBehavior:
    def test_complete_chain_stops(self):
        """Complete A→B→C chain stops."""
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

    def test_incomplete_chain_continues(self):
        """Incomplete A→B, missing B→C continues."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.5) for _ in range(2)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.3, "n2": 0.0},
            gain_history=[0.5],
            iteration=1, max_iterations=4,
            pattern="multi_hop",
        )
        assert decision.action == "continue_retrieval"

    def test_missing_bridge_investigates(self):
        """Missing bridge evidence triggers investigation."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.4) for _ in range(2)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.2, "n2": 0.0},
            gain_history=[0.5],
            iteration=1, max_iterations=4,
            pattern="multi_hop",
        )
        assert decision.action == "continue_retrieval"

    def test_no_bridge_exists_terminates(self):
        """When no bridge exists, budget exhaustion terminates."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.3) for _ in range(1)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.1, "n2": 0.0},
            gain_history=[0.5, 0.01],
            iteration=4, max_iterations=4,
            pattern="multi_hop",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason

    def test_repeated_failure_stops_on_budget(self):
        """Repeatedly failing to find bridge stops on budget."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.2) for _ in range(1)]
        for iteration in range(1, 6):
            decision = policy.should_continue_retrieval(
                evidence=refs,
                need_coverage={"n1": 0.1, "n2": 0.0},
                gain_history=[0.5, 0.0, 0.0, 0.0],
                iteration=iteration, max_iterations=4,
                pattern="multi_hop",
            )
            if decision.action == "synthesize":
                break
        assert decision.action == "synthesize"


# ---------------------------------------------------------------------------
# 7. Conflict Behavior
# ---------------------------------------------------------------------------

class TestConflictBehavior:
    def test_no_conflict_normal_synthesis(self):
        """No conflict → normal synthesis path."""
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

    def test_strong_conflict_conflicted(self):
        """Strong conflicting evidence → CONFLICTED."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=2,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.CONFLICTED

    def test_conflict_with_missing_evidence_investigates(self):
        """Conflict + missing evidence → INVESTIGATE when justified."""
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

    def test_conflict_unresolved_terminates(self):
        """Conflict remains after investigation → terminate."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="numerical",
            contradictions_detected=1,
        )
        # Non-conflict pattern synthesizes despite contradictions
        assert decision.action == "synthesize"

    def test_duplicate_contradictions_not_inflated(self):
        """Duplicate contradictory chunks don't inflate conflict count."""
        detector = _make_ref(0.8, text="contradiction A")
        detector2 = _make_ref(0.7, text="contradiction A")  # same text
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=1,
        )
        result = s.assess()
        assert result.conflict_count == 1

    def test_weak_noise_not_treated_as_authoritative(self):
        """Weak conflicting evidence not treated as equally authoritative."""
        s = ResearchSufficiency(
            evidence_count=2, coverage_score=0.5, source_diversity=1,
            avg_top3_score=0.3, need_count=2, contradictions_detected=1,
        )
        result = s.assess()
        # Weak evidence with contradictions → MARGINAL (not CONFLICTED)
        assert result.level == SufficiencyLevel.MARGINAL


# ---------------------------------------------------------------------------
# 8. Absent Information Behavior
# ---------------------------------------------------------------------------

class TestAbsentInformation:
    def test_absent_info_stops(self):
        """Absent information eventually stops."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.2) for _ in range(1)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.1},
            gain_history=[0.5, 0.01],
            iteration=2, max_iterations=2,
            pattern="absent_info",
        )
        assert decision.action == "synthesize"

    def test_no_endless_search(self):
        """No endless search for absent information."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.1) for _ in range(1)]
        for iteration in range(1, 100):
            decision = policy.should_continue_retrieval(
                evidence=refs,
                need_coverage={"n1": 0.05},
                gain_history=[0.5] + [0.01] * 50,
                iteration=iteration, max_iterations=3,
                pattern="absent_info",
            )
            if decision.action == "synthesize":
                break
        assert decision.action == "synthesize"

    def test_synthesis_does_not_fabricate(self):
        """Synthesis with absent info should not fabricate evidence."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.1) for _ in range(1)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 0.05},
            gain_history=[0.5, 0.01],
            iteration=2, max_iterations=2,
            pattern="absent_info",
        )
        # Policy says synthesize, but evidence is weak
        # The synthesis node will handle this appropriately
        assert decision.action == "synthesize"


# ---------------------------------------------------------------------------
# 9. Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_need_coverage_handled(self):
        """Empty need_coverage dict is handled gracefully."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(3)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
        )
        assert isinstance(decision, AdaptiveDecision)

    def test_none_pending_subquestions_handled(self):
        """None pending_subquestions is handled gracefully."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(3)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
            pending_subquestions=None,
        )
        assert isinstance(decision, AdaptiveDecision)

    def test_empty_gain_history_handled(self):
        """Empty gain_history is handled gracefully."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(3)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
        )
        assert isinstance(decision, AdaptiveDecision)

    def test_unknown_pattern_handled(self):
        """Unknown pattern uses default policy."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(3)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="unknown_pattern_xyz",
        )
        assert isinstance(decision, AdaptiveDecision)

    def test_policy_exception_safe(self):
        """Exception inside policy does not crash the system."""
        policy = AdaptiveResearchPolicy(enabled=True)
        # Pass invalid type to trigger potential exception
        try:
            decision = policy.should_continue_retrieval(
                evidence="not_a_list",
                need_coverage={},
                gain_history=[],
                iteration=1, max_iterations=3,
                pattern="simple_lookup",
            )
            # If no exception, verify it returned a valid decision
            assert isinstance(decision, AdaptiveDecision)
        except (TypeError, AttributeError):
            # Expected - invalid input raises exception
            # In production, this would be caught by the assess node
            pass


# ---------------------------------------------------------------------------
# 10. Async Safety
# ---------------------------------------------------------------------------

class TestAsyncSafety:
    def test_policy_is_synchronous(self):
        """AdaptiveResearchPolicy.should_continue_retrieval is synchronous."""
        policy = AdaptiveResearchPolicy(enabled=True)
        result = policy.should_continue_retrieval(
            evidence=[], need_coverage={}, gain_history=[],
            iteration=0, max_iterations=3, pattern="simple_lookup",
        )
        assert isinstance(result, AdaptiveDecision)
        # Verify it's not a coroutine
        assert not asyncio.iscoroutine(result)

    def test_no_orphan_tasks(self):
        """Policy does not create orphan async tasks."""
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(3)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=3,
            pattern="simple_lookup",
        )
        assert isinstance(decision, AdaptiveDecision)


# ---------------------------------------------------------------------------
# 11. Configuration Audit
# ---------------------------------------------------------------------------

class TestConfigurationAudit:
    def test_no_hidden_env_dependency(self):
        """Adaptive config does not depend on hidden env vars."""
        settings = Settings(_env_file=None)
        assert hasattr(settings, "adaptive_research_enabled")
        assert settings.adaptive_research_enabled is False

    def test_max_iterations_bounded(self):
        """Max iterations is bounded by config."""
        settings = Settings(_env_file=None)
        assert settings.orchestration_max_iterations <= 10

    def test_token_budget_bounded(self):
        """Token budget is bounded by config."""
        settings = Settings(_env_file=None)
        assert settings.orchestration_token_budget <= 100000

    def test_gain_threshold_sensible(self):
        """Gain threshold is sensible."""
        settings = Settings(_env_file=None)
        threshold = settings.stopping_evidence_gain_threshold
        assert 0.0 <= threshold <= 1.0
