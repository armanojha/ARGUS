"""Tests for Adaptive Research Orchestration (Phase 24.1)."""
from __future__ import annotations

import pytest
from uuid import uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.adaptive_research import (
    AdaptiveDecision,
    AdaptiveResearchPolicy,
    MarginalGainCalculator,
    MarginalGainResult,
    PatternPolicy,
    PatternSpecificPolicies,
    ResearchSufficiency,
    SufficiencyLevel,
    SufficiencyResult,
    SynthesisGate,
    SynthesisGateResult,
)


def _make_ref(score: float = 0.8, doc_id: str | None = None) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4() if doc_id is None else doc_id,
        source_id=uuid4(),
        source_path="test.md",
        source_type=SourceType.MARKDOWN,
        text="Test evidence chunk with relevant information about the topic.",
        score=score,
        rank=1,
        metadata={},
    )


# ---------------------------------------------------------------------------
# ResearchSufficiency
# ---------------------------------------------------------------------------

class TestResearchSufficiency:
    def test_empty_evidence_insufficient(self):
        s = ResearchSufficiency(
            evidence_count=0, coverage_score=0.0, source_diversity=0,
            avg_top3_score=0.0, need_count=0, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.INSUFFICIENT
        assert not result.conflict_detected

    def test_strong_evidence_no_conflict(self):
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.STRONG
        assert not result.conflict_detected
        assert result.coverage_ok

    def test_sufficient_evidence_no_conflict(self):
        s = ResearchSufficiency(
            evidence_count=3, coverage_score=0.8, source_diversity=2,
            avg_top3_score=0.55, need_count=2, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.SUFFICIENT
        assert not result.conflict_detected

    def test_marginal_evidence(self):
        s = ResearchSufficiency(
            evidence_count=2, coverage_score=0.5, source_diversity=1,
            avg_top3_score=0.4, need_count=2, contradictions_detected=0,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.MARGINAL

    def test_strong_evidence_with_contradiction_is_conflicted(self):
        """Strong evidence + contradictions = CONFLICTED, not INSUFFICIENT."""
        s = ResearchSufficiency(
            evidence_count=6, coverage_score=1.0, source_diversity=3,
            avg_top3_score=0.75, need_count=2, contradictions_detected=1,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.CONFLICTED
        assert result.conflict_detected
        assert result.coverage_ok
        assert result.conflict_count == 1

    def test_sufficient_evidence_with_contradiction_is_conflicted(self):
        """Sufficient evidence + contradictions = CONFLICTED."""
        s = ResearchSufficiency(
            evidence_count=3, coverage_score=0.8, source_diversity=2,
            avg_top3_score=0.55, need_count=2, contradictions_detected=2,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.CONFLICTED
        assert result.conflict_detected

    def test_insufficient_evidence_with_contradiction_stays_insufficient(self):
        """Insufficient evidence + contradictions = INSUFFICIENT (coverage is primary)."""
        s = ResearchSufficiency(
            evidence_count=1, coverage_score=0.1, source_diversity=1,
            avg_top3_score=0.3, need_count=3, contradictions_detected=1,
        )
        result = s.assess()
        assert result.level == SufficiencyLevel.INSUFFICIENT
        assert result.conflict_detected

    def test_complex_needs_low_coverage_marginal(self):
        s = ResearchSufficiency(
            evidence_count=3, coverage_score=0.3, source_diversity=2,
            avg_top3_score=0.55, need_count=5, contradictions_detected=0,
        )
        result = s.assess()
        # 3 evidence=2, 0.3 cov=1, 2 sources=2, 0.55 score=2 -> 7 = SUFFICIENT
        # need_count > 3 and coverage < 0.5 -> downgrades to MARGINAL
        assert result.level == SufficiencyLevel.MARGINAL

    def test_from_state_with_unplanned(self):
        evidence = [_make_ref(0.8)]
        need_coverage = {"_unplanned": 1.0}
        s = ResearchSufficiency.from_state(evidence, need_coverage)
        assert s.coverage_score == 1.0
        assert s.evidence_count == 1

    def test_from_state_with_planned(self):
        evidence = [_make_ref(0.8), _make_ref(0.7)]
        need_coverage = {"abc12345": 1.0, "def67890": 0.5}
        s = ResearchSufficiency.from_state(evidence, need_coverage)
        assert s.coverage_score == pytest.approx(0.75)
        assert s.need_count == 2


# ---------------------------------------------------------------------------
# MarginalGainCalculator
# ---------------------------------------------------------------------------

class TestMarginalGainCalculator:
    def test_short_history_no_stop(self):
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.5])
        assert not result.should_stop
        assert "too short" in result.reason

    def test_high_gain_no_stop(self):
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.3, 0.2])
        assert not result.should_stop

    def test_low_gain_stops(self):
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.3, 0.05])
        assert result.should_stop
        assert result.gain_ratio == 0.05

    def test_boundary_gain(self):
        calc = MarginalGainCalculator(threshold=0.10)
        result = calc.evaluate([0.3, 0.10])
        assert result.should_stop
        assert result.gain_ratio == 0.10


# ---------------------------------------------------------------------------
# PatternSpecificPolicies
# ---------------------------------------------------------------------------

class TestPatternSpecificPolicies:
    def test_conflict_policy(self):
        p = PatternSpecificPolicies.get_policy("conflict")
        assert p.min_iterations == 2
        assert p.require_contradiction_resolution is True
        assert p.allow_fast_path is False

    def test_multi_hop_policy(self):
        p = PatternSpecificPolicies.get_policy("multi_hop")
        assert p.min_iterations == 2
        assert p.require_coverage_above == 0.7

    def test_simple_lookup_policy(self):
        p = PatternSpecificPolicies.get_policy("simple_lookup")
        assert p.max_iterations == 1
        assert p.allow_fast_path is True

    def test_absent_info_policy(self):
        p = PatternSpecificPolicies.get_policy("absent_info")
        assert p.max_iterations == 2
        assert p.require_coverage_above == 0.0

    def test_unknown_pattern_defaults(self):
        p = PatternSpecificPolicies.get_policy("unknown_pattern")
        assert p.min_iterations == 1
        assert p.max_iterations == 3


# ---------------------------------------------------------------------------
# SynthesisGate
# ---------------------------------------------------------------------------

class TestSynthesisGate:
    def test_empty_evidence_blocks(self):
        gate = SynthesisGate(min_evidence_count=2)
        result = gate.check([], {})
        assert not result.should_synthesize
        assert "Insufficient" in result.reason

    def test_enough_evidence_passes(self):
        gate = SynthesisGate(min_evidence_count=2, min_coverage=0.0, min_sources=1)
        refs = [_make_ref(0.8), _make_ref(0.7)]
        result = gate.check(refs, {"_unplanned": 1.0})
        assert result.should_synthesize

    def test_low_coverage_blocks(self):
        gate = SynthesisGate(min_evidence_count=1, min_coverage=0.5)
        refs = [_make_ref(0.8)]
        result = gate.check(refs, {"need1": 0.2})
        assert not result.should_synthesize
        assert "Coverage" in result.reason

    def test_insufficient_sources_blocks(self):
        gate = SynthesisGate(min_evidence_count=1, min_coverage=0.0, min_sources=2)
        refs = [_make_ref(0.8)]
        result = gate.check(refs, {"_unplanned": 1.0})
        assert not result.should_synthesize
        assert "source diversity" in result.reason


# ---------------------------------------------------------------------------
# AdaptiveResearchPolicy
# ---------------------------------------------------------------------------

class TestAdaptiveResearchPolicy:
    def test_disabled_always_continues(self):
        policy = AdaptiveResearchPolicy(enabled=False)
        decision = policy.should_continue_retrieval(
            evidence=[], need_coverage={}, gain_history=[],
            iteration=0, max_iterations=3, pattern="simple_lookup",
        )
        assert decision.action == "continue_retrieval"
        assert "disabled" in decision.reason

    def test_budget_exhausted_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.8)], need_coverage={"_unplanned": 1.0},
            gain_history=[0.5], iteration=3, max_iterations=3,
            pattern="simple_lookup",
        )
        assert decision.action == "synthesize"
        assert "limit" in decision.reason

    def test_strong_evidence_synthesizes(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5, 0.3],
            iteration=2, max_iterations=4,
            pattern="complex_research",
        )
        assert decision.action == "synthesize"

    def test_pending_subqueries_continue(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        decision = policy.should_continue_retrieval(
            evidence=[_make_ref(0.8)], need_coverage={"_unplanned": 1.0},
            gain_history=[0.5], iteration=1, max_iterations=3,
            pattern="complex_research",
            pending_subquestions=["what is X"],
        )
        assert decision.action == "continue_retrieval"
        assert "pending" in decision.reason

    def test_min_iterations_enforced(self):
        policy = AdaptiveResearchPolicy(enabled=True)
        refs = [_make_ref(0.8) for _ in range(6)]
        decision = policy.should_continue_retrieval(
            evidence=refs,
            need_coverage={"n1": 1.0, "n2": 1.0},
            gain_history=[0.5],
            iteration=1, max_iterations=4,
            pattern="conflict",
        )
        assert decision.action == "continue_retrieval"
        assert "min" in decision.reason

    def test_conflict_pattern_investigates(self):
        """Conflict pattern with sufficient evidence should trigger investigation."""
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

    def test_non_conflict_pattern_synthesizes_despite_contradictions(self):
        """Non-conflict pattern with sufficient evidence synthesizes despite contradictions."""
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
        assert decision.action == "synthesize"
