"""Adaptive Research Orchestration (Phase 24).

Provides deterministic, zero-LLM research sufficiency assessment that
replaces the LLM-based assess node for simple and moderate queries. The
system decides how much research is necessary before producing an answer,
avoiding both premature stopping and unnecessary research loops.

Components:
    ResearchSufficiency — deterministic sufficiency model
    AdaptiveResearchPolicy — decides whether to loop, synthesize, or skip
    MarginalGainCalculator — detects diminishing returns
    PatternSpecificPolicies — query-pattern-specific research behaviors
    SynthesisGate — pre-synthesis quality gate

All components are feature-flagged via `adaptive_research_enabled` (default
False) and must not change behavior when disabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.planner import QueryPlan

logger = get_logger("argus.orchestration.adaptive_research")


# ---------------------------------------------------------------------------
# Research Sufficiency Model
# ---------------------------------------------------------------------------

class SufficiencyLevel(Enum):
    INSUFFICIENT = "insufficient"
    MARGINAL = "marginal"
    SUFFICIENT = "sufficient"
    STRONG = "strong"


@dataclass
class ResearchSufficiency:
    """Deterministic evidence sufficiency assessment (no LLM).

    Combines four signals into a single sufficiency level:
      1. Evidence count — enough chunks to ground an answer
      2. Coverage score — evidence needs are met
      3. Source diversity — multiple documents support the answer
      4. Score quality — top evidence is high-confidence
    """

    evidence_count: int = 0
    coverage_score: float = 0.0
    source_diversity: int = 0
    avg_top3_score: float = 0.0
    need_count: int = 0
    contradictions_detected: int = 0
    pattern: str = ""

    _MIN_EVIDENCE = 2
    _MIN_SOURCES = 2
    _STRONG_EVIDENCE = 5
    _STRONG_SOURCES = 3
    _GOOD_COVERAGE = 0.8
    _STRONG_COVERAGE = 1.0
    _GOOD_SCORE = 0.5
    _STRONG_SCORE = 0.7

    def assess(self) -> SufficiencyLevel:
        """Classify overall sufficiency from multiple evidence signals."""
        level = self._assess_base()

        if self.contradictions_detected > 0:
            level = self._downgrade_for_contradictions(level)

        if self.need_count > 3 and level == SufficiencyLevel.SUFFICIENT:
            level = self._adjust_for_complex_needs(level)

        return level

    def _assess_base(self) -> SufficiencyLevel:
        score = 0

        if self.evidence_count >= self._STRONG_EVIDENCE:
            score += 3
        elif self.evidence_count >= self._MIN_EVIDENCE:
            score += 2
        elif self.evidence_count >= 1:
            score += 1

        if self.coverage_score >= self._STRONG_COVERAGE:
            score += 3
        elif self.coverage_score >= self._GOOD_COVERAGE:
            score += 2
        elif self.coverage_score > 0:
            score += 1

        if self.source_diversity >= self._STRONG_SOURCES:
            score += 3
        elif self.source_diversity >= self._MIN_SOURCES:
            score += 2
        elif self.source_diversity >= 1:
            score += 1

        if self.avg_top3_score >= self._STRONG_SCORE:
            score += 3
        elif self.avg_top3_score >= self._GOOD_SCORE:
            score += 2
        elif self.avg_top3_score > 0:
            score += 1

        if score >= 10:
            return SufficiencyLevel.STRONG
        if score >= 7:
            return SufficiencyLevel.SUFFICIENT
        if score >= 4:
            return SufficiencyLevel.MARGINAL
        return SufficiencyLevel.INSUFFICIENT

    def _downgrade_for_contradictions(self, level: SufficiencyLevel) -> SufficiencyLevel:
        if self.contradictions_detected >= 2:
            return SufficiencyLevel.INSUFFICIENT
        if level == SufficiencyLevel.STRONG:
            return SufficiencyLevel.SUFFICIENT
        if level == SufficiencyLevel.SUFFICIENT:
            return SufficiencyLevel.MARGINAL
        return level

    def _adjust_for_complex_needs(self, level: SufficiencyLevel) -> SufficiencyLevel:
        if self.coverage_score < 0.5:
            return SufficiencyLevel.MARGINAL
        return level

    @classmethod
    def from_state(
        cls,
        evidence: list[EvidenceRef],
        need_coverage: dict[str, float],
        contradictions_detected: int = 0,
        pattern: str = "",
    ) -> ResearchSufficiency:
        """Build sufficiency from accumulated retrieval state."""
        avg_top3 = 0.0
        if evidence:
            top = sorted(evidence, key=lambda r: r.score, reverse=True)[:3]
            avg_top3 = sum(r.score for r in top) / len(top)

        coverage_vals = [v for k, v in need_coverage.items() if not k.startswith("_")]
        avg_coverage = sum(coverage_vals) / len(coverage_vals) if coverage_vals else (
            1.0 if need_coverage.get("_unplanned") == 1.0 else 0.0
        )

        return cls(
            evidence_count=len(evidence),
            coverage_score=avg_coverage,
            source_diversity=len({r.document_id for r in evidence}),
            avg_top3_score=avg_top3,
            need_count=len(coverage_vals),
            contradictions_detected=contradictions_detected,
            pattern=pattern,
        )


# ---------------------------------------------------------------------------
# Marginal Gain Calculator
# ---------------------------------------------------------------------------

@dataclass
class MarginalGainResult:
    should_stop: bool
    gain_ratio: float
    threshold: float
    reason: str


class MarginalGainCalculator:
    """Detect diminishing returns from retrieval gain history."""

    def __init__(self, threshold: float = 0.10, min_window: int = 2) -> None:
        self.threshold = threshold
        self.min_window = min_window

    def evaluate(self, gain_history: list[float]) -> MarginalGainResult:
        if len(gain_history) < self.min_window:
            return MarginalGainResult(
                should_stop=False,
                gain_ratio=0.0,
                threshold=self.threshold,
                reason=f"Gain history too short ({len(gain_history)} < {self.min_window})",
            )

        last_gain = gain_history[-1]
        if last_gain <= self.threshold:
            return MarginalGainResult(
                should_stop=True,
                gain_ratio=last_gain,
                threshold=self.threshold,
                reason=f"Negligible gain ({last_gain:.4f} <= {self.threshold})",
            )

        if len(gain_history) >= 3:
            recent_avg = sum(gain_history[-2:]) / 2
            if recent_avg <= self.threshold * 0.5:
                return MarginalGainResult(
                    should_stop=True,
                    gain_ratio=recent_avg,
                    threshold=self.threshold,
                    reason=f"Sustained low gain (avg {recent_avg:.4f})",
                )

        return MarginalGainResult(
            should_stop=False,
            gain_ratio=last_gain,
            threshold=self.threshold,
            reason=f"Gain still meaningful ({last_gain:.4f} > {self.threshold})",
        )


# ---------------------------------------------------------------------------
# Pattern-Specific Policies
# ---------------------------------------------------------------------------

@dataclass
class PatternPolicy:
    min_iterations: int = 1
    max_iterations: int = 3
    require_coverage_above: float = 0.5
    require_contradiction_resolution: bool = False
    allow_fast_path: bool = True


class PatternSpecificPolicies:
    """Query-pattern-specific research behaviors."""

    _POLICIES: dict[str, PatternPolicy] = {
        "conflict": PatternPolicy(
            min_iterations=2,
            max_iterations=4,
            require_coverage_above=0.6,
            require_contradiction_resolution=True,
            allow_fast_path=False,
        ),
        "multi_hop": PatternPolicy(
            min_iterations=2,
            max_iterations=4,
            require_coverage_above=0.7,
            require_contradiction_resolution=False,
            allow_fast_path=False,
        ),
        "complex_research": PatternPolicy(
            min_iterations=1,
            max_iterations=3,
            require_coverage_above=0.6,
            require_contradiction_resolution=False,
            allow_fast_path=False,
        ),
        "absent_info": PatternPolicy(
            min_iterations=1,
            max_iterations=2,
            require_coverage_above=0.0,
            require_contradiction_resolution=False,
            allow_fast_path=False,
        ),
        "simple_lookup": PatternPolicy(
            min_iterations=1,
            max_iterations=1,
            require_coverage_above=0.0,
            require_contradiction_resolution=False,
            allow_fast_path=True,
        ),
    }

    _DEFAULT = PatternPolicy()

    @classmethod
    def get_policy(cls, pattern: str) -> PatternPolicy:
        return cls._POLICIES.get(pattern, cls._DEFAULT)


# ---------------------------------------------------------------------------
# Synthesis Gate
# ---------------------------------------------------------------------------

@dataclass
class SynthesisGateResult:
    should_synthesize: bool
    reason: str
    min_evidence_count: int
    min_coverage: float
    min_sources: int


class SynthesisGate:
    """Pre-synthesis quality gate: prevents synthesis when evidence is clearly insufficient."""

    def __init__(
        self,
        min_evidence_count: int = 1,
        min_coverage: float = 0.0,
        min_sources: int = 1,
    ) -> None:
        self.min_evidence_count = min_evidence_count
        self.min_coverage = min_coverage
        self.min_sources = min_sources

    def check(
        self,
        evidence: list[EvidenceRef],
        need_coverage: dict[str, float],
    ) -> SynthesisGateResult:
        if len(evidence) < self.min_evidence_count:
            return SynthesisGateResult(
                should_synthesize=False,
                reason=f"Insufficient evidence ({len(evidence)} < {self.min_evidence_count})",
                min_evidence_count=self.min_evidence_count,
                min_coverage=self.min_coverage,
                min_sources=self.min_sources,
            )

        coverage_vals = [v for k, v in need_coverage.items() if not k.startswith("_")]
        avg_coverage = sum(coverage_vals) / len(coverage_vals) if coverage_vals else (
            1.0 if need_coverage.get("_unplanned") == 1.0 else 0.0
        )
        if avg_coverage < self.min_coverage:
            return SynthesisGateResult(
                should_synthesize=False,
                reason=f"Coverage too low ({avg_coverage:.2f} < {self.min_coverage})",
                min_evidence_count=self.min_evidence_count,
                min_coverage=self.min_coverage,
                min_sources=self.min_sources,
            )

        sources = len({r.document_id for r in evidence})
        if sources < self.min_sources:
            return SynthesisGateResult(
                should_synthesize=False,
                reason=f"Insufficient source diversity ({sources} < {self.min_sources})",
                min_evidence_count=self.min_evidence_count,
                min_coverage=self.min_coverage,
                min_sources=self.min_sources,
            )

        return SynthesisGateResult(
            should_synthesize=True,
            reason="Evidence meets minimum quality thresholds",
            min_evidence_count=self.min_evidence_count,
            min_coverage=self.min_coverage,
            min_sources=self.min_sources,
        )


# ---------------------------------------------------------------------------
# Adaptive Research Policy
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDecision:
    action: str  # "continue_retrieval", "synthesize", "skip_to_synthesis", "force_synthesis"
    reason: str
    sufficiency_level: str
    iteration: int
    max_iterations: int


class AdaptiveResearchPolicy:
    """Decides whether to continue retrieval or synthesize.

    Combines:
      - ResearchSufficiency assessment
      - MarginalGainCalculator
      - PatternSpecificPolicies
      - SynthesisGate
    """

    def __init__(
        self,
        *,
        marginal_gain_threshold: float = 0.10,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.gain_calculator = MarginalGainCalculator(threshold=marginal_gain_threshold)
        self.synthesis_gate = SynthesisGate()

    def should_continue_retrieval(
        self,
        evidence: list[EvidenceRef],
        need_coverage: dict[str, float],
        gain_history: list[float],
        iteration: int,
        max_iterations: int,
        pattern: str,
        contradictions_detected: int = 0,
        pending_subquestions: list[str] | None = None,
    ) -> AdaptiveDecision:
        """Make a deterministic research decision."""
        if not self.enabled:
            return AdaptiveDecision(
                action="continue_retrieval",
                reason="Adaptive research disabled; using default loop",
                sufficiency_level="unknown",
                iteration=iteration,
                max_iterations=max_iterations,
            )

        policy = PatternSpecificPolicies.get_policy(pattern)

        if iteration >= max_iterations:
            return AdaptiveDecision(
                action="synthesize",
                reason=f"Iteration limit reached ({iteration}/{max_iterations})",
                sufficiency_level="budget_exhausted",
                iteration=iteration,
                max_iterations=max_iterations,
            )

        if pending_subquestions and iteration < policy.max_iterations:
            return AdaptiveDecision(
                action="continue_retrieval",
                reason=f"{len(pending_subquestions)} pending subquery(ies) remaining",
                sufficiency_level="pending_work",
                iteration=iteration,
                max_iterations=max_iterations,
            )

        sufficiency = ResearchSufficiency.from_state(
            evidence, need_coverage, contradictions_detected, pattern
        )
        level = sufficiency.assess()

        if iteration < policy.min_iterations:
            return AdaptiveDecision(
                action="continue_retrieval",
                reason=f"Pattern '{pattern}' requires min {policy.min_iterations} iterations",
                sufficiency_level=level.value,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        gain_result = self.gain_calculator.evaluate(gain_history)
        if gain_result.should_stop and level in (SufficiencyLevel.SUFFICIENT, SufficiencyLevel.STRONG):
            return AdaptiveDecision(
                action="synthesize",
                reason=f"Sufficient evidence + {gain_result.reason}",
                sufficiency_level=level.value,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        if level == SufficiencyLevel.STRONG:
            return AdaptiveDecision(
                action="synthesize",
                reason=f"Strong evidence (score={sufficiency.avg_top3_score:.3f}, "
                       f"cov={sufficiency.coverage_score:.2f}, src={sufficiency.source_diversity})",
                sufficiency_level=level.value,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        if level == SufficiencyLevel.SUFFICIENT and not pending_subquestions:
            return AdaptiveDecision(
                action="synthesize",
                reason=f"Sufficient evidence, no pending work",
                sufficiency_level=level.value,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        if level == SufficiencyLevel.MARGINAL:
            if gain_result.should_stop:
                return AdaptiveDecision(
                    action="force_synthesis",
                    reason=f"Marginal evidence but {gain_result.reason}; forcing synthesis",
                    sufficiency_level=level.value,
                    iteration=iteration,
                    max_iterations=max_iterations,
                )
            return AdaptiveDecision(
                action="continue_retrieval",
                reason=f"Marginal evidence, gain still meaningful",
                sufficiency_level=level.value,
                iteration=iteration,
                max_iterations=max_iterations,
            )

        return AdaptiveDecision(
            action="continue_retrieval",
            reason=f"Insufficient evidence (level={level.value})",
            sufficiency_level=level.value,
            iteration=iteration,
            max_iterations=max_iterations,
        )


def create_adaptive_research_policy(settings: Any = None) -> AdaptiveResearchPolicy:
    """Create an AdaptiveResearchPolicy from settings."""
    enabled = False
    gain_threshold = 0.10

    if settings is not None:
        enabled = getattr(settings, "adaptive_research_enabled", False)
        gain_threshold = getattr(settings, "stopping_evidence_gain_threshold", 0.05)

    return AdaptiveResearchPolicy(
        marginal_gain_threshold=gain_threshold,
        enabled=enabled,
    )
