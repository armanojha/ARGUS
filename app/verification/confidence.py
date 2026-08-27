"""Confidence Scoring (Phase 04).

Implements diagnostic composite confidence scoring per V2 §9.3.
Scores are explicitly diagnostic, not calibrated probabilities.
"""

from __future__ import annotations

from app.graph.models import Claim
from app.logging_config import get_logger
from app.verification.models import ConfidenceComponents, VerificationEvidence

logger = get_logger("argus.verification.confidence")


class ConfidenceScorer:
    """Computes diagnostic confidence components for verification results."""

    def __init__(self):
        pass

    def score_verification(
        self,
        claim: Claim,
        supporting_evidence: list[VerificationEvidence],
        contradicting_evidence: list[VerificationEvidence],
        verifier_judgment: float,
    ) -> ConfidenceComponents:
        """Compute all confidence components for a verification result."""
        # Evidence coverage: fraction of claim propositions covered by evidence
        evidence_coverage = self._compute_evidence_coverage(
            claim, supporting_evidence, contradicting_evidence
        )

        # Source quality: average quality of sources
        source_quality = self._compute_source_quality(supporting_evidence)

        # Cross-source agreement: agreement among independent sources
        cross_source_agreement = self._compute_cross_source_agreement(
            supporting_evidence, contradicting_evidence
        )

        # Temporal relevance: how current/relevant the evidence is
        temporal_relevance = self._compute_temporal_relevance(
            claim, supporting_evidence
        )

        # Retrieval rank: quality of retrieval ranking
        retrieval_rank = self._compute_retrieval_rank(supporting_evidence)

        return ConfidenceComponents(
            evidence_coverage=evidence_coverage,
            source_quality=source_quality,
            cross_source_agreement=cross_source_agreement,
            temporal_relevance=temporal_relevance,
            retrieval_rank=retrieval_rank,
            verifier_judgment=verifier_judgment,
        )

    def _compute_evidence_coverage(
        self,
        claim: Claim,
        supporting_evidence: list[VerificationEvidence],
        contradicting_evidence: list[VerificationEvidence],
    ) -> float:
        """Compute fraction of claim covered by evidence.

        Simplified: ratio of supporting evidence to total evidence,
        adjusted for claim complexity.
        """
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        if total_evidence == 0:
            return 0.0

        # Base coverage from supporting evidence
        support_ratio = len(supporting_evidence) / total_evidence

        # Adjust for claim complexity (more entities = harder to cover)
        complexity_factor = 1.0
        if claim.subject_entity_id and claim.object_entity_id:
            complexity_factor = 0.8  # Binary relation harder to fully cover
        elif claim.subject_entity_id:
            complexity_factor = 0.9

        return min(1.0, support_ratio * complexity_factor)

    def _compute_source_quality(self, evidence: list[VerificationEvidence]) -> float:
        """Compute average source quality score.

        Based on source type, retrieval score, and domain authority.
        """
        if not evidence:
            return 0.0

        quality_scores = []
        for ev in evidence:
            # Base quality from retrieval score
            base_quality = ev.relevance_score

            # Adjust for source type
            source_type = ev.source_type.lower() if isinstance(ev.source_type, str) else str(ev.source_type).lower()
            type_multiplier = {
                "pdf": 0.9,        # Academic/official docs
                "markdown": 0.7,   # Notes
                "text": 0.6,       # Plain text
                "html": 0.5,       # Web pages
                "other": 0.5,
            }.get(source_type, 0.5)

            quality_scores.append(base_quality * type_multiplier)

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    def _compute_cross_source_agreement(
        self,
        supporting_evidence: list[VerificationEvidence],
        contradicting_evidence: list[VerificationEvidence],
    ) -> float:
        """Compute agreement across independent sources.

        High when multiple independent sources agree, low when they contradict.
        """
        total = len(supporting_evidence) + len(contradicting_evidence)
        if total == 0:
            return 0.0

        # Simple agreement ratio
        agreement = len(supporting_evidence) / total

        # Boost if multiple independent sources (different source_ids)
        source_ids = set()
        for ev in supporting_evidence:
            source_ids.add(ev.source_id)

        independence_boost = min(0.2, len(source_ids) * 0.05)  # Up to 0.2 boost

        return min(1.0, agreement + independence_boost)

    def _compute_temporal_relevance(
        self,
        claim: Claim,
        evidence: list[VerificationEvidence],
    ) -> float:
        """Compute temporal relevance of evidence to claim validity period."""
        if not evidence:
            return 0.0

        if not claim.valid_from:
            # No temporal constraint on claim
            return 0.7  # Neutral-good default

        # Check how many evidence items are temporally relevant
        relevant_count = 0
        for ev in evidence:
            # Would need publication date from evidence
            # Simplified: assume evidence is relevant if claim has validity period
            relevant_count += 1

        logger.debug("temporal_relevance_stub", note="implementation deferred to Phase 09")
        return relevant_count / len(evidence) if evidence else 0.0

    def _compute_retrieval_rank(self, evidence: list[VerificationEvidence]) -> float:
        """Compute retrieval rank quality.

        Based on retrieval scores of evidence items.
        """
        if not evidence:
            return 0.0

        # Average retrieval score (normalized)
        scores = [ev.relevance_score for ev in evidence]
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-1 (assuming scores are roughly 0-1)
        return min(1.0, avg_score)


def compute_composite_confidence(components: ConfidenceComponents) -> float:
    """Compute composite diagnostic confidence score.

    Simple average of all components. Explicitly diagnostic,
    not a calibrated probability.
    """
    return components.composite()


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create the singleton confidence scorer."""
    return ConfidenceScorer()