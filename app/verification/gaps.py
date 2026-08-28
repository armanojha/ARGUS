"""Evidence Gap Detection & Re-Retrieval Trigger (Phase 04).

Detects evidence gaps from verification results and triggers
additional retrieval cycles through the Phase 02 orchestration loop.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.graph.store import EvidenceGraphStore
from app.logging_config import get_logger
from app.orchestration.graph import run_query
from app.orchestration.models import OrchestrationResult
from app.verification.models import (
    EvidenceGap,
    ReRetrievalTrigger,
    VerificationBatchResult,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger("argus.verification.gaps")


class EvidenceGapDetector:
    """Detects evidence gaps from verification results."""

    def __init__(self, graph_store: EvidenceGraphStore | None = None):
        self.graph_store = graph_store

    def detect_gaps(self, batch_result: VerificationBatchResult) -> list[EvidenceGap]:
        """Detect evidence gaps from a batch of verification results."""
        gaps = []

        for result in batch_result.results:
            claim_gaps = self._analyze_claim_gaps(result)
            gaps.extend(claim_gaps)

        logger.info("evidence_gap_detection_completed", total_gaps=len(gaps))
        return gaps

    def _analyze_claim_gaps(self, result: VerificationResult) -> list[EvidenceGap]:
        """Analyze a single verification result for evidence gaps."""
        gaps = []

        if result.status == VerificationStatus.UNSUPPORTED:
            gaps.append(EvidenceGap(
                claim_id=result.claim_id,
                gap_type="no_evidence",
                description="No supporting evidence found for claim",
                suggested_query=self._generate_query_for_claim(result),
                priority=0.9,
            ))

        elif result.status == VerificationStatus.PARTIAL:
            # Check what's missing
            if result.evidence_coverage < 0.5:
                gaps.append(EvidenceGap(
                    claim_id=result.claim_id,
                    gap_type="partial_coverage",
                    description=f"Low evidence coverage ({result.evidence_coverage:.0%})",
                    suggested_query=self._generate_query_for_claim(result),
                    priority=0.7,
                ))

            if result.source_quality < 0.5:
                gaps.append(EvidenceGap(
                    claim_id=result.claim_id,
                    gap_type="source_quality",
                    description=f"Low source quality ({result.source_quality:.0%})",
                    suggested_query=self._generate_query_for_claim(result, focus="authoritative"),
                    priority=0.6,
                ))

            if result.temporal_relevance < 0.5:
                gaps.append(EvidenceGap(
                    claim_id=result.claim_id,
                    gap_type="temporal_mismatch",
                    description=f"Low temporal relevance ({result.temporal_relevance:.0%})",
                    suggested_query=self._generate_query_for_claim(result, focus="recent"),
                    priority=0.6,
                ))

        elif result.status == VerificationStatus.CONTRADICTED:
            # Contradicted claims may need more evidence to resolve
            gaps.append(EvidenceGap(
                claim_id=result.claim_id,
                gap_type="contradiction_resolution",
                description="Claim contradicted by evidence; needs resolution",
                suggested_query=self._generate_query_for_claim(result, focus="resolve"),
                priority=0.8,
            ))

        return gaps

    def _generate_query_for_claim(
        self,
        result: VerificationResult,
        focus: str = "general",
    ) -> str:
        """Generate a targeted retrieval query for a claim with gaps."""
        # Use claim text when available, fall back to claim ID
        base_query = f"Evidence for: {result.claim_text}" if result.claim_text else f"Evidence for: {result.claim_id}"

        if focus == "authoritative":
            return f"Authoritative sources for: {result.claim_text}" if result.claim_text else f"Authoritative sources for: {result.claim_id}"
        elif focus == "recent":
            return f"Recent evidence for: {result.claim_text}" if result.claim_text else f"Recent evidence for: {result.claim_id}"
        elif focus == "resolve":
            return f"Resolving contradictory evidence for: {result.claim_text}" if result.claim_text else f"Resolving contradictory evidence for: {result.claim_id}"

        return base_query


class ReRetrievalManager:
    """Manages re-retrieval triggered by evidence gaps.

    MVP requirement: exactly one additional retrieval cycle
    when evidence is inadequate.
    """

    def __init__(
        self,
        graph_store: EvidenceGraphStore | None = None,
        settings: Settings | None = None,
    ):
        self.graph_store = graph_store
        self.settings = settings or get_settings()

    async def trigger_re_retrieval(
        self,
        trigger: ReRetrievalTrigger,
        router: Any = None,  # LLMRouter
    ) -> OrchestrationResult | None:
        """Trigger re-retrieval for evidence gaps.

        MVP: exactly one additional retrieval cycle.
        """
        if not trigger.gaps:
            logger.info("re_retrieval_no_gaps")
            return None

        # For MVP, only trigger ONE additional cycle
        # Use the highest priority gap
        priority_gaps = sorted(trigger.gaps, key=lambda g: g.priority, reverse=True)
        top_gap = priority_gaps[0]

        if not top_gap.suggested_query:
            logger.warning("re_retrieval_no_query", gap=top_gap.claim_id)
            return None

        logger.info("re_retrieval_triggered", gap=top_gap.claim_id, query=top_gap.suggested_query[:100])

        # Run the orchestration loop with the gap query
        # This uses the existing Phase 02 run_query function
        try:
            result = await run_query(
                top_gap.suggested_query,
                request_id=trigger.context.get("request_id"),
            )
            logger.info("re_retrieval_completed", iterations=result.iterations_used, evidence=len(result.citations))
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("re_retrieval_failed", error=str(exc))
            return None

    def create_trigger_from_batch(
        self,
        batch_result: VerificationBatchResult,
        original_query: str,
        request_id: str | None = None,
    ) -> ReRetrievalTrigger:
        """Create a re-retrieval trigger from verification batch results."""
        detector = EvidenceGapDetector(self.graph_store)
        gaps = detector.detect_gaps(batch_result)

        return ReRetrievalTrigger(
            gaps=gaps,
            max_additional_queries=1,  # MVP: exactly one
            original_query=original_query,
            context={"request_id": request_id} if request_id else {},
        )


def get_gap_detector(graph_store: EvidenceGraphStore | None = None) -> EvidenceGapDetector:
    """Get or create the singleton evidence gap detector."""
    return EvidenceGapDetector(graph_store)


def get_re_retrieval_manager(
    graph_store: EvidenceGraphStore | None = None,
    settings: Settings | None = None,
) -> ReRetrievalManager:
    """Get or create the singleton re-retrieval manager."""
    return ReRetrievalManager(graph_store, settings)