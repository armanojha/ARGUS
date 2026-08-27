"""Verification exports (Phase 04)."""

from app.verification.confidence import (
    ConfidenceScorer,
    compute_composite_confidence,
    get_confidence_scorer,
)
from app.verification.contradiction import ContradictionDetector, get_contradiction_detector
from app.verification.engine import (
    VerificationFn,
    make_verification_node,
    verify_claim,
    verify_claims_batch,
)
from app.verification.gaps import (
    EvidenceGapDetector,
    ReRetrievalManager,
    get_gap_detector,
    get_re_retrieval_manager,
)
from app.verification.models import (
    ConfidenceComponents,
    ContradictionDetail,
    ContradictionType,
    EvidenceGap,
    ReRetrievalTrigger,
    VerificationBatchResult,
    VerificationEvidence,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "ConfidenceComponents",
    "ConfidenceScorer",
    "ContradictionDetail",
    "ContradictionDetector",
    "ContradictionType",
    "EvidenceGap",
    "EvidenceGapDetector",
    "ReRetrievalManager",
    "ReRetrievalTrigger",
    "VerificationBatchResult",
    "VerificationEvidence",
    "VerificationFn",
    "VerificationRequest",
    "VerificationResult",
    "VerificationStatus",
    "compute_composite_confidence",
    "get_confidence_scorer",
    "get_contradiction_detector",
    "get_gap_detector",
    "get_re_retrieval_manager",
    "make_verification_node",
    "verify_claim",
    "verify_claims_batch",
]