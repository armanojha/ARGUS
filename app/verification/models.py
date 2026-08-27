"""Verification data models (Phase 04).

Canonical data models for claim verification, contradiction detection,
and confidence scoring. All verification results remain traceable to
Phase 01/03 evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerificationStatus(str, Enum):
    """Verification outcome for a claim (V2 §9.1)."""

    SUPPORTED = "supported"           # Claim is well-supported by evidence
    PARTIAL = "partial"               # Claim has some support but gaps remain
    CONTRADICTED = "contradicted"     # Evidence contradicts the claim
    UNSUPPORTED = "unsupported"       # Insufficient evidence to verify
    ERROR = "error"                   # Verification failed (LLM error, etc.)


class ContradictionType(str, Enum):
    """Types of contradictions detected (V2 §9.2)."""

    PUBLICATION_DATE = "publication_date"       # Different publication dates/versions
    METRIC_DEFINITION = "metric_definition"     # Different metric/entity definitions
    GEOGRAPHIC_SCOPE = "geographic_scope"       # Different geographic/jurisdictional scope
    TIME_PERIOD = "time_period"                 # Different time periods
    REVISED_NUMBERS = "revised_numbers"         # Revised/restated numbers
    ENTITY_MISMATCH = "entity_mismatch"         # Different entities referred to
    SOURCE_CONFLICT = "source_conflict"         # Direct source disagreement
    TEMPORAL_CONFLICT = "temporal_conflict"     # Validity time conflicts


class VerificationEvidence(BaseModel):
    """Evidence reference used in verification."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    source_path: str
    source_type: str
    text: str
    supports: bool = Field(description="Whether this evidence supports (True) or contradicts (False) the claim")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance of this evidence to the claim")
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None


class ContradictionDetail(BaseModel):
    """Details of a detected contradiction."""

    model_config = ConfigDict(extra="forbid")

    contradiction_type: ContradictionType
    description: str = Field(description="Human-readable description of the contradiction")
    claim_a_id: UUID = Field(description="First claim involved")
    claim_b_id: UUID = Field(description="Second claim involved")
    evidence_a_ids: list[UUID] = Field(default_factory=list, description="Evidence supporting claim A")
    evidence_b_ids: list[UUID] = Field(default_factory=list, description="Evidence supporting claim B")
    severity: float = Field(default=0.5, ge=0.0, le=1.0, description="Severity of contradiction (0-1)")
    resolution_suggestion: str | None = Field(default=None, description="Suggested resolution if applicable")


class VerificationResult(BaseModel):
    """Result of verifying a single claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: UUID
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0, description="Overall verification confidence (0-1)")
    reasoning: str = Field(description="Explanation of the verification decision")
    # Evidence used
    supporting_evidence: list[VerificationEvidence] = Field(default_factory=list)
    contradicting_evidence: list[VerificationEvidence] = Field(default_factory=list)
    # Contradictions detected
    contradictions: list[ContradictionDetail] = Field(default_factory=list)
    # Diagnostic confidence components (V2 §9.3)
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claim covered by evidence")
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality/reliability of sources")
    cross_source_agreement: float = Field(default=0.0, ge=0.0, le=1.0, description="Agreement across independent sources")
    temporal_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Temporal relevance of evidence")
    retrieval_rank: float = Field(default=0.0, ge=0.0, le=1.0, description="Retrieval rank quality")
    verifier_judgment: float = Field(default=0.0, ge=0.0, le=1.0, description="Verifier LLM's own judgment")
    # Metadata
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verifier_model: str | None = Field(default=None, description="Model used for verification")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationRequest(BaseModel):
    """Request to verify a claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: UUID
    claim_text: str
    # Evidence from graph
    supporting_chunk_ids: list[UUID] = Field(default_factory=list)
    contradicting_chunk_ids: list[UUID] = Field(default_factory=list)
    # Related claims for contradiction detection
    related_claim_ids: list[UUID] = Field(default_factory=list)
    # Context
    entity_names: list[str] = Field(default_factory=list)
    temporal_context: str | None = None
    # Options
    max_evidence_items: int = Field(default=20, ge=1, le=50)
    require_cross_source: bool = Field(default=True)


class VerificationBatchResult(BaseModel):
    """Result of verifying multiple claims."""

    model_config = ConfigDict(extra="forbid")

    results: list[VerificationResult] = Field(default_factory=list)
    # Overall stats (auto-computed from results)
    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    contradicted_count: int = 0
    unsupported_count: int = 0
    error_count: int = 0
    # Contradictions found across batch
    all_contradictions: list[ContradictionDetail] = Field(default_factory=list)
    # Evidence gaps detected
    evidence_gaps: list[UUID] = Field(default_factory=list, description="Claim IDs with UNSUPPORTED/PARTIAL status")

    @model_validator(mode="after")
    def compute_counts(self) -> VerificationBatchResult:
        """Auto-compute counts from results."""
        self.total_claims = len(self.results)
        self.supported_count = sum(1 for r in self.results if r.status == VerificationStatus.SUPPORTED)
        self.partial_count = sum(1 for r in self.results if r.status == VerificationStatus.PARTIAL)
        self.contradicted_count = sum(1 for r in self.results if r.status == VerificationStatus.CONTRADICTED)
        self.unsupported_count = sum(1 for r in self.results if r.status == VerificationStatus.UNSUPPORTED)
        self.error_count = sum(1 for r in self.results if r.status == VerificationStatus.ERROR)
        return self


class ConfidenceComponents(BaseModel):
    """Diagnostic confidence components (V2 §9.3).

    These are explicitly diagnostic, not calibrated probabilities.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_coverage: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    cross_source_agreement: float = Field(ge=0.0, le=1.0)
    temporal_relevance: float = Field(ge=0.0, le=1.0)
    retrieval_rank: float = Field(ge=0.0, le=1.0)
    verifier_judgment: float = Field(ge=0.0, le=1.0)

    def composite(self) -> float:
        """Compute composite diagnostic score (simple average)."""
        return (
            self.evidence_coverage
            + self.source_quality
            + self.cross_source_agreement
            + self.temporal_relevance
            + self.retrieval_rank
            + self.verifier_judgment
        ) / 6.0


class EvidenceGap(BaseModel):
    """An evidence gap detected during verification."""

    model_config = ConfigDict(extra="forbid")

    claim_id: UUID
    gap_type: str = Field(description="Type of gap: 'no_evidence', 'partial_coverage', 'temporal_mismatch', 'source_quality'")
    description: str
    suggested_query: str | None = Field(default=None, description="Suggested retrieval query to fill gap")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority for re-retrieval")


class ReRetrievalTrigger(BaseModel):
    """Trigger for re-retrieval based on evidence gaps."""

    model_config = ConfigDict(extra="forbid")

    gaps: list[EvidenceGap]
    max_additional_queries: int = Field(default=1, ge=1, le=3, description="Max additional retrieval cycles (MVP: 1)")
    original_query: str
    context: dict[str, Any] = Field(default_factory=dict)