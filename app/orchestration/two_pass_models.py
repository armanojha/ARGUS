"""Two-pass verified synthesis models (Phase 29).

Structured claim representation for evidence-grounded synthesis.
Every claim must have an explicit relationship to evidence.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClaimSupportStatus(str, Enum):
    """Verification status of a generated claim against evidence."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ClaimType(str, Enum):
    """Type of factual claim."""

    FACTUAL = "factual"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    ABSENT = "absent"  # Evidence shows info is not available


class GeneratedClaim(BaseModel):
    """A single claim generated in Pass 1.

    Every claim must reference at least one evidence chunk.
    No evidence ID = invalid claim.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The factual claim text.")
    evidence_ids: list[int] = Field(
        description="1-based indices into the evidence list that support this claim."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model's self-assessed confidence in this claim.",
    )
    claim_type: ClaimType = Field(
        default=ClaimType.FACTUAL,
        description="Type of claim (factual, numerical, comparison, etc.).",
    )
    numerical_values: list[str] = Field(
        default_factory=list,
        description="Specific numerical values referenced in the claim, if any.",
    )


class ClaimVerificationResult(BaseModel):
    """Result of deterministically verifying a single claim against evidence."""

    model_config = ConfigDict(extra="forbid")

    claim: GeneratedClaim
    status: ClaimSupportStatus
    reason: str = Field(default="")
    supported_evidence_ids: list[int] = Field(default_factory=list)
    unsupported_parts: list[str] = Field(default_factory=list)


class ClaimSet(BaseModel):
    """Collection of verified claims ready for Pass 2 synthesis."""

    model_config = ConfigDict(extra="forbid")

    claims: list[GeneratedClaim]
    verified: list[ClaimVerificationResult] = Field(default_factory=list)
    contradictions_found: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def supported_claims(self) -> list[GeneratedClaim]:
        return [
            v.claim
            for v in self.verified
            if v.status in (ClaimSupportStatus.SUPPORTED, ClaimSupportStatus.PARTIALLY_SUPPORTED)
        ]

    @property
    def rejected_claims(self) -> list[ClaimVerificationResult]:
        return [
            v
            for v in self.verified
            if v.status in (
                ClaimSupportStatus.UNSUPPORTED,
                ClaimSupportStatus.CONTRADICTED,
                ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
            )
        ]
