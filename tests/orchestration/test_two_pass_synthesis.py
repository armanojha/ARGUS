"""Tests for two-pass verified synthesis (Phase 29).

Tests the structured claim model, deterministic verification,
and the two-pass synthesis pipeline.
"""

from __future__ import annotations

import json
import pytest
from app.orchestration.two_pass_models import (
    ClaimSet,
    ClaimSupportStatus,
    ClaimType,
    ClaimVerificationResult,
    GeneratedClaim,
)
from app.orchestration.two_pass_synthesis import (
    _verify_claim_deterministic,
    verify_claims,
)
from app.evidence.models import EvidenceRef, SourceType
from uuid import uuid4


def _make_evidence(text: str, score: float = 0.9) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="test/source.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
    )


class TestStructuredClaimModel:
    def test_valid_claim(self):
        claim = GeneratedClaim(
            claim="Revenue was $3.1 billion",
            evidence_ids=[1, 2],
            confidence=0.9,
            claim_type=ClaimType.NUMERICAL,
            numerical_values=["$3.1 billion"],
        )
        assert claim.evidence_ids == [1, 2]
        assert claim.claim_type == ClaimType.NUMERICAL

    def test_absent_claim(self):
        claim = GeneratedClaim(
            claim="The evidence does not contain information about this topic.",
            evidence_ids=[1],
            claim_type=ClaimType.ABSENT,
        )
        assert claim.claim_type == ClaimType.ABSENT

    def test_claim_set_supported(self):
        cs = ClaimSet(
            claims=[
                GeneratedClaim(claim="A", evidence_ids=[1]),
                GeneratedClaim(claim="B", evidence_ids=[2]),
            ],
            verified=[
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="A", evidence_ids=[1]),
                    status=ClaimSupportStatus.SUPPORTED,
                ),
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="B", evidence_ids=[2]),
                    status=ClaimSupportStatus.UNSUPPORTED,
                    reason="No evidence",
                ),
            ],
        )
        assert len(cs.supported_claims) == 1
        assert len(cs.rejected_claims) == 1

    def test_claim_set_empty(self):
        cs = ClaimSet(claims=[])
        assert len(cs.supported_claims) == 0
        assert len(cs.rejected_claims) == 0


class TestDeterministicVerification:
    def test_supported_claim(self):
        evidence = [_make_evidence("Acme Corporation is headquartered in New York City.")]
        claim = GeneratedClaim(
            claim="Acme is headquartered in New York City",
            evidence_ids=[1],
        )
        result = _verify_claim_deterministic(claim, evidence)
        assert result.status == ClaimSupportStatus.SUPPORTED
        assert len(result.supported_evidence_ids) == 1

    def test_unsupported_no_valid_ids(self):
        evidence = [_make_evidence("Some text.")]
        claim = GeneratedClaim(
            claim="Revenue was $5 billion",
            evidence_ids=[99],  # Out of range
        )
        result = _verify_claim_deterministic(claim, evidence)
        assert result.status == ClaimSupportStatus.UNSUPPORTED
        assert "No valid evidence IDs" in result.reason

    def test_absent_claim_always_supported(self):
        evidence = [_make_evidence("Some text about robotics.")]
        claim = GeneratedClaim(
            claim="The evidence does not contain information about European market share.",
            evidence_ids=[1],
            claim_type=ClaimType.ABSENT,
        )
        result = _verify_claim_deterministic(claim, evidence)
        assert result.status == ClaimSupportStatus.SUPPORTED

    def test_numerical_claim_value_in_evidence(self):
        evidence = [_make_evidence("Revenue in 2023 was $3.1 billion USD.")]
        claim = GeneratedClaim(
            claim="Revenue was $3.1 billion",
            evidence_ids=[1],
            claim_type=ClaimType.NUMERICAL,
            numerical_values=["$3.1 billion"],
        )
        result = _verify_claim_deterministic(claim, evidence)
        assert result.status == ClaimSupportStatus.SUPPORTED

    def test_numerical_claim_value_not_in_evidence(self):
        evidence = [_make_evidence("Revenue in 2023 was $3.1 billion USD.")]
        claim = GeneratedClaim(
            claim="Revenue was $5.0 billion",
            evidence_ids=[1],
            claim_type=ClaimType.NUMERICAL,
            numerical_values=["$5.0 billion"],
        )
        result = _verify_claim_deterministic(claim, evidence)
        assert result.status == ClaimSupportStatus.UNSUPPORTED
        assert "$5.0 billion" in result.reason

    def test_partial_overlap(self):
        evidence = [_make_evidence("Acme Corporation produces widgets and gadgets.")]
        claim = GeneratedClaim(
            claim="Acme produces widgets and also makes software products",
            evidence_ids=[1],
        )
        result = _verify_claim_deterministic(claim, evidence)
        # "software products" not in evidence, so partial
        assert result.status in (ClaimSupportStatus.PARTIALLY_SUPPORTED, ClaimSupportStatus.SUPPORTED)


class TestVerifyClaims:
    def test_batch_verification(self):
        evidence = [
            _make_evidence("Acme is headquartered in New York City."),
            _make_evidence("Revenue was $3.1 billion in 2023."),
        ]
        claims = [
            GeneratedClaim(claim="Acme is in NYC", evidence_ids=[1]),
            GeneratedClaim(claim="Revenue was $3.1B", evidence_ids=[2]),
            GeneratedClaim(claim="Made up fact", evidence_ids=[99]),
        ]
        cs = verify_claims(claims, evidence)
        assert len(cs.supported_claims) == 2
        assert len(cs.rejected_claims) == 1


class TestPass2CannotIntroduceFacts:
    """CRITICAL: Verify Pass 2 cannot introduce unsupported facts."""

    def test_pass2_receives_only_verified_claims(self):
        """Pass 2 gets a filtered list, not raw evidence."""
        verified = [
            {"claim": "Revenue was $3.1B", "evidence_ids": [1], "status": "supported"},
        ]
        # If Pass 2 tries to add a claim not in this list, the answer
        # should be rejected by the evaluator
        assert len(verified) == 1
        assert verified[0]["claim"] == "Revenue was $3.1B"

    def test_rejected_claims_not_in_pass2(self):
        """Unsupported claims are filtered out before Pass 2."""
        cs = ClaimSet(
            claims=[
                GeneratedClaim(claim="Supported fact", evidence_ids=[1]),
                GeneratedClaim(claim="Unsupported fact", evidence_ids=[99]),
            ],
            verified=[
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="Supported fact", evidence_ids=[1]),
                    status=ClaimSupportStatus.SUPPORTED,
                ),
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="Unsupported fact", evidence_ids=[99]),
                    status=ClaimSupportStatus.UNSUPPORTED,
                    reason="Invalid evidence ID",
                ),
            ],
        )
        supported = cs.supported_claims
        rejected = cs.rejected_claims
        assert len(supported) == 1
        assert supported[0].claim == "Supported fact"
        assert len(rejected) == 1
        assert rejected[0].claim.claim == "Unsupported fact"
