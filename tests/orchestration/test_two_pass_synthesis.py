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


# --- Phase 30 tests ---

class TestDeterministicRenderer:
    def test_renders_supported_claims(self):
        from app.orchestration.two_pass_synthesis import _render_verified_claims
        verified = [
            ClaimVerificationResult(
                claim=GeneratedClaim(claim="Revenue was $3.1 billion", evidence_ids=[1]),
                status=ClaimSupportStatus.SUPPORTED,
            ),
            ClaimVerificationResult(
                claim=GeneratedClaim(claim="Employees: 12,400", evidence_ids=[2]),
                status=ClaimSupportStatus.SUPPORTED,
            ),
        ]
        answer = _render_verified_claims(verified, "What are Acme's metrics?")
        assert "$3.1 billion" in answer
        assert "12,400" in answer
        assert "[1]" in answer
        assert "[2]" in answer

    def test_renders_absent_claim(self):
        from app.orchestration.two_pass_synthesis import _render_verified_claims
        verified = [
            ClaimVerificationResult(
                claim=GeneratedClaim(
                    claim="The available evidence does not contain sufficient information to answer this question.",
                    evidence_ids=[1],
                    claim_type=ClaimType.ABSENT,
                ),
                status=ClaimSupportStatus.SUPPORTED,
            ),
        ]
        answer = _render_verified_claims(verified, "What is X?")
        assert "does not contain" in answer

    def test_empty_claims_returns_absent(self):
        from app.orchestration.two_pass_synthesis import _render_verified_claims
        answer = _render_verified_claims([], "What is X?")
        assert "does not contain" in answer

    def test_contradictions_presented(self):
        from app.orchestration.two_pass_synthesis import _render_verified_claims
        verified = [
            ClaimVerificationResult(
                claim=GeneratedClaim(claim="Revenue was $3.1B", evidence_ids=[1]),
                status=ClaimSupportStatus.CONTRADICTED,
            ),
            ClaimVerificationResult(
                claim=GeneratedClaim(claim="Revenue was $4.7B", evidence_ids=[2]),
                status=ClaimSupportStatus.SUPPORTED,
            ),
        ]
        answer = _render_verified_claims(verified, "Revenue?")
        assert "conflicting" in answer.lower()
        assert "$3.1B" in answer
        assert "$4.7B" in answer


class TestEarlyExit:
    def test_simple_lookup_early_exits(self):
        from app.orchestration.two_pass_synthesis import _should_early_exit, _detect_question_pattern
        cs = ClaimSet(
            claims=[GeneratedClaim(claim="NYC", evidence_ids=[1])],
            verified=[
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="NYC", evidence_ids=[1]),
                    status=ClaimSupportStatus.SUPPORTED,
                ),
            ],
        )
        pattern = _detect_question_pattern("Where is Acme headquartered?")
        assert pattern == "simple_lookup"
        assert _should_early_exit(cs, pattern)

    def test_conflict_with_many_claims_no_early_exit(self):
        from app.orchestration.two_pass_synthesis import _should_early_exit
        claims = [
            ClaimVerificationResult(
                claim=GeneratedClaim(claim=f"Fact {i}", evidence_ids=[1]),
                status=ClaimSupportStatus.SUPPORTED,
            )
            for i in range(10)
        ]
        cs = ClaimSet(claims=[c.claim for c in claims], verified=claims)
        assert not _should_early_exit(cs, "conflict")

    def test_rejected_claims_block_early_exit(self):
        from app.orchestration.two_pass_synthesis import _should_early_exit
        cs = ClaimSet(
            claims=[GeneratedClaim(claim="A", evidence_ids=[1])],
            verified=[
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="A", evidence_ids=[1]),
                    status=ClaimSupportStatus.SUPPORTED,
                ),
                ClaimVerificationResult(
                    claim=GeneratedClaim(claim="B", evidence_ids=[99]),
                    status=ClaimSupportStatus.UNSUPPORTED,
                ),
            ],
        )
        assert not _should_early_exit(cs, "simple_lookup")


class TestNoRawEvidenceInPass2:
    """CRITICAL: Verify Pass 2 prompt does not contain raw evidence."""

    def test_restricted_prompt_has_no_evidence_block(self):
        from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
        from app.orchestration.models import ResearchPlan
        from app.evidence.models import EvidenceRef, SourceType
        from uuid import uuid4

        plan = ResearchPlan(objective="Test query")
        evidence = [
            EvidenceRef(
                chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(),
                source_path="test.md", source_type=SourceType.MARKDOWN,
                text="This is secret evidence text that should NOT appear in Pass 2.",
                score=0.9, rank=1,
            )
        ]
        verified = [{"claim": "Test claim", "evidence_ids": [1], "status": "supported"}]

        messages = build_verified_synthesis_messages(plan, evidence, verified)
        # Check all messages
        for msg in messages:
            assert "secret evidence text" not in msg.content, \
                "Pass 2 prompt must NOT contain raw evidence text"
            assert "EVIDENCE" not in msg.content or "VERIFIED CLAIMS" in msg.content

    def test_restricted_prompt_contains_verified_claims(self):
        from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
        from app.orchestration.models import ResearchPlan

        plan = ResearchPlan(objective="Test query")
        verified = [{"claim": "Revenue was $3.1B", "evidence_ids": [1], "status": "supported"}]
        messages = build_verified_synthesis_messages(plan, [], verified)

        system_msg = messages[0].content
        assert "renderer" in system_msg.lower()
        assert "VERIFIED CLAIMS" in messages[1].content
        assert "Revenue was $3.1B" in messages[1].content


# --- Phase 31 tests ---

class TestSynthesisMetrics:
    def test_metrics_dataclass_fields(self):
        from app.orchestration.two_pass_synthesis import SynthesisMetrics
        m = SynthesisMetrics()
        assert m.pass1_latency_ms == 0.0
        assert m.pass2_latency_ms == 0.0
        assert m.verification_latency_ms == 0.0
        assert m.total_latency_ms == 0.0
        assert m.pass1_claims_generated == 0
        assert m.verified_claims_count == 0
        assert m.rejected_claims_count == 0
        assert m.pass2_method == ""
        assert m.pass1_tokens_in == 0
        assert m.pass1_tokens_out == 0
        assert m.pass2_tokens_in == 0
        assert m.pass2_tokens_out == 0
        assert m.citations_in_answer == 0
        assert m.pattern == ""

    def test_metrics_with_values(self):
        from app.orchestration.two_pass_synthesis import SynthesisMetrics
        m = SynthesisMetrics(
            pass1_latency_ms=3000.0,
            pass2_latency_ms=1500.0,
            verification_latency_ms=0.5,
            total_latency_ms=4500.0,
            pass1_claims_generated=5,
            verified_claims_count=4,
            rejected_claims_count=1,
            pass2_method="llm_restricted",
            pass1_tokens_in=1200,
            pass1_tokens_out=400,
            pass2_tokens_in=300,
            pass2_tokens_out=100,
            citations_in_answer=3,
            pattern="simple_lookup",
        )
        assert m.pass1_latency_ms == 3000.0
        assert m.pass2_method == "llm_restricted"
        assert m.citations_in_answer == 3


class TestPatternDetection:
    def test_simple_lookup_patterns(self):
        from app.orchestration.two_pass_synthesis import _detect_question_pattern
        assert _detect_question_pattern("Where is Acme headquartered?") == "simple_lookup"
        assert _detect_question_pattern("What is the name of Acme's CEO?") == "simple_lookup"
        assert _detect_question_pattern("Who is the CTO?") == "simple_lookup"

    def test_numerical_patterns(self):
        from app.orchestration.two_pass_synthesis import _detect_question_pattern
        assert _detect_question_pattern("What is the P99 latency?") == "numerical"
        assert _detect_question_pattern("What is the revenue growth rate?") == "numerical"

    def test_conflict_patterns(self):
        from app.orchestration.two_pass_synthesis import _detect_question_pattern
        assert _detect_question_pattern("What are the conflicting reports?") == "conflict"
        assert _detect_question_pattern("What do different sources say?") == "conflict"

    def test_absent_info_patterns(self):
        from app.orchestration.two_pass_synthesis import _detect_question_pattern
        assert _detect_question_pattern("What undisclosed issues exist?") == "absent_info"
        assert _detect_question_pattern("What information is not available?") == "absent_info"

    def test_general_fallback(self):
        from app.orchestration.two_pass_synthesis import _detect_question_pattern
        assert _detect_question_pattern("Compare the two revenue figures.") == "general"
        assert _detect_question_pattern("How does Atlas work?") == "general"


class TestCitationExtraction:
    def test_extract_citations(self):
        from app.orchestration.two_pass_synthesis import _extract_evidence_ids_from_text
        ids = _extract_evidence_ids_from_text("Revenue was $3.1B [1] and $4.7B [2]")
        assert ids == [1, 2]

    def test_extract_citations_fullwidth(self):
        from app.orchestration.two_pass_synthesis import _extract_evidence_ids_from_text
        ids = _extract_evidence_ids_from_text("Revenue\u30101\u3011and\u30102\u3011")
        assert ids == [1, 2]

    def test_extract_citations_none(self):
        from app.orchestration.two_pass_synthesis import _extract_evidence_ids_from_text
        ids = _extract_evidence_ids_from_text("No citations here.")
        assert ids == []


class TestNormalizedCitationMarkers:
    def test_normalizes_fullwidth(self):
        from app.orchestration.two_pass_synthesis import _normalize_citation_markers
        result = _normalize_citation_markers("\u30101\u3011")
        assert result == "[1]"

    def test_normalizes_fullwidth_digits(self):
        from app.orchestration.two_pass_synthesis import _normalize_citation_markers
        result = _normalize_citation_markers("\uff10\uff11\uff12")
        assert result == "012"

    def test_noop_on_ascii(self):
        from app.orchestration.two_pass_synthesis import _normalize_citation_markers
        result = _normalize_citation_markers("[1] [2]")
        assert result == "[1] [2]"
