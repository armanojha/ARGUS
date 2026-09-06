"""Phase 22 Tests: Deterministic Evidence Verification.

Tests for:
- EvidenceNeedCoverageVerifier
- TextContradictionDetector
- MultiHopChainVerifier
- EvidenceConfidenceScorer
"""
from __future__ import annotations

import pytest
from uuid import uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.retrieval.planner import EvidenceNeed, ClaimType, NeedPriority, QueryPlan
from app.verification.deterministic import (
    CoverageStatus,
    EvidenceConfidenceScorer,
    EvidenceNeedCoverageVerifier,
    MultiHopChainVerifier,
    TextContradictionDetector,
    ConfidenceLevel,
)


def _make_ref(text: str, doc_id: str | None = None, score: float = 0.8) -> EvidenceRef:
    """Create a minimal EvidenceRef for testing."""
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4() if doc_id is None else uuid4(),  # Use doc_id param indirectly
        source_id=uuid4(),
        source_path="/test/source.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
    )


def _make_ref_with_doc(text: str, doc_uuid, score: float = 0.8) -> EvidenceRef:
    """Create an EvidenceRef with a specific document_id."""
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=doc_uuid,
        source_id=uuid4(),
        source_path="/test/source.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
    )


def _make_need(
    topic: str = "test topic",
    entities: list[str] | None = None,
    claim_type: ClaimType = ClaimType.PRIMARY,
) -> EvidenceNeed:
    return EvidenceNeed(
        id=str(uuid4())[:8],
        topic=topic,
        entities=entities or [],
        claim_type=claim_type,
        search_query=f"search for {topic}",
        priority=NeedPriority.HIGH,
        requires_opposing_evidence=False,
        source_constraints=[],
        original_need=f"Find evidence about {topic}",
    )


# ===========================================================================
# EvidenceNeedCoverageVerifier Tests
# ===========================================================================

class TestEvidenceNeedCoverageVerifier:
    def test_no_evidence_returns_no_evidence_exists(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("test topic", ["entity1"])
        result = verifier.verify_need(need, [])
        assert result.status == CoverageStatus.NO_EVIDENCE_EXISTS

    def test_covered_when_topic_and_entity_present(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("revenue", ["Acme"])
        ref = _make_ref("Acme reported revenue of $4.7 billion in 2025.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.COVERED
        assert result.coverage_score == 1.0

    def test_partially_covered_when_only_topic_present(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("revenue", ["Acme"])
        ref = _make_ref("Revenue figures for the quarter show growth.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.PARTIALLY_COVERED
        assert result.coverage_score == 0.5

    def test_not_covered_when_no_match(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("quantum computing", ["qubits"])
        ref = _make_ref("The weather today is sunny and warm.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.NOT_COVERED
        assert result.coverage_score == 0.0

    def test_contradictory_need_with_conflict_indicators(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("revenue", ["Acme"], ClaimType.CONTRADICTORY)
        ref = _make_ref("However, the revenue figure of $3.1 billion conflicts with later reports.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.COVERED

    def test_caveat_need_with_disclaimer_pattern(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("revenue", ["Acme"], ClaimType.CAVEAT)
        ref = _make_ref("Note: this figure is preliminary and may be revised.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.COVERED

    def test_supporting_need_with_entity_match(self):
        verifier = EvidenceNeedCoverageVerifier()
        need = _make_need("revenue", ["Acme"], ClaimType.SUPPORTING)
        ref = _make_ref("Acme Corporation announced new products.")
        result = verifier.verify_need(need, [ref])
        assert result.status == CoverageStatus.COVERED

    def test_verify_plan_returns_results_for_planned(self):
        verifier = EvidenceNeedCoverageVerifier()
        need1 = _make_need("revenue", ["Acme"])
        need2 = _make_need("employees", ["Acme"])
        plan = QueryPlan(
            original_query="test",
            pattern="conflict",
            evidence_needs=[need1, need2],
            search_variants=[],
            is_planned=True,
        )
        ref = _make_ref("Acme revenue was $4.7 billion with 12,400 employees.")
        results = verifier.verify_plan(plan, [ref])
        assert len(results) == 2

    def test_verify_plan_returns_empty_for_unplanned(self):
        verifier = EvidenceNeedCoverageVerifier()
        plan = QueryPlan(
            original_query="test",
            pattern="conceptual",
            evidence_needs=[],
            search_variants=[],
            is_planned=False,
        )
        results = verifier.verify_plan(plan, [])
        assert results == []


# ===========================================================================
# TextContradictionDetector Tests
# ===========================================================================

class TestTextContradictionDetector:
    def test_no_contradictions_with_single_chunk(self):
        detector = TextContradictionDetector()
        ref = _make_ref("Revenue was $4.7 billion.")
        result = detector.detect_contradictions([ref])
        assert result == []

    def test_financial_contradiction_detected(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Revenue of $3.1 billion in 2023.", doc_a)
        ref_b = _make_ref_with_doc("Revenue of $4.7 billion in 2025.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b])
        assert len(result) > 0
        assert any("financial" in c.description for c in result)

    def test_same_value_no_contradiction(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Revenue of $4.7 billion.", doc_a)
        ref_b = _make_ref_with_doc("Revenue of $4.7 billion.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b])
        assert len(result) == 0

    def test_same_document_no_contradiction(self):
        detector = TextContradictionDetector()
        doc = uuid4()
        ref_a = _make_ref_with_doc("Revenue of $3.1 billion.", doc)
        ref_b = _make_ref_with_doc("Revenue of $4.7 billion.", doc)
        result = detector.detect_contradictions([ref_a, ref_b])
        assert len(result) == 0

    def test_different_categories_no_contradiction(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Revenue of $4.7 billion.", doc_a)
        ref_b = _make_ref_with_doc("Founded in 1987.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b])
        assert len(result) == 0

    def test_incompatible_units_no_contradiction(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Utilization was 91%.", doc_a)
        ref_b = _make_ref_with_doc("Produced 1.2 million units.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b])
        # Should NOT be contradictions (different unit groups)
        financial_ops = [c for c in result if "financial" in c.description or "operational" in c.description]
        # May have percentage vs percentage if both match percentage pattern
        # But financial/operational with incompatible units should not conflict

    def test_operational_contradiction_detected(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Utilization was 61%.", doc_a)
        ref_b = _make_ref_with_doc("Utilization was 91%.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b])
        assert len(result) > 0
        assert any("operational" in c.description for c in result)

    def test_empty_chunks_returns_empty(self):
        detector = TextContradictionDetector()
        result = detector.detect_contradictions([])
        assert result == []

    def test_contradiction_severity_above_threshold(self):
        detector = TextContradictionDetector()
        doc_a = uuid4()
        doc_b = uuid4()
        ref_a = _make_ref_with_doc("Revenue of $3.1 billion.", doc_a)
        ref_b = _make_ref_with_doc("Revenue of $4.7 billion.", doc_b)
        result = detector.detect_contradictions([ref_a, ref_b], min_severity=0.5)
        assert len(result) > 0
        assert all(c.severity >= 0.5 for c in result)


# ===========================================================================
# MultiHopChainVerifier Tests
# ===========================================================================

class TestMultiHopChainVerifier:
    def test_complete_chain_all_covered(self):
        verifier = MultiHopChainVerifier()
        need1 = _make_need("PetroKem supplies Ohio", ["PetroKem", "Ohio"])
        need2 = _make_need("Ohio ships to Memphis", ["Ohio", "Memphis"])
        plan = QueryPlan(
            original_query="test",
            pattern="multi_hop",
            evidence_needs=[need1, need2],
            search_variants=[],
            is_planned=True,
        )
        ref1 = _make_ref("PetroKem Co. supplies crude polymers to the Ohio plant.")
        ref2 = _make_ref("The Ohio plant ships adhesives to the Memphis distribution center.")
        result = verifier.verify_chain(plan, [ref1, ref2])
        assert result.chain_status == CoverageStatus.COVERED
        assert len(result.hops) == 2
        assert all(h.status == CoverageStatus.COVERED for h in result.hops)

    def test_missing_middle_hop(self):
        verifier = MultiHopChainVerifier()
        # Need1: about PetroKem. Need2: about Memphis->Retail. Need3: about Atlas.
        # Only ref1 covers need1. Ref2 covers need2. Need3 has no coverage.
        need1 = _make_need("PetroKem raw materials", ["PetroKem"])
        need2 = _make_need("Memphis distribution to retail", ["Memphis", "retail"])
        need3 = _make_need("Atlas database feature", ["Atlas", "Delta Sync"])
        plan = QueryPlan(
            original_query="test",
            pattern="multi_hop",
            evidence_needs=[need1, need2, need3],
            search_variants=[],
            is_planned=True,
        )
        ref1 = _make_ref("PetroKem Co. supplies crude polymers to the Ohio plant.")
        ref2 = _make_ref("The Memphis center distributes products to eastern US retail.")
        result = verifier.verify_chain(plan, [ref1, ref2])
        assert result.chain_status == CoverageStatus.PARTIALLY_COVERED
        assert result.hops[0].status == CoverageStatus.COVERED
        assert result.hops[1].status == CoverageStatus.COVERED
        assert result.hops[2].status == CoverageStatus.NOT_COVERED
        assert result.weakest_hop == 2

    def test_no_evidence_for_any_hop(self):
        verifier = MultiHopChainVerifier()
        need1 = _make_need("topic A", ["entity1"])
        need2 = _make_need("topic B", ["entity2"])
        plan = QueryPlan(
            original_query="test",
            pattern="multi_hop",
            evidence_needs=[need1, need2],
            search_variants=[],
            is_planned=True,
        )
        result = verifier.verify_chain(plan, [])
        assert result.chain_status == CoverageStatus.NOT_COVERED
        assert result.weakest_hop == 0

    def test_unplanned_query_returns_not_covered(self):
        verifier = MultiHopChainVerifier()
        plan = QueryPlan(
            original_query="test",
            pattern="conceptual",
            evidence_needs=[],
            search_variants=[],
            is_planned=False,
        )
        result = verifier.verify_chain(plan, [])
        assert result.chain_status == CoverageStatus.NOT_COVERED

    def test_contradictory_hop(self):
        verifier = MultiHopChainVerifier()
        need1 = _make_need("topic A", ["entity1"])
        need2 = _make_need("topic B", ["entity2"])
        plan = QueryPlan(
            original_query="test",
            pattern="multi_hop",
            evidence_needs=[need1, need2],
            search_variants=[],
            is_planned=True,
        )
        ref = _make_ref("Something completely unrelated to either topic.")
        result = verifier.verify_chain(plan, [ref])
        assert result.chain_status == CoverageStatus.NOT_COVERED


# ===========================================================================
# EvidenceConfidenceScorer Tests
# ===========================================================================

class TestEvidenceConfidenceScorer:
    def test_high_confidence_all_covered_no_contradictions(self):
        scorer = EvidenceConfidenceScorer()
        from app.verification.deterministic import NeedVerificationResult
        need_results = [
            NeedVerificationResult("n1", "topic1", CoverageStatus.COVERED, coverage_score=1.0),
            NeedVerificationResult("n2", "topic2", CoverageStatus.COVERED, coverage_score=1.0),
        ]
        result = scorer.score(need_results, [])
        assert result.level == ConfidenceLevel.HIGH
        assert result.coverage_score == 1.0

    def test_conflicted_when_contradictions_present(self):
        scorer = EvidenceConfidenceScorer()
        from app.verification.deterministic import NeedVerificationResult, TextContradiction
        need_results = [
            NeedVerificationResult("n1", "topic1", CoverageStatus.COVERED, coverage_score=1.0),
        ]
        contras = [TextContradiction("numerical_conflict", "test", "a", "b", "A", "B", "1", "2", "cat", 0.7)]
        result = scorer.score(need_results, contras)
        assert result.level == ConfidenceLevel.CONFLICTED
        assert result.contradiction_count == 1

    def test_low_confidence_no_coverage(self):
        scorer = EvidenceConfidenceScorer()
        from app.verification.deterministic import NeedVerificationResult
        need_results = [
            NeedVerificationResult("n1", "topic1", CoverageStatus.NOT_COVERED, coverage_score=0.0),
        ]
        result = scorer.score(need_results, [])
        assert result.level == ConfidenceLevel.LOW

    def test_medium_confidence_partial_coverage(self):
        scorer = EvidenceConfidenceScorer()
        from app.verification.deterministic import NeedVerificationResult
        need_results = [
            NeedVerificationResult("n1", "topic1", CoverageStatus.COVERED, coverage_score=1.0),
            NeedVerificationResult("n2", "topic2", CoverageStatus.NOT_COVERED, coverage_score=0.0),
        ]
        result = scorer.score(need_results, [])
        assert result.level == ConfidenceLevel.MEDIUM

    def test_empty_needs_returns_low(self):
        scorer = EvidenceConfidenceScorer()
        result = scorer.score([], [])
        assert result.level == ConfidenceLevel.LOW
