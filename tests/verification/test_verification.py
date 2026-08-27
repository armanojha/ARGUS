"""Lightweight sanity tests for Verification (Phase 04).

Not exhaustive by design (see vault Phase 04 testing policy: deferred
to a later stabilization pass). Covers the phase's own acceptance
criteria: verification statuses, contradiction detection, evidence gaps,
and confidence scoring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.graph.models import Claim
from app.graph.store import EvidenceGraphStore
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.vector import assign_embedding_indices
from app.verification.confidence import ConfidenceScorer, compute_composite_confidence
from app.verification.contradiction import ContradictionDetector
from app.verification.gaps import EvidenceGapDetector, ReRetrievalManager
from app.verification.models import (
    ConfidenceComponents,
    ContradictionDetail,
    ContradictionType,
    EvidenceGap,
    ReRetrievalTrigger,
    VerificationBatchResult,
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
)


@pytest.fixture
def temp_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield tmp


@pytest.fixture
def evidence_store(temp_paths) -> EvidenceStore:
    return EvidenceStore(
        db_path=temp_paths / "evidence.db",
        bm25_index_path=temp_paths / "bm25.pkl",
        faiss_index_path=temp_paths / "faiss.index",
    )


@pytest.fixture
def populated_evidence_store(evidence_store: EvidenceStore) -> EvidenceStore:
    source = Source(type=SourceType.TEXT, path="/test/corpus.txt", checksum="c1")
    evidence_store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d1", chunking_strategy="fixed")
    evidence_store.insert_document(doc)
    evidence_store.insert_chunks(
        [
            Chunk(document_id=doc.id, ordinal=0, text="John Smith works at Acme Corp.", token_count=8),
            Chunk(document_id=doc.id, ordinal=1, text="Acme Corp is located in New York.", token_count=8),
            Chunk(document_id=doc.id, ordinal=2, text="The meeting happened on 2024-01-15.", token_count=8),
        ]
    )
    assign_bm25_doc_ids(evidence_store)
    assign_embedding_indices(evidence_store)
    return evidence_store


@pytest.fixture
def graph_store(temp_paths, populated_evidence_store: EvidenceStore) -> EvidenceGraphStore:
    return EvidenceGraphStore(
        graph_path=temp_paths / "graph.pkl",
        evidence_store=populated_evidence_store,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


class TestVerificationModels:
    def test_verification_status_enum(self):
        assert VerificationStatus.SUPPORTED.value == "supported"
        assert VerificationStatus.PARTIAL.value == "partial"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.UNSUPPORTED.value == "unsupported"
        assert VerificationStatus.ERROR.value == "error"

    def test_contradiction_type_enum(self):
        assert ContradictionType.PUBLICATION_DATE.value == "publication_date"
        assert ContradictionType.METRIC_DEFINITION.value == "metric_definition"
        assert ContradictionType.GEOGRAPHIC_SCOPE.value == "geographic_scope"
        assert ContradictionType.TIME_PERIOD.value == "time_period"
        assert ContradictionType.REVISED_NUMBERS.value == "revised_numbers"
        assert ContradictionType.ENTITY_MISMATCH.value == "entity_mismatch"
        assert ContradictionType.SOURCE_CONFLICT.value == "source_conflict"
        assert ContradictionType.TEMPORAL_CONFLICT.value == "temporal_conflict"

    def test_verification_result_creation(self):
        claim_id = uuid4()
        result = VerificationResult(
            claim_id=claim_id,
            status=VerificationStatus.SUPPORTED,
            confidence=0.9,
            reasoning="Well supported by evidence",
        )
        assert result.claim_id == claim_id
        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 0.9

    def test_verification_evidence_creation(self):
        chunk_id = uuid4()
        evidence = VerificationEvidence(
            chunk_id=chunk_id,
            document_id=uuid4(),
            source_id=uuid4(),
            source_path="/test/doc.pdf",
            source_type="pdf",
            text="Supporting text",
            supports=True,
            relevance_score=0.8,
        )
        assert evidence.supports is True
        assert evidence.relevance_score == 0.8

    def test_contradiction_detail_creation(self):
        detail = ContradictionDetail(
            contradiction_type=ContradictionType.SOURCE_CONFLICT,
            description="Sources disagree on value",
            claim_a_id=uuid4(),
            claim_b_id=uuid4(),
            severity=0.8,
        )
        assert detail.contradiction_type == ContradictionType.SOURCE_CONFLICT
        assert detail.severity == 0.8

    def test_confidence_components_composite(self):
        components = ConfidenceComponents(
            evidence_coverage=0.8,
            source_quality=0.7,
            cross_source_agreement=0.9,
            temporal_relevance=0.6,
            retrieval_rank=0.8,
            verifier_judgment=0.7,
        )
        composite = components.composite()
        expected = (0.8 + 0.7 + 0.9 + 0.6 + 0.8 + 0.7) / 6.0
        assert abs(composite - expected) < 0.001

    def test_evidence_gap_creation(self):
        gap = EvidenceGap(
            claim_id=uuid4(),
            gap_type="no_evidence",
            description="No evidence found",
            suggested_query="Find evidence for claim",
            priority=0.9,
        )
        assert gap.gap_type == "no_evidence"
        assert gap.priority == 0.9

    def test_re_retrieval_trigger_creation(self):
        gap = EvidenceGap(
            claim_id=uuid4(),
            gap_type="no_evidence",
            description="No evidence",
        )
        trigger = ReRetrievalTrigger(
            gaps=[gap],
            max_additional_queries=1,
            original_query="test query",
        )
        assert trigger.max_additional_queries == 1
        assert len(trigger.gaps) == 1


class TestConfidenceScorer:
    def test_score_verification_basic(self):
        scorer = ConfidenceScorer()
        claim = Claim(
            text="Test claim",
            predicate="is",
            object_value="true",
            supporting_chunk_ids=[uuid4()],
            confidence=0.8,
        )

        evidence = [
            VerificationEvidence(
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_id=uuid4(),
                source_path="/test/doc.pdf",
                source_type="pdf",
                text="Supporting text",
                supports=True,
                relevance_score=0.9,
            )
        ]

        components = scorer.score_verification(
            claim=claim,
            supporting_evidence=evidence,
            contradicting_evidence=[],
            verifier_judgment=0.8,
        )

        assert 0.0 <= components.evidence_coverage <= 1.0
        assert 0.0 <= components.source_quality <= 1.0
        assert 0.0 <= components.cross_source_agreement <= 1.0
        assert 0.0 <= components.temporal_relevance <= 1.0
        assert 0.0 <= components.retrieval_rank <= 1.0
        assert components.verifier_judgment == 0.8

    def test_composite_confidence(self):
        components = ConfidenceComponents(
            evidence_coverage=0.8,
            source_quality=0.7,
            cross_source_agreement=0.9,
            temporal_relevance=0.6,
            retrieval_rank=0.8,
            verifier_judgment=0.7,
        )
        composite = compute_composite_confidence(components)
        assert 0.0 <= composite <= 1.0


class TestContradictionDetector:
    def test_detector_creation(self):
        detector = ContradictionDetector()
        assert detector is not None

    def test_claims_overlap_same_subject(self):
        detector = ContradictionDetector()
        claim_a = Claim(
            text="A",
            predicate="works at",
            subject_entity_id=uuid4(),
            object_value="Acme",
        )
        claim_b = Claim(
            text="B",
            predicate="employed by",
            subject_entity_id=claim_a.subject_entity_id,
            object_value="Beta",
        )
        assert detector._claims_overlap(claim_a, claim_b) is True

    def test_claims_overlap_different_subject(self):
        detector = ContradictionDetector()
        claim_a = Claim(
            text="A",
            predicate="works at",
            subject_entity_id=uuid4(),
            object_value="Acme",
        )
        claim_b = Claim(
            text="B",
            predicate="works at",
            subject_entity_id=uuid4(),
            object_value="Beta",
        )
        assert detector._claims_overlap(claim_a, claim_b) is False

    def test_predicates_similar(self):
        detector = ContradictionDetector()
        assert detector._predicates_similar("works at", "employed by") is True
        assert detector._predicates_similar("is", "was") is True
        assert detector._predicates_similar("works at", "located in") is False

    def test_detect_contradictions_empty(self):
        detector = ContradictionDetector()
        contradictions = detector.detect_contradictions([])
        assert contradictions == []

    def test_detect_source_conflict(self):
        detector = ContradictionDetector()
        subject_id = uuid4()
        claim_a = Claim(
            text="John works at Acme",
            predicate="works at",
            subject_entity_id=subject_id,
            object_value="Acme Corp",
            supporting_chunk_ids=[uuid4()],
        )
        claim_b = Claim(
            text="John works at Beta",
            predicate="employed by",
            subject_entity_id=subject_id,
            object_value="Beta Inc",
            supporting_chunk_ids=[uuid4()],
        )
        contradictions = detector.detect_contradictions([claim_a, claim_b])
        # Should detect source conflict
        source_conflicts = [c for c in contradictions if c.contradiction_type == ContradictionType.SOURCE_CONFLICT]
        assert len(source_conflicts) >= 1


class TestEvidenceGapDetector:
    def test_detector_creation(self):
        detector = EvidenceGapDetector()
        assert detector is not None

    def test_detect_gaps_unsupported(self):
        detector = EvidenceGapDetector()
        result = VerificationResult(
            claim_id=uuid4(),
            status=VerificationStatus.UNSUPPORTED,
            confidence=0.1,
            reasoning="No evidence",
        )
        batch = VerificationBatchResult(results=[result])
        gaps = detector.detect_gaps(batch)
        assert len(gaps) >= 1
        assert gaps[0].gap_type == "no_evidence"

    def test_detect_gaps_partial(self):
        detector = EvidenceGapDetector()
        result = VerificationResult(
            claim_id=uuid4(),
            status=VerificationStatus.PARTIAL,
            confidence=0.5,
            reasoning="Partial support",
            evidence_coverage=0.3,
        )
        batch = VerificationBatchResult(results=[result])
        gaps = detector.detect_gaps(batch)
        assert len(gaps) >= 1
        assert any(g.gap_type == "partial_coverage" for g in gaps)

    def test_generate_query(self):
        detector = EvidenceGapDetector()
        result = VerificationResult(
            claim_id=uuid4(),
            status=VerificationStatus.UNSUPPORTED,
            confidence=0.1,
            reasoning="No evidence",
        )
        query = detector._generate_query_for_claim(result)
        assert "Evidence for:" in query


class TestReRetrievalManager:
    def test_manager_creation(self, settings: Settings):
        manager = ReRetrievalManager(settings=settings)
        assert manager is not None

    def test_create_trigger_from_batch(self, settings: Settings):
        manager = ReRetrievalManager(settings=settings)
        result = VerificationResult(
            claim_id=uuid4(),
            status=VerificationStatus.UNSUPPORTED,
            confidence=0.1,
            reasoning="No evidence",
        )
        batch = VerificationBatchResult(results=[result])
        trigger = manager.create_trigger_from_batch(batch, "original query")
        assert trigger.max_additional_queries == 1
        assert len(trigger.gaps) >= 1


class TestVerificationBatchResult:
    def test_batch_result_counts(self):
        results = [
            VerificationResult(claim_id=uuid4(), status=VerificationStatus.SUPPORTED, confidence=0.9, reasoning=""),
            VerificationResult(claim_id=uuid4(), status=VerificationStatus.PARTIAL, confidence=0.5, reasoning=""),
            VerificationResult(claim_id=uuid4(), status=VerificationStatus.CONTRADICTED, confidence=0.3, reasoning=""),
            VerificationResult(claim_id=uuid4(), status=VerificationStatus.UNSUPPORTED, confidence=0.1, reasoning=""),
        ]
        batch = VerificationBatchResult(results=results)
        assert batch.total_claims == 4
        assert batch.supported_count == 1
        assert batch.partial_count == 1
        assert batch.contradicted_count == 1
        assert batch.unsupported_count == 1