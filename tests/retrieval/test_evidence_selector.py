"""Evidence Selector tests (Phase 20).

Tests the EvidenceSelector class: semantic dedup, source diversity,
token budget enforcement, and graceful degradation.
"""
from __future__ import annotations

import numpy as np
import pytest
from uuid import UUID, uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.retrieval.evidence_selector import EvidenceSelector, SelectionMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ref(
    score: float = 0.9,
    chunk_id: UUID | None = None,
    doc_id: UUID | None = None,
    text: str = "Acme Corp was founded in 2001.",
) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id or uuid4(),
        document_id=doc_id or uuid4(),
        source_id=uuid4(),
        source_path="test.pdf",
        source_type=SourceType.PDF,
        text=text,
        score=score,
        rank=1,
    )


class _FakeVectorStore:
    """In-memory fake vector store for testing."""

    def __init__(self, embeddings: dict[UUID, np.ndarray]):
        self._embeddings = embeddings

    def get_embedding(self, chunk_id: UUID) -> np.ndarray | None:
        return self._embeddings.get(chunk_id)


# ---------------------------------------------------------------------------
# Tests: EvidenceSelector
# ---------------------------------------------------------------------------
class TestEvidenceSelector:
    def test_empty_evidence(self):
        sel = EvidenceSelector()
        result = sel.select([], None)
        assert result == []

    def test_single_chunk_passes_through(self):
        ref = _make_ref()
        sel = EvidenceSelector()
        result = sel.select([ref], None)
        assert len(result) == 1
        assert result[0].chunk_id == ref.chunk_id

    def test_semantic_dedup_identical_embeddings(self):
        """Two chunks with identical embeddings → keep one."""
        id1, id2 = uuid4(), uuid4()
        doc1, doc2 = uuid4(), uuid4()
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1, doc_id=doc1)
        ref2 = _make_ref(score=0.80, chunk_id=id2, doc_id=doc2)
        store = _FakeVectorStore({id1: emb, id2: emb})
        sel = EvidenceSelector(similarity_threshold=0.85)
        result = sel.select([ref1, ref2], store)
        assert len(result) == 1
        assert result[0].chunk_id == id1  # higher score kept

    def test_semantic_dedup_orthogonal_embeddings(self):
        """Two chunks with orthogonal embeddings → keep both."""
        id1, id2 = uuid4(), uuid4()
        doc1, doc2 = uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1, doc_id=doc1)
        ref2 = _make_ref(score=0.80, chunk_id=id2, doc_id=doc2)
        store = _FakeVectorStore({id1: emb1, id2: emb2})
        sel = EvidenceSelector(similarity_threshold=0.85)
        result = sel.select([ref1, ref2], store)
        assert len(result) == 2

    def test_source_diversity(self):
        """5 chunks from 2 docs → prefer chunks from both docs."""
        doc_a = uuid4()
        doc_b = uuid4()
        refs = [
            _make_ref(score=0.95, doc_id=doc_a, text="Chunk A1"),
            _make_ref(score=0.93, doc_id=doc_a, text="Chunk A2"),
            _make_ref(score=0.91, doc_id=doc_a, text="Chunk A3"),
            _make_ref(score=0.85, doc_id=doc_b, text="Chunk B1"),
            _make_ref(score=0.80, doc_id=doc_b, text="Chunk B2"),
        ]
        sel = EvidenceSelector(max_chunks=3, min_sources=2)
        result = sel.select(refs, None)
        doc_ids = {r.document_id for r in result}
        assert len(doc_ids) >= 2  # both sources represented

    def test_max_chunks_respected(self):
        """Never exceed max_chunks."""
        refs = [_make_ref(score=0.9 - i * 0.05) for i in range(12)]
        sel = EvidenceSelector(max_chunks=5)
        result = sel.select(refs, None)
        assert len(result) <= 5

    def test_token_budget_respected(self):
        """Never exceed max_tokens."""
        # Each chunk ~1200 tokens (4800 chars / 4)
        refs = [
            _make_ref(score=0.9, text="word " * 1200),
            _make_ref(score=0.85, text="word " * 1200),
            _make_ref(score=0.80, text="word " * 1200),
        ]
        sel = EvidenceSelector(max_tokens=2000, min_chunks=1)
        result = sel.select(refs, None)
        total_tokens = sum(max(1, len(r.text) // 4) for r in result)
        assert total_tokens <= 2000

    def test_min_chunks_preserved(self):
        """Always keep at least min_chunks even over budget."""
        refs = [
            _make_ref(score=0.9, text="word " * 2000),
            _make_ref(score=0.85, text="word " * 2000),
        ]
        sel = EvidenceSelector(max_tokens=100, min_chunks=2)
        result = sel.select(refs, None)
        assert len(result) >= 2

    def test_deterministic_output(self):
        """Same inputs → same output."""
        refs = [_make_ref(score=0.9 - i * 0.05) for i in range(5)]
        sel = EvidenceSelector(max_chunks=3)
        r1 = sel.select(refs, None)
        r2 = sel.select(refs, None)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_missing_embeddings_graceful_fallback(self):
        """If embeddings are missing, skip semantic dedup but still apply other rules."""
        id1, id2 = uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb1})  # id2 missing
        sel = EvidenceSelector(similarity_threshold=0.85, max_chunks=1)
        result = sel.select([ref1, ref2], store)
        # Semantic dedup skipped (fallback), but max_chunks still applies
        assert len(result) == 1

    def test_high_relevance_beats_diversity(self):
        """Highly relevant chunks should not be discarded for diversity."""
        doc_a = uuid4()
        doc_b = uuid4()
        refs = [
            _make_ref(score=0.99, doc_id=doc_a, text="Critical evidence"),
            _make_ref(score=0.98, doc_id=doc_a, text="Also critical"),
            _make_ref(score=0.50, doc_id=doc_b, text="Weak evidence"),
        ]
        sel = EvidenceSelector(max_chunks=2, min_sources=1)
        result = sel.select(refs, None)
        scores = [r.score for r in result]
        assert scores[0] >= 0.98  # top chunks preserved

    def test_metrics_populated(self):
        """SelectionMetrics is populated correctly."""
        refs = [_make_ref(score=0.9 - i * 0.05) for i in range(5)]
        sel = EvidenceSelector(max_chunks=3)
        metrics = SelectionMetrics()
        sel.select(refs, None, metrics=metrics)
        assert metrics.candidate_count == 5
        assert metrics.selected_count <= 3
        assert metrics.selection_latency_ms >= 0
        assert metrics.estimated_tokens > 0

    def test_threshold_zero_disables_dedup(self):
        """Threshold 0 disables semantic dedup."""
        id1, id2 = uuid4(), uuid4()
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb, id2: emb})
        sel = EvidenceSelector(similarity_threshold=0.0)
        result = sel.select([ref1, ref2], store)
        assert len(result) == 2

    def test_three_chunks_one_cluster(self):
        """Three chunks: two similar, one different → keep 2."""
        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        doc1, doc2, doc3 = uuid4(), uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.98, 0.20, 0.0, 0.0], dtype=np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        emb3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1, doc_id=doc1)
        ref2 = _make_ref(score=0.80, chunk_id=id2, doc_id=doc2)
        ref3 = _make_ref(score=0.70, chunk_id=id3, doc_id=doc3)
        store = _FakeVectorStore({id1: emb1, id2: emb2, id3: emb3})
        sel = EvidenceSelector(similarity_threshold=0.85, max_chunks=5)
        result = sel.select([ref1, ref2, ref3], store)
        assert len(result) == 2
        ids_kept = {r.chunk_id for r in result}
        assert id1 in ids_kept
        assert id3 in ids_kept

    def test_preserves_evidence_metadata(self):
        """Selected evidence preserves all original metadata."""
        ref = _make_ref(score=0.9, text="Important evidence")
        sel = EvidenceSelector()
        result = sel.select([ref], None)
        assert result[0].chunk_id == ref.chunk_id
        assert result[0].source_path == ref.source_path
        assert result[0].document_id == ref.document_id
        assert result[0].score == ref.score
