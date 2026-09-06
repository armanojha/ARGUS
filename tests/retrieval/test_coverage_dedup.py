"""Evidence coverage deduplication tests.

Tests the _coverage_deduplicate function and the get_embedding method
on FAISSVectorStore. Verifies that near-duplicate evidence chunks are
dropped to maximize coverage of distinct claims.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.nodes import _coverage_deduplicate, _merge_evidence
from app.retrieval.vector import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ref(score: float = 0.9, chunk_id: UUID | None = None) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="test.pdf",
        source_type=SourceType.PDF,
        text="Acme Corp was founded in 2001.",
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
# Tests: _coverage_deduplicate
# ---------------------------------------------------------------------------
class TestCoverageDeduplicate:
    def test_empty_list_returns_empty(self):
        store = _FakeVectorStore({})
        result = _coverage_deduplicate([], store, threshold=0.85)
        assert result == []

    def test_single_chunk_passes_through(self):
        ref = _make_ref()
        store = _FakeVectorStore({ref.chunk_id: np.ones(8)})
        result = _coverage_deduplicate([ref], store, threshold=0.85)
        assert len(result) == 1

    def test_identical_embeddings_dedup(self):
        """Two chunks with identical embeddings should be deduped."""
        id1, id2 = uuid4(), uuid4()
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb, id2: emb})
        result = _coverage_deduplicate([ref1, ref2], store, threshold=0.85)
        assert len(result) == 1
        assert result[0].chunk_id == id1  # higher score kept

    def test_orthogonal_embeddings_both_kept(self):
        """Two chunks with orthogonal embeddings should both be kept."""
        id1, id2 = uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb1, id2: emb2})
        result = _coverage_deduplicate([ref1, ref2], store, threshold=0.85)
        assert len(result) == 2

    def test_similar_embeddings_dedup(self):
        """Two chunks with cosine sim ~0.95 should be deduped at threshold 0.85."""
        id1, id2 = uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Slightly rotated → cos sim ≈ 0.95
        emb2 = np.array([0.95, 0.312, 0.0, 0.0], dtype=np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb1, id2: emb2})
        result = _coverage_deduplicate([ref1, ref2], store, threshold=0.85)
        assert len(result) == 1
        assert result[0].chunk_id == id1

    def test_threshold_zero_disables(self):
        """Threshold 0 disables dedup entirely."""
        id1, id2 = uuid4(), uuid4()
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb, id2: emb})
        result = _coverage_deduplicate([ref1, ref2], store, threshold=0.0)
        assert len(result) == 2

    def test_missing_embedding_falls_back(self):
        """If any chunk lacks an embedding, fall back to original list."""
        id1, id2 = uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        store = _FakeVectorStore({id1: emb1})  # id2 missing
        result = _coverage_deduplicate([ref1, ref2], store, threshold=0.85)
        assert len(result) == 2  # fallback, no dedup

    def test_three_chunks_one_cluster(self):
        """Three chunks: two similar, one different → keep 2."""
        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.98, 0.20, 0.0, 0.0], dtype=np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        emb3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        ref1 = _make_ref(score=0.95, chunk_id=id1)
        ref2 = _make_ref(score=0.80, chunk_id=id2)
        ref3 = _make_ref(score=0.70, chunk_id=id3)
        store = _FakeVectorStore({id1: emb1, id2: emb2, id3: emb3})
        result = _coverage_deduplicate([ref1, ref2, ref3], store, threshold=0.85)
        assert len(result) == 2
        ids_kept = {r.chunk_id for r in result}
        assert id1 in ids_kept
        assert id3 in ids_kept

    def test_preserves_order_by_score(self):
        """Output preserves the score-descending order of selected chunks."""
        ids = [uuid4() for _ in range(4)]
        # All different embeddings
        embs = [np.zeros(4, dtype=np.float32) for _ in range(4)]
        for i in range(4):
            embs[i][i] = 1.0
        refs = [_make_ref(score=0.9 - i * 0.1, chunk_id=ids[i]) for i in range(4)]
        store = _FakeVectorStore({ids[i]: embs[i] for i in range(4)})
        result = _coverage_deduplicate(refs, store, threshold=0.85)
        assert len(result) == 4
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: get_embedding on FAISSVectorStore
# ---------------------------------------------------------------------------
class TestGetEmbedding:
    def test_returns_none_when_no_index(self):
        store = FAISSVectorStore()
        assert store.get_embedding(uuid4()) is None

    def test_returns_embedding_for_indexed_chunk(self):
        import faiss

        id1, id2 = uuid4(), uuid4()
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(3)
        idx.add(embeddings)
        store = FAISSVectorStore()
        store._index = idx
        store._chunk_ids = [str(id1), str(id2)]

        emb = store.get_embedding(id1)
        assert emb is not None
        np.testing.assert_allclose(emb, embeddings[0], atol=1e-6)

    def test_returns_none_for_unknown_chunk(self):
        import faiss

        id1 = uuid4()
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(3)
        idx.add(embeddings)
        store = FAISSVectorStore()
        store._index = idx
        store._chunk_ids = [str(id1)]

        assert store.get_embedding(uuid4()) is None


# ---------------------------------------------------------------------------
# Tests: _merge_evidence (unchanged behavior)
# ---------------------------------------------------------------------------
class TestMergeEvidence:
    def test_dedup_by_chunk_id(self):
        id1 = uuid4()
        ref_a = _make_ref(score=0.8, chunk_id=id1)
        ref_b = _make_ref(score=0.9, chunk_id=id1)
        merged, count = _merge_evidence([ref_a], [ref_b])
        assert count == 0  # not new, just updated
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_new_chunks_added(self):
        ref1 = _make_ref(score=0.8)
        ref2 = _make_ref(score=0.9)
        merged, count = _merge_evidence([ref1], [ref2])
        assert count == 1
        assert len(merged) == 2
