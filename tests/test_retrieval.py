"""Tests for Retrieval (Phase 01)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def store(temp_db):
    return EvidenceStore(temp_db)


@pytest.fixture
def populated_store(store):
    """Create a store with test data."""
    # Create source
    source = Source(
        type=SourceType.TEXT,
        path="/test/corpus.txt",
        checksum="corpus_checksum",
    )
    store.upsert_source(source)

    # Create document
    doc = Document(
        source_id=source.id,
        version=1,
        checksum="doc_checksum",
        chunking_strategy="semantic_v1",
    )
    store.insert_document(doc)

    # Create chunks with varied content
    chunks = [
        Chunk(
            document_id=doc.id,
            ordinal=0,
            text="The quick brown fox jumps over the lazy dog.",
            token_count=10,
            page_start=1,
        ),
        Chunk(
            document_id=doc.id,
            ordinal=1,
            text="Machine learning models require large datasets for training.",
            token_count=12,
            page_start=1,
        ),
        Chunk(
            document_id=doc.id,
            ordinal=2,
            text="Neural networks are inspired by biological neurons.",
            token_count=10,
            page_start=2,
        ),
        Chunk(
            document_id=doc.id,
            ordinal=3,
            text="The fox is a clever animal that adapts to environments.",
            token_count=12,
            page_start=2,
        ),
        Chunk(
            document_id=doc.id,
            ordinal=4,
            text="Deep learning uses multiple layers of neural networks.",
            token_count=10,
            page_start=3,
        ),
    ]
    store.insert_chunks(chunks)

    # Assign BM25 doc IDs
    assign_bm25_doc_ids(store)

    # Assign embedding indices
    assign_embedding_indices(store)

    return store


class TestBM25Retriever:
    def test_build_index(self, populated_store):
        bm25 = BM25Retriever(populated_store)
        bm25.build_index()

        assert bm25._bm25 is not None
        assert len(bm25._chunk_ids) == 5

    def test_search_exact_match(self, populated_store):
        bm25 = BM25Retriever(populated_store)
        bm25.build_index()

        results = bm25.search("fox", top_k=3)
        assert len(results) > 0
        # Should find chunks mentioning "fox"
        # Verify results are relevant
        for cid, score in results:
            assert score > 0

    def test_search_no_results(self, populated_store):
        bm25 = BM25Retriever(populated_store)
        bm25.build_index()

        results = bm25.search("xyzqwerty", top_k=5)
        assert len(results) == 0

    def test_save_load_index(self, populated_store):
        bm25 = BM25Retriever(populated_store)
        bm25.build_index()

        # Create new retriever with same index path
        bm25_2 = BM25Retriever(populated_store)
        loaded = bm25_2.load_index()

        assert loaded is True
        assert len(bm25_2._chunk_ids) == 5


class TestFAISSVectorStore:
    def test_build_index(self, populated_store):
        import numpy as np

        vector_store = FAISSVectorStore(populated_store)

        # Create dummy embeddings
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunk_ids = [c.id for c in populated_store.get_chunks_by_document(
            populated_store.get_latest_document_for_source(
                populated_store.get_source_by_checksum("corpus_checksum").id
            ).id
        )]

        vector_store.build_index(embeddings, chunk_ids)

        assert vector_store._index is not None
        assert vector_store._index.ntotal == 5

    def test_search(self, populated_store):
        import numpy as np

        vector_store = FAISSVectorStore(populated_store)
        embeddings = np.random.rand(5, 384).astype(np.float32)
        chunk_ids = [c.id for c in populated_store.get_chunks_by_document(
            populated_store.get_latest_document_for_source(
                populated_store.get_source_by_checksum("corpus_checksum").id
            ).id
        )]

        vector_store.build_index(embeddings, chunk_ids)

        # Search with a random query embedding
        query_emb = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_emb, top_k=3)

        assert len(results) <= 3
        for cid, score in results:
            assert 0 <= score <= 1  # Cosine similarity


class TestHybridRetriever:
    def test_hybrid_search(self, populated_store):
        retriever = HybridRetriever(populated_store)

        # Build indexes
        retriever.ensure_indexes()

        # Search
        results = retriever.search("fox", top_k=3)

        assert len(results) > 0
        for ref in results:
            assert ref.score > 0
            assert ref.rank > 0
            assert ref.text is not None
            assert ref.source_path is not None

    def test_bm25_only_search(self, populated_store):
        retriever = HybridRetriever(populated_store)
        retriever.ensure_indexes()

        results = retriever.search_bm25_only("machine learning", top_k=3)
        assert len(results) > 0

    def test_vector_only_search(self, populated_store):
        """Test vector-only search with matching embeddings."""
        from app.retrieval.embeddings import EmbeddingGenerator
        from app.retrieval.vector import FAISSVectorStore

        store = populated_store
        vector_store = FAISSVectorStore(store)
        embedder = EmbeddingGenerator()

        # Get chunks and generate real embeddings
        chunks = store.get_chunks_by_document(
            store.get_latest_document_for_source(
                store.get_source_by_checksum("corpus_checksum").id
            ).id
        )
        chunk_ids = [c.id for c in chunks]
        embeddings = embedder.embed_chunks(chunks)

        # Build index with real embeddings
        vector_store = FAISSVectorStore(store)
        vector_store.build_index(embeddings, chunk_ids)

        # Search with a query that should match one of the chunks
        query_emb = embedder.embed_texts(["neural networks"])[0]
        results = vector_store.search(query_emb, top_k=3)

        assert len(results) > 0
        for cid, score in results:
            assert 0 <= score <= 1

    def test_hybrid_fusion_weights(self, populated_store):
        retriever = HybridRetriever(populated_store)
        retriever.ensure_indexes()

        # Test with different weights
        results_bm25 = retriever.search("fox", top_k=3, bm25_weight=1.0, vector_weight=0.0)
        results_hybrid = retriever.search("fox", top_k=3, bm25_weight=0.5, vector_weight=0.5)

        # Both should return results
        assert len(results_bm25) > 0
        assert len(results_hybrid) > 0