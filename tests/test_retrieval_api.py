"""Tests for Retrieval API (Phase 01)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.vector import assign_embedding_indices


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def index_dir():
    """Create a temporary directory for indexes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_db, index_dir):
    settings = Settings(_env_file=None, data_dir=temp_db.parent, config_dir=temp_db.parent / "configs")
    # Override index paths to use temporary directory
    settings.bm25_index_path = index_dir / "bm25.pkl"
    settings.faiss_index_path = index_dir / "faiss.index"
    settings.evidence_db_path = temp_db
    return EvidenceStore(temp_db, bm25_index_path=settings.bm25_index_path, faiss_index_path=settings.faiss_index_path)


@pytest.fixture
def populated_store(store):
    """Create a store with test data."""
    source = Source(
        type=SourceType.TEXT,
        path="/test/corpus.txt",
        checksum="corpus_checksum",
    )
    store.upsert_source(source)

    doc = Document(
        source_id=source.id,
        version=1,
        checksum="doc_checksum",
        chunking_strategy="semantic_v1",
    )
    store.insert_document(doc)

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
    ]
    store.insert_chunks(chunks)
    assign_bm25_doc_ids(store)
    assign_embedding_indices(store)

    return store


@pytest.fixture
def client(populated_store, index_dir):
    """Create test client with populated store."""
    # Override the store dependency
    import app.evidence.store as store_module
    original_get_store = store_module.get_evidence_store

    def mock_get_store():
        return populated_store

    store_module.get_evidence_store = mock_get_store

    # Also override retrieval store - create retrievers with correct index paths
    import app.api.retrieval as api_retrieval_module
    import app.retrieval.bm25 as bm25_module
    import app.retrieval.embeddings as embeddings_module
    import app.retrieval.hybrid as hybrid_module
    import app.retrieval.vector as vector_module
    original_get_retriever = hybrid_module.get_hybrid_retriever
    original_api_get_retriever = api_retrieval_module.get_hybrid_retriever

    # Build retriever and indexes ONCE during fixture setup
    bm25 = bm25_module.BM25Retriever(populated_store, index_path=index_dir / "bm25.pkl")
    vector = vector_module.FAISSVectorStore(populated_store, index_path=index_dir / "faiss.index")
    embedder = embeddings_module.EmbeddingGenerator()
    retriever = hybrid_module.HybridRetriever(populated_store, bm25=bm25, vector=vector, embedder=embedder)
    retriever.ensure_indexes()

    # Make ensure_indexes a no-op for subsequent calls (API calls it on every request)
    retriever.ensure_indexes = lambda: None

    def mock_get_retriever():
        return retriever

    # Patch BOTH the source module and the API module's imported reference
    hybrid_module.get_hybrid_retriever = mock_get_retriever
    api_retrieval_module.get_hybrid_retriever = mock_get_retriever

    app = create_app()
    with TestClient(app) as client:
        yield client

    # Restore
    store_module.get_evidence_store = original_get_store
    hybrid_module.get_hybrid_retriever = original_get_retriever
    api_retrieval_module.get_hybrid_retriever = original_api_get_retriever


class TestRetrievalAPI:
    def test_retrieve_post(self, client):
        response = client.post("/api/v1/retrieve", json={
            "query": "fox jumps",
            "top_k": 3,
            "mode": "hybrid",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "fox jumps"
        assert data["mode"] == "hybrid"
        assert "citations" in data
        assert len(data["citations"]) > 0

        # Check citation structure
        citation = data["citations"][0]
        assert "chunk_id" in citation
        assert "document_id" in citation
        assert "source_id" in citation
        assert "source_path" in citation
        assert "source_type" in citation
        assert "text" in citation
        assert "score" in citation
        assert "rank" in citation

    def test_retrieve_get(self, client):
        response = client.get("/api/v1/retrieve", params={
            "query": "machine learning",
            "top_k": 2,
            "mode": "bm25",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert data["mode"] == "bm25"
        assert len(data["citations"]) <= 2

    def test_retrieve_empty_query(self, client):
        response = client.post("/api/v1/retrieve", json={
            "query": "",
            "top_k": 3,
        })

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422

    def test_retrieve_modes(self, client):
        """Test all retrieval modes work."""
        for mode in ["hybrid", "bm25", "vector"]:
            response = client.post("/api/v1/retrieve", json={
                "query": "test query",
                "top_k": 2,
                "mode": mode,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["mode"] == mode

    def test_retrieve_weights(self, client):
        """Test custom weights."""
        response = client.post("/api/v1/retrieve", json={
            "query": "test",
            "top_k": 2,
            "bm25_weight": 0.8,
            "vector_weight": 0.2,
        })
        assert response.status_code == 200

    def test_retrieve_reranker_toggle(self, client):
        """Test reranker on/off."""
        response = client.post("/api/v1/retrieve", json={
            "query": "test",
            "top_k": 2,
            "use_reranker": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["reranked"] is False

    def test_citation_structure(self, client):
        """Verify citation has all required fields for Phase 01."""
        response = client.post("/api/v1/retrieve", json={
            "query": "fox",
            "top_k": 1,
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["citations"]) > 0
        citation = data["citations"][0]

        # Required fields for citation mapping
        assert citation["chunk_id"] is not None
        assert citation["document_id"] is not None
        assert citation["source_id"] is not None
        assert citation["source_path"] == "/test/corpus.txt"
        assert citation["source_type"] == "text"
        assert citation["text"] is not None
        assert citation["score"] > 0
        assert citation["rank"] == 1