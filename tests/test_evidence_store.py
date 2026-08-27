"""Tests for Evidence Store (Phase 01)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def store(temp_db):
    """Create an EvidenceStore with a temporary database."""
    return EvidenceStore(temp_db)


@pytest.fixture
def sample_source():
    return Source(
        type=SourceType.PDF,
        path="/test/document.pdf",
        checksum="abc123",
        metadata={"author": "Test Author"},
    )


@pytest.fixture
def sample_document(sample_source):
    return Document(
        source_id=sample_source.id,
        version=1,
        checksum="def456",
        chunking_strategy="semantic_v1",
        metadata={"pages": 10},
    )


@pytest.fixture
def sample_chunks(sample_document):
    return [
        Chunk(
            document_id=sample_document.id,
            ordinal=0,
            text="This is the first chunk of text.",
            token_count=10,
            page_start=1,
            page_end=1,
            section_path="Introduction",
        ),
        Chunk(
            document_id=sample_document.id,
            ordinal=1,
            text="This is the second chunk with more content.",
            token_count=12,
            page_start=1,
            page_end=2,
            section_path="Background",
        ),
    ]


class TestSourceOperations:
    def test_upsert_source(self, store, sample_source):
        result = store.upsert_source(sample_source)
        assert result.id == sample_source.id
        assert result.checksum == sample_source.checksum

        # Fetch and verify
        fetched = store.get_source(sample_source.id)
        assert fetched is not None
        assert fetched.id == sample_source.id
        assert fetched.path == sample_source.path

    def test_get_source_by_checksum(self, store, sample_source):
        store.upsert_source(sample_source)
        fetched = store.get_source_by_checksum(sample_source.checksum)
        assert fetched is not None
        assert fetched.id == sample_source.id

    def test_source_deduplication(self, store, sample_source):
        """Same checksum should return existing source."""
        store.upsert_source(sample_source)
        duplicate = Source(
            type=SourceType.PDF,
            path="/different/path.pdf",
            checksum=sample_source.checksum,
        )
        result = store.upsert_source(duplicate)
        assert result.id == sample_source.id  # Same ID returned


class TestDocumentOperations:
    def test_insert_document(self, store, sample_source, sample_document):
        store.upsert_source(sample_source)
        result = store.insert_document(sample_document)
        assert result.id == sample_document.id

        fetched = store.get_document(sample_document.id)
        assert fetched is not None
        assert fetched.source_id == sample_document.source_id
        assert fetched.version == sample_document.version

    def test_get_latest_document_for_source(self, store, sample_source):
        store.upsert_source(sample_source)

        doc_v1 = Document(source_id=sample_source.id, version=1, checksum="v1", chunking_strategy="v1")
        doc_v2 = Document(source_id=sample_source.id, version=2, checksum="v2", chunking_strategy="v2")

        store.insert_document(doc_v1)
        store.insert_document(doc_v2)

        latest = store.get_latest_document_for_source(sample_source.id)
        assert latest is not None
        assert latest.version == 2


class TestChunkOperations:
    def test_insert_chunks(self, store, sample_source, sample_document, sample_chunks):
        store.upsert_source(sample_source)
        store.insert_document(sample_document)

        # Update chunk document_ids
        for chunk in sample_chunks:
            chunk.document_id = sample_document.id

        result = store.insert_chunks(sample_chunks)
        assert len(result) == 2

        # Verify retrieval
        fetched = store.get_chunks_by_document(sample_document.id)
        assert len(fetched) == 2
        assert fetched[0].ordinal == 0
        assert fetched[1].ordinal == 1

    def test_get_chunk(self, store, sample_source, sample_document, sample_chunks):
        store.upsert_source(sample_source)
        store.insert_document(sample_document)
        for chunk in sample_chunks:
            chunk.document_id = sample_document.id
        store.insert_chunks(sample_chunks)

        chunk = store.get_chunk(sample_chunks[0].id)
        assert chunk is not None
        assert chunk.text == sample_chunks[0].text

    def test_get_chunks_by_ids(self, store, sample_source, sample_document, sample_chunks):
        store.upsert_source(sample_source)
        store.insert_document(sample_document)
        for chunk in sample_chunks:
            chunk.document_id = sample_document.id
        store.insert_chunks(sample_chunks)

        ids = [c.id for c in sample_chunks]
        fetched = store.get_chunks_by_ids(ids)
        assert len(fetched) == 2


class TestEvidenceRefs:
    def test_get_evidence_refs(self, store, sample_source, sample_document, sample_chunks):
        store.upsert_source(sample_source)
        store.insert_document(sample_document)
        for chunk in sample_chunks:
            chunk.document_id = sample_document.id
        store.insert_chunks(sample_chunks)

        chunk_ids = [c.id for c in sample_chunks]
        scores = [0.9, 0.8]

        refs = store.get_evidence_refs(chunk_ids, scores)

        assert len(refs) == 2
        assert refs[0].chunk_id == sample_chunks[0].id
        assert refs[0].score == 0.9
        assert refs[0].rank == 1
        assert refs[1].rank == 2
        assert refs[0].source_path == sample_source.path
        assert refs[0].source_type == sample_source.type


class TestChecksum:
    def test_compute_checksum(self):
        content = b"test content"
        checksum = EvidenceStore.compute_checksum(content)
        assert len(checksum) == 64  # SHA256 hex
        assert checksum == EvidenceStore.compute_checksum(content)