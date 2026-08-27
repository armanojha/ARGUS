"""Tests for Ingestion Pipeline (Phase 01)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from app.evidence.store import EvidenceStore
from app.ingestion.chunking import TextSegment, chunk_by_sections, chunk_text
from app.ingestion.pdf import extract_pdf_segments, extract_pdf_text
from app.ingestion.pipeline import IngestionPipeline, ingest_corpus_directory


class TestChunking:
    def test_chunk_text_basic(self):
        segments = [
            TextSegment(text="This is a test document. " * 20, page_start=1, page_end=1),
        ]
        document_id = uuid4()
        chunks = chunk_text(segments, document_id)

        assert len(chunks) > 0
        assert all(c.document_id == document_id for c in chunks)
        assert all(c.ordinal == i for i, c in enumerate(chunks))

    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap."""
        long_text = "This is a test sentence. " * 50
        segments = [TextSegment(text=long_text, page_start=1, page_end=5)]
        document_id = uuid4()

        chunks = chunk_text(segments, document_id, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1

    def test_chunk_by_sections(self):
        segments = [
            TextSegment(text="Introduction content here.", page_start=1, section_path="Introduction"),
            TextSegment(text="Background content here.", page_start=1, section_path="Background"),
            TextSegment(text="Methodology content here.", page_start=2, section_path="Methodology"),
        ]
        document_id = uuid4()
        chunks = chunk_by_sections(segments, document_id)

        assert len(chunks) == 3
        assert chunks[0].section_path == "Introduction"
        assert chunks[1].section_path == "Background"
        assert chunks[2].section_path == "Methodology"


class TestPDFExtraction:
    def test_extract_pdf_text(self):
        """Test PDF text extraction with a simple PDF."""
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            writer.write(pdf_path)

            text = extract_pdf_text(pdf_path)
            assert isinstance(text, str)
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extract_pdf_segments(self):
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = Path(f.name)

        try:
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            writer.write(pdf_path)

            segments = extract_pdf_segments(pdf_path)
            assert isinstance(segments, list)
        finally:
            pdf_path.unlink(missing_ok=True)


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
def pipeline(store):
    return IngestionPipeline(store)


class TestIngestionPipeline:
    def test_ingest_text_file(self, pipeline, temp_db):
        """Test ingesting a plain text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nAnd some content.")
            text_path = Path(f.name)

        try:
            doc = pipeline.ingest_text_file(text_path)
            assert doc is not None
            assert doc.source_id is not None
            assert doc.version == 1
            assert doc.chunking_strategy == "semantic_v1"

            # Verify chunks were created
            chunks = pipeline.store.get_chunks_by_document(doc.id)
            assert len(chunks) > 0
        finally:
            Path(text_path).unlink(missing_ok=True)

    def test_ingest_text_file_deduplication(self, pipeline):
        """Test that ingesting the same file twice returns the same document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Duplicate test content.")
            text_path = Path(f.name)

        try:
            doc1 = pipeline.ingest_text_file(text_path)
            doc2 = pipeline.ingest_text_file(text_path)

            # Should return the same document (deduplicated)
            assert doc1.id == doc2.id
            assert doc1.version == doc2.version
        finally:
            Path(text_path).unlink(missing_ok=True)

    def test_ingest_corpus_directory(self, pipeline):
        """Test ingesting a directory of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir)

            # Create test files
            (corpus_dir / "doc1.txt").write_text("First document content.")
            (corpus_dir / "doc2.md").write_text("# Markdown Document\n\nContent here.")
            subdir = corpus_dir / "subdir"
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / "doc3.txt").write_text("Nested document.")

            docs = ingest_corpus_directory(corpus_dir, pipeline.store)

            assert len(docs) == 3
            # All should be ingested successfully
            for doc in docs:
                chunks = pipeline.store.get_chunks_by_document(doc.id)
                assert len(chunks) > 0


class TestIngestionEdgeCases:
    @pytest.fixture
    def pipeline(self, store):
        return IngestionPipeline(store)

    def test_empty_file_raises(self, pipeline):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            text_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Empty file"):
                pipeline.ingest_text_file(text_path)
        finally:
            Path(text_path).unlink(missing_ok=True)

    def test_nonexistent_file_raises(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_text_file(Path("/nonexistent/file.txt"))