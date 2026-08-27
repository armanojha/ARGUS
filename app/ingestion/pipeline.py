"""Document ingestion pipeline (Phase 01).

Orchestrates extraction, chunking, embedding, and storage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from app.config import get_settings
from app.evidence.models import Document, Source, SourceType
from app.evidence.store import EvidenceStore, _json_dumps, get_evidence_store
from app.ingestion.chunking import TextSegment, chunk_by_sections
from app.ingestion.pdf import extract_pdf_segments, extract_pdf_text
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.pipeline")


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(self, store: EvidenceStore | None = None):
        self.store = store or get_evidence_store()
        self.settings = get_settings()

    def ingest_pdf(
        self,
        pdf_path: Path,
        source_type: SourceType = SourceType.PDF,
        chunking_strategy: str = "semantic_v1",
    ) -> Document:
        """Ingest a PDF file through the full pipeline.

        Steps:
        1. Compute source checksum
        2. Upsert source
        3. Check if document version exists
        4. Extract text segments with provenance
        4. Chunk with provenance preservation
        5. Store chunks
        6. Create document record

        Returns the created Document.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 1. Compute source checksum
        pdf_bytes = pdf_path.read_bytes()
        source_checksum = hashlib.sha256(pdf_bytes).hexdigest()

        # 2. Upsert source
        existing_source = self.store.get_source_by_checksum(source_checksum)
        if existing_source:
            source = existing_source
            logger.info("source_exists", source_id=str(source.id), path=str(pdf_path))
        else:
            source = Source(
                type=source_type,
                path=str(pdf_path),
                checksum=source_checksum,
                metadata={"filename": pdf_path.name, "size_bytes": len(pdf_bytes)},
            )
            source = self.store.upsert_source(source)
            logger.info("source_created", source_id=str(source.id), path=str(pdf_path))

        # 3. Check for existing document version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        if existing_doc and existing_doc.chunking_strategy == chunking_strategy:
            # Verify checksum matches
            full_text = extract_pdf_text(pdf_path)
            doc_checksum = hashlib.sha256(full_text.encode()).hexdigest()
            if existing_doc.checksum == doc_checksum:
                logger.info("document_unchanged", document_id=str(existing_doc.id))
                return existing_doc

        # 4. Extract text segments with provenance
        segments = extract_pdf_segments(pdf_path)
        if not segments:
            raise ValueError(f"No extractable text found in {pdf_path}")

        # 5. Chunk with provenance preservation
        # Use section-aware chunking for better context preservation
        chunks = chunk_by_sections(segments, UUID(int=0))  # placeholder doc_id

        # 6. Create document record
        full_text = extract_pdf_text(pdf_path)
        doc_checksum = hashlib.sha256(full_text.encode()).hexdigest()

        # Determine next version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        next_version = (existing_doc.version + 1) if existing_doc else 1

        document = Document(
            source_id=source.id,
            version=next_version,
            checksum=doc_checksum,
            chunking_strategy=chunking_strategy,
            metadata={
                "source_path": str(pdf_path),
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
            },
        )

        # 7. Update chunks with correct document_id and store atomically
        for i, chunk in enumerate(chunks):
            chunk.document_id = document.id
            chunk.ordinal = i

        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source_id, version, checksum, chunking_strategy, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    str(document.source_id),
                    document.version,
                    document.checksum,
                    document.chunking_strategy,
                    _json_dumps(document.metadata),
                    document.created_at.isoformat(),
                ),
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, ordinal, text, token_count,
                        page_start, page_end, char_start, char_end, section_path,
                        embedding_index, bm25_doc_id, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.ordinal,
                        chunk.text,
                        chunk.token_count,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section_path,
                        chunk.embedding_index,
                        chunk.bm25_doc_id,
                        _json_dumps(chunk.metadata),
                        chunk.created_at.isoformat(),
                    ),
                )

        logger.info(
            "document_ingested",
            document_id=str(document.id),
            source_id=str(source.id),
            version=document.version,
            chunk_count=len(chunks),
        )
        return document

    def ingest_text_file(
        self,
        file_path: Path,
        source_type: SourceType = SourceType.TEXT,
        chunking_strategy: str = "semantic_v1",
    ) -> Document:
        """Ingest a plain text or markdown file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty file: {file_path}")

        # Compute checksums
        source_checksum = hashlib.sha256(text.encode()).hexdigest()
        doc_checksum = source_checksum  # Same for text files

        # Upsert source
        existing_source = self.store.get_source_by_checksum(source_checksum)
        if existing_source:
            source = existing_source
        else:
            source = Source(
                type=source_type,
                path=str(file_path),
                checksum=source_checksum,
                metadata={"filename": file_path.name},
            )
            source = self.store.upsert_source(source)

        # Check existing document
        existing_doc = self.store.get_latest_document_for_source(source.id)
        if (
            existing_doc
            and existing_doc.chunking_strategy == chunking_strategy
            and existing_doc.checksum == doc_checksum
        ):
            return existing_doc

        # Create segments (single segment for text file)
        segments = [TextSegment(
            text=text,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(text),
            section_path=None,
        )]

        # Chunk
        chunks = chunk_by_sections(segments, UUID(int=0))

        # Create document
        existing_doc = self.store.get_latest_document_for_source(source.id)
        next_version = (existing_doc.version + 1) if existing_doc else 1

        document = Document(
            source_id=source.id,
            version=next_version,
            checksum=doc_checksum,
            chunking_strategy=chunking_strategy,
            metadata={
                "source_path": str(file_path),
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
            },
        )

        # Update and store chunks atomically
        for i, chunk in enumerate(chunks):
            chunk.document_id = document.id
            chunk.ordinal = i

        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source_id, version, checksum, chunking_strategy, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    str(document.source_id),
                    document.version,
                    document.checksum,
                    document.chunking_strategy,
                    _json_dumps(document.metadata),
                    document.created_at.isoformat(),
                ),
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, ordinal, text, token_count,
                        page_start, page_end, char_start, char_end, section_path,
                        embedding_index, bm25_doc_id, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.ordinal,
                        chunk.text,
                        chunk.token_count,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section_path,
                        chunk.embedding_index,
                        chunk.bm25_doc_id,
                        _json_dumps(chunk.metadata),
                        chunk.created_at.isoformat(),
                    ),
                )

        logger.info(
            "text_file_ingested",
            document_id=str(document.id),
            chunk_count=len(chunks),
        )
        return document


def ingest_corpus_directory(
    corpus_dir: Path,
    store: EvidenceStore | None = None,
) -> list[Document]:
    """Ingest all supported files in a directory recursively.

    Supports: .pdf, .txt, .md
    """
    store = store or get_evidence_store()
    pipeline = IngestionPipeline(store)

    supported_extensions = {".pdf", ".txt", ".md"}
    documents = []

    for file_path in corpus_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                if file_path.suffix.lower() == ".pdf":
                    doc = pipeline.ingest_pdf(file_path)
                else:
                    doc = pipeline.ingest_text_file(file_path)
                documents.append(doc)
            except (OSError, ValueError, RuntimeError) as e:
                logger.error("ingestion_failed", path=str(file_path), error=str(e))
                # Continue with other files

    logger.info("corpus_ingested", directory=str(corpus_dir), document_count=len(documents))
    return documents