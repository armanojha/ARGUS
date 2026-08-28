"""Document ingestion pipeline (Phase 01 + Phase 11 Multimodal).

Orchestrates extraction, chunking, embedding, and storage.
Supports PDF, text, web pages, spreadsheets, and multimodal content.
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
from app.ingestion.ocr import extract_pdf_segments_with_ocr, extract_pdf_text_for_checksum
from app.ingestion.tables import extract_pdf_tables, tables_to_multimodal, tables_to_text_segments
from app.ingestion.web import fetch_web_page, web_page_to_multimodal, web_page_to_text_segments, compute_web_page_checksum, is_valid_web_url
from app.ingestion.spreadsheets import ingest_spreadsheet, spreadsheet_to_multimodal, spreadsheet_to_text_segments, compute_spreadsheet_checksum, is_valid_spreadsheet
from app.ingestion.images import extract_pdf_images, extract_pdf_charts, images_to_multimodal, charts_to_multimodal, images_to_text_segments, charts_to_text_segments
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
        """Ingest a PDF file through the full pipeline with multimodal support.

        Steps:
        1. Compute source checksum
        2. Upsert source
        3. Check if document version exists
        4. Extract text segments with provenance (OCR fallback for scanned PDFs)
        5. Extract multimodal content (tables, images, charts)
        6. Chunk with provenance preservation
        7. Store chunks
        8. Create document record

        Returns the created Document.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 1. Compute source checksum (using OCR-aware text for checksum)
        full_text = extract_pdf_text_for_checksum(pdf_path)
        source_checksum = hashlib.sha256(full_text.encode()).hexdigest()

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
                metadata={"filename": pdf_path.name, "size_bytes": len(pdf_path.read_bytes())},
            )
            source = self.store.upsert_source(source)
            logger.info("source_created", source_id=str(source.id), path=str(pdf_path))

        # 3. Check for existing document version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        if existing_doc and existing_doc.chunking_strategy == chunking_strategy:
            # Verify checksum matches
            doc_checksum = hashlib.sha256(full_text.encode()).hexdigest()
            if existing_doc.checksum == doc_checksum:
                logger.info("document_unchanged", document_id=str(existing_doc.id))
                return existing_doc

        # 4. Extract text segments with provenance (OCR fallback for scanned PDFs)
        segments = extract_pdf_segments_with_ocr(pdf_path)
        if not segments:
            raise ValueError(f"No extractable text found in {pdf_path}")

        # 5. Extract multimodal content (tables, images, charts)
        tables = extract_pdf_tables(pdf_path)
        images = extract_pdf_images(pdf_path)
        charts = extract_pdf_charts(pdf_path)

        # 6. Convert multimodal content to text segments for chunking
        table_segments = tables_to_text_segments(tables)
        image_segments = images_to_text_segments(images)
        chart_segments = charts_to_text_segments(charts)
        segments.extend(table_segments + image_segments + chart_segments)

        # 7. Chunk with provenance preservation
        # Use section-aware chunking for better context preservation
        chunks = chunk_by_sections(segments, UUID(int=0))  # placeholder doc_id

        # 8. Create document record
        doc_checksum = hashlib.sha256(full_text.encode()).hexdigest()

        # Determine next version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        next_version = (existing_doc.version + 1) if existing_doc else 1

        # Collect multimodal metadata
        multimodal_metadata = {
            "table_count": len(extract_pdf_tables(pdf_path)),
            "image_count": len(images),
            "chart_count": len(charts),
        }

        document = Document(
            source_id=source.id,
            version=next_version,
            checksum=doc_checksum,
            chunking_strategy=chunking_strategy,
            metadata={
                "source_path": str(pdf_path),
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "multimodal": multimodal_metadata,
            },
        )

        # 9. Update chunks with correct document_id and store atomically
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
            **multimodal_metadata,
        )
        return document

    def ingest_web_page(
        self,
        url: str,
        source_type: SourceType = SourceType.HTML,
        chunking_strategy: str = "semantic_v1",
    ) -> Document:
        """Ingest a web page through the full pipeline.

        Steps:
        1. Fetch and parse web page
        2. Compute source checksum
        3. Upsert source
        4. Check if document version exists
        5. Extract text segments with provenance
        6. Chunk with provenance preservation
        7. Store chunks
        8. Create document record
        """
        settings = get_settings()
        
        if not settings.multimodal_web_ingestion_enabled:
            raise RuntimeError("Web ingestion disabled via configuration")
        
        if not is_valid_web_url(url):
            raise ValueError(f"Invalid URL: {url}")

        # 1. Fetch and parse web page
        web_result = fetch_web_page(url)

        # 2. Compute source checksum
        source_checksum = compute_web_page_checksum(url, web_result.html_content)

        # 3. Upsert source
        existing_source = self.store.get_source_by_checksum(source_checksum)
        if existing_source:
            source = existing_source
            logger.info("source_exists", source_id=str(source.id), url=url)
        else:
            source = Source(
                type=source_type,
                path=web_result.canonical_url,
                checksum=source_checksum,
                metadata={
                    "url": url,
                    "canonical_url": web_result.canonical_url,
                    "title": web_result.title,
                    "retrieved_date": web_result.retrieved_date.isoformat(),
                },
            )
            source = self.store.upsert_source(source)
            logger.info("source_created", source_id=str(source.id), url=url)

        # 4. Check for existing document version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        doc_checksum = source_checksum
        if existing_doc and existing_doc.chunking_strategy == chunking_strategy:
            if existing_doc.checksum == doc_checksum:
                logger.info("document_unchanged", document_id=str(existing_doc.id))
                return existing_doc

        # 5. Extract text segments with provenance
        segments = web_page_to_text_segments(web_result)
        if not segments:
            raise ValueError(f"No extractable content found at {url}")

        # 6. Chunk with provenance preservation
        chunks = chunk_by_sections(segments, UUID(int=0))

        # 7. Create document record
        existing_doc = self.store.get_latest_document_for_source(source.id)
        next_version = (existing_doc.version + 1) if existing_doc else 1

        document = Document(
            source_id=source.id,
            version=next_version,
            checksum=doc_checksum,
            chunking_strategy=chunking_strategy,
            metadata={
                "source_path": web_result.canonical_url,
                "url": url,
                "title": web_result.title,
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "multimodal": {"web_pages": 1},
            },
        )

        # 8. Update chunks with correct document_id and store atomically
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
            "web_page_ingested",
            document_id=str(document.id),
            source_id=str(source.id),
            version=document.version,
            chunk_count=len(chunks),
            url=url,
        )
        return document

    def ingest_spreadsheet_file(
        self,
        file_path: Path,
        source_type: SourceType = SourceType.SPREADSHEET,
        chunking_strategy: str = "semantic_v1",
    ) -> Document:
        """Ingest a spreadsheet file (Excel/CSV) through the full pipeline."""
        settings = get_settings()
        
        if not settings.multimodal_spreadsheet_enabled:
            raise RuntimeError("Spreadsheet ingestion disabled via configuration")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Spreadsheet not found: {file_path}")

        if not is_valid_spreadsheet(file_path):
            raise ValueError(f"Unsupported spreadsheet format: {file_path.suffix}")

        # 1. Ingest spreadsheet
        spreadsheet_result = ingest_spreadsheet(file_path)

        # 2. Compute source checksum
        source_checksum = compute_spreadsheet_checksum(file_path)

        # 3. Upsert source
        existing_source = self.store.get_source_by_checksum(source_checksum)
        if existing_source:
            source = existing_source
            logger.info("source_exists", source_id=str(source.id), path=str(file_path))
        else:
            source = Source(
                type=source_type,
                path=str(file_path),
                checksum=source_checksum,
                metadata={
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "sheet_count": spreadsheet_result.metadata.get("sheet_count", 0),
                },
            )
            source = self.store.upsert_source(source)
            logger.info("source_created", source_id=str(source.id), path=str(file_path))

        # 4. Check for existing document version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        doc_checksum = source_checksum
        if existing_doc and existing_doc.chunking_strategy == chunking_strategy:
            if existing_doc.checksum == doc_checksum:
                logger.info("document_unchanged", document_id=str(existing_doc.id))
                return existing_doc

        # 5. Extract text segments with provenance
        segments = spreadsheet_to_text_segments(spreadsheet_result)
        if not segments:
            raise ValueError(f"No extractable content found in {file_path}")

        # 6. Chunk with provenance preservation
        chunks = chunk_by_sections(segments, UUID(int=0))

        # 7. Create document record
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
                "multimodal": {
                    "spreadsheets": 1,
                    "sheet_count": spreadsheet_result.metadata.get("sheet_count", 0),
                },
            },
        )

        # 8. Update chunks with correct document_id and store atomically
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
            "spreadsheet_ingested",
            document_id=str(document.id),
            source_id=str(source.id),
            version=document.version,
            chunk_count=len(chunks),
            path=str(file_path),
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

    Supports: .pdf, .txt, .md, .xlsx, .xls, .xlsm, .csv
    """
    settings = get_settings()
    store = store or get_evidence_store()
    pipeline = IngestionPipeline(store)

    supported_extensions = {".pdf", ".txt", ".md"}
    if settings.multimodal_spreadsheet_enabled:
        supported_extensions.update({".xlsx", ".xls", ".xlsm", ".csv"})
    
    documents = []

    for file_path in corpus_dir.rglob("*"):
        if not file_path.is_file():
            continue
        
        suffix = file_path.suffix.lower()
        if suffix not in supported_extensions:
            continue
            
        try:
            if suffix == ".pdf":
                doc = pipeline.ingest_pdf(file_path)
            elif suffix in (".xlsx", ".xls", ".xlsm", ".csv"):
                if not settings.multimodal_spreadsheet_enabled:
                    continue
                doc = pipeline.ingest_spreadsheet_file(file_path)
            else:
                doc = pipeline.ingest_text_file(file_path)
            documents.append(doc)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("ingestion_failed", path=str(file_path), error=str(e))
            # Continue with other files

    logger.info("corpus_ingested", directory=str(corpus_dir), document_count=len(documents))
    return documents