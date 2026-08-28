"""Ingestion pipeline exports (Phase 01 + Phase 11)."""

from app.ingestion.chunking import TextSegment, chunk_by_sections, chunk_text
from app.ingestion.images import (
    ExtractedChart,
    ExtractedImage,
    charts_to_text_segments,
    extract_pdf_charts,
    extract_pdf_images,
    images_to_text_segments,
)
from app.ingestion.ocr import (
    OCRResult,
    extract_pdf_segments_with_ocr,
    extract_pdf_text_layer,
    extract_pdf_with_ocr_fallback,
    has_usable_text_layer,
)
from app.ingestion.pdf import ExtractedPage, extract_pdf_segments, extract_pdf_text
from app.ingestion.pipeline import IngestionPipeline, ingest_corpus_directory
from app.ingestion.spreadsheets import (
    SpreadsheetResult,
    ingest_spreadsheet,
    is_valid_spreadsheet,
    spreadsheet_to_text_segments,
)
from app.ingestion.tables import (
    ExtractedTable,
    extract_pdf_tables,
    tables_to_text_segments,
)
from app.ingestion.web import (
    WebPageResult,
    fetch_web_page,
    is_valid_web_url,
    web_page_to_text_segments,
)

__all__ = [
    # Phase 01
    "ExtractedPage",
    "IngestionPipeline",
    "TextSegment",
    "chunk_by_sections",
    "chunk_text",
    "extract_pdf_segments",
    "extract_pdf_text",
    "ingest_corpus_directory",
    # Phase 11.1 — OCR
    "OCRResult",
    "extract_pdf_segments_with_ocr",
    "extract_pdf_with_ocr_fallback",
    "extract_pdf_text_layer",
    "has_usable_text_layer",
    # Phase 11.2 — Tables
    "ExtractedTable",
    "extract_pdf_tables",
    "tables_to_text_segments",
    # Phase 11.3 — Web
    "WebPageResult",
    "fetch_web_page",
    "is_valid_web_url",
    "web_page_to_text_segments",
    # Phase 11.4 — Spreadsheets
    "SpreadsheetResult",
    "ingest_spreadsheet",
    "is_valid_spreadsheet",
    "spreadsheet_to_text_segments",
    # Phase 11.5 — Charts/Images
    "ExtractedChart",
    "ExtractedImage",
    "charts_to_text_segments",
    "extract_pdf_charts",
    "extract_pdf_images",
    "images_to_text_segments",
]