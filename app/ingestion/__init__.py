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
    "ExtractedChart",
    "ExtractedImage",
    "ExtractedPage",
    "ExtractedTable",
    "IngestionPipeline",
    "OCRResult",
    "SpreadsheetResult",
    "TextSegment",
    "WebPageResult",
    "charts_to_text_segments",
    "chunk_by_sections",
    "chunk_text",
    "extract_pdf_charts",
    "extract_pdf_images",
    "extract_pdf_segments",
    "extract_pdf_segments_with_ocr",
    "extract_pdf_tables",
    "extract_pdf_text",
    "extract_pdf_text_layer",
    "extract_pdf_with_ocr_fallback",
    "fetch_web_page",
    "has_usable_text_layer",
    "images_to_text_segments",
    "ingest_corpus_directory",
    "ingest_spreadsheet",
    "is_valid_spreadsheet",
    "is_valid_web_url",
    "spreadsheet_to_text_segments",
    "tables_to_text_segments",
    "web_page_to_text_segments",
]