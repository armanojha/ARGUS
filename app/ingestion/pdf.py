"""PDF text extraction with structure preservation (Phase 01).

Extracts text, headings, tables, and page anchors from PDFs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pypdf

from app.ingestion.chunking import TextSegment
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.pdf")


@dataclass
class ExtractedPage:
    """A page of extracted content with structure."""
    page_number: int  # 1-based
    text: str
    sections: list[tuple[str, int]]  # (section_title, char_offset)
    tables: list[list[list[str]]]  # List of tables, each table is list of rows, each row is list of cells


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    rows: list[list[str]]
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom)


def extract_pdf_pages(pdf_path: Path) -> Iterator[ExtractedPage]:
    """Extract text from PDF pages with basic structure detection.

    Yields ExtractedPage objects with text, section hints, and tables.
    """
    # Use pdfplumber for text + table extraction
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""

            # Extract tables
            tables = page.extract_tables() or []
            # Clean up table cells
            cleaned_tables = []
            for table in tables:
                cleaned_table = []
                for row in table:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    cleaned_table.append(cleaned_row)
                cleaned_tables.append(cleaned_table)

            # Simple section detection: lines that look like headings
            sections = []
            lines = text.split("\n")
            char_offset = 0
            for line in lines:
                stripped = line.strip()
                if stripped and (
                    stripped.isupper() or
                    stripped.endswith(":") or
                    (len(stripped) < 80 and stripped[0].isupper() and stripped.count(" ") < 5)
                ):
                    sections.append((stripped, char_offset))
                char_offset += len(line) + 1

            yield ExtractedPage(
                page_number=page_num,
                text=text,
                sections=sections,
                tables=cleaned_tables,
            )


def extract_pdf_tables(pdf_path: Path) -> list[ExtractedTable]:
    """Extract all tables from a PDF with page and position info.

    Returns a list of ExtractedTable objects.
    """
    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables() or []
            for table_idx, table in enumerate(page_tables):
                cleaned_table = []
                for row in table:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    cleaned_table.append(cleaned_row)
                # Get table bbox if available
                bbox = None
                try:
                    table_objects = page.find_tables()
                    if table_idx < len(table_objects):
                        t = table_objects[table_idx]
                        bbox = (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
                except (AttributeError, IndexError, TypeError):
                    # find_tables() may not be available or return unexpected format
                    pass

                tables.append(ExtractedTable(
                    page_number=page_num,
                    table_index=table_idx,
                    rows=cleaned_table,
                    bbox=bbox,
                ))

    logger.info("extracted_pdf_tables", path=str(pdf_path), table_count=len(tables))
    return tables


def extract_pdf_segments(pdf_path: Path) -> list[TextSegment]:
    """Extract text segments from PDF with page/section provenance.

    Returns a list of TextSegment objects suitable for chunking.
    """
    segments = []

    for page in extract_pdf_pages(pdf_path):
        if not page.text.strip() and not page.tables:
            continue

        # If we have sections, split by them
        if page.sections:
            for i, (section_title, offset) in enumerate(page.sections):
                start = offset
                end = page.sections[i + 1][1] if i + 1 < len(page.sections) else len(page.text)
                segment_text = page.text[start:end].strip()
                if segment_text:
                    segments.append(TextSegment(
                        text=segment_text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        char_start=start,
                        char_end=end,
                        section_path=section_title,
                    ))
        else:
            # No sections detected, treat whole page as one segment
            if page.text.strip():
                segments.append(TextSegment(
                    text=page.text.strip(),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    char_start=0,
                    char_end=len(page.text),
                    section_path=None,
                ))

        # Add tables as separate segments with special section_path
        for table_idx, table in enumerate(page.tables):
            # Convert table to markdown-like text representation
            table_text = _table_to_text(table)
            if table_text.strip():
                segments.append(TextSegment(
                    text=table_text,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    char_start=None,
                    char_end=None,
                    section_path=f"Table {table_idx + 1}",
                ))

    logger.info("extracted_pdf_segments", path=str(pdf_path), segment_count=len(segments))
    return segments


def _table_to_text(table: list[list[str]]) -> str:
    """Convert a table (list of rows) to a text representation."""
    if not table:
        return ""

    # Find max columns
    max_cols = max(len(row) for row in table)

    # Pad rows to same length
    padded_table = [row + [""] * (max_cols - len(row)) for row in table]

    # Create markdown-style table
    lines = []
    for i, row in enumerate(padded_table):
        lines.append(" | ".join(row))
        if i == 0:
            # Header separator
            lines.append(" | ".join(["---"] * max_cols))

    return "\n".join(lines)


def extract_pdf_text(pdf_path: Path) -> str:
    """Simple full-text extraction for checksumming."""
    reader = pypdf.PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n".join(texts)