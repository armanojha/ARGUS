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


def _detect_headings_from_chars(page, text: str) -> tuple[list[tuple[str, int]], list[int]]:
    """Detect headings using pdfplumber page.chars font size analysis.

    Groups characters into lines by y-position, computes median font size,
    and identifies lines with significantly larger font as headings.

    Returns (sections, heading_levels) where sections is list of
    (heading_text, char_offset) and heading_levels is list of ints (1=h1, 2=h2, 3=h3).
    """
    chars = page.chars
    if not chars:
        # Fallback to text-based heuristic if no char data available
        fallback = _fallback_heading_detection(text)
        return fallback, [2] * len(fallback)

    # Group chars into lines by approximate y-position (top coordinate)
    # Use a tolerance of half the median character height
    char_heights = [c.get("top", 0) - c.get("bottom", 0) for c in chars if c.get("text", "").strip()]
    if not char_heights:
        fallback = _fallback_heading_detection(text)
        return fallback, [2] * len(fallback)

    abs_heights = [abs(h) for h in char_heights if h != 0]
    if not abs_heights:
        fallback = _fallback_heading_detection(text)
        return fallback, [2] * len(fallback)

    median_char_height = sorted(abs_heights)[len(abs_heights) // 2]
    tolerance = max(median_char_height * 0.5, 2.0)

    # Build line groups: sort chars by top position, then group by y-proximity
    text_chars = [c for c in chars if c.get("text", "").strip()]
    text_chars.sort(key=lambda c: (c.get("top", 0), c.get("x0", 0)))

    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_top = None

    for char in text_chars:
        char_top = char.get("top", 0)
        if current_top is None or abs(char_top - current_top) <= tolerance:
            current_line.append(char)
            current_top = char_top if current_top is None else current_top
        else:
            if current_line:
                lines.append(current_line)
            current_line = [char]
            current_top = char_top
    if current_line:
        lines.append(current_line)

    if not lines:
        fallback = _fallback_heading_detection(text)
        return fallback, [2] * len(fallback)

    # Compute font size per line (median of character font sizes)
    line_font_sizes: list[float] = []
    for line_chars in lines:
        sizes = [c.get("size", 12) for c in line_chars if c.get("size")]
        if sizes:
            sizes.sort()
            line_font_sizes.append(sizes[len(sizes) // 2])
        else:
            line_font_sizes.append(12.0)

    if not line_font_sizes:
        fallback = _fallback_heading_detection(text)
        return fallback, [2] * len(fallback)

    # Overall median font size (body text)
    sorted_sizes = sorted(line_font_sizes)
    median_font_size = sorted_sizes[len(sorted_sizes) // 2]

    # Identify headings: lines with font size > 1.2x median
    sections = []
    heading_levels = []
    char_offset = 0

    for i, line_chars in enumerate(lines):
        line_text = "".join(c.get("text", "") for c in line_chars).strip()
        font_size = line_font_sizes[i] if i < len(line_font_sizes) else 12.0

        if line_text and font_size > median_font_size * 1.2 and len(line_text) < 120:
            # Determine heading level based on font size ratio
            ratio = font_size / median_font_size
            if ratio >= 1.8:
                level = 1
            elif ratio >= 1.4:
                level = 2
            else:
                level = 3

            # Find the offset of this text in the full text
            offset = text.find(line_text, char_offset)
            if offset >= 0:
                sections.append((line_text, offset))
                heading_levels.append(level)
                char_offset = offset + len(line_text)
            else:
                sections.append((line_text, char_offset))
                heading_levels.append(level)

    return sections, heading_levels


def _fallback_heading_detection(text: str) -> list[tuple[str, int]]:
    """Fallback text-based heading detection when font info is unavailable."""
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
    return sections


@dataclass
class ExtractedPage:
    """A page of extracted content with structure."""
    page_number: int  # 1-based
    text: str
    sections: list[tuple[str, int]]  # (section_title, char_offset)
    tables: list[list[list[str]]]  # List of tables, each table is list of rows, each row is list of cells
    heading_levels: list[int] | None = None  # heading level per section (1=h1, 2=h2, 3=h3)


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    rows: list[list[str]]
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom)


def extract_pdf_pages(pdf_path: Path) -> Iterator[ExtractedPage]:
    """Extract text from PDF pages with structure detection.

    Uses font size analysis from page.chars to detect headings.
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

            # Font-size-based heading detection using page.chars
            sections, heading_levels = _detect_headings_from_chars(page, text)

            yield ExtractedPage(
                page_number=page_num,
                text=text,
                sections=sections,
                tables=cleaned_tables,
                heading_levels=heading_levels,
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


def _build_hierarchical_path(heading_stack: list[tuple[str, int]], level: int) -> str:
    """Build a hierarchical section path from the heading stack.

    Trims the stack to only include headings at level < current level,
    then joins with " > " separator.

    Example: heading_stack = [("Chapter 1", 1), ("Results", 2), ("Data", 3)]
             level = 2 -> "Chapter 1 > Results"
    """
    # Keep only headings that are at a higher level (smaller number) than current
    while heading_stack and heading_stack[-1][1] >= level:
        heading_stack.pop()

    parts = [title for title, _ in heading_stack]
    return " > ".join(parts) if parts else None


def extract_pdf_segments(pdf_path: Path) -> list[TextSegment]:
    """Extract text segments from PDF with page/section provenance.

    Returns a list of TextSegment objects suitable for chunking.
    Builds hierarchical section_path from heading levels.
    """
    segments = []
    # Heading stack: list of (title, level) tracking active hierarchy
    heading_stack: list[tuple[str, int]] = []

    for page in extract_pdf_pages(pdf_path):
        if not page.text.strip() and not page.tables:
            continue

        # If we have sections, split by them
        if page.sections:
            heading_levels = page.heading_levels or [2] * len(page.sections)

            for i, (section_title, offset) in enumerate(page.sections):
                level = heading_levels[i] if i < len(heading_levels) else 2

                # Build hierarchical path
                section_path = _build_hierarchical_path(heading_stack, level)
                if section_path:
                    section_path = f"{section_path} > {section_title}"
                else:
                    section_path = section_title

                # Push this heading onto the stack
                heading_stack.append((section_title, level))

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
                        section_path=section_path,
                    ))
        else:
            # No sections detected, treat whole page as one segment
            if page.text.strip():
                # Use accumulated heading stack for context
                section_path = " > ".join(t for t, _ in heading_stack) if heading_stack else None
                segments.append(TextSegment(
                    text=page.text.strip(),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    char_start=0,
                    char_end=len(page.text),
                    section_path=section_path,
                ))

        # Add tables as separate segments with special section_path
        for table_idx, table in enumerate(page.tables):
            # Convert table to markdown-like text representation
            table_text = _table_to_text(table)
            if table_text.strip():
                # Tables get their parent section as context
                table_section = " > ".join(t for t, _ in heading_stack) if heading_stack else None
                if table_section:
                    table_section = f"{table_section} > Table {table_idx + 1}"
                else:
                    table_section = f"Table {table_idx + 1}"
                segments.append(TextSegment(
                    text=table_text,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    char_start=None,
                    char_end=None,
                    section_path=table_section,
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