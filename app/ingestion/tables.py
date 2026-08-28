"""Table Extraction with Row/Column Semantics (Phase 11.2).

Extracts tables from PDFs preserving row/column structure and page provenance.
Extends the basic table extraction in pdf.py with richer semantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import pdfplumber

from app.config import get_settings
from app.ingestion.multimodal import Table, TableCell, MultimodalType
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.tables")


@dataclass(frozen=True)
class ExtractedTable:
    """A table extracted from a PDF with full provenance."""
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    cells: list[TableCell]
    caption: str | None = None
    num_rows: int = 0
    num_cols: int = 0
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom)


def _extract_table_cells(pdf_table: list[list[str]], page_num: int, table_idx: int) -> list[TableCell]:
    """Convert raw pdfplumber table to structured TableCell objects with row/col semantics."""
    cells = []
    
    if not pdf_table:
        return cells
    
    # First row as headers if it looks like a header row
    headers = pdf_table[0] if pdf_table else []
    
    for row_idx, row in enumerate(pdf_table):
        for col_idx, cell_value in enumerate(row):
            cell = TableCell(
                row=row_idx,
                col=col_idx,
                value=cell_value.strip() if cell_value else "",
                row_span=1,
                col_span=1,
                is_header=(row_idx == 0 and _looks_like_header(row)),
            )
            cells.append(cell)
    
    return cells


def _looks_like_header(row: list[str]) -> bool:
    """Heuristic to detect if a row is a header row."""
    if not row:
        return False
    
    # Check if all cells are non-empty and relatively short (typical headers)
    non_empty = [c for c in row if c and c.strip()]
    if len(non_empty) < len(row) * 0.5:
        return False
    
    # Headers often have specific formatting
    for cell in non_empty:
        if len(cell.strip()) > 50:
            return False
    
    return True


def extract_pdf_tables(pdf_path: Path) -> list[ExtractedTable]:
    """Extract all tables from a PDF with full row/column semantics and provenance.
    
    Returns list of ExtractedTable objects with structured cells, headers, rows,
    and page/bbox provenance.
    """
    settings = get_settings()
    
    if not settings.multimodal_table_extraction_enabled:
        logger.info("table_extraction_disabled_via_config")
        return []
    
    tables = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables() or []
            
            for table_idx, raw_table in enumerate(page_tables):
                if not raw_table:
                    continue
                
                # Clean table cells
                cleaned_rows = []
                for row in raw_table:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    cleaned_rows.append(cleaned_row)
                
                if not cleaned_rows:
                    continue
                
                # Detect headers and data rows
                headers = cleaned_rows[0] if _looks_like_header(cleaned_rows[0]) else []
                data_rows = cleaned_rows[1:] if headers else cleaned_rows
                
                # Extract cells with semantics
                cells = _extract_table_cells(cleaned_rows, page_num, table_idx)
                
                # Get bbox if available
                bbox = None
                try:
                    table_objects = page.find_tables()
                    if table_idx < len(table_objects):
                        t = table_objects[table_idx]
                        bbox = (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
                except (AttributeError, IndexError, TypeError):
                    pass
                
                # Try to find caption (text near the table)
                caption = _find_table_caption(page, bbox) if bbox else None
                
                table = ExtractedTable(
                    page_number=page_num,
                    table_index=table_idx,
                    headers=headers,
                    rows=data_rows,
                    cells=cells,
                    caption=caption,
                    num_rows=len(cleaned_rows),
                    num_cols=max(len(r) for r in cleaned_rows) if cleaned_rows else 0,
                    bbox=bbox,
                )
                tables.append(table)
    
    logger.info("extracted_pdf_tables", path=str(pdf_path), table_count=len(tables))
    return tables


def _find_table_caption(page, bbox: tuple[float, float, float, float] | None) -> str | None:
    """Attempt to find a caption for the table by looking at nearby text."""
    if not bbox:
        return None
    
    try:
        # Look for text elements near the table (above or below)
        x0, top, x1, bottom = bbox
        page_height = page.height
        
        # Search in regions above and below the table
        search_regions = [
            (x0, max(0, top - 100), x1, top),  # Above
            (x0, bottom, x1, min(page_height, bottom + 100)),  # Below
        ]
        
        for region in search_regions:
            texts = page.within_bbox(region).extract_text()
            if texts:
                lines = [l.strip() for l in texts.split("\n") if l.strip()]
                for line in lines:
                    # Check if line looks like a caption
                    if any(keyword in line.lower() for keyword in ["table", "figure", "chart"]):
                        if len(line) < 200:
                            return line
    except Exception:
        pass
    
    return None


def tables_to_multimodal(
    tables: list[ExtractedTable],
    document_id: UUID,
    source_chunk_ids: list[UUID] | None = None,
) -> list[Table]:
    """Convert ExtractedTable objects to Multimodal Table objects for storage."""
    multimodal_tables = []
    
    for table in tables:
        multimodal = Table(
            id=UUID(int=0),  # Will be assigned by store
            source_path="",  # Will be set by pipeline
            content_type=MultimodalType.TABLE,
            source_chunk_ids=source_chunk_ids or [],
            page_number=table.page_number,
            page_range=(table.page_number, table.page_number),
            headers=table.headers,
            rows=table.rows,
            cells=table.cells,
            caption=table.caption,
            num_rows=table.num_rows,
            num_cols=table.num_cols,
            metadata={
                "table_index": table.table_index,
                "bbox": table.bbox,
            } if table.bbox else {"table_index": table.table_index},
        )
        multimodal_tables.append(multimodal)
    
    return multimodal_tables


def tables_to_text_segments(tables: list[ExtractedTable]) -> list:
    """Convert tables to text segments for chunking pipeline.
    
    Creates a markdown-style representation that preserves structure.
    """
    from app.ingestion.chunking import TextSegment
    
    segments = []
    
    for table in tables:
        if not table.rows and not table.headers:
            continue
        
        # Build markdown-style table
        lines = []
        
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append(" | ".join(["---"] * len(table.headers)))
        
        for row in table.rows:
            # Pad row to match header count
            padded_row = row + [""] * (len(table.headers) - len(row)) if table.headers else row
            lines.append(" | ".join(padded_row))
        
        table_text = "\n".join(lines)
        
        if table_text.strip():
            segments.append(TextSegment(
                text=table_text,
                page_start=table.page_number,
                page_end=table.page_number,
                char_start=None,
                char_end=None,
                section_path=f"Table {table.table_index + 1}",
            ))
    
    return segments


__all__ = [
    "ExtractedTable",
    "extract_pdf_tables",
    "tables_to_multimodal",
    "tables_to_text_segments",
]