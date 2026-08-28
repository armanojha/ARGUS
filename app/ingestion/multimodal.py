"""Multimodal Content Representation (Phase 11).

Defines the data models for multimodal content ingestion:
- OCR text
- Tables (with row/column semantics)
- Web pages
- Spreadsheets
- Charts/images

Phase 11 implements the ingestion pipelines. The evidence store
and retrieval components depend on these models for provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# =============================================================================
# Multimodal Content Types
# =============================================================================

class MultimodalType(str, Enum):
    """Types of multimodal content."""
    OCR_TEXT = "ocr_text"
    TABLE = "table"
    WEB_PAGE = "web_page"
    SPREADSHEET = "spreadsheet"
    CHART = "chart"
    IMAGE = "image"


class IngestionPriority(str, Enum):
    """Priority for ingestion (matches V2/V3 §8 priority column)."""
    HIGH = "high"
    MEDIUM = "medium"
    LATER = "later"


# =============================================================================
# Base Multimodal Content
# =============================================================================

@dataclass(frozen=True)
class MultimodalContent:
    """Base class for all multimodal content."""
    id: UUID
    source_path: str
    content_type: MultimodalType
    source_uri: str | None = None
    source_chunk_ids: list[UUID] = field(default_factory=list)
    text_content: str | None = None
    page_number: int | None = None
    page_range: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# OCR Text (Phase 11.1)
# =============================================================================

@dataclass(frozen=True)
class OCRText(MultimodalContent):
    content_type: MultimodalType = MultimodalType.OCR_TEXT
    language: str | None = None
    confidence: float | None = None
    ocr_used: bool = True


# =============================================================================
# Tables (Phase 11.2)
# =============================================================================

@dataclass(frozen=True)
class TableCell:
    row: int
    col: int
    value: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Table(MultimodalContent):
    content_type: MultimodalType = MultimodalType.TABLE
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    cells: list[Any] = field(default_factory=list)
    caption: str | None = None
    num_rows: int = 0
    num_cols: int = 0


# =============================================================================
# Web Pages (Phase 11.3)
# =============================================================================

@dataclass(frozen=True)
class WebPage(MultimodalContent):
    url: str = ""
    content_type: MultimodalType = MultimodalType.WEB_PAGE
    canonical_url: str | None = None
    title: str | None = None
    author: str | None = None
    published_date: datetime | None = None
    retrieved_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    html_content: str | None = None
    text_content: str | None = None
    description: str | None = None
    keywords: list[str] = field(default_factory=list)


# =============================================================================
# Spreadsheets (Phase 11.4)
# =============================================================================

@dataclass(frozen=True)
class SpreadsheetCell:
    sheet: str
    row: int
    col: int
    value: Any
    formula: str | None = None
    data_type: str | None = None


@dataclass(frozen=True)
class SpreadsheetSheet:
    name: str
    rows: int
    cols: int
    cells: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Spreadsheet(MultimodalContent):
    content_type: MultimodalType = MultimodalType.SPREADSHEET
    sheets: list[Any] = field(default_factory=list)
    author: str | None = None
    created_date: datetime | None = None
    modified_date: datetime | None = None


# =============================================================================
# Charts/Images (Phase 11.5)
# =============================================================================

class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    OTHER = "other"


@dataclass(frozen=True)
class ChartDataPoint:
    x: Any
    y: Any
    series: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class Chart(MultimodalContent):
    content_type: MultimodalType = MultimodalType.CHART
    chart_type: ChartType = ChartType.OTHER
    title: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    data_points: list[Any] = field(default_factory=list)
    series_names: list[str] = field(default_factory=list)
    model_description: str | None = None


@dataclass(frozen=True)
class Image(MultimodalContent):
    content_type: MultimodalType = MultimodalType.IMAGE
    width: int | None = None
    height: int | None = None
    format: str | None = None
    model_description: str | None = None
    ocr_text: str | None = None


# =============================================================================
# Multimodal Evidence Reference
# =============================================================================

class MultimodalEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    source_path: str
    source_type: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    content_type: str | None = None
    table_headers: list[str] | None = None
    table_rows: list[list[str]] | None = None
    web_url: str | None = None
    web_title: str | None = None
    chart_type: str | None = None
    chart_data_points: list[dict[str, Any]] | None = None
    image_description: str | None = None
    ocr_confidence: float | None = None


@dataclass(frozen=True)
class MultimodalIngestionResult:
    content_id: UUID
    content_type: MultimodalType
    source_path: str
    text_content: str | None = None
    tables: list[Any] = field(default_factory=list)
    web_pages: list[Any] = field(default_factory=list)
    spreadsheets: list[Any] = field(default_factory=list)
    charts: list[Any] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    ocr_text: list[Any] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MultimodalIngestionInterface(ABC):
    @property
    @abstractmethod
    def content_type(self) -> MultimodalType:
        ...

    @property
    @abstractmethod
    def priority(self) -> str:
        ...

    @abstractmethod
    async def can_ingest(self, file_path: Path) -> bool:
        ...

    @abstractmethod
    async def ingest(
        self,
        file_path: Path,
        document_id: UUID,
    ) -> Any:
        ...


class MultimodalIngestionFactoryInterface(ABC):
    @abstractmethod
    def create_ingester(self, content_type: MultimodalType) -> Any | None:
        ...

    @abstractmethod
    def get_all_ingesters(self) -> list[Any]:
        ...


class DefaultMultimodalFactory:
    def create_ingester(self, content_type: MultimodalType) -> None:
        return None

    def get_all_ingesters(self) -> list:
        return []


_multimodal_factory: Any = None


def get_multimodal_factory() -> Any:
    global _multimodal_factory
    if _multimodal_factory is None:
        _multimodal_factory = DefaultMultimodalFactory()
    return _multimodal_factory


def set_multimodal_factory(factory: Any) -> None:
    global _multimodal_factory
    _multimodal_factory = factory


__all__ = [
    "Chart",
    "ChartDataPoint",
    "ChartType",
    "DefaultMultimodalFactory",
    "Image",
    "IngestionPriority",
    "MultimodalContent",
    "MultimodalEvidenceRef",
    "MultimodalIngestionFactoryInterface",
    "MultimodalIngestionInterface",
    "MultimodalIngestionResult",
    "MultimodalType",
    "OCRText",
    "Spreadsheet",
    "SpreadsheetCell",
    "SpreadsheetSheet",
    "Table",
    "TableCell",
    "WebPage",
    "get_multimodal_factory",
    "set_multimodal_factory",
]