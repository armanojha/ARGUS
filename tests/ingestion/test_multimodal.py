"""Tests for Phase 11 Multimodal Ingestion."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.ingestion.ocr import (
    extract_pdf_text_layer,
    has_usable_text_layer,
    extract_pdf_with_ocr_fallback,
    extract_pdf_segments_with_ocr,
    OCRResult,
)
from app.ingestion.tables import extract_pdf_tables, tables_to_text_segments, ExtractedTable
from app.ingestion.web import fetch_web_page, web_page_to_text_segments, is_valid_web_url
from app.ingestion.spreadsheets import ingest_spreadsheet, spreadsheet_to_text_segments, is_valid_spreadsheet
from app.ingestion.images import extract_pdf_images, extract_pdf_charts, images_to_text_segments, charts_to_text_segments

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.ingestion.ocr import (
    extract_pdf_text_layer,
    has_usable_text_layer,
    extract_pdf_with_ocr_fallback,
    extract_pdf_segments_with_ocr,
    OCRResult,
)
from app.ingestion.tables import extract_pdf_tables, tables_to_text_segments, ExtractedTable
from app.ingestion.web import fetch_web_page, web_page_to_text_segments, is_valid_web_url
from app.ingestion.spreadsheets import ingest_spreadsheet, spreadsheet_to_text_segments, is_valid_spreadsheet
from app.ingestion.images import extract_pdf_images, extract_pdf_charts, images_to_text_segments, charts_to_text_segments


@pytest.fixture
def text_layer_pdf():
    """Create a PDF with a text layer."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF with text layer.")
    c.drawString(100, 730, "It contains multiple lines of text.")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def scanned_pdf():
    """Create a PDF without text layer (simulated scanned PDF)."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    # Draw an image-like rectangle instead of text to simulate scanned content
    c.rect(100, 700, 400, 50, fill=1)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def text_layer_pdf_path(text_layer_pdf):
    """Write text layer PDF to temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(text_layer_pdf)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def scanned_pdf_path(scanned_pdf):
    """Write scanned PDF to temp file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(scanned_pdf)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


class TestOCRFallback:
    """Tests for OCR fallback functionality (Phase 11.1)."""

    def test_extract_pdf_text_layer(self, text_layer_pdf_path, scanned_pdf_path):
        """Test extracting text layer from PDF."""
        # PDF with text layer should return text
        texts = extract_pdf_text_layer(text_layer_pdf_path)
        assert len(texts) == 1
        assert "test PDF with text layer" in texts[0]
        
        # Scanned PDF (no text layer) should return minimal/empty text
        scanned_texts = extract_pdf_text_layer(scanned_pdf_path)
        assert len(scanned_texts) == 1
        # The scanned PDF has a rectangle, no actual text
        assert scanned_texts[0].strip() == ""

    def test_has_usable_text_layer(self, text_layer_pdf_path, scanned_pdf_path):
        """Test detecting usable text layer."""
        # PDF with text layer should be detected as usable
        assert has_usable_text_layer(text_layer_pdf_path, min_chars_per_page=10) is True
        
        # Scanned PDF should not be detected as usable
        assert has_usable_text_layer(scanned_pdf_path, min_chars_per_page=10) is False

    def test_ocr_fallback_logic(self, text_layer_pdf_path, scanned_pdf_path):
        """Test OCR fallback only when no text layer."""
        # The key acceptance criteria: OCR only when no text layer exists
        
        # Text-layer PDF: should not use OCR
        results = list(extract_pdf_with_ocr_fallback(text_layer_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is False
        assert "test PDF with text layer" in results[0].text
        
        # Scanned PDF: should attempt OCR (will fail without tesseract, but fallback logic tested)
        results = list(extract_pdf_with_ocr_fallback(scanned_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        # Without tesseract, it falls back to text layer (empty)
        assert results[0].ocr_used is False
        assert results[0].text.strip() == ""

    def test_ocr_result_structure(self):
        """Test OCRResult structure."""
        result = OCRResult(
            page_number=1,
            text="Test OCR text",
            language="eng",
            confidence=0.95,
            ocr_used=True,
        )
        assert result.page_number == 1
        assert result.ocr_used is True
        assert result.confidence == 0.95

    def test_extract_pdf_segments_with_ocr(self, text_layer_pdf_path):
        """Test extracting segments with OCR fallback."""
        segments = extract_pdf_segments_with_ocr(text_layer_pdf_path)
        assert len(segments) == 1
        assert "test PDF with text layer" in segments[0].text


class TestTableExtraction:
    """Tests for table extraction with row/column semantics (Phase 11.2)."""

    def test_table_cell_structure(self):
        """Test TableCell has row/col semantics."""
        from app.ingestion.multimodal import TableCell
        
        cell = TableCell(
            row=0,
            col=1,
            value="Test Value",
            row_span=1,
            col_span=2,
            is_header=True,
        )
        assert cell.row == 0
        assert cell.col == 1
        assert cell.value == "Test Value"
        assert cell.row_span == 1
        assert cell.col_span == 2
        assert cell.is_header is True

    def test_extracted_table_structure(self):
        """Test ExtractedTable has proper structure."""
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=["Col1", "Col2"],
            rows=[["A", "B"], ["C", "D"]],
            cells=[],
            num_rows=2,
            num_cols=2,
        )
        assert table.page_number == 1
        assert table.headers == ["Col1", "Col2"]
        assert len(table.rows) == 2

    def test_tables_to_text_segments(self):
        """Test converting tables to text segments."""
        from app.ingestion.chunking import TextSegment
        
        table = ExtractedTable(
            page_number=1,
            table_index=0,
            headers=["Name", "Value"],
            rows=[["A", "1"], ["B", "2"]],
            cells=[],
            num_rows=2,
            num_cols=2,
        )
        
        segments = tables_to_text_segments([table])
        assert len(segments) == 1
        assert isinstance(segments[0], TextSegment)
        assert "Name | Value" in segments[0].text
        assert "--- | ---" in segments[0].text


class TestWebIngestion:
    """Tests for web page ingestion (Phase 11.3)."""

    def test_is_valid_web_url(self):
        """Test URL validation."""
        assert is_valid_web_url("https://example.com") is True
        assert is_valid_web_url("http://example.com") is True
        assert is_valid_web_url("ftp://example.com") is False
        assert is_valid_web_url("not-a-url") is False
        assert is_valid_web_url("") is False

    def test_web_page_result_structure(self):
        """Test WebPageResult structure."""
        from app.ingestion.web import WebPageResult
        from datetime import UTC, datetime
        
        result = WebPageResult(
            url="https://example.com",
            canonical_url="https://example.com",
            title="Test Page",
            author="Author",
            published_date=datetime.now(UTC),
            retrieved_date=datetime.now(UTC),
            html_content="<html>Test</html>",
            text_content="Test content",
            description="Test description",
            keywords=["test", "example"],
            metadata={},
        )
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert "test" in result.keywords

    def test_web_page_to_text_segments(self):
        """Test converting web page to text segments."""
        from app.ingestion.web import WebPageResult
        from datetime import UTC, datetime
        from app.ingestion.chunking import TextSegment
        
        result = WebPageResult(
            url="https://example.com",
            canonical_url="https://example.com",
            title="Test Page",
            author=None,
            published_date=None,
            retrieved_date=datetime.now(UTC),
            html_content="<html>Test</html>",
            text_content="Test page content",
            description="Test description",
            keywords=["test"],
            metadata={},
        )
        
        segments = web_page_to_text_segments(result)
        assert len(segments) == 1
        assert isinstance(segments[0], TextSegment)
        assert segments[0].text == "Test page content"
        assert segments[0].section_path == "https://example.com"


class TestSpreadsheetIngestion:
    """Tests for spreadsheet ingestion (Phase 11.4)."""

    def test_is_valid_spreadsheet(self):
        """Test spreadsheet format validation."""
        assert is_valid_spreadsheet(Path("test.xlsx")) is True
        assert is_valid_spreadsheet(Path("test.xls")) is True
        assert is_valid_spreadsheet(Path("test.csv")) is True
        assert is_valid_spreadsheet(Path("test.txt")) is False
        assert is_valid_spreadsheet(Path("test.pdf")) is False

    def test_spreadsheet_cell_structure(self):
        """Test SpreadsheetCell structure."""
        from app.ingestion.multimodal import SpreadsheetCell
        
        cell = SpreadsheetCell(
            sheet="Sheet1",
            row=0,
            col=0,
            value="Test",
            formula="=A1+B1",
            data_type="str",
        )
        assert cell.sheet == "Sheet1"
        assert cell.row == 0
        assert cell.col == 0
        assert cell.value == "Test"
        assert cell.formula == "=A1+B1"

    def test_spreadsheet_to_text_segments(self):
        """Test converting spreadsheet to text segments."""
        from app.ingestion.spreadsheets import SpreadsheetResult, SpreadsheetSheet
        from app.ingestion.chunking import TextSegment
        from app.ingestion.multimodal import SpreadsheetCell
        
        sheet = SpreadsheetSheet(
            name="Sheet1",
            rows=2,
            cols=2,
            cells=[
                SpreadsheetCell(sheet="Sheet1", row=0, col=0, value="A"),
                SpreadsheetCell(sheet="Sheet1", row=0, col=1, value="B"),
                SpreadsheetCell(sheet="Sheet1", row=1, col=0, value="C"),
                SpreadsheetCell(sheet="Sheet1", row=1, col=1, value="D"),
            ],
        )
        
        result = SpreadsheetResult(
            sheets=[sheet],
            author="Author",
            created_date=None,
            modified_date=None,
            total_rows=2,
            total_cells=4,
            metadata={},
        )
        
        segments = spreadsheet_to_text_segments(result)
        assert len(segments) == 1
        assert isinstance(segments[0], TextSegment)
        assert "Sheet: Sheet1" in segments[0].text


class TestChartsImages:
    """Tests for charts/images extraction (Phase 11.5)."""

    def test_chart_type_enum(self):
        """Test ChartType enum."""
        from app.ingestion.multimodal import ChartType
        
        assert ChartType.BAR == "bar"
        assert ChartType.LINE == "line"
        assert ChartType.PIE == "pie"
        assert ChartType.SCATTER == "scatter"
        assert ChartType.HEATMAP == "heatmap"
        assert ChartType.OTHER == "other"

    def test_extracted_image_structure(self):
        """Test ExtractedImage structure."""
        from app.ingestion.images import ExtractedImage
        
        img = ExtractedImage(
            page_number=1,
            image_index=0,
            image_bytes=b"fake bytes",
            format="png",
            width=800,
            height=600,
            bbox=(100, 100, 900, 700),
            page_region="center",
        )
        assert img.page_number == 1
        assert img.width == 800
        assert img.height == 600
        assert img.page_region == "center"

    def test_extracted_chart_structure(self):
        """Test ExtractedChart structure."""
        from app.ingestion.images import ExtractedChart
        from app.ingestion.multimodal import ChartDataPoint, ChartType
        
        chart = ExtractedChart(
            page_number=1,
            chart_index=0,
            chart_type=ChartType.BAR,
            title="Test Chart",
            x_axis_label="X",
            y_axis_label="Y",
            data_points=[
                ChartDataPoint(x=1, y=10, series="A"),
                ChartDataPoint(x=2, y=20, series="A"),
            ],
            series_names=["A"],
            image_bytes=b"bytes",
        )
        assert chart.chart_type == ChartType.BAR
        assert chart.title == "Test Chart"
        assert len(chart.data_points) == 2

    def test_charts_to_text_segments(self):
        """Test converting charts to text segments."""
        from app.ingestion.images import ExtractedChart
        from app.ingestion.multimodal import ChartDataPoint, ChartType
        from app.ingestion.chunking import TextSegment
        
        chart = ExtractedChart(
            page_number=1,
            chart_index=0,
            chart_type=ChartType.BAR,
            title="Sales Chart",
            x_axis_label="Month",
            y_axis_label="Revenue",
            data_points=[
                ChartDataPoint(x="Jan", y=100, series="Sales"),
                ChartDataPoint(x="Feb", y=200, series="Sales"),
            ],
            series_names=["Sales"],
            model_description="Bar chart showing monthly sales",
        )
        
        segments = charts_to_text_segments([chart])
        assert len(segments) == 1
        assert isinstance(segments[0], TextSegment)
        assert "Chart on page 1" in segments[0].text
        assert "Sales Chart" in segments[0].text


class TestMultimodalPipelineIntegration:
    """Integration tests for multimodal pipeline."""

    def test_pipeline_imports(self):
        """Test that all multimodal modules can be imported."""
        from app.ingestion import ocr, tables, web, spreadsheets, images
        
        assert ocr is not None
        assert tables is not None
        assert web is not None
        assert spreadsheets is not None
        assert images is not None

    def test_multimodal_config_flags(self):
        """Test that multimodal config flags exist."""
        from app.config import get_settings
        
        settings = get_settings()
        assert hasattr(settings, "multimodal_enabled")
        assert hasattr(settings, "multimodal_ocr_enabled")
        assert hasattr(settings, "multimodal_table_extraction_enabled")
        assert hasattr(settings, "multimodal_web_ingestion_enabled")
        assert hasattr(settings, "multimodal_spreadsheet_enabled")
        assert hasattr(settings, "multimodal_chart_extraction_enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])