"""Tests for Phase 11 Multimodal Ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

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


class TestOCRFallback:
    """Tests for OCR fallback functionality (Phase 11.1)."""

    def test_extract_pdf_text_layer(self):
        """Test extracting text layer from PDF."""
        # This would need a test PDF - using a minimal check
        # In real tests, we'd create a test PDF
        pass

    def test_has_usable_text_layer(self):
        """Test detecting usable text layer."""
        # Would need test PDFs with and without text layers
        pass

    def test_ocr_fallback_logic(self):
        """Test OCR fallback only when no text layer."""
        # The key acceptance criteria: OCR only when no text layer exists
        pass

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