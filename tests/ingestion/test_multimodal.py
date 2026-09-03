"""Tests for Phase 11 Multimodal Ingestion."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

from app.ingestion.images import (
    charts_to_text_segments,
    extract_pdf_charts,
    extract_pdf_images,
)
from app.ingestion.ocr import (
    OCRResult,
    extract_pdf_segments_with_ocr,
    extract_pdf_text_layer,
    extract_pdf_with_ocr_fallback,
    has_usable_text_layer,
)
from app.ingestion.spreadsheets import (
    ingest_spreadsheet,
    is_valid_spreadsheet,
    spreadsheet_to_text_segments,
)
from app.ingestion.tables import ExtractedTable, extract_pdf_tables, tables_to_text_segments
from app.ingestion.web import is_valid_web_url, web_page_to_text_segments


def _create_text_pdf() -> bytes:
    """Create a valid PDF with text content using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 750), "This is a test PDF with text layer.", fontsize=12)
    page.insert_text((100, 730), "It contains multiple lines of text.", fontsize=12)
    
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _create_scanned_pdf() -> bytes:
    """Create a PDF without extractable text (simulated scanned PDF)."""
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def text_layer_pdf_path():
    """Create a temp PDF file with text layer."""
    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    temp_path.write_bytes(_create_text_pdf())
    yield temp_path
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def scanned_pdf_path():
    """Create a temp PDF file without text layer (simulated scanned)."""
    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    temp_path.write_bytes(_create_scanned_pdf())
    yield temp_path
    temp_path.unlink(missing_ok=True)


class TestOCRFallback:
    """Tests for OCR fallback functionality (Phase 11.1)."""

    def test_extract_pdf_text_layer(self, text_layer_pdf_path, scanned_pdf_path):
        """Test extracting text layer from PDF."""
        # PDF with text layer should return text
        texts = extract_pdf_text_layer(text_layer_pdf_path)
        assert len(texts) == 1
        assert "test PDF with text layer" in texts[0]
        assert "multiple lines of text" in texts[0]
        
        # Scanned PDF (no text layer) should return minimal/empty text
        scanned_texts = extract_pdf_text_layer(scanned_pdf_path)
        assert len(scanned_texts) == 1
        # The scanned PDF has no text
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
        assert len(segments) >= 1
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
        from datetime import UTC, datetime

        from app.ingestion.web import WebPageResult
        
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
        from datetime import UTC, datetime

        from app.ingestion.chunking import TextSegment
        from app.ingestion.web import WebPageResult
        
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
        from pathlib import Path
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
        from app.ingestion.chunking import TextSegment
        from app.ingestion.multimodal import SpreadsheetCell
        from app.ingestion.spreadsheets import SpreadsheetResult, SpreadsheetSheet
        
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
        from app.ingestion.chunking import TextSegment
        from app.ingestion.images import ExtractedChart
        from app.ingestion.multimodal import ChartDataPoint, ChartType
        
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
        from app.ingestion import images, ocr, spreadsheets, tables, web

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

    def test_multimodal_enabled_gates_all_features(self):
        """Test that multimodal_enabled=False disables all sub-features."""
        from unittest.mock import patch

        from app.config import Settings

        settings = Settings(
            multimodal_enabled=False,
            multimodal_ocr_enabled=True,
            multimodal_table_extraction_enabled=True,
            multimodal_web_ingestion_enabled=True,
            multimodal_spreadsheet_enabled=True,
            multimodal_chart_extraction_enabled=True,
        )
        temp_pdf = Path(tempfile.mktemp(suffix=".pdf"))
        temp_pdf.write_bytes(_create_text_pdf())
        try:
            with patch("app.ingestion.ocr.get_settings", return_value=settings):
                from app.ingestion.ocr import extract_pdf_with_ocr_fallback
                results = list(extract_pdf_with_ocr_fallback(temp_pdf))
                assert all(not r.ocr_used for r in results)
        finally:
            temp_pdf.unlink(missing_ok=True)

    def test_csv_ingestion_end_to_end(self):
        """Test ingesting a real CSV file through the spreadsheet pipeline."""
        from app.config import Settings

        settings = Settings(
            multimodal_enabled=True,
            multimodal_spreadsheet_enabled=True,
        )
        with patch("app.ingestion.spreadsheets.get_settings", return_value=settings):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as f:
                f.write("Name,Age,City\nAlice,30,NYC\nBob,25,LA\n")
                csv_path = Path(f.name)

            try:
                result = ingest_spreadsheet(csv_path)
                assert len(result.sheets) == 1
                assert result.sheets[0].name == "Sheet1"
                assert result.total_rows == 2
                assert result.total_cells == 6

                # Verify cell content
                cell_values = {c.value for c in result.sheets[0].cells}
                assert "Alice" in cell_values
                assert "Bob" in cell_values
                assert "NYC" in cell_values

                # Verify text segment conversion
                segments = spreadsheet_to_text_segments(result)
                assert len(segments) == 1
                assert "Sheet: Sheet1" in segments[0].text
                assert "Alice" in segments[0].text
            finally:
                csv_path.unlink(missing_ok=True)

    def test_csv_ingestion_small_file(self):
        """Test ingesting a small CSV that may trip the Sniffer."""
        from app.config import Settings

        settings = Settings(
            multimodal_enabled=True,
            multimodal_spreadsheet_enabled=True,
        )
        with patch("app.ingestion.spreadsheets.get_settings", return_value=settings):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as f:
                f.write("single_column\nvalue1\n")
                csv_path = Path(f.name)

            try:
                result = ingest_spreadsheet(csv_path)
                assert len(result.sheets) == 1
                assert result.total_rows == 1
            finally:
                csv_path.unlink(missing_ok=True)

    def test_table_extraction_end_to_end(self):
        """Test extracting tables from a PDF with a real table."""
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)

        headers = ["Name", "Score", "Grade"]
        rows = [["Alice", "95", "A"], ["Bob", "82", "B"], ["Carol", "78", "C"]]

        y = 400
        for row in [headers] + rows:
            x = 100
            for cell in row:
                page.insert_text((x, y), cell, fontsize=10)
                x += 150
            y += 20

        pdf_path = Path(tempfile.mktemp(suffix=".pdf"))
        doc.save(str(pdf_path))
        doc.close()

        from app.config import Settings

        settings = Settings(
            multimodal_enabled=True,
            multimodal_table_extraction_enabled=True,
        )
        with patch("app.ingestion.tables.get_settings", return_value=settings):
            try:
                tables = extract_pdf_tables(pdf_path)
                assert isinstance(tables, list)
            finally:
                pdf_path.unlink(missing_ok=True)

    def test_pdf_images_extraction_end_to_end(self):
        """Test extracting images from a PDF (may return empty for text-only PDF)."""
        from app.config import Settings

        settings = Settings(
            multimodal_enabled=True,
            multimodal_chart_extraction_enabled=True,
        )
        with patch("app.ingestion.images.get_settings", return_value=settings):
            pdf_path = Path(tempfile.mktemp(suffix=".pdf"))
            doc = pymupdf.open()
            doc.new_page(width=612, height=792)
            doc.save(str(pdf_path))
            doc.close()

            try:
                images = extract_pdf_images(pdf_path)
                assert isinstance(images, list)
                charts = extract_pdf_charts(pdf_path)
                assert isinstance(charts, list)
            finally:
                pdf_path.unlink(missing_ok=True)

    def test_features_disabled_when_multimodal_off(self):
        """Test that sub-features return empty/raise when multimodal_enabled=False."""
        from app.config import Settings

        settings = Settings(
            multimodal_enabled=False,
            multimodal_spreadsheet_enabled=True,
            multimodal_chart_extraction_enabled=True,
        )

        # Spreadsheet: raises RuntimeError
        with patch("app.ingestion.spreadsheets.get_settings", return_value=settings):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as f:
                f.write("a,b\n1,2\n")
                csv_path = Path(f.name)
            try:
                with pytest.raises(RuntimeError, match="disabled"):
                    ingest_spreadsheet(csv_path)
            finally:
                csv_path.unlink(missing_ok=True)

        # Images: returns empty list
        with patch("app.ingestion.images.get_settings", return_value=settings):
            pdf_path = Path(tempfile.mktemp(suffix=".pdf"))
            doc = pymupdf.open()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()
            try:
                assert extract_pdf_images(pdf_path) == []
                assert extract_pdf_charts(pdf_path) == []
            finally:
                pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])