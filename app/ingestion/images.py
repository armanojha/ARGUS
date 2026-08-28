"""Chart/Image Extraction (Phase 11.5 - Later Stage).

Extracts images/charts from PDFs, stores region + model description/extracted values.
Connects narrative claims to chart/table/image evidence (multimodal claim retrieval).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pdfplumber
from PIL import Image as PILImage

from app.config import get_settings
from app.ingestion.chunking import TextSegment
from app.ingestion.multimodal import Chart, ChartDataPoint, ChartType, Image as MultimodalImage, MultimodalType
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.images")


@dataclass(frozen=True)
class ExtractedImage:
    """An image extracted from a PDF page."""
    page_number: int
    image_index: int
    image_bytes: bytes
    format: str
    width: int
    height: int
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom)
    page_region: str | None = None  # e.g., "top-left", "center"


@dataclass(frozen=True)
class ExtractedChart:
    """A chart detected in a PDF page."""
    page_number: int
    chart_index: int
    chart_type: ChartType
    title: str | None
    x_axis_label: str | None
    y_axis_label: str | None
    data_points: list[ChartDataPoint]
    series_names: list[str]
    image_bytes: bytes | None = None
    bbox: tuple[float, float, float, float] | None = None
    model_description: str | None = None


def _detect_chart_type_from_image(image: Image.Image) -> ChartType:
    """Attempt to detect chart type from image using simple heuristics.
    
    In a full implementation, this would use a vision model.
    For now, we use simple heuristics based on image characteristics.
    """
    # This is a placeholder - a real implementation would use a vision model
    # or more sophisticated detection
    return ChartType.OTHER


def _extract_images_from_pdf(pdf_path: Path) -> list[ExtractedImage]:
    """Extract all images from a PDF using pdfplumber."""
    images = []
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Get images from page
            page_images = page.images
            
            for img_idx, img_info in enumerate(page_images):
                try:
                    # Extract image bytes
                    stream = img_info.get("stream")
                    if not stream:
                        continue
                    
                    image_bytes = stream.get_data()
                    
                    # Determine format
                    filter_name = img_info.get("filter", [])
                    if isinstance(filter_name, list):
                        filter_name = filter_name[0] if filter_name else ""
                    
                    fmt = "png"  # default
                    if "DCTDecode" in str(filter_name):
                        fmt = "jpeg"
                    elif "FlateDecode" in str(filter_name):
                        fmt = "png"
                    elif "JPXDecode" in str(filter_name):
                        fmt = "jp2"
                    
                    # Get dimensions
                    width = img_info.get("width", 0)
                    height = img_info.get("height", 0)
                    
                    # Get bbox if available
                    bbox = None
                    if "x0" in img_info and "y0" in img_info:
                        bbox = (
                            img_info["x0"],
                            img_info["y0"],
                            img_info.get("x1", img_info["x0"] + width),
                            img_info.get("y1", img_info["y0"] + height),
                        )
                    
                    # Determine page region
                    page_region = None
                    if bbox and width > 0 and height > 0:
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        page_width = page.width
                        page_height = page.height
                        
                        h_region = "left" if x_center < page_width / 3 else ("right" if x_center > 2 * page_width / 3 else "center")
                        v_region = "top" if y_center < page_height / 3 else ("bottom" if y_center > 2 * page_height / 3 else "center")
                        page_region = f"{v_region}-{h_region}"
                    
                    images.append(ExtractedImage(
                        page_number=page_num,
                        image_index=img_idx,
                        image_bytes=image_bytes,
                        format=fmt,
                        width=width,
                        height=height,
                        bbox=bbox,
                        page_region=page_region,
                    ))
                except Exception as e:
                    logger.warning("image_extraction_failed", page=page_num, idx=img_idx, error=str(e))
    
    return images


def _extract_charts_from_pdf(pdf_path: Path) -> list[ExtractedChart]:
    """Extract charts from PDF.
    
    This is a placeholder implementation. A full implementation would:
    1. Use image extraction to get chart images
    2. Use a vision model (via LLM gateway) to detect chart type and extract data
    3. Store model description and extracted data points
    
    For now, we extract images that look like charts based on size/position.
    """
    charts = []
    
    if not get_settings().multimodal_chart_extraction_enabled:
        return charts
    
    # Extract all images first
    images = _extract_images_from_pdf(pdf_path)
    
    # Heuristic: images that are large and centered might be charts
    for img in images:
        if img.width < 200 or img.height < 150:
            continue  # Too small to be a meaningful chart
        
        # Check aspect ratio - charts often have specific ratios
        aspect = img.width / img.height if img.height > 0 else 0
        if 0.5 < aspect < 3.0:  # Reasonable chart aspect ratio
            # Create chart object (data points would come from vision model)
            chart = ExtractedChart(
                page_number=img.page_number,
                chart_index=img.image_index,
                chart_type=_detect_chart_type_from_image(
                    PILImage.open(io.BytesIO(img.image_bytes))
                ) if img.image_bytes else ChartType.OTHER,
                title=None,  # Would be extracted by vision model
                x_axis_label=None,
                y_axis_label=None,
                data_points=[],
                series_names=[],
                image_bytes=img.image_bytes,
                bbox=img.bbox,
                model_description=None,  # Would be generated by vision model
            )
            charts.append(chart)
    
    return charts


def extract_pdf_images(pdf_path: Path) -> list[ExtractedImage]:
    """Extract all images from a PDF document."""
    settings = get_settings()
    
    if not settings.multimodal_chart_extraction_enabled:
        logger.info("image_extraction_disabled_via_config")
        return []
    
    return _extract_images_from_pdf(pdf_path)


def extract_pdf_charts(pdf_path: Path) -> list[ExtractedChart]:
    """Extract charts from a PDF document."""
    settings = get_settings()
    
    if not settings.multimodal_chart_extraction_enabled:
        logger.info("chart_extraction_disabled_via_config")
        return []
    
    return _extract_charts_from_pdf(pdf_path)


def images_to_multimodal(
    images: list[ExtractedImage],
    document_id: UUID,
    source_path: str,
    source_chunk_ids: list[UUID] | None = None,
) -> list[Image]:
    """Convert ExtractedImage objects to Multimodal Image objects."""
    multimodal_images = []
    
    for img in images:
        multimodal = Image(
            id=UUID(int=0),  # Will be assigned by store
            source_path=source_path,
            content_type=MultimodalType.IMAGE,
            source_chunk_ids=source_chunk_ids or [],
            page_number=img.page_number,
            page_range=(img.page_number, img.page_number),
            width=img.width,
            height=img.height,
            format=img.format,
            metadata={
                "image_index": img.image_index,
                "bbox": img.bbox,
                "page_region": img.page_region,
            },
        )
        multimodal_images.append(multimodal)
    
    return multimodal_images


def charts_to_multimodal(
    charts: list[ExtractedChart],
    document_id: UUID,
    source_path: str,
    source_chunk_ids: list[UUID] | None = None,
) -> list[Chart]:
    """Convert ExtractedChart objects to Multimodal Chart objects."""
    multimodal_charts = []
    
    for chart in charts:
        multimodal = Chart(
            id=UUID(int=0),  # Will be assigned by store
            source_path=source_path,
            content_type=MultimodalType.CHART,
            source_chunk_ids=source_chunk_ids or [],
            page_number=chart.page_number,
            page_range=(chart.page_number, chart.page_number),
            chart_type=chart.chart_type,
            title=chart.title,
            x_axis_label=chart.x_axis_label,
            y_axis_label=chart.y_axis_label,
            data_points=chart.data_points,
            series_names=chart.series_names,
            model_description=chart.model_description,
            metadata={
                "chart_index": chart.chart_index,
                "bbox": chart.bbox,
            } if chart.bbox else {"chart_index": chart.chart_index},
        )
        multimodal_charts.append(multimodal)
    
    return multimodal_charts


def images_to_text_segments(images: list[ExtractedImage]) -> list[TextSegment]:
    """Convert images to text segments for chunking pipeline (placeholder descriptions)."""
    segments = []
    
    for img in images:
        desc = f"[Image on page {img.page_number}, {img.width}x{img.height} {img.format}]"
        if img.page_region:
            desc += f" ({img.page_region})"
        
        segments.append(TextSegment(
            text=desc,
            page_start=img.page_number,
            page_end=img.page_number,
            char_start=None,
            char_end=None,
            section_path=f"Image {img.image_index + 1}",
        ))
    
    return segments


def charts_to_text_segments(charts: list[ExtractedChart]) -> list[TextSegment]:
    """Convert charts to text segments for chunking pipeline."""
    segments = []
    
    for chart in charts:
        desc = f"[Chart on page {chart.page_number}, type: {chart.chart_type.value}]"
        if chart.title:
            desc += f" - {chart.title}"
        if chart.model_description:
            desc += f" - {chart.model_description}"
        
        segments.append(TextSegment(
            text=desc,
            page_start=chart.page_number,
            page_end=chart.page_number,
            char_start=None,
            char_end=None,
            section_path=f"Chart {chart.chart_index + 1}",
        ))
    
    return segments


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute hash for image deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


__all__ = [
    "ExtractedChart",
    "ExtractedImage",
    "charts_to_multimodal",
    "charts_to_text_segments",
    "compute_image_hash",
    "extract_pdf_charts",
    "extract_pdf_images",
    "images_to_multimodal",
    "images_to_text_segments",
]