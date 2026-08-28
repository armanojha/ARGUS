"""OCR Fallback for Scanned PDFs (Phase 11.1).

Provides OCR text extraction only when no usable text layer exists.
Uses Tesseract via pytesseract with graceful fallback if unavailable.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber
from PIL import Image

from app.config import get_settings
from app.ingestion.chunking import TextSegment
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.ocr")


@dataclass(frozen=True)
class OCRResult:
    """Result of OCR processing on a PDF page."""
    page_number: int
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    ocr_used: bool = True


def _tesseract_available() -> bool:
    """Check if tesseract OCR engine is available on the system."""
    return shutil.which("tesseract") is not None


def _get_ocr_languages() -> list[str]:
    """Get configured OCR languages from settings."""
    settings = get_settings()
    # Default to English, can be extended via settings
    return getattr(settings, "ocr_languages", ["eng"])


def extract_pdf_text_layer(pdf_path: Path) -> list[str]:
    """Extract text from PDF using pdfplumber (text layer only, no OCR).
    
    Returns list of text per page. Empty string for pages with no text layer.
    """
    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return texts


def has_usable_text_layer(pdf_path: Path, min_chars_per_page: int = 50) -> bool:
    """Check if PDF has a usable text layer.
    
    A text layer is considered usable if the average characters per page
    exceeds the minimum threshold.
    """
    texts = extract_pdf_text_layer(pdf_path)
    if not texts:
        return False
    
    total_chars = sum(len(t) for t in texts)
    avg_chars = total_chars / len(texts) if texts else 0
    return avg_chars >= min_chars_per_page


def _pdf_page_to_image(page, dpi: int = 300) -> Image.Image:
    """Convert a pdfplumber page to a PIL Image for OCR."""
    # pdfplumber page objects can be rendered to images
    # Use the page's to_image method with specified resolution
    im = page.to_image(resolution=dpi)
    return im.original


def _run_tesseract_ocr(image: Image.Image, languages: list[str]) -> tuple[str, Optional[float]]:
    """Run Tesseract OCR on a PIL Image.
    
    Returns tuple of (extracted_text, confidence).
    """
    try:
        import pytesseract
        
        # Configure tesseract
        lang_str = "+".join(languages)
        
        # Get detailed output including confidence
        data = pytesseract.image_to_data(
            image,
            lang=lang_str,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate average confidence
        words = [w for w in data["text"] if w.strip()]
        text = " ".join(words)
        
        confidences = [c for c in data["conf"] if c > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        return text, avg_confidence
    
    except ImportError:
        logger.warning("pytesseract not available for OCR")
        return "", None
    except Exception as e:
        logger.warning("tesseract_ocr_failed", error=str(e))
        return "", None


def extract_pdf_with_ocr_fallback(
    pdf_path: Path,
    min_chars_per_page: int = 50,
    dpi: int = 300,
) -> Iterator[OCRResult]:
    """Extract text from PDF with OCR fallback for scanned pages.
    
    Only applies OCR to pages that don't have a usable text layer.
    Yields OCRResult for each page.
    
    Per Phase 11.1: OCR only when no usable text layer exists.
    """
    settings = get_settings()
    
    if not settings.multimodal_ocr_enabled:
        logger.info("ocr_disabled_via_config")
        # Fall back to text layer only
        texts = extract_pdf_text_layer(pdf_path)
        for i, text in enumerate(texts):
            yield OCRResult(
                page_number=i + 1,
                text=text,
                ocr_used=False,
            )
        return
    
    if not _tesseract_available():
        logger.warning("tesseract_not_available_using_text_layer_only")
        texts = extract_pdf_text_layer(pdf_path)
        for i, text in enumerate(texts):
            yield OCRResult(
                page_number=i + 1,
                text=text,
                ocr_used=False,
            )
        return
    
    languages = _get_ocr_languages()
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # First try to extract text layer
            text_layer = page.extract_text() or ""
            
            if len(text_layer.strip()) >= min_chars_per_page:
                # Text layer is sufficient, no OCR needed
                yield OCRResult(
                    page_number=page_num,
                    text=text_layer,
                    ocr_used=False,
                )
            else:
                # No usable text layer - apply OCR
                logger.info("applying_ocr", page=page_num, path=str(pdf_path))
                
                try:
                    image = _pdf_page_to_image(page, dpi=dpi)
                    ocr_text, confidence = _run_tesseract_ocr(image, languages)
                    
                    if ocr_text.strip():
                        yield OCRResult(
                            page_number=page_num,
                            text=ocr_text,
                            language="+".join(languages),
                            confidence=confidence,
                            ocr_used=True,
                        )
                    else:
                        # OCR produced nothing, fall back to empty text layer
                        yield OCRResult(
                            page_number=page_num,
                            text=text_layer,
                            ocr_used=False,
                        )
                except Exception as e:
                    logger.warning("ocr_page_failed", page=page_num, error=str(e))
                    # Fall back to text layer even if minimal
                    yield OCRResult(
                        page_number=page_num,
                        text=text_layer,
                        ocr_used=False,
                    )


def extract_pdf_segments_with_ocr(
    pdf_path: Path,
    min_chars_per_page: int = 50,
    dpi: int = 300,
) -> list[TextSegment]:
    """Extract text segments from PDF with OCR fallback.
    
    Returns TextSegment objects suitable for the existing chunking pipeline.
    OCR is only applied to pages without usable text layers.
    """
    segments = []
    
    for ocr_result in extract_pdf_with_ocr_fallback(pdf_path, min_chars_per_page, dpi):
        if not ocr_result.text.strip():
            continue
        
        segments.append(TextSegment(
            text=ocr_result.text.strip(),
            page_start=ocr_result.page_number,
            page_end=ocr_result.page_number,
            char_start=0,
            char_end=len(ocr_result.text),
            section_path=None,
        ))
    
    logger.info("pdf_segments_extracted_with_ocr", 
                path=str(pdf_path), 
                segment_count=len(segments))
    return segments


def extract_pdf_text_for_checksum(pdf_path: Path) -> str:
    """Extract full text from PDF for checksumming.
    
    Uses text layer primarily, OCR only for pages without text.
    """
    texts = []
    for result in extract_pdf_with_ocr_fallback(pdf_path):
        texts.append(result.text)
    return "\n".join(texts)


__all__ = [
    "OCRResult",
    "extract_pdf_text_layer",
    "has_usable_text_layer",
    "extract_pdf_with_ocr_fallback",
    "extract_pdf_segments_with_ocr",
    "extract_pdf_text_for_checksum",
]