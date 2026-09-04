"""OCR Fallback for Scanned PDFs (Phase 11.1).

Provides OCR text extraction only when no usable text layer exists.

Two engines are supported:

* **PaddleOCR** (preferred) — state-of-the-art deep-learning OCR that runs in
  an *isolated* Python 3.11+ venv because PaddlePaddle (PaddleOCR's backend)
  has no wheels for this application's Python 3.14 interpreter. The main app
  communicates with a persistent worker (`scripts/paddle_ocr_runner.py`) over
  JSON lines on stdin/stdout, which lets the heavy models load once and be
  reused for every page.
* **Tesseract** (legacy fallback) — via pytesseract when Paddle is unavailable
  or explicitly chosen.

OCR is applied only to pages without a usable text layer (per Phase 11.1).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from PIL import Image

from app.config import REPO_ROOT, get_settings
from app.ingestion.chunking import TextSegment
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.ocr")


@dataclass(frozen=True)
class OCRResult:
    """Result of OCR processing on a PDF page."""
    page_number: int
    text: str
    language: str | None = None
    confidence: float | None = None
    ocr_used: bool = True
    engine: str | None = None
    metadata: dict | None = None


def _tesseract_available() -> bool:
    """Check if tesseract OCR engine is available on the system."""
    return shutil.which("tesseract") is not None


def _get_ocr_languages() -> list[str]:
    """Get configured OCR languages from settings."""
    settings = get_settings()
    # Default to English, can be extended via settings
    return getattr(settings, "ocr_languages", ["eng"])


# ---------------------------------------------------------------------------
# PaddleOCR runner (persistent subprocess in an isolated Python 3.11+ venv)
# ---------------------------------------------------------------------------

def _default_runner_python() -> Path | None:
    """Default PaddleOCR venv interpreter next to the repo."""
    exe = REPO_ROOT / ".venv-ocr" / "Scripts" / "python.exe"
    return exe if exe.exists() else None


def _default_runner_script() -> Path | None:
    script = REPO_ROOT / "scripts" / "paddle_ocr_runner.py"
    return script if script.exists() else None


def _resolve_runner_python() -> Path | None:
    settings = get_settings()
    cfg = getattr(settings, "ocr_runner_python", None)
    if cfg:
        p = Path(cfg)
        return p if p.exists() else None
    return _default_runner_python()


def _resolve_runner_script() -> Path | None:
    settings = get_settings()
    cfg = getattr(settings, "ocr_runner_script", None)
    if cfg:
        p = Path(cfg)
        return p if p.exists() else None
    return _default_runner_script()


def _paddle_available() -> bool:
    """PaddleOCR is available if runner interpreter and script both exist."""
    return _resolve_runner_python() is not None and _resolve_runner_script() is not None


def _resolve_ocr_engine() -> str | None:
    """Resolve the effective OCR engine per config, or None if none usable."""
    settings = get_settings()
    engine = getattr(settings, "ocr_engine", "auto").lower()
    if engine == "tesseract":
        return "tesseract" if _tesseract_available() else None
    if engine == "paddle":
        return "paddle" if _paddle_available() else None
    # auto: prefer PaddleOCR, fall back to Tesseract.
    if _paddle_available():
        return "paddle"
    return "tesseract" if _tesseract_available() else None


class PaddleOCRRunner:
    """Manages a persistent PaddleOCR worker subprocess (JSON-lines over stdio).

    The heavy OCR models are loaded once at worker startup and reused for every
    page, avoiding per-page model reloads. The worker is transparently restarted
    if it crashes or disconnects.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @property
    def available(self) -> bool:
        return _paddle_available()

    def _start(self) -> None:
        python = _resolve_runner_python()
        script = _resolve_runner_script()
        if python is None or script is None:
            raise FileNotFoundError("PaddleOCR runner interpreter/script not found")
        self._proc = subprocess.Popen(
            [str(python), "-u", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        # Wait for the worker's readiness handshake. Paddle emits diagnostic
        # noise on stderr during model warm-up, so skip non-JSON lines.
        ready = self._read_json_line()
        if not ready or ready.get("event") != "ready":
            raise RuntimeError("PaddleOCR worker did not report ready")

    def _read_json_line(self) -> dict | None:
        """Read the next JSON object from the worker, skipping noise lines."""
        assert self._proc is not None and self._proc.stderr is not None
        import json

        while True:
            line = self._proc.stderr.readline()
            if not line:
                return None  # EOF / worker gone
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # diagnostic noise from Paddle; skip
            if isinstance(obj, dict):
                return obj

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None
        self._start()

    def run_ocr(self, image: Image.Image, language: str = "en") -> tuple[str, float | None]:
        """Run PaddleOCR on a PIL image. Returns (text, avg_confidence)."""
        if not self.available:
            return "", None
        settings = get_settings()
        timeout = float(getattr(settings, "ocr_runner_timeout", 120.0))

        # NOTE: on Windows a NamedTemporaryFile keeps the handle open/locked,
        # so the worker subprocess could not write to it. Use mkstemp (handle
        # released immediately) and clean up explicitly.
        import os

        fd, tf_name = tempfile.mkstemp(suffix=".png", prefix="argus_ocr_")
        os.close(fd)
        try:
            image.save(tf_name, format="PNG")
        except Exception:  # noqa: BLE001
            os.unlink(tf_name)
            raise
        try:
            req = {"path": tf_name, "lang": language or "en"}
            self._ensure_started()
            try:
                return self._exchange(req, timeout)
            except (BrokenPipeError, ConnectionResetError, subprocess.SubprocessError):
                # Worker died mid-call: restart once and retry.
                self._proc = None
                self._ensure_started()
                return self._exchange(req, timeout)
        finally:
            try:
                os.unlink(tf_name)
            except OSError:
                pass

    def _exchange(self, req: dict, timeout: float) -> tuple[str, float | None]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stderr is not None
        import json

        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = self._read_json_line()
        if resp is None:
            raise BrokenPipeError("PaddleOCR worker closed unexpectedly")
        if not resp.get("ok"):
            logger.warning("paddle_ocr_error", error=resp.get("error"))
            return "", None
        return resp.get("text", ""), resp.get("confidence")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None


_runner_singleton: PaddleOCRRunner | None = None


def _get_runner() -> PaddleOCRRunner | None:
    global _runner_singleton
    if _paddle_available():
        if _runner_singleton is None:
            _runner_singleton = PaddleOCRRunner()
        return _runner_singleton
    return None


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


def _run_tesseract_ocr(image: Image.Image, languages: list[str]) -> tuple[str, float | None]:
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
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("tesseract_ocr_failed", error=str(e))
        return "", None


def _run_paddle_ocr(image: Image.Image, languages: list[str]) -> tuple[str, float | None]:
    """Run PaddleOCR on a PIL Image via the isolated worker."""
    runner = _get_runner()
    if runner is None:
        logger.warning("paddle_ocr_runner_unavailable")
        return "", None
    raw_lang = languages[0] if languages else "en"
    # Map ARGUS-style language codes to PaddleOCR-friendly names.
    lang = {"eng": "en", "en": "en"}.get(raw_lang, raw_lang)
    try:
        text, confidence = runner.run_ocr(image, lang or "en")
        return text, confidence
    except Exception as e:  # noqa: BLE001
        logger.warning("paddle_ocr_failed", error=str(e))
        return "", None


def _run_ocr(image: Image.Image, languages: list[str], engine: str | None) -> tuple[str, float | None, str | None]:
    """Dispatch OCR to the resolved engine. Returns (text, confidence, engine)."""
    if engine == "paddle":
        text, conf = _run_paddle_ocr(image, languages)
        if text.strip():
            return text, conf, "paddle"
        # fall through to tesseract if OCR produced nothing meaningful
        if _tesseract_available():
            text, conf = _run_tesseract_ocr(image, languages)
            return text, conf, "tesseract"
        return text, conf, "paddle"
    text, conf = _run_tesseract_ocr(image, languages)
    return text, conf, "tesseract"


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

    if not settings.multimodal_enabled or not settings.multimodal_ocr_enabled:
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

    engine = _resolve_ocr_engine()
    if engine is None:
        logger.warning("ocr_engine_unavailable_using_text_layer_only")
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
                logger.info("applying_ocr", page=page_num, path=str(pdf_path), engine=engine)

                try:
                    image = _pdf_page_to_image(page, dpi=dpi)
                    ocr_text, confidence, used_engine = _run_ocr(image, languages, engine)

                    if ocr_text.strip():
                        yield OCRResult(
                            page_number=page_num,
                            text=ocr_text,
                            language="+".join(languages),
                            confidence=confidence,
                            ocr_used=True,
                            engine=used_engine,
                        )
                    else:
                        # OCR produced nothing, fall back to empty text layer
                        yield OCRResult(
                            page_number=page_num,
                            text=text_layer,
                            ocr_used=False,
                        )
                except (OSError, ValueError, RuntimeError) as e:
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
    "PaddleOCRRunner",
    "extract_pdf_segments_with_ocr",
    "extract_pdf_text_for_checksum",
    "extract_pdf_text_layer",
    "extract_pdf_with_ocr_fallback",
    "has_usable_text_layer",
]
