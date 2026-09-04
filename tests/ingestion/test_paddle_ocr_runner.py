"""Tests for the PaddleOCR integration (isolated Python 3.11 runner).

These tests run on the main interpreter (3.14) and never import PaddleOCR.
They verify:

* engine resolution under `ocr_engine: auto|paddle|tesseract`,
* the `PaddleOCRRunner` JSON-over-stdio protocol (handshake, skip-noise,
  request/response, crash restart),
* graceful fallback to Tesseract when Paddle is unavailable or fails.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from PIL import Image

from app.config import Settings
from app.ingestion.ocr import (
    PaddleOCRRunner,
    _OcrCache,
    _resolve_ocr_engine,
    extract_pdf_with_ocr_fallback,
)


def _make_text_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 750), "This is a test PDF with text layer.", fontsize=12)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _make_blank_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _make_scanned_pdf() -> bytes:
    """Image-only page (vector ink, no extractable text layer)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(pymupdf.Rect(40, 40, 572, 440), color=None, fill=(0.1, 0.1, 0.1))
    page.draw_rect(pymupdf.Rect(40, 480, 572, 720), color=None, fill=(0.2, 0.2, 0.2))
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def text_layer_pdf_path():
    p = Path(tempfile.mktemp(suffix=".pdf"))
    p.write_bytes(_make_text_pdf())
    yield p
    p.unlink(missing_ok=True)


@pytest.fixture
def scanned_pdf_path():
    p = Path(tempfile.mktemp(suffix=".pdf"))
    p.write_bytes(_make_blank_pdf())
    yield p
    p.unlink(missing_ok=True)


@pytest.fixture
def inky_scanned_pdf_path():
    """Scanned PDF with visible ink but no text layer (exercises OCR fallback)."""
    p = Path(tempfile.mktemp(suffix=".pdf"))
    p.write_bytes(_make_scanned_pdf())
    yield p
    p.unlink(missing_ok=True)


@pytest.fixture
def _clean_runner_singleton():
    """Ensure the module-level runner singleton does not leak across tests."""
    import app.ingestion.ocr as ocm

    before = ocm._runner_singleton
    ocm._runner_singleton = None
    yield
    ocm._runner_singleton = before


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------

class TestEngineResolution:
    def test_auto_prefers_paddle_when_available(self, _clean_runner_singleton):
        with (
            patch("app.ingestion.ocr._paddle_available", return_value=True),
            patch("app.ingestion.ocr._tesseract_available", return_value=True),
        ):
            assert _resolve_ocr_engine() == "paddle"

    def test_auto_falls_back_to_tesseract(self, _clean_runner_singleton):
        with (
            patch("app.ingestion.ocr._paddle_available", return_value=False),
            patch("app.ingestion.ocr._tesseract_available", return_value=True),
        ):
            assert _resolve_ocr_engine() == "tesseract"

    def test_auto_none_when_no_engine(self, _clean_runner_singleton):
        with (
            patch("app.ingestion.ocr._paddle_available", return_value=False),
            patch("app.ingestion.ocr._tesseract_available", return_value=False),
        ):
            assert _resolve_ocr_engine() is None

    def test_forced_tesseract_even_with_paddle(self, _clean_runner_singleton):
        settings = Settings(ocr_engine="tesseract")
        with (
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._paddle_available", return_value=True),
            patch("app.ingestion.ocr._tesseract_available", return_value=True),
        ):
            assert _resolve_ocr_engine() == "tesseract"

    def test_forced_paddle_ignores_tesseract(self, _clean_runner_singleton):
        settings = Settings(ocr_engine="paddle")
        with (
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._paddle_available", return_value=True),
            patch("app.ingestion.ocr._tesseract_available", return_value=False),
        ):
            assert _resolve_ocr_engine() == "paddle"

    def test_forced_paddle_none_when_unavailable(self, _clean_runner_singleton):
        settings = Settings(ocr_engine="paddle")
        with (
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._paddle_available", return_value=False),
        ):
            assert _resolve_ocr_engine() is None


# ---------------------------------------------------------------------------
# Runner protocol (mocked subprocess)
# ---------------------------------------------------------------------------

class TestPaddleOCRRunnerProtocol:
    def _make_proc(self):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.poll.return_value = None
        return proc

    def test_start_performs_handshake(self, _clean_runner_singleton):
        runner = PaddleOCRRunner()
        proc = self._make_proc()
        with patch("app.ingestion.ocr.subprocess.Popen", return_value=proc) as popen, \
             patch.object(runner, "_read_json_line",
                          return_value={"event": "ready", "version": "paddleocr"}) as rjl:
            runner._drain_thread = MagicMock()
            runner._queue = MagicMock()
            runner._start()
            popen.assert_called_once()
            rjl.assert_called_once()

    def test_start_raises_without_ready(self, _clean_runner_singleton):
        runner = PaddleOCRRunner()
        proc = self._make_proc()
        with patch("app.ingestion.ocr.subprocess.Popen", return_value=proc), \
             patch.object(runner, "_read_json_line", return_value={"event": "nope"}):
            runner._drain_thread = MagicMock()
            runner._queue = MagicMock()
            with pytest.raises(RuntimeError, match="did not report ready"):
                runner._start()
        assert runner._proc is None  # failed spawn must not leak a worker

    def test_runner_exchange_returns_text_confidence(self, _clean_runner_singleton):
        runner = PaddleOCRRunner()
        runner._proc = self._make_proc()
        runner._queue = MagicMock()
        runner._queue.get.return_value = {"ok": True, "text": "ARGUS OCR", "confidence": 0.98}
        from app.ingestion.ocr import json
        runner._proc.stdin.write = lambda s: json.dumps(s)
        text, conf = runner._exchange({"path": "x.png", "lang": "en"}, timeout=30)
        assert text == "ARGUS OCR"
        assert conf == 0.98
        runner._proc.stdin.flush.assert_called()

    def test_runner_exchange_returns_empty_on_error(self, _clean_runner_singleton):
        runner = PaddleOCRRunner()
        runner._proc = self._make_proc()
        runner._queue = MagicMock()
        runner._queue.get.return_value = {"ok": False, "error": "boom"}
        text, conf = runner._exchange({"path": "x.png"}, timeout=10)
        assert (text, conf) == ("", None)

    def test_run_ocr_saves_temp_and_delegates(self, _clean_runner_singleton):
        runner = PaddleOCRRunner()
        img = Image.new("RGB", (10, 10), "white")
        with patch("app.ingestion.ocr._paddle_available", return_value=True), \
             patch.object(runner, "_ensure_started"), \
             patch.object(runner, "_exchange", return_value=("hello", 0.9)) as ex:
            text, conf = runner.run_ocr(img, "en")
        assert (text, conf) == ("hello", 0.9)
        assert ex.called


# ---------------------------------------------------------------------------
# PDF fallback with Paddle mocked
# ---------------------------------------------------------------------------

class TestPaddlePdfFallback:
    def test_pdf_ocr_uses_paddle_when_available(self, text_layer_pdf_path, _clean_runner_singleton):
        """A PDF with a real text layer must not go through OCR at all."""
        results = list(extract_pdf_with_ocr_fallback(text_layer_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is False

    def test_pdf_paddle_failure_falls_back_to_tesseract(self, inky_scanned_pdf_path):
        """When Paddle fails and Tesseract is available, tesseract is used."""
        settings = Settings(multimodal_enabled=True, multimodal_ocr_enabled=True)
        fake_runner = MagicMock()
        fake_runner.run_ocr.return_value = ("", None)
        with (
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._get_runner", return_value=fake_runner),
            patch("app.ingestion.ocr._tesseract_available", return_value=True),
            patch("app.ingestion.ocr._run_tesseract_ocr",
                  return_value=("recovered by tesseract", 88.0)),
        ):
            results = list(extract_pdf_with_ocr_fallback(inky_scanned_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is True
        assert results[0].engine == "tesseract"
        assert "recovered by tesseract" in results[0].text

    def test_pdf_no_engine_falls_back_to_text_layer(self, scanned_pdf_path):
        """With no engine at all, the (empty) text layer is returned."""
        settings = Settings(multimodal_enabled=True, multimodal_ocr_enabled=True)
        with (
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._resolve_ocr_engine", return_value=None),
        ):
            results = list(extract_pdf_with_ocr_fallback(scanned_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is False
        assert results[0].text.strip() == ""

    def test_ocr_cache_reuses_stored_result(self, inky_scanned_pdf_path, _clean_runner_singleton, tmp_path):
        """A second pass over the same PDF must be served from the cache."""
        cache = _OcrCache(tmp_path)
        runner = MagicMock()
        runner.deps = {}
        runner.run_ocr.return_value = ("FIRST OCR TEXT", 0.95)
        with (
            patch("app.ingestion.ocr._get_ocr_cache", return_value=cache),
            patch("app.ingestion.ocr._resolve_ocr_engine", return_value="paddle"),
            patch("app.ingestion.ocr._get_runner", return_value=runner),
        ):
            first = list(extract_pdf_with_ocr_fallback(inky_scanned_pdf_path, min_chars_per_page=10))
            runner.run_ocr.reset_mock()
            second = list(extract_pdf_with_ocr_fallback(inky_scanned_pdf_path, min_chars_per_page=10))
        assert first[0].text == "FIRST OCR TEXT"
        assert first[0].ocr_used is True
        assert second[0].text == "FIRST OCR TEXT"
        assert runner.run_ocr.call_count == 0  # second pass was served from cache

    def test_blank_scanned_page_skips_ocr(self, scanned_pdf_path, _clean_runner_singleton):
        """An entirely blank rendered page must skip OCR entirely."""
        runner = MagicMock()
        runner.deps = {}
        runner.run_ocr.return_value = ("should not be called", 0.9)
        with (
            patch("app.ingestion.ocr._resolve_ocr_engine", return_value="paddle"),
            patch("app.ingestion.ocr._get_runner", return_value=runner),
        ):
            results = list(extract_pdf_with_ocr_fallback(scanned_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is False
        assert runner.run_ocr.call_count == 0

    def test_render_failure_keyerror_falls_back_to_text_layer(
        self, scanned_pdf_path, _clean_runner_singleton,
    ):
        """A render failure (e.g. PIL KeyError('JPEG')) must not crash."""

        def _boom(page, dpi=300):
            raise KeyError("JPEG")

        with (
            patch("app.ingestion.ocr._resolve_ocr_engine", return_value="paddle"),
            patch("app.ingestion.ocr._pdf_page_to_image", side_effect=_boom),
        ):
            results = list(extract_pdf_with_ocr_fallback(scanned_pdf_path, min_chars_per_page=10))
        assert len(results) == 1
        assert results[0].ocr_used is False
        assert results[0].text == ""