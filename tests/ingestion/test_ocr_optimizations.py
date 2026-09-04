"""Tests for the OCR optimization layer (caching, escalation, blank-page skip).

These run on the main interpreter (3.14) and never import PaddleOCR.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from app.config import Settings
from app.ingestion.ocr import (
    _is_nearly_blank,
    _ocr_stack_versions,
    _OcrCache,
    _run_ocr,
    _run_paddle_ocr,
)


def _settings(**overrides) -> Settings:
    base = {"multimodal_enabled": True, "multimodal_ocr_enabled": True}
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestOcrCache:
    def test_roundtrip(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        entry = {"text": "hello", "confidence": 0.95}
        cache.set("abc", entry)
        assert cache.get("abc") == entry

    def test_missing_key(self, tmp_path: Path):
        assert _OcrCache(tmp_path).get("nope") is None

    def test_key_changes_with_models(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        s1 = _settings(ocr_text_detection_model="PP-OCRv5_mobile_det")
        s2 = _settings(ocr_text_detection_model="PP-OCRv6_medium_det")
        k1 = cache._key("h", 1, 300, "paddle", ["eng"], s1, {}, strong_tier=False)
        k2 = cache._key("h", 1, 300, "paddle", ["eng"], s2, {}, strong_tier=False)
        assert k1 != k2

    def test_key_changes_with_pdf_and_page(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        s = _settings()
        k1 = cache._key("hashA", 1, 300, "paddle", ["eng"], s, {}, strong_tier=False)
        k2 = cache._key("hashA", 2, 300, "paddle", ["eng"], s, {}, strong_tier=False)
        k3 = cache._key("hashB", 1, 300, "paddle", ["eng"], s, {}, strong_tier=False)
        assert len({k1, k2, k3}) == 3

    def test_key_changes_with_stack_versions(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        s = _settings()
        k1 = cache._key("h", 1, 300, "paddle", ["eng"], s, {"paddleocr": "3.7.0"}, strong_tier=False)
        k2 = cache._key("h", 1, 300, "paddle", ["eng"], s, {"paddleocr": "3.8.0"}, strong_tier=False)
        assert k1 != k2

    def test_corrupt_entry_returns_none(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        assert cache.get("bad") is None

    def test_clear(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        cache.set("a", {"text": "x"})
        cache.set("b", {"text": "y"})
        assert cache.clear() == 2
        assert cache.get("a") is None

    def test_prune_bounds_growth(self, tmp_path: Path):
        cache = _OcrCache(tmp_path)
        cache.MAX_ENTRIES = 4
        for i in range(8):
            cache.set(f"key-{i}", {"text": str(i)})
        kept = {p.name for p in tmp_path.glob("*.json")}
        assert len(kept) == 4
        assert "key-0.json" not in kept and "key-1.json" not in kept
        assert "key-7.json" in kept


# ---------------------------------------------------------------------------
# Blank-page detection
# ---------------------------------------------------------------------------

class TestBlankPage:
    def test_white_image_is_blank(self):
        img = Image.new("RGB", (100, 100), "white")
        assert _is_nearly_blank(img) is True

    def test_content_image_not_blank(self):
        img = Image.new("RGB", (100, 100), "white")
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.rectangle((10, 10, 60, 40), fill="black")
        assert _is_nearly_blank(img) is False


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def _runner(self):
        runner = MagicMock()
        runner.run_ocr.side_effect = [
            ("text", 0.30),   # fast tier: low confidence
            ("strong text", 0.99),  # strong tier
        ]
        return runner

    def test_escalates_on_low_confidence(self):
        settings = _settings(ocr_escalate_on_low_confidence=0.8)
        runner = self._runner()
        with (
            patch("app.ingestion.ocr._get_runner", return_value=runner),
            patch("app.ingestion.ocr.get_settings", return_value=settings),
        ):
            text, _conf, engine = _run_paddle_ocr(Image.new("RGB", (8, 8)), ["eng"])
        assert text == "strong text"
        assert engine == "paddle"
        assert runner.run_ocr.call_count == 2
        assert runner.run_ocr.call_args_list[0].kwargs["tier"] == "fast"
        assert runner.run_ocr.call_args_list[1].kwargs["tier"] == "strong"

    def test_no_escalation_when_confidence_ok(self):
        settings = _settings(ocr_escalate_on_low_confidence=0.8)
        runner = MagicMock()
        runner.run_ocr.return_value = ("good", 0.95)
        with (
            patch("app.ingestion.ocr._get_runner", return_value=runner),
            patch("app.ingestion.ocr.get_settings", return_value=settings),
        ):
            text, _conf, _ = _run_paddle_ocr(Image.new("RGB", (8, 8)), ["eng"])
        assert runner.run_ocr.call_count == 1
        assert text == "good"

    def test_no_escalation_when_disabled(self):
        settings = _settings(ocr_escalate_on_low_confidence=0.0)
        runner = MagicMock()
        runner.run_ocr.return_value = ("weak", 0.10)
        with (
            patch("app.ingestion.ocr._get_runner", return_value=runner),
            patch("app.ingestion.ocr.get_settings", return_value=settings),
        ):
            _run_paddle_ocr(Image.new("RGB", (8, 8)), ["eng"])
        assert runner.run_ocr.call_count == 1

    def test_no_escalation_when_tiers_identical(self):
        settings = _settings(
            ocr_escalate_on_low_confidence=0.9,
            ocr_text_detection_model="PP-OCRv6_medium_det",
            ocr_text_recognition_model="PP-OCRv6_medium_rec",
            ocr_text_detection_model_strong="PP-OCRv6_medium_det",
            ocr_text_recognition_model_strong="PP-OCRv6_medium_rec",
        )
        runner = MagicMock()
        runner.run_ocr.return_value = ("weak", 0.10)
        with (
            patch("app.ingestion.ocr._get_runner", return_value=runner),
            patch("app.ingestion.ocr.get_settings", return_value=settings),
        ):
            _run_paddle_ocr(Image.new("RGB", (8, 8)), ["eng"])
        assert runner.run_ocr.call_count == 1

    def test_dispatch_returns_escalated_engine(self):
        settings = _settings(ocr_escalate_on_low_confidence=0.9)
        runner = self._runner()
        with (
            patch("app.ingestion.ocr._get_runner", return_value=runner),
            patch("app.ingestion.ocr.get_settings", return_value=settings),
            patch("app.ingestion.ocr._tesseract_available", return_value=False),
        ):
            text, _conf, engine = _run_ocr(Image.new("RGB", (8, 8)), ["eng"], "paddle")
        assert (text, engine) == ("strong text", "paddle")


# ---------------------------------------------------------------------------
# Stack versions (filesystem-based, no import of PaddleOCR)
# ---------------------------------------------------------------------------

class TestStackVersions:
    def test_returns_present_or_empty(self):
        versions = _ocr_stack_versions()
        assert isinstance(versions, dict)
        # In this repo the isolated 3.11 venv exists with a pinned stack.
        assert versions.get("paddlepaddle") is None or versions["paddlepaddle"]

    def test_missing_python_returns_empty(self):
        with patch("app.ingestion.ocr._resolve_runner_python", return_value=None):
            assert _ocr_stack_versions() == {}