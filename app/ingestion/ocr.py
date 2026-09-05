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

import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
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
# OCR result cache (content-hash keyed; auto-invalidated)
# ---------------------------------------------------------------------------

class _OcrCache:
    """File-backed per-page OCR cache.

    Keys are content hashes derived from the PDF bytes + page number + every
    setting/version that affects OCR output, so stale entries are structurally
    impossible: changing the document, model tier, preprocessing, threshold,
    or the installed OCR stack all produce a different key.
    """

    SCHEMA = 1
    MAX_ENTRIES = 2000

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, pdf_hash: str, page: int, dpi: int, engine: str,
             languages: list[str], settings, deps: dict, strong_tier: bool) -> str:
        payload = {
            "v": self.SCHEMA,
            "pdf": pdf_hash,
            "page": page,
            "dpi": dpi,
            "engine": engine,
            "lang": "+".join(languages),
            "det": getattr(settings, "ocr_text_detection_model", "") or "",
            "rec": getattr(settings, "ocr_text_recognition_model", "") or "",
            "det_strong": getattr(settings, "ocr_text_detection_model_strong", "") or "",
            "rec_strong": getattr(settings, "ocr_text_recognition_model_strong", "") or "",
            "orient": bool(getattr(settings, "ocr_doc_orientation_classify", True)),
            "unwarp": bool(getattr(settings, "ocr_doc_unwarping", True)),
            "textline": bool(getattr(settings, "ocr_textline_orientation", True)),
            "rec_thresh": getattr(settings, "ocr_rec_score_thresh", 0.4),
            "det_limit": int(getattr(settings, "ocr_det_limit_side_len", 0) or 0),
            "esc": getattr(settings, "ocr_escalate_on_low_confidence", 0.0),
            "strong": bool(strong_tier),
            "deps": deps,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> dict | None:
        p = self._dir / f"{key}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, key: str, entry: dict) -> None:
        p = self._dir / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        self._prune()

    def _prune(self) -> None:
        """Bound cache growth: drop the oldest entries beyond the cap."""
        entries = []
        for p in self._dir.glob("*.json"):
            try:
                entries.append((p.stat().st_mtime, p))
            except OSError:
                continue
        entries.sort()
        if len(entries) <= self.MAX_ENTRIES:
            return
        for _, p in entries[:-self.MAX_ENTRIES]:
            try:
                p.unlink()
            except OSError:
                pass

    def clear(self) -> int:
        removed = 0
        for p in self._dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        return removed


def _pdf_content_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ocr_stack_versions() -> dict:
    """Installed OCR-stack versions read from the venv's dist-info dirs.

    Filesystem-based (no worker import), so cache keys are stable across
    processes and automatically change when the OCR stack is upgraded.
    """
    python = _resolve_runner_python()
    if python is None:
        return {}
    site = Path(python).resolve().parent.parent / "Lib" / "site-packages"
    versions: dict = {}
    for dist_name in ("paddlepaddle", "paddleocr", "paddlex"):
        try:
            dists = list(site.glob(f"{dist_name}-*.dist-info"))
        except OSError:
            dists = []
        if dists:
            versions[dist_name] = dists[0].name[len(dist_name) + 1:-len(".dist-info")]
    return versions


def _get_ocr_cache() -> _OcrCache | None:
    settings = get_settings()
    if not getattr(settings, "ocr_cache_enabled", True):
        return None
    configured = getattr(settings, "ocr_cache_dir", None)
    directory = Path(configured) if configured else REPO_ROOT / ".cache" / "ocr"
    return _OcrCache(directory)


def _is_nearly_blank(image: Image.Image, ink_fraction: float = 0.0015) -> bool:
    """True when the rendered page is effectively blank (no ink).

    A cheap downscaled grayscale check skips OCR (and its several-second cost)
    for empty/scanned-blank pages.
    """
    g = image.convert("L")
    g.thumbnail((256, 256))
    raw = g.tobytes()
    if not raw:
        return True
    dark = sum(1 for b in raw if b < 128)
    return dark / len(raw) < ink_fraction


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
        self._queue: queue.Queue[dict | None] | None = None
        self._drain_thread: threading.Thread | None = None
        self._deps: dict = {}  # OCR-stack versions reported by the worker
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return _paddle_available()

    @property
    def deps(self) -> dict:
        """Installed OCR-stack versions (paddlepaddle/paddleocr/paddlex)."""
        return dict(self._deps)

    def _kill_proc(self) -> None:
        """Best-effort terminate of the current worker subprocess."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("paddle_ocr_close_stdin_error", error=str(e))
        try:
            proc.wait(timeout=5)
        except Exception as e:  # noqa: BLE001
            logger.debug("paddle_ocr_wait_timeout", error=str(e))
            try:
                proc.kill()
            except Exception as e2:  # noqa: BLE001
                logger.debug("paddle_ocr_kill_failed", error=str(e2))

    def _start(self) -> None:
        python = _resolve_runner_python()
        script = _resolve_runner_script()
        if python is None or script is None:
            raise FileNotFoundError("PaddleOCR runner interpreter/script not found")
        settings = get_settings()
        env = dict(os.environ)
        model_dir = getattr(settings, "ocr_model_cache_dir", None)
        if model_dir:
            env["PADDLE_PDX_CACHE_HOME"] = str(model_dir)
        env.update({
            "OCR_DET_MODEL": getattr(settings, "ocr_text_detection_model", "") or "",
            "OCR_REC_MODEL": getattr(settings, "ocr_text_recognition_model", "") or "",
            "OCR_DET_MODEL_STRONG": getattr(settings, "ocr_text_detection_model_strong", "PP-OCRv6_medium_det") or "PP-OCRv6_medium_det",
            "OCR_REC_MODEL_STRONG": getattr(settings, "ocr_text_recognition_model_strong", "PP-OCRv6_medium_rec") or "PP-OCRv6_medium_rec",
            "OCR_DOC_ORIENT": "1" if getattr(settings, "ocr_doc_orientation_classify", True) else "0",
            "OCR_DOC_UNWARP": "1" if getattr(settings, "ocr_doc_unwarping", True) else "0",
            "OCR_TEXTLINE_ORIENT": "1" if getattr(settings, "ocr_textline_orientation", True) else "0",
            "OCR_DEVICE": getattr(settings, "ocr_device", "cpu") or "cpu",
            "OCR_REC_SCORE_THRESH": str(getattr(settings, "ocr_rec_score_thresh", 0.4)),
            "OCR_DET_LIMIT_SIDE_LEN": str(int(getattr(settings, "ocr_det_limit_side_len", 0) or 0)),
        })
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
            env=env,
        )
        # Drain the worker's stderr continuously so Paddle's diagnostic noise
        # can never fill the pipe and deadlock the exchange (the worker also
        # writes protocol JSON on stderr). Non-JSON noise lines are discarded.
        self._queue = queue.Queue()
        self._drain_thread = threading.Thread(target=self._drain, daemon=True)
        self._drain_thread.start()
        # Wait for the worker's readiness handshake.
        ready = self._read_json_line(timeout=180.0)
        if not ready or ready.get("event") != "ready":
            self._kill_proc()
            self._proc = None
            failed = (ready or {}).get("error")
            raise RuntimeError("PaddleOCR worker did not report ready"
                               + (f": {failed}" if failed else ""))
        self._deps = dict(ready.get("deps") or {})

    def _drain(self) -> None:
        assert self._queue is not None and self._proc is not None and self._proc.stderr is not None
        # Capture the queue and stream at thread start: a worker restart
        # replaces self._queue / self._proc, and the dying thread must keep
        # pushing to the queue it belongs to, never the new one.
        q = self._queue
        stderr = self._proc.stderr
        for raw in stderr:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # diagnostic noise from Paddle; discard
            if isinstance(obj, dict):
                q.put(obj)
        # EOF: worker exited.
        q.put(None)

    def _read_json_line(self, timeout: float = 180.0) -> dict | None:
        """Read the next JSON object from the worker's drained stream."""
        assert self._queue is not None
        try:
            return self._queue.get(timeout=timeout)
        except Exception:  # noqa: BLE001 (queue.Empty / undefined)
            return None

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None
        self._start()

    def run_ocr(self, image: Image.Image, language: str = "en", tier: str = "fast") -> tuple[str, float | None]:
        """Run PaddleOCR on a PIL image. Returns (text, avg_confidence).

        ``tier`` selects the worker's model set: "fast" (mobile, the default)
        or "strong" (max-accuracy, used on low-confidence escalation).

        The JSON-lines exchange is serialized with a lock: the worker is
        single-threaded and concurrent requests would interleave on stdin.
        """
        if not self.available:
            return "", None
        settings = get_settings()
        timeout = float(getattr(settings, "ocr_runner_timeout", 120.0))

        # NOTE: on Windows a NamedTemporaryFile keeps the handle open/locked,
        # so the worker subprocess could not write to it. Use mkstemp (handle
        # released immediately) and clean up explicitly.
        fd, tf_name = tempfile.mkstemp(suffix=".png", prefix="argus_ocr_")
        os.close(fd)
        try:
            image.save(tf_name, format="PNG")
        except Exception:
            os.unlink(tf_name)
            raise
        try:
            req = {"path": tf_name, "lang": language or "en", "tier": tier}
            with self._lock:
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

        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        resp = self._read_json_line(timeout=timeout)
        if resp is None:
            raise BrokenPipeError("PaddleOCR worker closed unexpectedly")
        if not resp.get("ok"):
            logger.warning("paddle_ocr_error", error=resp.get("error"))
            return "", None
        return resp.get("text", ""), resp.get("confidence")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._kill_proc()
        self._proc = None


_runner_singleton: PaddleOCRRunner | None = None
_runner_singleton_lock = threading.Lock()


def _get_runner() -> PaddleOCRRunner | None:
    global _runner_singleton
    if not _paddle_available():
        return None
    if _runner_singleton is None:
        with _runner_singleton_lock:
            if _runner_singleton is None:
                _runner_singleton = PaddleOCRRunner()
    return _runner_singleton


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


def _run_paddle_ocr(image: Image.Image, languages: list[str]) -> tuple[str, float | None, str | None]:
    """Run PaddleOCR on a PIL Image via the isolated worker.

    Runs the fast tier first; if the average confidence falls below the
    configured escalation threshold and the strong tier is a *different* model
    set, the page is re-run with the strong (max-accuracy) tier.
    """
    runner = _get_runner()
    if runner is None:
        logger.warning("paddle_ocr_runner_unavailable")
        return "", None, None
    raw_lang = languages[0] if languages else "en"
    # Map ARGUS-style language codes to PaddleOCR-friendly names.
    lang = {"eng": "en", "en": "en"}.get(raw_lang, raw_lang)
    try:
        text, confidence = runner.run_ocr(image, lang or "en", tier="fast")
        if not text.strip():
            return text, confidence, "paddle"
        settings = get_settings()
        threshold = float(getattr(settings, "ocr_escalate_on_low_confidence", 0.0) or 0.0)
        strong_det = getattr(settings, "ocr_text_detection_model_strong", "PP-OCRv6_medium_det") or ""
        strong_rec = getattr(settings, "ocr_text_recognition_model_strong", "PP-OCRv6_medium_rec") or ""
        fast_det = getattr(settings, "ocr_text_detection_model", "") or ""
        fast_rec = getattr(settings, "ocr_text_recognition_model", "") or ""
        escalated = (
            threshold > 0.0
            and confidence is not None
            and confidence < threshold
            and (strong_det, strong_rec) != (fast_det, fast_rec)
        )
        if escalated:
            logger.info("paddle_ocr_escalating_to_strong", confidence=confidence, threshold=threshold)
            text, confidence = runner.run_ocr(image, lang or "en", tier="strong")
        return text, confidence, "paddle"
    except Exception as e:  # noqa: BLE001
        logger.warning("paddle_ocr_failed", error=str(e))
        return "", None, None


def _run_ocr(image: Image.Image, languages: list[str], engine: str | None) -> tuple[str, float | None, str | None]:
    """Dispatch OCR to the resolved engine. Returns (text, confidence, engine)."""
    if engine == "paddle":
        text, conf, _ = _run_paddle_ocr(image, languages)
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
    cache = _get_ocr_cache()
    pdf_hash: str | None = None
    stack_versions = _ocr_stack_versions()

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
                continue

            # No usable text layer - apply OCR (check the content-hash cache first)
            logger.info("applying_ocr", page=page_num, path=str(pdf_path), engine=engine)

            key = None
            if cache is not None:
                if pdf_hash is None:
                    pdf_hash = _pdf_content_hash(pdf_path)
                key = cache._key(pdf_hash, page_num, dpi, engine, languages,
                                 settings, stack_versions, strong_tier=False)
                hit = cache.get(key)
                if hit is not None:
                    logger.info("ocr_cache_hit", page=page_num, path=str(pdf_path))
                    yield OCRResult(
                        page_number=page_num,
                        text=hit.get("text", ""),
                        language="+".join(languages),
                        confidence=hit.get("confidence"),
                        ocr_used=True,
                        engine=hit.get("engine") or engine,
                        metadata=hit.get("metadata"),
                    )
                    continue

            try:
                image = _pdf_page_to_image(page, dpi=dpi)
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.warning("ocr_page_failed", page=page_num, error=str(e))
                # Fall back to text layer even if minimal
                yield OCRResult(
                    page_number=page_num,
                    text=text_layer,
                    ocr_used=False,
                )
                continue

            if _is_nearly_blank(image):
                # Blank rendered page: nothing to OCR.
                yield OCRResult(
                    page_number=page_num,
                    text=text_layer,
                    ocr_used=False,
                )
                continue

            try:
                ocr_text, confidence, used_engine = _run_ocr(image, languages, engine)
            except (OSError, ValueError, RuntimeError, KeyError) as e:
                logger.warning("ocr_page_failed", page=page_num, error=str(e))
                # Fall back to text layer even if minimal
                yield OCRResult(
                    page_number=page_num,
                    text=text_layer,
                    ocr_used=False,
                )
                continue

            if ocr_text.strip():
                result = OCRResult(
                    page_number=page_num,
                    text=ocr_text,
                    language="+".join(languages),
                    confidence=confidence,
                    ocr_used=True,
                    engine=used_engine,
                )
                if cache is not None and key is not None and used_engine == "paddle":
                    cache.set(key, {
                        "text": result.text,
                        "confidence": result.confidence,
                        "engine": result.engine,
                        "metadata": result.metadata,
                    })
                yield result
            else:
                # OCR produced nothing, fall back to empty text layer
                yield OCRResult(
                    page_number=page_num,
                    text=text_layer,
                    ocr_used=False,
                )


def _parse_ocr_text_with_headings(text: str) -> list[tuple[str, str | None]]:
    """Parse OCR text that may contain heading markers (# ## ###).

    Returns list of (text, section_path_or_None) tuples.
    Heading markers are stripped and used to build hierarchical section paths.
    """
    result: list[tuple[str, str | None]] = []
    # Heading stack: list of (title, level) for hierarchical path building
    heading_stack: list[tuple[str, int]] = []
    heading_prefixes = {3: "### ", 2: "## ", 1: "# "}

    def _build_path(level: int) -> str | None:
        """Build hierarchical path, trimming stack to headings above current level."""
        while heading_stack and heading_stack[-1][1] >= level:
            heading_stack.pop()
        parts = [title for title, _ in heading_stack]
        return " > ".join(parts) if parts else None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Check for heading markers (longest prefix first to avoid ## matching #)
        matched = False
        for level in (3, 2, 1):
            prefix = heading_prefixes[level]
            if stripped.startswith(prefix):
                title = stripped[len(prefix):].strip()
                _build_path(level)  # trims heading_stack as side effect
                heading_stack.append((title, level))
                # Don't emit heading lines as segments — they're metadata
                matched = True
                break

        if not matched:
            current_path = " > ".join(t for t, _ in heading_stack) if heading_stack else None
            result.append((stripped, current_path))

    return result


def extract_pdf_segments_with_ocr(
    pdf_path: Path,
    min_chars_per_page: int = 50,
    dpi: int = 300,
) -> list[TextSegment]:
    """Extract text segments from PDF with OCR fallback.

    Returns TextSegment objects suitable for the existing chunking pipeline.
    OCR is only applied to pages without usable text layers.
    Heading markers from PaddleOCR bounding box analysis are parsed into
    section_path for structure-aware chunking.
    """
    segments = []

    for ocr_result in extract_pdf_with_ocr_fallback(pdf_path, min_chars_per_page, dpi):
        if not ocr_result.text.strip():
            continue

        # Parse heading markers from OCR text
        parsed = _parse_ocr_text_with_headings(ocr_result.text)
        if not parsed:
            continue

        # Group consecutive lines under the same section into segments
        current_section_text: list[str] = []
        current_section_title: str | None = None

        for line_text, section_title in parsed:
            if section_title != current_section_title and current_section_text:
                # Section changed — flush previous section as a segment
                combined = "\n".join(current_section_text)
                if combined.strip():
                    segments.append(TextSegment(
                        text=combined.strip(),
                        page_start=ocr_result.page_number,
                        page_end=ocr_result.page_number,
                        char_start=0,
                        char_end=len(combined),
                        section_path=current_section_title,
                    ))
                current_section_text = []
            current_section_title = section_title
            current_section_text.append(line_text)

        # Flush remaining lines
        if current_section_text:
            combined = "\n".join(current_section_text)
            if combined.strip():
                segments.append(TextSegment(
                    text=combined.strip(),
                    page_start=ocr_result.page_number,
                    page_end=ocr_result.page_number,
                    char_start=0,
                    char_end=len(combined),
                    section_path=current_section_title,
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
