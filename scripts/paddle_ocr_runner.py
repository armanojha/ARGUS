"""PaddleOCR worker (runs on a Python 3.11+ venv).

PaddleOCR is built on PaddlePaddle, which has no wheels for Python 3.14
(the interpreter the main ARGUS app runs on). To use PaddleOCR without
disrupting the running app, this worker runs as a *persistent subprocess*
in a dedicated Python 3.11 venv that has `paddlepaddle` + `paddleocr`
installed. The main app (`app.ingestion.ocr.PaddleOCRRunner`) talks to it
over JSON lines on stdin/stdout.

Protocol
--------
Requests are single-line JSON objects on stdin:
    {"path": "<image file path>", "lang": "<lang>", "tier": "fast|strong"}

Responses are single-line JSON objects on *stderr*:
    {"ok": true, "text": "...", "confidence": 0.99,
     "language": "en", "lines": [...] , "error": null}

The worker redirects its own OS-level stdout to a null sink, because
PaddlePaddle/PaddleX writes diagnostic lines (e.g. "ReduceMeanCheckIf..." )
directly to file descriptor 1, which would otherwise corrupt a stdout-based
JSON protocol. All protocol traffic therefore travels over *stderr*; the
parent reads responses from the worker's stderr pipe.

The worker keeps running and reuses its loaded PaddleOCR models for every
request (model download/load happens once at startup). It exits cleanly
when stdin is closed (EOF).

Configuration (via environment variables, set by the parent process)
--------------------------------------------------------------------
    OCR_DET_MODEL             detection model name (default "PP-OCRv5_mobile_det";
                              empty -> PaddleOCR per-language default)
OCR_REC_MODEL              recognition model name (default "" -> auto-resolve;
                              English maps to the fast "en_PP-OCRv5_mobile_rec")
    OCR_DET_MODEL_STRONG       strong-tier detection model (default "PP-OCRv6_medium_det")
    OCR_REC_MODEL_STRONG       strong-tier recognition model (default "PP-OCRv6_medium_rec")
    OCR_DOC_ORIENT            "1"/"0" page orientation classify (default 1)
    OCR_DOC_UNWARP            "1"/"0" scan unwarping (default 1)
    OCR_TEXTLINE_ORIENT       "1"/"0" textline orientation (default 1)
    OCR_DEVICE                inference device (default "cpu")
    OCR_REC_SCORE_THRESH      min line confidence (default 0.4)
    OCR_DET_LIMIT_SIDE_LEN    max det input side in px (default 0 = framework)

Note: for non-English languages the fast mobile default only applies when
model names are explicitly provided (empty defaults keep PaddleOCR's own
language resolution). Benchmarking showed the language-default English
models resolve to PP-OCRv6_medium, ~17x slower than PP-OCRv5 mobile with
no measured accuracy loss.

Model cache location: PaddleX stores downloaded models under
`$PADDLE_PDX_CACHE_HOME` (default `~/.paddlex`); the parent process sets
that env var to keep models out of the global home directory.

The startup "ready" handshake also reports installed OCR-stack versions
(`deps`: paddlepaddle/paddleocr/paddlex) so the parent can invalidate any
cached OCR results when the stack is upgraded.

This module is intentionally self-contained (only stdlib + paddleocr,
PIL, numpy) so it can run inside the isolated OCR venv without importing
any ARGUS app code.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Protocol stream: redirect OS-level stdout (fd 1) to a null sink and emit
# JSON responses on *stderr*. Paddle internals spam fd 1; keeping our protocol
# on stderr makes it immune to that noise.
# ---------------------------------------------------------------------------
_NULL_FD = None
_PROTO = None


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _init_streams() -> None:
    global _NULL_FD, _PROTO
    if _NULL_FD is not None:
        return
    _NULL_FD = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_NULL_FD, 1)  # redirect fd 1 (stdout) -> devnull
    # Python-level stdout/tqdm that still uses sys.stdout must also go nowhere.
    sys.stdout = io.StringIO()
    _PROTO = os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8")


def _emit(payload: dict) -> None:
    if _PROTO is None:
        return
    _PROTO.write(json.dumps(payload) + "\n")
    _PROTO.flush()


# ---------------------------------------------------------------------------
# Module-level singleton so the heavy models are loaded exactly once per tier.
# Keyed by (lang, tier); "fast" and "strong" hold separate loaded models.
# ---------------------------------------------------------------------------
_OCR = {}


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _get_ocr(lang: str, tier: str = "fast"):
    lang = lang or "en"
    tier = "strong" if tier == "strong" else "fast"
    cached = _OCR.get((lang, tier))
    if cached is None:
        with contextlib.redirect_stdout(io.StringIO()):
            from paddleocr import PaddleOCR

            if tier == "strong":
                # Max-accuracy tier, only entered on low-confidence escalation.
                det_model = _env_str("OCR_DET_MODEL_STRONG", "PP-OCRv6_medium_det")
                rec_model = _env_str("OCR_REC_MODEL_STRONG", "PP-OCRv6_medium_rec")
            else:
                # Fast tier: PP-OCRv5 mobile for English (the language-default
                # English models resolve to PP-OCRv6 medium, ~17x slower on CPU
                # with no measured accuracy gain). Other languages keep
                # PaddleOCR's own model resolution unless explicitly overridden.
                det_model = _env_str("OCR_DET_MODEL")
                rec_model = _env_str("OCR_REC_MODEL")
                if lang == "en":
                    if not det_model:
                        det_model = "PP-OCRv5_mobile_det"
                    if not rec_model:
                        rec_model = "en_PP-OCRv5_mobile_rec"

            kwargs = {
                "lang": lang,
                "device": _env_str("OCR_DEVICE", "cpu") or "cpu",
                "text_rec_score_thresh": _env_float("OCR_REC_SCORE_THRESH", 0.4),
                # Preprocessing (each adds model load + inference time; the
                # unwarping model alone costs ~10 s at startup).
                "use_doc_orientation_classify": _env_flag("OCR_DOC_ORIENT", True),
                "use_doc_unwarping": _env_flag("OCR_DOC_UNWARP", True),
                "use_textline_orientation": _env_flag("OCR_TEXTLINE_ORIENT", True),
            }
            if det_model:
                kwargs["text_detection_model_name"] = det_model
            if rec_model:
                kwargs["text_recognition_model_name"] = rec_model
            det_limit = _env_int("OCR_DET_LIMIT_SIDE_LEN", 0)
            if det_limit > 0:
                kwargs["text_det_limit_side_len"] = det_limit

            _OCR[(lang, tier)] = {"lang": lang, "ocr": PaddleOCR(**kwargs)}
    return _OCR[(lang, tier)]["ocr"]


def _run_ocr(image_path: str, lang: str, tier: str = "fast") -> dict:
    ocr = _get_ocr(lang, tier)
    result = ocr.predict(str(image_path))
    if not result:
        return {"ok": False, "error": "no prediction returned", "text": "",
                "confidence": None, "lines": []}

    res = result[0]
    rec_texts = res.get("rec_texts") or []
    rec_scores = res.get("rec_scores") or []
    rec_polys = res.get("rec_polys") or []

    lines = []
    for i, raw in enumerate(rec_texts):
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        score = float(rec_scores[i]) if i < len(rec_scores) else None
        poly = None
        if i < len(rec_polys) and rec_polys[i] is not None:
            poly = [[float(p[0]), float(p[1])] for p in rec_polys[i]]
        lines.append({"text": text, "confidence": score, "box": poly})

    # Sort top-to-bottom, then left-to-right for natural reading order.
    lines.sort(key=lambda l: (_centroid_y(l["box"]), _centroid_x(l["box"])))

    text = "\n".join(l["text"] for l in lines)
    confs = [l["confidence"] for l in lines if l["confidence"] is not None]
    avg_conf = (sum(confs) / len(confs)) if confs else None

    return {
        "ok": True,
        "text": text,
        "confidence": avg_conf,
        "language": lang or "en",
        "lines": lines,
        "error": None,
    }


def _centroid_x(box):
    if not box:
        return 0.0
    return sum(p[0] for p in box) / len(box)


def _centroid_y(box):
    if not box:
        return 0.0
    return sum(p[1] for p in box) / len(box)


def _deps() -> dict:
    """Report installed OCR-stack versions (used for cache invalidation)."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return {}
    return {
        "paddlepaddle": metadata.version("paddlepaddle"),
        "paddleocr": metadata.version("paddleocr"),
        "paddlex": metadata.version("paddlex"),
    }


def _main() -> int:
    _init_streams()  # silence Paddle's stdout noise; protocol goes to stderr
    for attempt in range(4):
        try:
            _get_ocr("en")  # warm the models once so the first request is fast
            break
        except Exception as e:  # noqa: BLE001 - retry on slow first model download
            # Model download/init can be slow on first run; retry briefly.
            _emit({"event": "warn", "message": "paddle warmup retry",
                   "error": f"{type(e).__name__}: {e}", "attempt": attempt + 1})
    _emit({"event": "ready", "version": "paddleocr", "deps": _deps()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"ok": False, "error": "invalid json"})
            continue

        image_path = req.get("path")
        lang = req.get("lang") or "en"
        tier = req.get("tier") or "fast"
        cleanup = None

        if req.get("data_b64"):
            try:
                raw = base64.b64decode(req["data_b64"])
                if not image_path:
                    import tempfile
                    fd, image_path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    cleanup = image_path
                Path(image_path).write_bytes(raw)
            except Exception as e:  # noqa: BLE001
                _emit({"ok": False, "error": f"decode: {e}"})
                continue

        if not image_path or not Path(image_path).exists():
            _emit({"ok": False, "error": "missing image path", "text": "",
                   "confidence": None, "lines": []})
            continue

        try:
            _emit(_run_ocr(image_path, lang, tier))
        except Exception as e:  # noqa: BLE001
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}",
                   "text": "", "confidence": None, "lines": []})
        finally:
            if cleanup:
                try:
                    Path(cleanup).unlink(missing_ok=True)
                except OSError:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
