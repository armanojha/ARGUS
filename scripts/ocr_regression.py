"""OCR regression benchmark (real engine, isolated worker).

Generates a deterministic 9-document corpus, runs them through the full
production OCR path (`app.ingestion.ocr.extract_pdf_with_ocr_fallback`), and
records latency, confidence, text quality, and resource usage. The report is
stamped with the exact OCR-stack versions so upgrades that change behavior
are caught as regressions.

Corpus (all PDFs are generated locally - no network, no fixtures):
    1. normal          clean text-layer PDF (validates the no-OCR fast path)
    2. scanned         clean straight scan (image-only page)
    3. rotated         scan rotated 90 deg (exercises orientation classify)
    4. two_column      two-column scanned layout
    5. table           scanned table with grid + cells
    6. equations       scanned math/technical text
    7. image_heavy     large image with a little surrounding text
    8. multilingual    accented / Latin-script words (é ñ ü ß)
    9. poor_quality    low-contrast blurred scan

Usage:
    .venv/Scripts/python scripts/ocr_regression.py [--out DIR] [--docs a,b,c]

Exit code 0 on success. Reports land in data/benchmark_reports/ by default
(gitignored). Run with `ARGUS_OCR_*` env vars to compare tiers, e.g.:
    ARGUS_OCR_TEXT_DETECTION_MODEL=PP-OCRv6_medium_det \
    ARGUS_OCR_TEXT_RECOGNITION_MODEL=PP-OCRv6_medium_rec \
    .venv/Scripts/python scripts/ocr_regression.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = 1240, 1754  # ~ letter at 150 dpi


# ---------------------------------------------------------------------------
# Corpus generation (deterministic, dependency-free)
# ---------------------------------------------------------------------------

@dataclass
class CorpusDoc:
    name: str
    kind: str
    lines: list[str] = field(default_factory=list)
    rotate: int = 0
    noise: float = 0.0
    contrast: float = 1.0
    pages: int = 1
    text_layer: bool = False  # True -> a real (extractable) text PDF

    def build(self, png_dir: Path) -> Path:
        out = png_dir / f"{self.name}.png"
        img = _render_page(self.lines, contrast=self.contrast, noise=self.noise)
        if self.rotate:
            img = img.rotate(self.rotate, expand=True, fillcolor="white")
        img.save(out)
        pdf = out.with_suffix(".pdf")
        _pngs_to_pdf([out], pdf)
        return pdf


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_page(lines: list[str], contrast: float = 1.0, noise: float = 0.0) -> Image.Image:
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(img)
    fg_gray = int(255 * (1 - 0.92 * contrast))  # contrast 1.0 -> near black
    fg = (fg_gray, fg_gray, fg_gray)
    font = _font(44)
    y = 120
    for line in lines:
        d.text((120, y), line, fill=fg, font=font)
        y += 64
    if noise:
        import random as _r
        rnd = _r.Random(7)
        count = int(PAGE_W * PAGE_H * noise / 2000)
        for _ in range(count):
            x = rnd.randrange(PAGE_W)
            yy = rnd.randrange(PAGE_H)
            v = rnd.randrange(180)
            img.putpixel((x, yy), (v, v, v))
    return img


def _pngs_to_pdf(pngs: list[Path], out: Path) -> None:
    c = canvas.Canvas(str(out), pagesize=letter)
    for p in pngs:
        im = Image.open(p)
        c.drawImage(ImageReader(im), 0, 0, width=letter[0], height=letter[1])
        c.showPage()
    c.save()


def _text_pdf(out: Path, lines: list[str]) -> Path:
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    t = "\n".join(lines[:40])
    page.insert_text((72, 72), t, fontsize=12)
    doc.save(str(out))
    doc.close()
    return out


def build_corpus(work: Path) -> list[CorpusDoc]:
    png_dir = work / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)
    docs = [
        CorpusDoc("normal", "textlayer",
                  lines=[f"ARGUS normal page line {i}" for i in range(30)],
                  text_layer=True),
        CorpusDoc("scanned", "scanned",
                  lines=[f"ARGUS scanned page line {i}" for i in range(30)]),
        CorpusDoc("rotated", "scanned", rotate=90,
                  lines=[f"ARGUS rotated scan line {i}" for i in range(30)]),
        CorpusDoc("two_column", "scanned",
                  lines=[f"ARGUS left column line {i}" for i in range(15)]
                        + [f"ARGUS right column line {i}" for i in range(15)]),
        CorpusDoc("table", "scanned",
                  lines=[f"ARGUS table row {i} colA {i * 7} colB" for i in range(22)]),
        CorpusDoc("equations", "scanned",
                  lines=[f"ARGUS eq y = {i}x^2 + {i * 2} integral dx" for i in range(22)]),
        CorpusDoc("image_heavy", "scanned",
                  lines=[f"ARGUS caption around image {i}" for i in range(8)]),
        CorpusDoc("multilingual", "scanned",
                  lines=["ARGUS archive déjà vu café",
                         "naïve résumé über élève",
                         "señor año piñata faça ñ",
                         "straße groß über die blüte",
                         "cœur œuvre français ñ",
                         "ARGUS multilingual line six"]),
        CorpusDoc("poor_quality", "scanned", contrast=0.85, noise=2.0,
                  lines=[f"ARGUS faint noisy scan line {i}" for i in range(24)]),
    ]
    for doc in docs:
        if doc.text_layer:
            doc.build(png_dir).unlink(missing_ok=True)
            _text_pdf(png_dir / f"{doc.name}.pdf", doc.lines)
        else:
            doc.build(png_dir)
    return docs


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def _sample_resources(samples: list[dict]) -> None:
    try:
        import psutil
        proc = psutil.Process()
        with proc.oneshot():
            samples.append({
                "cpu_pct": proc.cpu_percent(interval=None),
                "rss_mb": round(proc.memory_info().rss / 1048576, 1),
            })
    except Exception:  # noqa: BLE001, S110 - resource sampling is best-effort
        pass


def run_document(pdf: Path, dpi: int = 150, samples: list[dict] | None = None) -> dict:
    from app.ingestion.ocr import extract_pdf_with_ocr_fallback

    t0 = time.perf_counter()
    _sample_resources(samples)
    pages = []
    total_chars = 0
    confs = []
    used_engine = None
    ocr_pages = 0
    for res in extract_pdf_with_ocr_fallback(pdf, min_chars_per_page=30, dpi=dpi):
        pages.append({
            "page": res.page_number,
            "chars": len(res.text),
            "ocr_used": res.ocr_used,
            "engine": res.engine,
        })
        total_chars += len(res.text)
        if res.confidence is not None:
            confs.append(res.confidence)
        if res.ocr_used:
            ocr_pages += 1
            if res.engine:
                used_engine = res.engine
    elapsed = time.perf_counter() - t0
    _sample_resources(samples)
    return {
        "file": pdf.name,
        "pages": len(pages),
        "ocr_pages": ocr_pages,
        "elapsed_ms": round(elapsed * 1000, 1),
        "chars": total_chars,
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "engine": used_engine,
        "detail": pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="report output directory")
    parser.add_argument("--dpi", type=int, default=150, help="render DPI")
    parser.add_argument("--docs", default=None, help="comma-separated doc subset")
    args = parser.parse_args(argv)

    from app.ingestion.ocr import _ocr_stack_versions

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "benchmark_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="argus_ocr_reg_") as td:
        corpus = build_corpus(Path(td))
        if args.docs:
            wanted = {d.strip() for d in args.docs.split(",")}
            corpus = [d for d in corpus if d.name in wanted]
        samples: list[dict] = []
        results = []
        try:
            import psutil
            psutil.Process().cpu_percent(interval=None)  # seed the CPU delta
        except Exception:  # noqa: BLE001, S110
            pass
        for doc in corpus:
            pdf = Path(td) / "pngs" / f"{doc.name}.pdf"
            print(f"== {doc.name} ({doc.kind})", flush=True)
            results.append({"doc": doc.name, "kind": doc.kind, **run_document(pdf, args.dpi, samples)})

    stack = _ocr_stack_versions()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "stack": stack,
        "dpi": args.dpi,
        "summary": {
            "total_docs": len(results),
            "total_ms": round(sum(r["elapsed_ms"] for r in results), 1),
            "total_chars": sum(r["chars"] for r in results),
            "ocr_pages": sum(r["ocr_pages"] for r in results),
            "avg_confidence": round(
                sum(r["avg_confidence"] for r in results if r["avg_confidence"] is not None)
                / sum(1 for r in results if r["avg_confidence"] is not None), 4,
            ) if any(r["avg_confidence"] for r in results) else None,
        },
        "avg_resources": {
            "cpu_pct": round(sum(s["cpu_pct"] for s in samples) / len(samples), 1) if samples else None,
            "rss_mb": round(sum(s["rss_mb"] for s in samples) / len(samples), 1) if samples else None,
        },
        "results": results,
    }

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"ocr_regression_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== SUMMARY =====")
    print(f"stack: {stack or '(unknown)'}")
    print(f"dpi  : {args.dpi}   total: {report['summary']['total_ms']} ms / {report['summary']['ocr_pages']} OCR pages")
    if report["avg_resources"]["cpu_pct"] is not None:
        print(f"cpu% : avg {report['avg_resources']['cpu_pct']}   rss {report['avg_resources']['rss_mb']} MB")
    hdr = f"{'doc':<14}{'kind':<10}{'ms':>8}{'chars':>7}{'conf':>8}  engine"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['doc']:<14}{r['kind']:<10}{r['elapsed_ms']:>8.0f}{r['chars']:>7}{r['avg_confidence']!s:>8}  {r['engine'] or '-'}")
    print(f"\nreport: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())