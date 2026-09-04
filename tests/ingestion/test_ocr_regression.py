"""Fast tests for the OCR regression benchmark helpers (no engine involved)."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pymupdf


def _load_module():
    return importlib.import_module("scripts.ocr_regression")


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hack_page_hash(pdf: Path) -> str:
    with pymupdf.open(str(pdf)) as doc:
        page = doc[0]
        return hashlib.sha256(page.get_pixmap().samples).hexdigest()


class TestCorpus:
    def test_builds_nine_docs_deterministically(self, tmp_path: Path):
        mod = _load_module()
        c1 = mod.build_corpus(Path(tmp_path))
        assert len(c1) == 9
        assert {d.name for d in c1} == {
            "normal", "scanned", "rotated", "two_column", "table",
            "equations", "image_heavy", "multilingual", "poor_quality",
        }
        some_dir = Path(tmp_path) / "pngs"
        hashes = {d.name: _hack_page_hash(some_dir / f"{d.name}.pdf") for d in c1}
        c2 = mod.build_corpus(Path(tmp_path))
        hashes2 = {d.name: _hack_page_hash(some_dir / f"{d.name}.pdf") for d in c2}
        assert hashes == hashes2

    def test_text_layer_doc_has_extractable_text(self, tmp_path: Path):
        mod = _load_module()
        mod.build_corpus(Path(tmp_path))
        pdf = Path(tmp_path) / "pngs" / "normal.pdf"
        with pymupdf.open(str(pdf)) as doc:
            assert len(doc) == 1
            text = doc[0].get_text()
        assert "ARGUS normal page line" in text

    def test_scanned_doc_has_no_text_layer(self, tmp_path: Path):
        mod = _load_module()
        mod.build_corpus(Path(tmp_path))
        pdf = Path(tmp_path) / "pngs" / "scanned.pdf"
        with pymupdf.open(str(pdf)) as doc:
            text = doc[0].get_text()
        assert text.strip() == ""

    def test_rotated_page_produces_landscape_size(self, tmp_path: Path):
        """90-degree rotation must change the rendered aspect ratio."""
        mod = _load_module()
        doc = mod.CorpusDoc("t", "scanned", rotate=90,
                            lines=["ARGUS x" for _ in range(10)])
        pdf = doc.build(Path(tmp_path))
        im = mod.Image.open(pdf if False else Path(tmp_path) / "t.png")
        w, h = im.size
        assert h < w  # landscape after rotating the portrait render