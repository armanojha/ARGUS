"""Tests for the user Knowledge Base ingestion + discovery service.

Covers the deterministic, no-LLM control plane that turns the user corpus
directory into the EvidenceStore via the shared ``IngestionPipeline`` with
idempotent checksum dedup.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from app.evidence.models import SourceType
from app.evidence.store import EvidenceStore
from app.ingestion.knowledge_base import (
    discover_files,
    ingest_knowledge_base,
    kind_of,
    supported_extensions,
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def store(temp_db):
    return EvidenceStore(temp_db)


@pytest.fixture
def kb_root(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text("User note content.\n", encoding="utf-8")
    (root / "sub" / "paper.md").write_text("# Paper\n\nSome markdown.\n", encoding="utf-8")
    (root / "report.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "ignore.bin").write_bytes(b"binary")
    return root


class TestDiscovery:
    def test_supported_extensions_always_include_core_types(self):
        exts = supported_extensions()
        for expected in (".pdf", ".txt", ".md"):
            assert expected in exts

    def test_discover_files_is_recursive_and_sorted(self, kb_root):
        files = discover_files(kb_root)
        paths = [str(p.relative_to(kb_root)) for p in files]
        nested = os.path.join("sub", "paper.md")
        assert "notes.txt" in paths
        assert nested in paths
        assert "report.csv" in paths
        # Unspported types are excluded; ordering is deterministic.
        assert "ignore.bin" not in paths
        assert paths == sorted(paths, key=str.lower)

    def test_discover_files_missing_root_returns_empty(self, kb_root):
        assert discover_files(kb_root / "nope") == []

    def test_kind_of_maps_suffixes(self):
        assert kind_of(Path("a.pdf"))[0] == SourceType.PDF
        assert kind_of(Path("a.md"))[0] == SourceType.TEXT
        assert kind_of(Path("a.txt"))[0] == SourceType.TEXT
        assert kind_of(Path("a.csv"))[0] == SourceType.SPREADSHEET


class TestIngest:
    def test_ingest_knowledge_base_first_run_all_new(self, store, kb_root):
        result = ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)
        assert result.ingested == 3
        assert result.unchanged == 0
        assert result.errors == 0
        assert len(result.documents_ingested) == 3
        assert store.count_documents() == 3
        assert store.count_sources() == 3
        assert store.count_chunks() > 0

    def test_ingest_knowledge_base_idempotent_second_run(self, store, kb_root):
        first = ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)
        assert first.ingested == 3

        second = ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)
        assert second.ingested == 0
        assert second.unchanged == 3
        assert second.errors == 0
        # No duplicate documents after re-sync.
        assert store.count_documents() == 3

    def test_ingest_content_change_is_reingested(self, store, kb_root):
        ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)

        # Modify an existing file -> should be ingested again (new version).
        target = kb_root / "notes.txt"
        before = store.list_documents(limit=100)
        assert before[0].version == 1
        target.write_text("Changed content now.\n", encoding="utf-8")
        result = ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)
        assert result.ingested == 1
        assert result.unchanged == 2
        # A changed file is a NEW document version (not an in-place update) ->
        # the corpus grows by one document row.
        assert store.count_documents() == 3 + 1
        # Changed content is re-derived (checksum differs); unchanged files are skipped.
        assert result.ingested == 1
        assert result.unchanged == 2

    def test_ingest_returns_correct_result_fields(self, store, kb_root):
        result = ingest_knowledge_base(root=kb_root, store=store, rebuild_indexes=False)
        assert result.total == result.ingested + result.unchanged
        assert result.knowledge_base_path == str(Path(kb_root).resolve())
        assert result.started_at != ""
        assert result.duration_s >= 0.0