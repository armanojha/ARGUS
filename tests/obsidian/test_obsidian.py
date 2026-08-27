"""Lightweight sanity tests for Obsidian Ingestion (Phase 05).

Not exhaustive by design (see vault Phase 05 testing policy: deferred
to a later stabilization pass). Covers the phase's own acceptance
criteria: vault scanning, parsing, incremental sync, and write-back.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.evidence.store import EvidenceStore
from app.integrations.obsidian.models import (
    NoteType,
    ObsidianFrontmatter,
    ObsidianSection,
    ObsidianWikilink,
    ResearchCaptureNote,
)
from app.integrations.obsidian.parser import (
    extract_callouts,
    extract_code_blocks,
    extract_sections,
    extract_tags,
    extract_wikilinks,
    parse_frontmatter,
    parse_obsidian_note,
)
from app.integrations.obsidian.scanner import VaultScanner
from app.integrations.obsidian.sync import SyncManager
from app.integrations.obsidian.writer import ObsidianWriter


@pytest.fixture
def temp_vault():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        # Create a test vault structure
        (vault / "90_ARGUS").mkdir(parents=True, exist_ok=True)
        (vault / ".obsidian").mkdir(parents=True, exist_ok=True)
        yield vault


@pytest.fixture
def sample_note_content() -> str:
    return """---
title: "Test Note"
tags: [test, example]
aliases: [Test Alias]
date: 2024-01-15
---

# Main Heading

This is the main content of the note.

## Subsection

More content here with a [[wikilink]] and a #tag.

> [!note]
> This is a callout.

```python
def hello():
    print("Hello, world!")
```

## Another Section

Final content with [[another link|alias]].
"""


@pytest.fixture
def populated_vault(temp_vault: Path, sample_note_content: str) -> Path:
    note_path = temp_vault / "test_note.md"
    note_path.write_text(sample_note_content, encoding="utf-8")
    return temp_vault


class TestObsidianModels:
    def test_frontmatter_creation(self):
        fm = ObsidianFrontmatter(
            title="Test",
            tags=["tag1", "tag2"],
            aliases=["alias1"],
        )
        assert fm.title == "Test"
        assert fm.tags == ["tag1", "tag2"]
        assert fm.aliases == ["alias1"]

    def test_section_creation(self):
        section = ObsidianSection(
            heading="Test Heading",
            level=2,
            content="Section content",
            char_start=0,
            char_end=20,
        )
        assert section.heading == "Test Heading"
        assert section.level == 2

    def test_wikilink_creation(self):
        link = ObsidianWikilink(target="Page", alias="Alias", char_start=0, char_end=15)
        assert link.target == "Page"
        assert link.alias == "Alias"

    def test_research_capture_note_to_markdown(self):
        capture = ResearchCaptureNote(
            research_id="test-123",
            title="Test Research",
            query="What is the answer?",
            answer="The answer is 42.",
            confidence=0.95,
            sources=["source1.md", "source2.md"],
            claims=["Claim 1", "Claim 2"],
        )
        markdown = capture.to_markdown()
        assert "test-123" in markdown
        assert "Test Research" in markdown
        assert "What is the answer?" in markdown
        assert "The answer is 42." in markdown
        assert "42" in markdown
        assert "source1.md" in markdown
        assert "Claim 1" in markdown


class TestObsidianParser:
    def test_parse_frontmatter(self, sample_note_content: str):
        frontmatter, content = parse_frontmatter(sample_note_content)
        assert frontmatter.title == "Test Note"
        assert frontmatter.tags == ["test", "example"]
        assert frontmatter.aliases == ["Test Alias"]
        assert frontmatter.date is not None
        assert "Main Heading" in content

    def test_parse_frontmatter_missing(self):
        content = "No frontmatter here"
        frontmatter, remaining = parse_frontmatter(content)
        assert frontmatter.title is None
        assert remaining == content

    def test_extract_sections(self, sample_note_content: str):
        _, content = parse_frontmatter(sample_note_content)
        sections = extract_sections(content)
        assert len(sections) >= 3  # Main Heading, Subsection, Another Section
        headings = [s.heading for s in sections]
        assert "Main Heading" in headings
        assert "Subsection" in headings
        assert "Another Section" in headings

    def test_extract_wikilinks(self, sample_note_content: str):
        _, content = parse_frontmatter(sample_note_content)
        wikilinks = extract_wikilinks(content)
        assert len(wikilinks) >= 2
        targets = [w.target for w in wikilinks]
        assert "wikilink" in targets
        assert "another link" in targets

    def test_extract_tags(self, sample_note_content: str):
        _, content = parse_frontmatter(sample_note_content)
        tags = extract_tags(content)
        assert len(tags) >= 1
        assert any(t.tag == "tag" for t in tags)

    def test_extract_callouts(self, sample_note_content: str):
        _, content = parse_frontmatter(sample_note_content)
        callouts = extract_callouts(content)
        assert len(callouts) >= 1
        assert callouts[0].type == "note"

    def test_extract_code_blocks(self, sample_note_content: str):
        _, content = parse_frontmatter(sample_note_content)
        code_blocks = extract_code_blocks(content)
        assert len(code_blocks) >= 1
        assert code_blocks[0].language == "python"
        assert "hello" in code_blocks[0].content

    def test_parse_obsidian_note(self, populated_vault: Path):
        note_path = populated_vault / "test_note.md"
        note = parse_obsidian_note(note_path, populated_vault)
        assert note.file_name == "test_note.md"
        assert note.frontmatter.title == "Test Note"
        assert len(note.sections) >= 3
        assert len(note.wikilinks) >= 2
        assert len(note.tags) >= 1
        assert len(note.callouts) >= 1
        assert len(note.code_blocks) >= 1
        assert note.content_checksum is not None


class TestVaultScanner:
    def test_scanner_creation(self, populated_vault: Path):
        scanner = VaultScanner(populated_vault)
        assert scanner.vault_root == populated_vault.resolve()

    def test_scan_vault(self, populated_vault: Path):
        scanner = VaultScanner(populated_vault)
        notes = scanner.scan()
        assert len(notes) >= 1
        assert notes[0].file_name == "test_note.md"

    def test_vault_identity(self, populated_vault: Path):
        scanner = VaultScanner(populated_vault)
        identity = scanner.get_vault_identity()
        assert len(identity) == 16  # 16 char hash


class TestSyncManager:
    def test_sync_manager_creation(self, populated_vault: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.pkl"
            manager = SyncManager(populated_vault, manifest_path)
            assert manager.manifest is not None
            assert manager.manifest.vault_path == str(populated_vault)

    def test_upsert_and_get_note_record(self, populated_vault: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.pkl"
            manager = SyncManager(populated_vault, manifest_path)

            from app.integrations.obsidian.models import ObsidianNoteRecord
            record = ObsidianNoteRecord(
                vault_relative_path="test.md",
                content_checksum="abc123",
                source_id=uuid4(),
                document_id=uuid4(),
                chunk_ids=[uuid4()],
                note_type=NoteType.PERSONAL_CONTEXT,
                file_modified=datetime.now(UTC),
                file_size=100,
            )
            manager.upsert_note_record(record)

            retrieved = manager.get_note_record("test.md")
            assert retrieved is not None
            assert retrieved.vault_relative_path == "test.md"
            assert retrieved.content_checksum == "abc123"

    def test_has_note_changed(self, populated_vault: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.pkl"
            manager = SyncManager(populated_vault, manifest_path)

            # New note should be changed
            assert manager.has_note_changed("new.md", "checksum123") is True

            # Add record
            from app.integrations.obsidian.models import ObsidianNoteRecord
            record = ObsidianNoteRecord(
                vault_relative_path="existing.md",
                content_checksum="same_checksum",
                source_id=uuid4(),
                document_id=uuid4(),
                chunk_ids=[],
                note_type=NoteType.PERSONAL_CONTEXT,
                file_modified=datetime.now(UTC),
                file_size=100,
            )
            manager.upsert_note_record(record)

            # Same checksum = not changed
            assert manager.has_note_changed("existing.md", "same_checksum") is False
            # Different checksum = changed
            assert manager.has_note_changed("existing.md", "different_checksum") is True

    def test_get_deleted_notes(self, populated_vault: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.pkl"
            manager = SyncManager(populated_vault, manifest_path)

            from app.integrations.obsidian.models import ObsidianNoteRecord
            record1 = ObsidianNoteRecord(
                vault_relative_path="note1.md",
                content_checksum="abc",
                source_id=uuid4(),
                document_id=uuid4(),
                chunk_ids=[],
                note_type=NoteType.PERSONAL_CONTEXT,
                file_modified=datetime.now(UTC),
                file_size=100,
            )
            record2 = ObsidianNoteRecord(
                vault_relative_path="note2.md",
                content_checksum="def",
                source_id=uuid4(),
                document_id=uuid4(),
                chunk_ids=[],
                note_type=NoteType.PERSONAL_CONTEXT,
                file_modified=datetime.now(UTC),
                file_size=100,
            )
            manager.upsert_note_record(record1)
            manager.upsert_note_record(record2)

            # Only note1 exists in current scan
            deleted = manager.get_deleted_notes({"note1.md"})
            assert "note2.md" in deleted
            assert "note1.md" not in deleted


class TestObsidianWriter:
    def test_writer_creation(self, populated_vault: Path):
        writer = ObsidianWriter(populated_vault)
        assert writer.write_back_root.exists()
        assert (writer.write_back_root / "Research_Output").exists()
        assert (writer.write_back_root / "Evidence_Reports").exists()
        assert (writer.write_back_root / "Research_Traces").exists()
        assert (writer.write_back_root / "Sync_Logs").exists()

    def test_write_research_capture(self, populated_vault: Path):
        writer = ObsidianWriter(populated_vault)
        capture = ResearchCaptureNote(
            research_id="test-123",
            title="Test Research",
            query="Test query",
            answer="Test answer",
            confidence=0.9,
            sources=["source1.md"],
            claims=["Claim 1"],
        )
        file_path = writer.write_research_capture(capture)
        assert file_path.exists()
        assert file_path.parent.name == "Research_Output"
        content = file_path.read_text(encoding="utf-8")
        assert "test-123" in content
        assert "Test Research" in content

    def test_write_evidence_report(self, populated_vault: Path):
        writer = ObsidianWriter(populated_vault)
        file_path = writer.write_evidence_report(
            research_id="test-123",
            title="Evidence Report",
            evidence_summary="Summary of evidence",
            citations=[{"source": "doc1", "text": "quote"}],
        )
        assert file_path.exists()
        assert file_path.parent.name == "Evidence_Reports"

    def test_write_research_trace(self, populated_vault: Path):
        writer = ObsidianWriter(populated_vault)
        file_path = writer.write_research_trace("test-123", {"step": "test"})
        assert file_path.exists()
        assert file_path.parent.name == "Research_Traces"

    def test_write_sync_log(self, populated_vault: Path):
        writer = ObsidianWriter(populated_vault)
        file_path = writer.write_sync_log({"notes": 5, "chunks": 10})
        assert file_path.exists()
        assert file_path.parent.name == "Sync_Logs"


class TestObsidianIngestionPipeline:
    def test_pipeline_creation(self, populated_vault: Path):
        from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(
                db_path=Path(tmpdir) / "evidence.db",
                bm25_index_path=Path(tmpdir) / "bm25.pkl",
                faiss_index_path=Path(tmpdir) / "faiss.index",
            )
            manifest_path = Path(tmpdir) / "manifest.pkl"
            pipeline = ObsidianIngestionPipeline(populated_vault, store=store, manifest_path=manifest_path)
            assert pipeline.vault_root == populated_vault.resolve()

    def test_ingest_vault_incremental(self, populated_vault: Path):
        from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(
                db_path=Path(tmpdir) / "evidence.db",
                bm25_index_path=Path(tmpdir) / "bm25.pkl",
                faiss_index_path=Path(tmpdir) / "faiss.index",
            )
            manifest_path = Path(tmpdir) / "manifest.pkl"
            pipeline = ObsidianIngestionPipeline(populated_vault, store=store, manifest_path=manifest_path)
            result = pipeline.ingest_vault(incremental=True)

            assert result.notes_discovered >= 1
            assert result.notes_new >= 1
            assert result.chunks_created >= 1
            assert result.completed_at is not None
            assert result.manifest is not None

    def test_ingest_vault_full(self, populated_vault: Path):
        from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(
                db_path=Path(tmpdir) / "evidence.db",
                bm25_index_path=Path(tmpdir) / "bm25.pkl",
                faiss_index_path=Path(tmpdir) / "faiss.index",
            )
            manifest_path = Path(tmpdir) / "manifest.pkl"
            pipeline = ObsidianIngestionPipeline(populated_vault, store=store, manifest_path=manifest_path)
            result = pipeline.ingest_vault(incremental=False)

            assert result.notes_discovered >= 1
            assert result.notes_new >= 1

    def test_unchanged_note_not_reprocessed(self, populated_vault: Path):
        from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(
                db_path=Path(tmpdir) / "evidence.db",
                bm25_index_path=Path(tmpdir) / "bm25.pkl",
                faiss_index_path=Path(tmpdir) / "faiss.index",
            )
            manifest_path = Path(tmpdir) / "manifest.pkl"
            pipeline = ObsidianIngestionPipeline(populated_vault, store=store, manifest_path=manifest_path)

            # First ingestion
            result1 = pipeline.ingest_vault(incremental=True)
            assert result1.notes_new >= 1

            # Second ingestion (incremental) - should be unchanged
            result2 = pipeline.ingest_vault(incremental=True)
            assert result2.notes_unchanged >= 1
            assert result2.notes_new == 0


