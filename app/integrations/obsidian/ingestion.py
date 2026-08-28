"""Obsidian Ingestion Pipeline (Phase 05).

Integrates Obsidian notes into the existing ARGUS evidence store and retrieval system.
Notes are stored as personal-context claims, distinct from external evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from app.config import get_settings
from app.evidence.models import Document, Source, SourceType
from app.evidence.store import EvidenceStore, get_evidence_store
from app.ingestion.chunking import chunk_by_sections
from app.integrations.obsidian.models import (
    ObsidianIngestionResult,
    ObsidianNoteRecord,
    ParsedObsidianNote,
)
from app.integrations.obsidian.scanner import VaultScanner
from app.integrations.obsidian.sync import SyncManager
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.ingestion")


class ObsidianIngestionPipeline:
    """End-to-end Obsidian vault ingestion pipeline.

    Integrates Obsidian notes into the ARGUS evidence store as personal-context
    claims, keeping them distinct from external evidence per V3 §2.
    """

    def __init__(
        self,
        vault_root: Path,
        store: EvidenceStore | None = None,
        manifest_path: Path | None = None,
        classifier: object | None = None,
        aligner: object | None = None,
    ):
        self.vault_root = Path(vault_root).resolve()
        self.store = store or get_evidence_store()
        self.settings = get_settings()
        self.scanner = VaultScanner(self.vault_root)
        self.sync_manager = SyncManager(self.vault_root, manifest_path)

        # Phase 09.1 extension points (backward compatible: absent by default).
        self.classifier: object | None = classifier
        if self.classifier is None and (
            self.settings.obsidian_full_enabled and self.settings.obsidian_classification_enabled
        ):
            from app.integrations.obsidian.classifier import RuleBasedObsidianClassifier

            self.classifier = RuleBasedObsidianClassifier()

        # Phase 09.3 graph alignment (backward compatible: absent by default).
        self.aligner: object | None = aligner
        if self.aligner is None and (
            self.settings.obsidian_full_enabled and self.settings.obsidian_vault_graph_alignment_enabled
        ):
            from app.graph.store import get_graph_store
            from app.integrations.obsidian.alignment import VaultGraphAligner

            self.aligner = VaultGraphAligner(get_graph_store())

    def ingest_vault(
        self,
        incremental: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> ObsidianIngestionResult:
        """Ingest the entire Obsidian vault.

        Args:
            incremental: If True, only process changed files. If False, full re-ingestion.
            exclude_patterns: Glob patterns to exclude from scanning.

        Returns:
            Ingestion result with statistics.
        """
        from datetime import UTC, datetime

        started_at = datetime.now(UTC)
        result = ObsidianIngestionResult(
            vault_path=str(self.vault_root),
            started_at=started_at,
        )

        # Scan vault
        notes = self.scanner.scan(exclude_patterns=exclude_patterns)
        result.notes_discovered = len(notes)

        current_note_paths = set()

        for note in notes:
            current_note_paths.add(note.vault_relative_path)

            try:
                if incremental and not self.sync_manager.has_note_changed(
                    note.vault_relative_path, note.content_checksum
                ):
                    result.notes_unchanged += 1
                    continue

                # Ingest the note
                existing = self.sync_manager.get_note_record(note.vault_relative_path)
                record = self._ingest_note(note)
                self.sync_manager.upsert_note_record(record)

                if existing is not None:
                    result.notes_updated += 1
                else:
                    result.notes_new += 1

                result.chunks_created += len(record.chunk_ids)

                # Phase 09.1/09.2: expose classification and hypothesis objectives.
                if record.knowledge_class:
                    result.notes_classified += 1
                if (
                    record.knowledge_class in {"hypothesis", "task_question"}
                    and self.settings.obsidian_full_enabled
                    and self.settings.obsidian_hypothesis_conversion_enabled
                ):
                    objective = self._build_hypothesis_objective(note, record)
                    if objective is not None:
                        result.hypothesis_objectives.append(objective)

            except Exception as e:  # noqa: BLE001
                logger.error("note_ingestion_failed", path=note.vault_relative_path, error=str(e))
                result.notes_failed += 1
                result.errors.append(f"{note.vault_relative_path}: {e}")

        # Handle deleted notes
        if incremental:
            deleted_paths = self.sync_manager.get_deleted_notes(current_note_paths)
            for deleted_path in deleted_paths:
                self._handle_deleted_note(deleted_path)
                result.notes_deleted += 1

        # Update sync timestamps
        if incremental:
            self.sync_manager.mark_incremental_sync()
        else:
            self.sync_manager.mark_full_sync()

        self.sync_manager.save()
        result.completed_at = datetime.now(UTC)
        result.manifest = self.sync_manager.manifest

        logger.info(
            "vault_ingestion_completed",
            vault=str(self.vault_root),
            new=result.notes_new,
            updated=result.notes_updated,
            unchanged=result.notes_unchanged,
            deleted=result.notes_deleted,
            failed=result.notes_failed,
            chunks=result.chunks_created,
        )

        return result

    def _ingest_note(self, note: ParsedObsidianNote) -> ObsidianNoteRecord:
        """Ingest a single parsed Obsidian note into the evidence store."""
        # 1. Create or get source
        source = self._upsert_source(note)

        # Phase 09.1: classify the note (no LLM; deterministic).
        knowledge_class: str | None = None
        treatment_rule: str | None = None
        if self.classifier is not None:
            classification = self._classify(note)
            knowledge_class = classification.knowledge_class
            treatment_rule = classification.treatment_rule

        # 2. Check for existing document version
        existing_doc = self.store.get_latest_document_for_source(source.id)
        doc_checksum = note.content_checksum

        if existing_doc and existing_doc.checksum == doc_checksum:
            # Document unchanged, return existing record
            existing_record = self.sync_manager.get_note_record(note.vault_relative_path)
            if existing_record:
                return existing_record

        # Delete old chunks if updating an existing note
        existing_record = self.sync_manager.get_note_record(note.vault_relative_path)
        if existing_record and existing_record.chunk_ids:
            for chunk_id in existing_record.chunk_ids:
                self.store.delete_chunk(chunk_id)
            logger.info(
                "old_chunks_deleted",
                note_path=note.vault_relative_path,
                count=len(existing_record.chunk_ids),
            )

        # 3. Create text segments from note sections
        segments = self._note_to_segments(note)

        # 4. Chunk the note content
        chunks = chunk_by_sections(segments, UUID(int=0))  # placeholder doc_id

        # 5. Create document record
        next_version = 1
        if existing_doc:
            next_version = existing_doc.version + 1

        document = Document(
            source_id=source.id,
            version=next_version,
            checksum=doc_checksum,
            chunking_strategy="obsidian_sections_v1",
            metadata={
                "vault_path": str(self.vault_root),
                "note_path": note.vault_relative_path,
                "note_type": note.note_type.value,
                "knowledge_class": knowledge_class,
                "treatment_rule": treatment_rule,
                "frontmatter_title": note.frontmatter.title,
                "tags": note.frontmatter.tags,
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
            },
        )
        document = self.store.insert_document(document)

        # 6. Update chunks with correct document_id and store
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk.document_id = document.id
            chunk.ordinal = i
            # Add Obsidian-specific metadata
            chunk.metadata.update({
                "vault_relative_path": note.vault_relative_path,
                "note_type": note.note_type.value,
                "knowledge_class": knowledge_class,
                "treatment_rule": treatment_rule,
                "section_path": chunk.section_path,
            })
            chunk_ids.append(chunk.id)

        self.store.insert_chunks(chunks)

        # 7. Create note record for manifest
        record = ObsidianNoteRecord(
            vault_relative_path=note.vault_relative_path,
            content_checksum=note.content_checksum,
            source_id=source.id,
            document_id=document.id,
            chunk_ids=chunk_ids,
            note_type=note.note_type,
            frontmatter=note.frontmatter,
            tags=note.frontmatter.tags,
            wikilink_targets=[w.target for w in note.wikilinks],
            knowledge_class=knowledge_class,
            treatment_rule=treatment_rule,
            file_modified=note.file_modified,
            file_size=note.file_size,
        )

        # Phase 09.3: align note with the evidence graph (best-effort).
        if self.aligner is not None and knowledge_class is not None:
            try:
                self.aligner.align_note(
                    note,
                    chunk_ids,
                    knowledge_class=knowledge_class,
                    treatment_rule=treatment_rule,
                )
            except Exception as exc:  # noqa: BLE001 - alignment must not break ingestion
                logger.warning("note_alignment_failed", note=note.vault_relative_path, error=str(exc))

        logger.info(
            "note_ingested",
            note_path=note.vault_relative_path,
            document_id=str(document.id),
            chunks=len(chunks),
        )

        return record

    def _upsert_source(self, note: ParsedObsidianNote) -> Source:
        """Create or get source for the Obsidian note."""
        source_checksum = hashlib.sha256(
            f"{self.vault_root}:{note.vault_relative_path}".encode()
        ).hexdigest()

        existing_source = self.store.get_source_by_checksum(source_checksum)
        if existing_source:
            return existing_source

        source = Source(
            type=SourceType.MARKDOWN,
            path=note.vault_relative_path,
            checksum=source_checksum,
            metadata={
                "vault_root": str(self.vault_root),
                "note_path": note.vault_relative_path,
                "note_type": "obsidian_note",
            },
        )
        return self.store.upsert_source(source)

    def _note_to_segments(self, note: ParsedObsidianNote) -> list:
        """Convert parsed note sections to TextSegments for chunking."""
        from app.ingestion.chunking import TextSegment

        segments = []

        # Add frontmatter as context if present
        if note.frontmatter.title:
            fm_text = f"Title: {note.frontmatter.title}\n"
            if note.frontmatter.tags:
                fm_text += f"Tags: {', '.join(note.frontmatter.tags)}\n"
            segments.append(TextSegment(
                text=fm_text,
                section_path="frontmatter",
            ))

        # Convert sections to segments
        for section in note.sections:
            segment_text = f"# {'#' * (section.level - 1)} {section.heading}\n\n{section.content}"
            segments.append(TextSegment(
                text=segment_text,
                section_path=section.heading,
                char_start=section.char_start,
                char_end=section.char_end,
            ))

        # If no sections, use the whole content
        if not segments:
            segments.append(TextSegment(
                text=note.content_without_frontmatter,
                section_path="body",
            ))

        return segments

    def _handle_deleted_note(self, vault_relative_path: str) -> None:
        """Handle a note that was deleted from the vault."""
        record = self.sync_manager.get_note_record(vault_relative_path)
        if not record:
            return

        # Note: We don't delete chunks from evidence store to preserve history
        # Just remove from manifest
        self.sync_manager.remove_note_record(vault_relative_path)
        logger.info("note_deleted_from_manifest", path=vault_relative_path)


def ingest_obsidian_vault(
    vault_root: Path,
    incremental: bool = True,
    store: EvidenceStore | None = None,
    manifest_path: Path | None = None,
) -> ObsidianIngestionResult:
    """Convenience function to ingest an Obsidian vault."""
    pipeline = ObsidianIngestionPipeline(vault_root, store, manifest_path)
    return pipeline.ingest_vault(incremental=incremental)