"""Obsidian Sync Manifest (Phase 05).

Manages incremental sync state for Obsidian vault ingestion.
Tracks checksums, document IDs, and chunk IDs for each note.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.integrations.obsidian.models import ObsidianNoteRecord, SyncManifest
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.sync")


class SyncManager:
    """Manages incremental sync state for Obsidian vault ingestion."""

    def __init__(
        self,
        vault_root: Path,
        manifest_path: Path | None = None,
    ):
        self.vault_root = Path(vault_root).resolve()
        settings = get_settings()
        self.manifest_path = manifest_path or (settings.data_dir / "obsidian_index" / "sync_manifest.pkl")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest: SyncManifest | None = None
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load manifest from disk if exists."""
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("rb") as f:
                    self._manifest = pickle.load(f)
                logger.info("manifest_loaded", path=str(self.manifest_path), notes=len(self._manifest.notes))
            except Exception as e:  # noqa: BLE001
                logger.warning("manifest_load_failed", error=str(e), path=str(self.manifest_path))
                self._manifest = None

        if self._manifest is None:
            vault_identity = self._compute_vault_identity()
            self._manifest = SyncManifest(
                vault_path=str(self.vault_root),
                vault_identity=vault_identity,
            )
            logger.info("manifest_created_new", vault=str(self.vault_root))

    def _compute_vault_identity(self) -> str:
        """Compute stable vault identity."""
        import hashlib
        return hashlib.sha256(str(self.vault_root).encode()).hexdigest()[:16]

    def save(self) -> None:
        """Persist manifest to disk."""
        if self._manifest is None:
            logger.warning("manifest_save_skipped", reason="no manifest to save")
            return
        try:
            with self.manifest_path.open("wb") as f:
                pickle.dump(self._manifest, f)
            logger.info("manifest_saved", path=str(self.manifest_path), notes=len(self._manifest.notes))
        except Exception as e:
            logger.error("manifest_save_failed", error=str(e), path=str(self.manifest_path))
            raise

    @property
    def manifest(self) -> SyncManifest:
        """Get the current manifest."""
        if self._manifest is None:
            self._load_manifest()
        assert self._manifest is not None  # guaranteed after _load_manifest
        return self._manifest

    def get_note_record(self, vault_relative_path: str) -> ObsidianNoteRecord | None:
        """Get existing note record from manifest."""
        return self.manifest.notes.get(vault_relative_path)

    def has_note_changed(self, vault_relative_path: str, content_checksum: str) -> bool:
        """Check if a note has changed since last sync."""
        record = self.get_note_record(vault_relative_path)
        if record is None:
            return True  # New note
        return record.content_checksum != content_checksum

    def get_deleted_notes(self, current_note_paths: set[str]) -> list[str]:
        """Find notes that exist in manifest but not in current scan."""
        manifest_paths = set(self.manifest.notes.keys())
        return list(manifest_paths - current_note_paths)

    def upsert_note_record(self, record: ObsidianNoteRecord) -> None:
        """Add or update a note record in the manifest."""
        self.manifest.notes[record.vault_relative_path] = record
        self.manifest.total_notes = len(self.manifest.notes)
        self.manifest.total_chunks = sum(len(r.chunk_ids) for r in self.manifest.notes.values())

    def remove_note_record(self, vault_relative_path: str) -> bool:
        """Remove a note record from manifest (for deleted files)."""
        if vault_relative_path in self.manifest.notes:
            del self.manifest.notes[vault_relative_path]
            self.manifest.total_notes = len(self.manifest.notes)
            self.manifest.total_chunks = sum(len(r.chunk_ids) for r in self.manifest.notes.values())
            return True
        return False

    def mark_full_sync(self) -> None:
        """Mark that a full sync was completed."""
        self.manifest.last_full_sync = datetime.now(UTC)

    def mark_incremental_sync(self) -> None:
        """Mark that an incremental sync was completed."""
        self.manifest.last_incremental_sync = datetime.now(UTC)

    def get_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            "total_notes": self.manifest.total_notes,
            "total_chunks": self.manifest.total_chunks,
            "last_full_sync": self.manifest.last_full_sync.isoformat() if self.manifest.last_full_sync else None,
            "last_incremental_sync": self.manifest.last_incremental_sync.isoformat() if self.manifest.last_incremental_sync else None,
            "vault_path": self.manifest.vault_path,
            "vault_identity": self.manifest.vault_identity,
        }


