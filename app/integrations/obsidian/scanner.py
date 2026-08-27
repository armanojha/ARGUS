"""Obsidian Vault Scanner (Phase 05).

Discovers and scans Obsidian vault for Markdown files.
Computes checksums and tracks file metadata for incremental sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.integrations.obsidian.models import ParsedObsidianNote
from app.integrations.obsidian.parser import parse_obsidian_note
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.scanner")


class VaultScanner:
    """Scans an Obsidian vault for Markdown files."""

    def __init__(self, vault_root: Path):
        self.vault_root = Path(vault_root).resolve()
        if not self.vault_root.exists():
            raise ValueError(f"Vault root does not exist: {self.vault_root}")
        if not self.vault_root.is_dir():
            raise ValueError(f"Vault root is not a directory: {self.vault_root}")

    def scan(self, exclude_patterns: list[str] | None = None) -> list[ParsedObsidianNote]:
        """Scan vault for Markdown files and parse them.

        Args:
            exclude_patterns: Glob patterns to exclude (e.g., ["90_ARGUS/**", ".obsidian/**"])

        Returns:
            List of parsed Obsidian notes
        """
        exclude_patterns = exclude_patterns or [
            "90_ARGUS/**",
            ".obsidian/**",
            ".git/**",
            ".trash/**",
        ]

        notes = []
        markdown_files = self._find_markdown_files(exclude_patterns)

        logger.info("vault_scan_started", vault=str(self.vault_root), files_found=len(markdown_files))

        for file_path in markdown_files:
            try:
                note = parse_obsidian_note(file_path, self.vault_root)
                notes.append(note)
            except Exception as e:  # noqa: BLE001
                logger.error("note_parse_failed", file=str(file_path), error=str(e))
                # Continue scanning other files

        logger.info("vault_scan_completed", vault=str(self.vault_root), notes_parsed=len(notes))
        return notes

    def _find_markdown_files(self, exclude_patterns: list[str]) -> list[Path]:
        """Find all .md files in vault, respecting exclude patterns."""
        all_files = list(self.vault_root.rglob("*.md"))

        if not exclude_patterns:
            return all_files

        # Filter out excluded files
        filtered = []
        for file_path in all_files:
            relative = file_path.relative_to(self.vault_root)
            excluded = False
            for pattern in exclude_patterns:
                if relative.match(pattern):
                    excluded = True
                    break
            if not excluded:
                filtered.append(file_path)

        return filtered

    def get_vault_identity(self) -> str:
        """Generate a stable identity hash for the vault."""
        import hashlib
        # Use the absolute path of the vault root for identity
        return hashlib.sha256(str(self.vault_root).encode()).hexdigest()[:16]

    def get_file_info(self, file_path: Path) -> dict[str, Any]:
        """Get basic file info without full parsing."""
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(self.vault_root)),
            "size": file_path.stat().st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
        }