"""Obsidian 90_ARGUS Writer (Phase 05).

Writes ARGUS research outputs to the dedicated 90_ARGUS/ area in the user's vault.
Per V3 §4.3, ARGUS never writes outside this folder.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.integrations.obsidian.models import ResearchCaptureNote
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.writer")


def _escape_yaml(value: str) -> str:
    """Escape special characters for YAML values."""
    return value.replace('"', '\\"').replace('\n', '\\n')


class ObsidianWriter:
    """Writes ARGUS research outputs to the 90_ARGUS/ area of the vault."""

    def __init__(self, vault_root: Path, write_back_root: str = "90_ARGUS"):
        self.vault_root = Path(vault_root).resolve()
        self.write_back_root = self.vault_root / write_back_root
        self._ensure_write_back_structure()

    def _ensure_write_back_structure(self) -> None:
        """Create the 90_ARGUS directory structure if it doesn't exist."""
        directories = [
            self.write_back_root / "Research_Output",
            self.write_back_root / "Evidence_Reports",
            self.write_back_root / "Research_Traces",
            self.write_back_root / "Sync_Logs",
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def write_research_capture(self, capture: ResearchCaptureNote) -> Path:
        """Write a research capture note to 90_ARGUS/Research_Output/.

        Args:
            capture: The research capture note to write.

        Returns:
            Path to the written file.
        """
        # Generate filename: research_id + timestamp (sanitized)
        import re
        timestamp = capture.created_at.strftime("%Y%m%d_%H%M%S")
        safe_id = re.sub(r'[/\\..]', '_', capture.research_id)
        filename = f"{safe_id}_{timestamp}.md"
        file_path = self.write_back_root / "Research_Output" / filename

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the markdown content
        content = capture.to_markdown()
        file_path.write_text(content, encoding="utf-8")

        logger.info(
            "research_capture_written",
            research_id=capture.research_id,
            path=str(file_path),
        )

        return file_path

    def write_evidence_report(
        self,
        research_id: str,
        title: str,
        evidence_summary: str,
        citations: list[dict[str, Any]],
    ) -> Path:
        """Write an evidence report to 90_ARGUS/Evidence_Reports/."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{research_id}_evidence_{timestamp}.md"
        file_path = self.write_back_root / "Evidence_Reports" / filename

        content = f"""---
title: "{_escape_yaml(title)}"
research_id: {research_id}
type: evidence_report
created_at: {datetime.now(UTC).isoformat()}
tags: [argus, evidence-report]
---

# {_escape_yaml(title)}

**Research ID:** {research_id}

## Evidence Summary

{evidence_summary}

## Citations

""" + "\n".join(f"{i}. {cit}" for i, cit in enumerate(citations, 1))

        file_path.write_text(content, encoding="utf-8")
        logger.info("evidence_report_written", research_id=research_id, path=str(file_path))
        return file_path

    def write_research_trace(
        self,
        research_id: str,
        trace_data: dict[str, Any],
    ) -> Path:
        """Write a research trace (debug/log) to 90_ARGUS/Research_Traces/."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{research_id}_trace_{timestamp}.json"
        file_path = self.write_back_root / "Research_Traces" / filename

        import json
        content = json.dumps(trace_data, indent=2, default=str)
        file_path.write_text(content, encoding="utf-8")

        logger.info("research_trace_written", research_id=research_id, path=str(file_path))
        return file_path

    def write_sync_log(self, log_data: dict[str, Any]) -> Path:
        """Write a sync log to 90_ARGUS/Sync_Logs/."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"sync_{timestamp}.json"
        file_path = self.write_back_root / "Sync_Logs" / filename

        import json
        content = json.dumps(log_data, indent=2, default=str)
        file_path.write_text(content, encoding="utf-8")

        logger.info("sync_log_written", path=str(file_path))
        return file_path

    def list_research_outputs(self) -> list[Path]:
        """List all research output files in 90_ARGUS/Research_Output/."""
        output_dir = self.write_back_root / "Research_Output"
        if not output_dir.exists():
            return []
        return sorted(output_dir.glob("*.md"))

    def read_research_capture(self, file_path: Path) -> str | None:
        """Read a research capture note from the vault."""
        # Path traversal check
        resolved = file_path.resolve()
        vault_resolved = self.vault_root.resolve()
        if not str(resolved).startswith(str(vault_resolved)):
            logger.warning("path_traversal_blocked", path=str(file_path))
            return None
        if not file_path.exists():
            return None
        return file_path.read_text(encoding="utf-8")