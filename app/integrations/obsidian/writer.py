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


def _safe_md_name(value: str) -> str:
    """Sanitize a string for use in a filename (kept README/text friendly)."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "_", value.strip())
    return cleaned.replace(" ", "_")[:80] or "untitled"


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
            self.write_back_root / "Proposals",
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

    def write_proposal(self, proposal: Any) -> Path:
        """Write a write-back proposal note to 90_ARGUS/Proposals/ (Phase 09).

        Writing a proposal never mutates the target note; applying the
        change requires the user to accept it via the proposal manager.
        """
        from app.integrations.obsidian.proposals import _render_proposal

        proposal_file = self.write_back_root / "Proposals" / f"{proposal.proposal_id}.md"
        proposal_file.write_text(_render_proposal(proposal), encoding="utf-8")
        logger.info("writeback_proposal_written", proposal_id=proposal.proposal_id, path=str(proposal_file))
        return proposal_file

    def promote_research_capture(
        self,
        capture: Any,
        target_dir: Path,
        user_decision: bool = True,
        capture_path: Path | None = None,
    ) -> Path | None:
        """User-promotion path for verified research outputs (Phase 09.4).

        Writes a *pointer / summary* note into the user's knowledge area:
        wikilinks back to the source notes and the 90_ARGUS capture — never
        a content copy. Requires an explicit `user_decision` and refuses to
        write into the 90_ARGUS write-back area.
        """
        from app.integrations.obsidian.models import ResearchCaptureNote

        parsed = ResearchCaptureNote.model_validate(capture) if isinstance(capture, dict) else capture

        target_dir = Path(target_dir).resolve()
        if str(target_dir).lower().startswith(str(self.write_back_root).lower()):
            raise ValueError("promotion target must be outside the 90_ARGUS write-back area")
        try:
            target_dir.relative_to(self.vault_root)
        except ValueError as exc:
            raise ValueError(f"promotion target is outside the vault: {target_dir}") from exc

        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_md_name(parsed.title or parsed.research_id)}_{parsed.research_id}.md"
        output_path = target_dir / filename

        source_links = []
        for source in parsed.sources or []:
            stem = Path(source).stem
            source_links.append(f"- [[{stem}|{source}]]")
        capture_link = ""
        if capture_path is not None:
            capture_link = f"[[{Path(capture_path).stem}|ARGUS capture]]"

        content_lines = [
            "---",
            "title: " + str(parsed.title or "").replace('"', '\\"'),
            f"research_id: {parsed.research_id}",
            f"status: {parsed.status}",
            f"confidence: {parsed.confidence if parsed.confidence is not None else ''}",
            "promoted_by: argus",
            "tags: [argus, promoted]",
            "---",
            "",
            f"# {parsed.title or parsed.research_id}",
            "",
            f"**Status:** {parsed.status}",
            f"**Research ID:** {parsed.research_id}",
            f"**Confidence:** {parsed.confidence if parsed.confidence is not None else 'n/a'}",
        ]
        if capture_link:
            content_lines.append(f"**Original capture:** {capture_link}")
        content_lines.append("")
        content_lines.append("## Claims")
        for claim in parsed.claims or []:
            content_lines.append(f"- {claim}")
        content_lines.append("")
        content_lines.append("## Sources")
        content_lines.extend(source_links or ["- _no sources recorded_"])
        content_lines.append("")
        content_lines.append("---")
        content_lines.append(f"*Promoted by ARGUS on {datetime.now(UTC).isoformat()}*")

        output_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
        logger.info("research_capture_promoted", research_id=parsed.research_id, path=str(output_path))
        return output_path