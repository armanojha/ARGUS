"""Phase 09.4: Safe write-back workflow (propose, don't mutate).

Proposals are stored as human-reviewable Markdown files under
90_ARGUS/Proposals/. The default workflow is `propose` — a proposal NEVER
touches the target note. Only an explicit `accept` applies the change to
the user's note; `reject` leaves it untouched.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from app.integrations.obsidian.contracts import WriteBackProposal, WriteBackProposalInterface
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.proposals")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _apply_change(content: str, proposed_content: str, section_heading: str | None) -> str:
    """Insert proposed content into a note body.

    When `section_heading` is provided, content is inserted inside that
    section (creating it at the end if missing). Otherwise content is
    appended to the end of the note.
    """
    block = proposed_content.strip()
    if not section_heading:
        return content.rstrip() + "\n\n" + block + "\n"

    headings = list(_HEADING_RE.finditer(content))
    target = next(
        (
            m for m in headings
            if m.group(2).strip().lower() == section_heading.strip().lower()
        ),
        None,
    )
    if target is None:
        clean_heading = section_heading.strip().lstrip("#").strip()
        return content.rstrip() + f"\n\n## {clean_heading}\n\n{block}\n"

    insert_at = len(content)
    target_level = len(target.group(1))
    for m in headings:
        if m.start() <= target.start():
            continue
        if len(m.group(1)) <= target_level:
            insert_at = m.start()
            break
    before = content[:insert_at].rstrip()
    return before + "\n\n" + block + "\n\n" + content[insert_at:].lstrip("\n")


def _render_proposal(proposal: WriteBackProposal) -> str:
    """Render a proposal to a reviewable Markdown note."""
    frontmatter = {
        "proposal_id": proposal.proposal_id,
        "target_note": proposal.target_note_path,
        "change_type": proposal.change_type,
        "section_heading": proposal.section_heading,
        "research_id": proposal.research_id,
        "confidence": proposal.confidence,
        "status": proposal.status,
        "created_at": proposal.created_at.isoformat(),
        "reviewed_at": proposal.reviewed_at.isoformat() if proposal.reviewed_at else None,
        "reviewed_by": proposal.reviewed_by,
        "evidence_citations": proposal.evidence_citations,
        "tags": ["argus", "writeback-proposal"],
    }
    keep = {k: v for k, v in frontmatter.items() if v is not None}
    yaml_block = yaml.safe_dump(keep, allow_unicode=True, sort_keys=False).strip()

    lines = [
        "---",
        yaml_block,
        "---",
        "",
        f"# Write-Back Proposal {proposal.proposal_id}",
        "",
        f"**Status:** {proposal.status}",
        f"**Target note:** {proposal.target_note_path}",
        f"**Change type:** {proposal.change_type}",
        f"**Research ID:** {proposal.research_id or 'n/a'}",
        f"**Confidence:** {proposal.confidence if proposal.confidence is not None else 'n/a'}",
        "",
        "## Proposed Change",
        "",
    ]
    if proposal.section_heading:
        lines.append(f"### {proposal.section_heading.strip().lstrip('#')}")
        lines.append("")
    lines.append(proposal.proposed_content.strip())
    lines.append("")
    if proposal.evidence_citations:
        lines.append("## Evidence Citations")
        for i, citation in enumerate(proposal.evidence_citations, 1):
            try:
                rendered = yaml.safe_dump(citation, allow_unicode=True, sort_keys=False).strip().replace("\n", " ")
            except Exception:  # noqa: BLE001
                rendered = str(citation)
            lines.append(f"{i}. {rendered}")
        lines.append("")
    lines.append("---")
    lines.append(f"*Created by ARGUS on {proposal.created_at.isoformat()}*")
    return "\n".join(lines) + "\n"


def _parse_proposal(file_path: Path) -> WriteBackProposal:
    """Parse a proposal note back into a WriteBackProposal."""
    text = file_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
    data: dict[str, Any] = {}
    body = text
    if match:
        try:
            loaded = yaml.safe_load(match.group(1)) or {}
            if isinstance(loaded, dict):
                data = loaded
        except yaml.YAMLError:
            data = {}
        body = text[match.end():]

    proposed = _extract_block(body, "Proposed Change")
    if proposed is None:
        proposed = data.get("proposed_content", "") if isinstance(data, dict) else ""

    created_at = data.get("created_at") or datetime.now(UTC)
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = datetime.now(UTC)
    reviewed_at = data.get("reviewed_at")
    if isinstance(reviewed_at, str):
        try:
            reviewed_at = datetime.fromisoformat(reviewed_at)
        except ValueError:
            reviewed_at = None

    return WriteBackProposal(
        proposal_id=str(data.get("proposal_id") or file_path.stem),
        target_note_path=str(data.get("target_note") or ""),
        change_type=str(data.get("change_type") or "update"),
        proposed_content=proposed,
        section_heading=data.get("section_heading"),
        research_id=data.get("research_id"),
        evidence_citations=list(data.get("evidence_citations") or []),
        confidence=data.get("confidence"),
        status=str(data.get("status") or "pending"),
        created_at=created_at,
        reviewed_at=reviewed_at,
        reviewed_by=data.get("reviewed_by"),
        metadata={"proposal_file": str(file_path)},
    )


def _extract_block(body: str, heading: str) -> str | None:
    """Extract content under a `## <heading>` block from a proposal body."""
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$.*?(?=^##\s+|\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(body)
    if not match:
        return None
    block = match.group(0).split("\n", 1)[-1].strip()
    return block or None


class WriteBackProposalManager(WriteBackProposalInterface):
    """File-based proposal manager under 90_ARGUS/Proposals/."""

    def __init__(
        self,
        vault_root: Path | str,
        write_back_root: str = "90_ARGUS",
        proposals_subdir: str = "Proposals",
    ) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.proposals_root = self.vault_root / write_back_root / proposals_subdir
        self.proposals_root.mkdir(parents=True, exist_ok=True)

    # -- path helpers --------------------------------------------------------

    def _target_note_path(self, target_note_path: str) -> Path:
        """Resolve a target note path, refusing to escape the vault."""
        path = Path(target_note_path)
        if not path.is_absolute():
            path = self.vault_root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.vault_root)
        except ValueError as exc:
            raise ValueError(f"target note is outside the vault: {target_note_path}") from exc
        if not resolved.suffix:
            resolved = resolved.with_suffix(".md")
        return resolved

    def _proposal_file(self, proposal_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", proposal_id)
        return self.proposals_root / f"{safe_id}.md"

    # -- interface -----------------------------------------------------------

    async def create_proposal(
        self,
        target_note_path: str,
        change_type: str,
        proposed_content: str,
        section_heading: str | None = None,
        research_id: str | None = None,
        evidence_citations: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
    ) -> WriteBackProposal:
        target = self._target_note_path(target_note_path)
        if not target.exists():
            raise FileNotFoundError(f"target note does not exist: {target}")

        proposal = WriteBackProposal(
            proposal_id=f"prop-{uuid4().hex[:10]}",
            target_note_path=str(target.relative_to(self.vault_root)),
            change_type=change_type,
            proposed_content=proposed_content.strip(),
            section_heading=section_heading,
            research_id=research_id,
            evidence_citations=list(evidence_citations or []),
            confidence=confidence,
            status="pending",
        )
        proposal_file = self._proposal_file(proposal.proposal_id)
        proposal_file.write_text(_render_proposal(proposal), encoding="utf-8")
        logger.info(
            "writeback_proposal_created",
            proposal_id=proposal.proposal_id,
            target=proposal.target_note_path,
        )
        return proposal

    async def get_proposal(self, proposal_id: str) -> WriteBackProposal | None:
        proposal_file = self._proposal_file(proposal_id)
        if not proposal_file.exists():
            return None
        return _parse_proposal(proposal_file)

    async def list_proposals(
        self,
        status: str | None = None,
        note_path: str | None = None,
    ) -> list[WriteBackProposal]:
        proposals: list[WriteBackProposal] = []
        for path in sorted(self.proposals_root.glob("*.md")):
            try:
                proposal = _parse_proposal(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("proposal_parse_failed", path=str(path), error=str(exc))
                continue
            if status and proposal.status != status:
                continue
            if note_path and proposal.target_note_path != str(note_path):
                continue
            proposals.append(proposal)
        return proposals

    async def accept_proposal(self, proposal_id: str, reviewer: str = "user") -> bool:
        """Apply the proposed change to the target note (only on accept)."""
        proposal = await self.get_proposal(proposal_id)
        if proposal is None or proposal.status != "pending":
            return False
        target = self._target_note_path(proposal.target_note_path)
        if not target.exists():
            logger.error("writeback_apply_failed", proposal_id=proposal_id, reason="target note missing")
            return False

        content = target.read_text(encoding="utf-8")
        try:
            updated = _apply_change(content, proposal.proposed_content, proposal.section_heading)
        except Exception as exc:  # noqa: BLE001
            logger.error("writeback_apply_failed", proposal_id=proposal_id, error=str(exc))
            return False
        target.write_text(updated, encoding="utf-8")

        proposal.status = "accepted"
        proposal.reviewed_at = datetime.now(UTC)
        proposal.reviewed_by = reviewer
        self._proposal_file(proposal_id).write_text(_render_proposal(proposal), encoding="utf-8")
        logger.info("writeback_proposal_accepted", proposal_id=proposal_id, target=str(target))
        return True

    async def reject_proposal(self, proposal_id: str, reviewer: str = "user") -> bool:
        """Reject a proposal; the target note stays untouched."""
        proposal = await self.get_proposal(proposal_id)
        if proposal is None or proposal.status != "pending":
            return False
        proposal.status = "rejected"
        proposal.reviewed_at = datetime.now(UTC)
        proposal.reviewed_by = reviewer
        self._proposal_file(proposal_id).write_text(_render_proposal(proposal), encoding="utf-8")
        logger.info("writeback_proposal_rejected", proposal_id=proposal_id)
        return True