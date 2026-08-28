"""Obsidian Ingestion data models (Phase 05).

Canonical data models for Obsidian vault ingestion. These models represent
parsed Obsidian notes with full provenance, keeping personal knowledge
separate from external evidence per V3 §2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ObsidianSourceType(str, Enum):
    """Type of Obsidian source."""
    MARKDOWN = "markdown"


class NoteType(str, Enum):
    """Classification of Obsidian note (minimal for MVP, full taxonomy in Phase 09)."""
    PERSONAL_CONTEXT = "personal_context"  # Generic personal knowledge
    RESEARCH_CAPTURE = "research_capture"  # ARGUS-generated research output


class ObsidianFrontmatter(BaseModel):
    """Parsed YAML frontmatter from Obsidian note."""

    model_config = ConfigDict(extra="allow")  # Allow arbitrary frontmatter fields

    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    date: datetime | None = None
    created: datetime | None = None
    modified: datetime | None = None
    # Custom fields preserved as-is
    custom: dict[str, Any] = Field(default_factory=dict)


class ObsidianSection(BaseModel):
    """A section within an Obsidian note (heading + content)."""

    model_config = ConfigDict(extra="forbid")

    heading: str
    level: int  # 1-6 for # through ######
    content: str
    char_start: int
    char_end: int
    subsections: list[ObsidianSection] = Field(default_factory=list)


class ObsidianWikilink(BaseModel):
    """A [[wikilink]] found in the note."""

    model_config = ConfigDict(extra="forbid")

    target: str  # The link target (page name)
    alias: str | None = None  # Optional display alias [[target|alias]]
    char_start: int
    char_end: int


class ObsidianTag(BaseModel):
    """A #tag found in the note."""

    model_config = ConfigDict(extra="forbid")

    tag: str  # The tag without the # prefix
    char_start: int
    char_end: int


class ObsidianCallout(BaseModel):
    """A >[!callout] block found in the note."""

    model_config = ConfigDict(extra="forbid")

    type: str  # e.g., "note", "warning", "tip", "abstract"
    title: str | None = None
    content: str
    char_start: int
    char_end: int


class ObsidianCodeBlock(BaseModel):
    """A fenced code block found in the note."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    content: str
    char_start: int
    char_end: int


class ParsedObsidianNote(BaseModel):
    """A fully parsed Obsidian note with all extracted features."""

    model_config = ConfigDict(extra="forbid")

    # File identity
    file_path: Path  # Relative to vault root
    absolute_path: Path
    file_name: str
    file_stem: str  # Without extension

    # Content
    raw_content: str
    content_without_frontmatter: str

    # Parsed components
    frontmatter: ObsidianFrontmatter
    sections: list[ObsidianSection] = Field(default_factory=list)
    wikilinks: list[ObsidianWikilink] = Field(default_factory=list)
    tags: list[ObsidianTag] = Field(default_factory=list)
    callouts: list[ObsidianCallout] = Field(default_factory=list)
    code_blocks: list[ObsidianCodeBlock] = Field(default_factory=list)

    # Metadata
    file_size: int
    file_modified: datetime
    file_created: datetime
    content_checksum: str  # SHA256 of raw_content

    # Ingestion metadata
    note_type: NoteType = NoteType.PERSONAL_CONTEXT
    vault_relative_path: str  # Relative path from vault root


class ObsidianNoteRecord(BaseModel):
    """Record of an ingested Obsidian note in the sync manifest."""

    model_config = ConfigDict(extra="forbid")

    note_id: UUID = Field(default_factory=uuid4)
    vault_relative_path: str  # Relative path from vault root
    content_checksum: str
    source_id: UUID  # Evidence Store source ID
    document_id: UUID  # Evidence Store document ID
    chunk_ids: list[UUID] = Field(default_factory=list)
    note_type: NoteType = NoteType.PERSONAL_CONTEXT
    frontmatter: ObsidianFrontmatter | None = None
    tags: list[str] = Field(default_factory=list)
    wikilink_targets: list[str] = Field(default_factory=list)
    # Phase 09.1: 7-class knowledge taxonomy
    knowledge_class: str | None = None
    treatment_rule: str | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_synced: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_modified: datetime
    file_size: int


class SyncManifest(BaseModel):
    """Manifest tracking all ingested notes for incremental sync."""

    model_config = ConfigDict(extra="forbid")

    vault_path: str
    vault_identity: str  # Hash of vault root path for identity
    notes: dict[str, ObsidianNoteRecord] = Field(default_factory=dict)  # vault_relative_path -> record
    last_full_sync: datetime | None = None
    last_incremental_sync: datetime | None = None
    total_notes: int = 0
    total_chunks: int = 0


class ObsidianIngestionResult(BaseModel):
    """Result of an Obsidian ingestion run."""

    model_config = ConfigDict(extra="forbid")

    vault_path: str
    started_at: datetime
    completed_at: datetime | None = None
    notes_discovered: int = 0
    notes_new: int = 0
    notes_updated: int = 0
    notes_unchanged: int = 0
    notes_deleted: int = 0
    notes_failed: int = 0
    notes_classified: int = 0
    hypothesis_objectives: list[Any] = Field(default_factory=list)  # HypothesisResearchObjective
    chunks_created: int = 0
    chunks_updated: int = 0
    errors: list[str] = Field(default_factory=list)
    manifest: SyncManifest | None = None


class ResearchCaptureNote(BaseModel):
    """ARGUS-generated research capture note for 90_ARGUS/Research_Output/."""

    model_config = ConfigDict(extra="forbid")

    research_id: str
    title: str
    status: str = "completed"  # completed, in_progress, archived
    query: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    caveats: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # Source paths
    claims: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    argus_version: str = "0.1.0"

    def to_markdown(self) -> str:
        """Generate the markdown content for the research capture note."""
        fm = {
            "title": self.title,
            "research_id": self.research_id,
            "status": self.status,
            "query": self.query,
            "confidence": self.confidence,
            "sources": self.sources,
            "claims": self.claims,
            "created_at": self.created_at.isoformat(),
            "argus_version": self.argus_version,
            "tags": ["argus", "research-capture"],
        }

        # Build frontmatter YAML
        fm_lines = ["---"]
        for key, value in fm.items():
            if isinstance(value, list):
                if value:
                    fm_lines.append(f"{key}:")
                    for v in value:
                        fm_lines.append(f"  - {v}")
                else:
                    fm_lines.append(f"{key}: []")
            elif value is not None:
                fm_lines.append(f"{key}: {value}")
        fm_lines.append("---")

        # Build body
        body_lines = [
            f"# {self.title}",
            "",
            f"**Research ID:** {self.research_id}",
            f"**Status:** {self.status}",
            f"**Query:** {self.query}",
            "",
            "## Answer",
            self.answer,
            "",
        ]

        if self.citations:
            body_lines.append("## Citations")
            for i, cit in enumerate(self.citations, 1):
                body_lines.append(f"{i}. {cit}")
            body_lines.append("")

        if self.confidence is not None:
            body_lines.append(f"## Confidence: {self.confidence:.0%}")
            body_lines.append("")

        if self.caveats:
            body_lines.append("## Caveats")
            for caveat in self.caveats:
                body_lines.append(f"- {caveat}")
            body_lines.append("")

        if self.claims:
            body_lines.append("## Claims")
            for claim in self.claims:
                body_lines.append(f"- {claim}")
            body_lines.append("")

        if self.sources:
            body_lines.append("## Linked Sources")
            for source in self.sources:
                stem = Path(source).stem
                body_lines.append(f"- [[{stem}|{source}]]")
            body_lines.append("")

        body_lines.append("---")
        body_lines.append(f"*Generated by ARGUS v{self.argus_version} on {self.created_at.isoformat()}*")

        return "\n".join(fm_lines) + "\n\n" + "\n".join(body_lines)


# Forward reference resolution
ObsidianSection.model_rebuild()