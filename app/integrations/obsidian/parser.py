"""Obsidian Markdown Parser (Phase 05).

Parses Obsidian-flavored Markdown files into structured note objects.
Handles YAML frontmatter, sections, wikilinks, tags, callouts, and code blocks.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from app.integrations.obsidian.models import (
    ObsidianCallout,
    ObsidianCodeBlock,
    ObsidianFrontmatter,
    ObsidianSection,
    ObsidianTag,
    ObsidianWikilink,
    ParsedObsidianNote,
)
from app.logging_config import get_logger

logger = get_logger("argus.obsidian.parser")

# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+?)(?:\|([^\]]+))?\]\]")
TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z0-9_\-/]+)")
CALLOUT_PATTERN = re.compile(r">\s*\[!(\w+)\](?:\s*-\s*(.+))?\n((?:>.*\n?)*)", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of content."""
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_frontmatter(content: str) -> tuple[ObsidianFrontmatter, str]:
    """Extract and parse YAML frontmatter from content.

    Returns:
        Tuple of (frontmatter, content_without_frontmatter)
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return ObsidianFrontmatter(), content

    frontmatter_text = match.group(1)
    remaining_content = content[match.end():].lstrip("\n")

    try:
        fm_data = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as e:
        logger.warning("frontmatter_parse_failed", error=str(e))
        fm_data = {}

    # Extract known fields
    frontmatter = ObsidianFrontmatter(
        title=fm_data.get("title"),
        tags=fm_data.get("tags", []) if isinstance(fm_data.get("tags"), list) else [],
        aliases=fm_data.get("aliases", []) if isinstance(fm_data.get("aliases"), list) else [],
        date=_parse_datetime(fm_data.get("date")),
        created=_parse_datetime(fm_data.get("created")),
        modified=_parse_datetime(fm_data.get("modified")),
        custom={k: v for k, v in fm_data.items()
                if k not in {"title", "tags", "aliases", "date", "created", "modified"}},
    )

    return frontmatter, remaining_content


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime from various formats."""
    import datetime as dt_mod

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    # YAML may parse dates as datetime.date objects
    if isinstance(value, dt_mod.date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                dt = datetime.strptime(value, fmt)  # noqa: DTZ007
                return dt.replace(tzinfo=UTC)
            except ValueError:
                continue
    return None


def extract_sections(content: str) -> list[ObsidianSection]:
    """Extract sections from markdown content based on headings."""
    sections = []
    heading_matches = list(HEADING_PATTERN.finditer(content))

    for i, match in enumerate(heading_matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        char_start = match.start()
        char_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(content)

        section_content = content[match.end():char_end].strip()

        # Recursively extract subsections
        subsections = extract_subsections(content, match.end(), char_end, level)

        sections.append(ObsidianSection(
            heading=heading_text,
            level=level,
            content=section_content,
            char_start=char_start,
            char_end=char_end,
            subsections=subsections,
        ))

    return sections


def extract_subsections(content: str, start: int, end: int, parent_level: int) -> list[ObsidianSection]:
    """Extract subsections within a range."""
    subsections = []
    # Find headings within the range that are deeper than parent
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    for match in pattern.finditer(content[start:end]):
        level = len(match.group(1))
        if level <= parent_level:
            continue
        heading_text = match.group(2).strip()
        char_start = start + match.start()
        # Find next heading at same or higher level
        next_match = None
        for m in pattern.finditer(content[start:end]):
            if m.start() > match.start():
                next_level = len(m.group(1))
                if next_level <= level:
                    next_match = m
                    break
        char_end = start + next_match.start() if next_match else end

        section_content = content[match.end():char_end].strip()

        subsections.append(ObsidianSection(
            heading=heading_text,
            level=level,
            content=section_content,
            char_start=char_start,
            char_end=char_end,
            subsections=[],
        ))

    return subsections


def extract_wikilinks(content: str) -> list[ObsidianWikilink]:
    """Extract [[wikilinks]] from content."""
    wikilinks = []
    for match in WIKILINK_PATTERN.finditer(content):
        target = match.group(1).strip()
        alias = match.group(2).strip() if match.group(2) else None
        wikilinks.append(ObsidianWikilink(
            target=target,
            alias=alias,
            char_start=match.start(),
            char_end=match.end(),
        ))
    return wikilinks


def extract_tags(content: str) -> list[ObsidianTag]:
    """Extract #tags from content."""
    tags = []
    for match in TAG_PATTERN.finditer(content):
        tag = match.group(1)
        tags.append(ObsidianTag(
            tag=tag,
            char_start=match.start(),
            char_end=match.end(),
        ))
    return tags


def extract_callouts(content: str) -> list[ObsidianCallout]:
    """Extract >[!callout] blocks from content."""
    callouts = []
    for match in CALLOUT_PATTERN.finditer(content):
        callout_type = match.group(1).lower()
        title = match.group(2).strip() if match.group(2) else None
        callout_content = match.group(3).strip()
        # Remove leading "> " from each line
        callout_content = re.sub(r"^>\s?", "", callout_content, flags=re.MULTILINE)

        callouts.append(ObsidianCallout(
            type=callout_type,
            title=title,
            content=callout_content,
            char_start=match.start(),
            char_end=match.end(),
        ))
    return callouts


def extract_code_blocks(content: str) -> list[ObsidianCodeBlock]:
    """Extract fenced code blocks from content."""
    code_blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(content):
        language = match.group(1).strip() if match.group(1) else None
        code_content = match.group(2)
        code_blocks.append(ObsidianCodeBlock(
            language=language,
            content=code_content,
            char_start=match.start(),
            char_end=match.end(),
        ))
    return code_blocks


def parse_obsidian_note(file_path: Path, vault_root: Path) -> ParsedObsidianNote:
    """Parse a single Obsidian markdown file into a structured note."""
    raw_content = file_path.read_text(encoding="utf-8")
    content_checksum = compute_checksum(raw_content)

    # Parse frontmatter
    frontmatter, content_without_fm = parse_frontmatter(raw_content)

    # Extract all components
    sections = extract_sections(content_without_fm)
    wikilinks = extract_wikilinks(content_without_fm)
    tags = extract_tags(content_without_fm)
    callouts = extract_callouts(content_without_fm)
    code_blocks = extract_code_blocks(content_without_fm)

    # File metadata
    stat = file_path.stat()
    vault_relative_path = str(file_path.relative_to(vault_root))

    return ParsedObsidianNote(
        file_path=file_path,
        absolute_path=file_path,
        file_name=file_path.name,
        file_stem=file_path.stem,
        raw_content=raw_content,
        content_without_frontmatter=content_without_fm,
        frontmatter=frontmatter,
        sections=sections,
        wikilinks=wikilinks,
        tags=tags,
        callouts=callouts,
        code_blocks=code_blocks,
        file_size=stat.st_size,
        file_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        file_created=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
        content_checksum=content_checksum,
        vault_relative_path=vault_relative_path,
    )