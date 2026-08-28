"""Document chunking utilities (Phase 01).

Provides token-aware chunking with overlap, preserving section/page context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.evidence.models import Chunk
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.chunking")


@dataclass
class TextSegment:
    """A segment of text with provenance info."""
    text: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_path: str | None = None
    metadata: dict[str, Any] | None = None


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def chunk_text(
    segments: list[TextSegment],
    document_id: UUID,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk text segments into character-bounded chunks with overlap.

    Args:
        segments: List of text segments with provenance info
        document_id: UUID of the parent document
        chunk_size: Target chunk size in characters (from settings if None)
        chunk_overlap: Overlap in characters (from settings if None)

    Returns:
        List of Chunk objects with provenance preserved
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Combine all segments into a single text stream with position tracking
    full_text = ""
    segment_bounds = []  # (segment_idx, start_char, end_char)
    for i, seg in enumerate(segments):
        start = len(full_text)
        full_text += seg.text
        end = len(full_text)
        segment_bounds.append((i, start, end))

    # Tokenize by characters (rough approximation)
    # In production, use a proper tokenizer like tiktoken
    tokens = list(full_text)
    total_tokens = len(tokens)

    chunks = []
    ordinal = 0
    start_token = 0

    while start_token < total_tokens:
        end_token = min(start_token + chunk_size, total_tokens)
        chunk_text = "".join(tokens[start_token:end_token])

        # Find which segments this chunk spans
        chunk_char_start = start_token
        chunk_char_end = end_token

        # Find provenance for this chunk
        page_start = None
        page_end = None
        char_start = chunk_char_start
        char_end = chunk_char_end
        section_path = None
        metadata: dict[str, Any] = {}

        for seg_idx, seg_start, seg_end in segment_bounds:
            seg = segments[seg_idx]
            # Check if this segment overlaps with the chunk
            if seg_end > chunk_char_start and seg_start < chunk_char_end:
                if page_start is None or (seg.page_start is not None and seg.page_start < page_start):
                    page_start = seg.page_start
                if page_end is None or (seg.page_end is not None and seg.page_end > page_end):
                    page_end = seg.page_end
                if section_path is None and seg.section_path:
                    section_path = seg.section_path
                if seg.metadata:
                    metadata.update(seg.metadata)

        chunk = Chunk(
            document_id=document_id,
            ordinal=ordinal,
            text=chunk_text,
            token_count=estimate_tokens(chunk_text),
            page_start=page_start,
            page_end=page_end,
            char_start=char_start,
            char_end=char_end,
            section_path=section_path,
            metadata=metadata,
        )
        if not chunk_text.strip():
            ordinal += 1
            start_token += chunk_size - chunk_overlap
            continue
        chunks.append(chunk)

        ordinal += 1
        start_token += chunk_size - chunk_overlap

    logger.info("chunked_document", document_id=str(document_id), chunk_count=len(chunks))
    return chunks


def chunk_by_sections(
    segments: list[TextSegment],
    document_id: UUID,
    max_chunk_size: int | None = None,
) -> list[Chunk]:
    """Chunk by section boundaries, respecting max size.

    Creates one chunk per section by default. Only combines adjacent sections
    if their combined size is within max_chunk_size.
    """
    settings = get_settings()
    max_chunk_size = max_chunk_size or settings.chunk_size

    chunks = []
    ordinal = 0

    for seg in segments:
        seg_tokens = estimate_tokens(seg.text)

        # If segment itself exceeds max size, fall back to token-based chunking
        if seg_tokens > max_chunk_size:
            # Split this segment using token-based chunking
            sub_chunks = chunk_text([seg], document_id, chunk_size=max_chunk_size, chunk_overlap=0)
            for sub_chunk in sub_chunks:
                sub_chunk.ordinal = ordinal
                chunks.append(sub_chunk)
                ordinal += 1
            continue

        # Create a new chunk for this section
        chunk = Chunk(
            document_id=document_id,
            ordinal=ordinal,
            text=seg.text.strip(),
            token_count=seg_tokens,
            page_start=seg.page_start,
            page_end=seg.page_end,
            char_start=seg.char_start,
            char_end=seg.char_end,
            section_path=seg.section_path,
            metadata=seg.metadata or {},
        )
        chunks.append(chunk)
        ordinal += 1

    logger.info("chunked_by_sections", document_id=str(document_id), chunk_count=len(chunks))
    return chunks