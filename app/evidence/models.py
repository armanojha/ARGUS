"""Evidence Store data models (Phase 01).

Canonical data models for the deterministic evidence store.
These models represent the Source → Document → Chunk hierarchy
with full provenance tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """Type of source document."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    SPREADSHEET = "spreadsheet"
    OTHER = "other"


class Source(BaseModel):
    """A source document origin (e.g., a PDF file, a markdown note)."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    type: SourceType
    path: str  # Original file path or identifier
    checksum: str  # SHA256 of the source content
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    """A versioned document derived from a source.

    A source can produce multiple document versions (e.g., re-processed
    with different chunking strategies). Each document is immutable once created.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    version: int = 1
    checksum: str  # SHA256 of the document content (all chunks concatenated)
    chunking_strategy: str  # e.g., "semantic_v1", "fixed_512_64"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A canonical text chunk with precise provenance anchors.

    Each chunk belongs to exactly one document version and carries
    precise location information for citation mapping.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    ordinal: int  # 0-based position within the document
    text: str
    token_count: int
    # Provenance anchors for citation mapping
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_path: str | None = None  # e.g., "Chapter 1 > Section 2.1"
    # Vector embedding reference (index into FAISS)
    embedding_index: int | None = None
    # BM25 document ID (for lexical index)
    bm25_doc_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceRef(BaseModel):
    """A citation reference returned by retrieval.

    Contains all information needed to display a citation and
    fetch the full chunk text if needed.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    source_path: str
    source_type: SourceType
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    score: float  # Retrieval/rerank score
    rank: int  # 1-based rank in results
    metadata: dict[str, Any] = Field(default_factory=dict)