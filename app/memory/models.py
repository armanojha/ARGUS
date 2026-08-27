"""Memory data models (Phase 08).

Extended models for the persistent multi-layer memory system.
All memory records maintain full provenance back to evidence chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.memory.interfaces import MemoryLayer, MemoryScope, MemoryRecord as InterfaceMemoryRecord


class MemoryPromotionStatus(str, Enum):
    """Status of a memory record in the promotion pipeline."""

    PROVISIONAL = "provisional"       # Low confidence, awaiting verification
    PROMOTED = "promoted"             # High confidence, auto-promoted
    REJECTED = "rejected"             # Contradicted or invalidated
    ARCHIVED = "archived"             # Superseded by newer memory


class GraphDeltaType(str, Enum):
    """Type of graph delta for versioned updates."""

    CLAIM_ADDED = "claim_added"
    CLAIM_REVISED = "claim_revised"
    CLAIM_SUPERSEDED = "claim_superseded"
    CLAIM_CONTRADICTED = "claim_contradicted"
    ENTITY_ADDED = "entity_added"
    ENTITY_MERGED = "entity_merged"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"


@dataclass(frozen=True)
class MemoryRecord(InterfaceMemoryRecord):
    """A single memory record with full provenance.

    Extends the interface record with promotion and versioning fields.
    """
    id: UUID = field(default_factory=uuid4)
    layer: MemoryLayer = MemoryLayer.WORKING
    scope: MemoryScope = MemoryScope.GLOBAL
    content: str = ""
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    supporting_chunk_ids: list[str] = field(default_factory=list)
    source_query: str | None = None
    confidence: float = field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Phase 08 extensions
    promotion_status: MemoryPromotionStatus = MemoryPromotionStatus.PROVISIONAL
    version: int = 1
    supersedes_id: UUID | None = None
    superseded_by_id: UUID | None = None

    def promote(self, new_confidence: float | None = None) -> MemoryRecord:
        """Create a promoted version of this memory record."""
        return MemoryRecord(
            id=uuid4(),
            layer=self.layer,
            scope=self.scope,
            content=self.content,
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            supporting_chunk_ids=self.supporting_chunk_ids,
            source_query=self.source_query,
            confidence=new_confidence or self.confidence,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            tags=self.tags,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
            promotion_status=MemoryPromotionStatus.PROMOTED,
            version=self.version + 1,
            supersedes_id=self.id,
            superseded_by_id=None,
        )

    def supersede(self, new_content: str, new_confidence: float, new_chunk_ids: list[str]) -> MemoryRecord:
        """Create a new memory record that supersedes this one."""
        new_record = MemoryRecord(
            id=uuid4(),
            layer=self.layer,
            scope=self.scope,
            content=new_content,
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            supporting_chunk_ids=new_chunk_ids,
            source_query=self.source_query,
            confidence=new_confidence,
            valid_from=datetime.now(UTC),
            valid_to=self.valid_to,
            tags=self.tags,
            metadata=self.metadata,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            promotion_status=MemoryPromotionStatus.PROMOTED if new_confidence >= 0.7 else MemoryPromotionStatus.PROVISIONAL,
            version=self.version + 1,
            supersedes_id=self.id,
            superseded_by_id=None,
        )
        # Note: the old record's superseded_by_id would need to be updated separately
        return new_record


class GraphDelta(BaseModel):
    """A versioned delta representing a change to the evidence graph.

    Per V2 §6.3 and V3 §6.2: new research creates versioned deltas,
    never blind overwrites. Existing claims remain traceable.
    """
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    delta_type: GraphDeltaType
    # The entity/claim/event ID that was affected
    target_id: UUID
    target_type: str  # "claim" | "entity" | "event" | "edge"
    # The new data (for additions/revisions) or the removed data
    new_data: dict[str, Any] = Field(default_factory=dict)
    old_data: dict[str, Any] | None = None
    # Provenance
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    source_query: str | None = None
    # Confidence of this delta
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Whether this delta is provisional (low confidence) or promoted
    is_provisional: bool = True
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Version chain
    version: int = 1
    previous_delta_id: UUID | None = None


class GraphVersion(BaseModel):
    """A versioned snapshot of the graph state at a point in time."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    version_number: int
    description: str
    # Deltas included in this version
    delta_ids: list[UUID] = Field(default_factory=list)
    # Graph statistics at this version
    node_counts: dict[str, int] = Field(default_factory=dict)
    edge_count: int = 0
    # Parent version
    parent_version_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by_query: str | None = None


class MemorySearchResult(BaseModel):
    """Result of a memory search with layer breakdown."""

    model_config = ConfigDict(extra="forbid")

    records: list[MemoryRecord] = Field(default_factory=list)
    total_count: int = 0
    query: MemoryQuery | None = None
    layer_counts: dict[str, int] = Field(default_factory=dict)


# Re-export from interfaces for convenience
MemoryQuery = MemoryQuery  # from interfaces


class MemoryStats(BaseModel):
    """Memory store statistics."""

    model_config = ConfigDict(extra="forbid")

    total_records: int = 0
    layer_counts: dict[str, int] = Field(default_factory=dict)
    promotion_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    db_size_bytes: int = 0


__all__ = [
    "MemoryRecord",
    "MemoryPromotionStatus",
    "GraphDelta",
    "GraphDeltaType",
    "GraphVersion",
    "MemorySearchResult",
    "MemoryStats",
]


# Fix circular import - MemoryQuery is imported from interfaces
from app.memory.interfaces import MemoryQuery  # noqa: E402