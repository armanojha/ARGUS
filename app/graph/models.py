"""Evidence Graph data models (Phase 03).

Canonical data models for the Evidence Graph. These models represent
entities, claims, events, and their relationships with full provenance
linking back to Phase 01 EvidenceRef/Chunk objects.

All graph facts remain traceable to their supporting evidence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EdgeType(str, Enum):
    """Types of edges in the Evidence Graph (V2 §6)."""

    SUPPORTS = "supports"           # Evidence supports a claim
    CONTRADICTS = "contradicts"     # Evidence contradicts a claim
    DERIVED_FROM = "derived_from"   # Claim/entity derived from chunk
    VALID_DURING = "valid_during"   # Claim/event valid during time period
    HAS_ASSUMPTION = "has_assumption"  # Claim has an assumption
    RELATES_TO = "relates_to"       # Entity-entity or entity-claim relation
    MENTIONS = "mentions"           # Chunk mentions entity
    INSTANCE_OF = "instance_of"     # Entity is instance of type


class EntityType(str, Enum):
    """Types of entities in the graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    DATE = "date"
    SOURCE = "source"
    OTHER = "other"


class TemporalPrecision(str, Enum):
    """Precision of temporal information."""

    EXACT = "exact"           # Exact date/time known
    DAY = "day"               # Day precision
    MONTH = "month"           # Month precision
    YEAR = "year"             # Year precision
    DECADE = "decade"         # Decade precision
    CENTURY = "century"       # Century precision
    UNKNOWN = "unknown"       # Temporal info not extractable


class Entity(BaseModel):
    """A canonical entity extracted from evidence.

    Entities are deduplicated by canonical name + type. Aliases track
    alternative surface forms found in the text.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    canonical_name: str = Field(description="Canonical name for the entity")
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list, description="Alternative surface forms")
    description: str | None = Field(default=None, description="Brief description from evidence")
    # Provenance: chunks that mention this entity
    supporting_chunk_ids: list[UUID] = Field(default_factory=list)
    # Confidence in entity existence (0-1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Claim(BaseModel):
    """A factual claim extracted from evidence.

    Claims represent propositions that can be supported or contradicted
    by evidence. Each claim is linked to its source chunks.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(description="The claim text in natural language")
    # Structured representation: subject-predicate-object
    subject_entity_id: UUID | None = Field(default=None, description="Subject entity if identified")
    predicate: str = Field(description="Relation/predicate (e.g., 'is CEO of', 'occurred in')")
    object_entity_id: UUID | None = Field(default=None, description="Object entity if identified")
    object_value: str | None = Field(default=None, description="Object value if not an entity")
    # Provenance
    supporting_chunk_ids: list[UUID] = Field(default_factory=list, description="Chunks supporting this claim")
    contradicting_chunk_ids: list[UUID] = Field(default_factory=list, description="Chunks contradicting this claim")
    # Confidence in claim (0-1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Temporal validity (when the claim is/was true)
    valid_from: datetime | None = Field(default=None, description="Claim validity start")
    valid_to: datetime | None = Field(default=None, description="Claim validity end")
    valid_precision: TemporalPrecision = Field(default=TemporalPrecision.UNKNOWN)
    # Publication time (when the claim was published/stated)
    published_at: datetime | None = Field(default=None, description="When the claim appeared in source")
    published_precision: TemporalPrecision = Field(default=TemporalPrecision.UNKNOWN)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Event(BaseModel):
    """An event extracted from evidence.

    Events are claims with specific temporal anchoring. They represent
    occurrences with participants, time, and location.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Event name/description")
    # Temporal anchoring (fact validity time)
    event_time: datetime | None = Field(default=None, description="When the event occurred")
    event_time_precision: TemporalPrecision = Field(default=TemporalPrecision.UNKNOWN)
    event_end_time: datetime | None = Field(default=None, description="Event end time if duration")
    # Participants (entities involved)
    participant_entity_ids: list[UUID] = Field(default_factory=list)
    # Location
    location_entity_id: UUID | None = Field(default=None)
    # Provenance
    supporting_chunk_ids: list[UUID] = Field(default_factory=list)
    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEdge(BaseModel):
    """An edge in the Evidence Graph.

    All edges carry provenance back to source chunks and confidence scores.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    edge_type: EdgeType
    source_node_id: UUID = Field(description="Source node ID (entity, claim, event, or chunk)")
    source_node_type: str = Field(description="Type of source node: 'entity', 'claim', 'event', 'chunk'")
    target_node_id: UUID = Field(description="Target node ID")
    target_node_type: str = Field(description="Type of target node: 'entity', 'claim', 'event', 'chunk'")
    # Provenance: chunks that support this edge
    supporting_chunk_ids: list[UUID] = Field(default_factory=list)
    # Confidence in this edge (0-1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # For VALID_DURING edges
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    valid_precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphNode(BaseModel):
    """A node in the Evidence Graph (union type for storage)."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    node_type: str  # 'entity', 'claim', 'event', 'chunk', 'source', 'document'
    data: dict[str, Any]  # Serialized node data
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Result of LLM-based extraction from a batch of chunks."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    # Chunk IDs that were processed
    processed_chunk_ids: list[UUID] = Field(default_factory=list)
    # Any warnings during extraction
    warnings: list[str] = Field(default_factory=list)


class GraphQuery(BaseModel):
    """A graph-based query for multi-hop retrieval."""

    model_config = ConfigDict(extra="forbid")

    # Starting entities (by name or ID)
    start_entity_names: list[str] = Field(default_factory=list)
    start_entity_ids: list[UUID] = Field(default_factory=list)
    # Edge types to traverse
    edge_types: list[EdgeType] = Field(default_factory=lambda: [EdgeType.RELATES_TO, EdgeType.MENTIONS, EdgeType.SUPPORTS])
    # Maximum hops
    max_hops: int = Field(default=2, ge=1, le=4)
    # Filter by temporal range
    time_window_start: datetime | None = None
    time_window_end: datetime | None = None
    # Limit results
    limit: int = Field(default=20, ge=1, le=100)


class GraphQueryResult(BaseModel):
    """Result of a graph query."""

    model_config = ConfigDict(extra="forbid")

    # Entities found
    entities: list[Entity] = Field(default_factory=list)
    # Claims found
    claims: list[Claim] = Field(default_factory=list)
    # Events found
    events: list[Event] = Field(default_factory=list)
    # Edges traversed
    edges: list[GraphEdge] = Field(default_factory=list)
    # EvidenceRefs for citations (from supporting chunks)
    evidence_refs: list[Any] = Field(default_factory=list)  # EvidenceRef from Phase 01
    # Path information for multi-hop
    paths: list[list[UUID]] = Field(default_factory=list, description="Node ID paths from start to result")