"""Graph Store Extension Contracts (Phase 08, 09).

Defines extension interfaces for the Evidence Graph Store that
Phase 08 (versioned deltas, self-evolution) and Phase 09
(vault-graph alignment) will implement.

Phase 03 provides the base EvidenceGraphStore. These contracts
define extension points without modifying the base store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# =============================================================================
# Phase 08: Versioned Graph Deltas (Self-Evolution)
# =============================================================================

class DeltaType(str, Enum):
    """Types of graph deltas (V2 §6.3, V3 §6.2)."""
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_DEPRECATED = "node_deprecated"  # Soft delete, not hard delete
    EDGE_ADDED = "edge_added"
    EDGE_UPDATED = "edge_updated"
    EDGE_DEPRECATED = "edge_deprecated"
    CONFIDENCE_CHANGED = "confidence_changed"
    MERGED = "merged"           # Two nodes merged
    SPLIT = "split"             # One node split into multiple


@dataclass(frozen=True)
class GraphDelta:
    """A versioned change to the graph.

    Phase 08 implements versioned deltas. All changes to the graph
    create deltas rather than blind overwrites.
    """
    id: UUID
    delta_type: DeltaType
    # Target
    node_type: str  # 'entity', 'claim', 'event', 'edge'
    node_id: UUID
    # Change details
    before: dict[str, Any] | None = None  # Previous state
    after: dict[str, Any] | None = None   # New state
    # Provenance
    source_chunk_ids: list[UUID] = field(default_factory=list)
    source_query: str | None = None
    # Confidence in this delta
    confidence: float = 1.0
    # Status
    status: str = "provisional"  # provisional, promoted, reverted
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    promoted_by: str | None = None  # query_id or 'auto'
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphDeltaStoreInterface(ABC):
    """Interface for storing and querying graph deltas.

    Phase 08 implements this. The base EvidenceGraphStore can
    optionally use this for versioned history.
    """

    @abstractmethod
    async def record_delta(self, delta: Any) -> None:  # GraphDelta
        """Record a new graph delta."""
        ...

    @abstractmethod
    async def get_deltas_for_node(
        self,
        node_type: str,
        node_id: UUID,
        limit: int = 100,
    ) -> list[Any]:  # list[GraphDelta]
        """Get all deltas for a specific node."""
        ...

    @abstractmethod
    async def get_deltas_since(
        self,
        since: datetime,
        delta_types: list[str] | None = None,
    ) -> list[Any]:  # list[GraphDelta]
        """Get all deltas since a timestamp."""
        ...

    @abstractmethod
    async def promote_delta(self, delta_id: UUID, promoted_by: str) -> bool:
        """Mark a provisional delta as promoted."""
        ...

    @abstractmethod
    async def revert_delta(self, delta_id: UUID) -> bool:
        """Revert a delta (mark as reverted, restore previous state)."""
        ...


class VersionedGraphStoreInterface(ABC):
    """Extended graph store with versioned deltas.

    Phase 08 implements this by extending EvidenceGraphStore.
    The base EvidenceGraphStore remains unchanged; this interface
    defines the additional methods for versioned operations.
    """

    @abstractmethod
    async def upsert_entity_versioned(self, entity: Any, delta: Any) -> Any:  # Entity, GraphDelta
        """Upsert entity with versioned delta tracking."""
        ...

    @abstractmethod
    async def upsert_claim_versioned(self, claim: Any, delta: Any) -> Any:  # Claim, GraphDelta
        """Upsert claim with versioned delta tracking."""
        ...

    @abstractmethod
    async def upsert_event_versioned(self, event: Any, delta: Any) -> Any:  # Event, GraphDelta
        """Upsert event with versioned delta tracking."""
        ...

    @abstractmethod
    async def add_edge_versioned(self, edge: Any, delta: Any) -> Any:  # GraphEdge, GraphDelta
        """Add edge with versioned delta tracking."""
        ...

    @abstractmethod
    async def deprecate_node(self, node_type: str, node_id: UUID, reason: str) -> bool:
        """Soft-deprecate a node (create DEPRECATED delta)."""
        ...

    @abstractmethod
    async def merge_nodes(
        self,
        primary_id: UUID,
        secondary_id: UUID,
        node_type: str,
        reason: str,
    ) -> bool:
        """Merge two nodes (create MERGED delta)."""
        ...

    @abstractmethod
    def get_node_history(self, node_type: str, node_id: UUID) -> list[Any]:
        """Get full version history for a node."""
        ...


# =============================================================================
# Phase 09: Vault-Graph Alignment
# =============================================================================

class ObsidianNodeType(str, Enum):
    """Obsidian node types mapped to graph node types (V3 §6 table)."""
    SOURCE_NOTE = "source_note"           # Maps to: Source/Document
    KNOWLEDGE_NOTE = "knowledge_note"     # Maps to: Entity/Claim
    HYPOTHESIS = "hypothesis"             # Maps to: Claim (with hypothesis tag)
    PROJECT_NOTE = "project_note"         # Maps to: Event/Project entity
    TASK_QUESTION = "task_question"       # Maps to: Claim/Event
    RESEARCH_CAPTURE = "research_capture" # Maps to: Claim/Event (ARGUS-generated)
    REFERENCE_INDEX = "reference_index"   # Maps to: Entity (index)


class VaultGraphAlignment(BaseModel):
    """Mapping between Obsidian note and graph nodes.

    Phase 09 uses this to maintain vault-graph alignment.
    """
    model_config = ConfigDict(extra="forbid")

    note_path: str
    note_id: str  # UUID string
    obsidian_node_type: ObsidianNodeType
    # Graph node mappings
    entity_ids: list[str] = field(default_factory=list)  # UUID strings
    claim_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    # Alignment metadata
    confidence: float = 1.0
    last_aligned: datetime
    alignment_method: str = "auto"  # auto, manual, inferred
    metadata: dict[str, Any] = field(default_factory=dict)


class VaultGraphAlignmentInterface(ABC):
    """Interface for vault-graph alignment.

    Phase 09 implements this. It maintains the mapping between
    Obsidian notes and Evidence Graph nodes.
    """

    @abstractmethod
    async def align_note(
        self,
        note_path: str,
        note_content: str,
        frontmatter: dict[str, Any],
    ) -> Any:  # VaultGraphAlignment
        """Align a vault note to graph nodes."""
        ...

    @abstractmethod
    async def get_alignment(self, note_path: str) -> Any | None:  # VaultGraphAlignment
        """Get existing alignment for a note."""
        ...

    @abstractmethod
    async def update_alignment(
        self,
        note_path: str,
        alignment: Any,  # VaultGraphAlignment
    ) -> None:
        """Update alignment for a note."""
        ...

    @abstractmethod
    async def remove_alignment(self, note_path: str) -> bool:
        """Remove alignment for a deleted note."""
        ...

    @abstractmethod
    async def resolve_canonical_entities(
        self,
        wikilink_targets: list[str],
    ) -> dict[str, str]:  # wikilink -> entity_id
        """Resolve wikilink targets to canonical entity IDs.

        Prevents graph fragmentation from duplicate entities.
        """
        ...

    @abstractmethod
    async def get_unaligned_notes(self) -> list[str]:
        """Get notes that haven't been aligned yet."""
        ...


# =============================================================================
# Graph Extension Factory
# =============================================================================

class GraphExtensionFactoryInterface(ABC):
    """Factory for creating graph extensions.

    Phase 08 and 09 implement this to provide their extensions.
    """

    @abstractmethod
    def create_delta_store(self) -> Any | None:  # GraphDeltaStoreInterface
        """Create Phase 08 delta store, or None."""
        ...

    @abstractmethod
    def create_versioned_store(self) -> Any | None:  # VersionedGraphStoreInterface
        """Create Phase 08 versioned store wrapper, or None."""
        ...

    @abstractmethod
    def create_vault_alignment(self) -> Any | None:  # VaultGraphAlignmentInterface
        """Create Phase 09 vault alignment, or None."""
        ...


class DefaultGraphExtensionFactory:
    """Default factory returning None (no extensions)."""

    def create_delta_store(self) -> None:
        return None

    def create_versioned_store(self) -> None:
        return None

    def create_vault_alignment(self) -> None:
        return None


# Global factory instance
_graph_extension_factory: Any = None


def get_graph_extension_factory() -> Any:
    global _graph_extension_factory
    if _graph_extension_factory is None:
        _graph_extension_factory = DefaultGraphExtensionFactory()
    return _graph_extension_factory


def set_graph_extension_factory(factory: Any) -> None:
    global _graph_extension_factory
    _graph_extension_factory = factory


__all__ = [
    "DefaultGraphExtensionFactory",
    # Phase 08
    "DeltaType",
    "GraphDelta",
    "GraphDeltaStoreInterface",
    # Factory
    "GraphExtensionFactoryInterface",
    # Phase 09
    "ObsidianNodeType",
    "VaultGraphAlignment",
    "VaultGraphAlignmentInterface",
    "VersionedGraphStoreInterface",
    "get_graph_extension_factory",
    "set_graph_extension_factory",
]