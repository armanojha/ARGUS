"""Memory Layer Interfaces (Phase 08, 09).

Defines the interfaces for the persistent multi-layer memory system.
Phase 08 implements these interfaces. Phase 09 (Obsidian Full) depends
on the vault memory layer. Phase 02 planner can optionally use memory.

These are CONTRACTS only - Phase 08 implements them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Memory Layer Types (V2 §6.2, V3 §6.1)
# =============================================================================

class MemoryLayer(str, Enum):
    """Memory layer types (V2 §6.2, V3 §6.1)."""
    WORKING = "working"                     # Current query context (short-term)
    LONG_TERM_KNOWLEDGE = "long_term_knowledge"   # Verified facts, high confidence
    RESEARCH_HISTORY = "research_history"         # Past queries, plans, results
    SOURCE_MEMORY = "source_memory"               # Source reliability, bias
    USER_MEMORY = "user_memory"                   # User preferences, corrections
    VAULT_MEMORY = "vault_memory"                 # Obsidian vault pointers (Phase 09)


class MemoryScope(str, Enum):
    """Scope of memory applicability."""
    GLOBAL = "global"           # Applicable across all queries
    SESSION = "session"         # Current research session only
    QUERY = "query"             # Single query only
    VAULT = "vault"             # Specific to a vault (Phase 09)


class MemoryRecord(BaseModel):
    """A single memory record with full provenance.

    All memory records trace back to evidence chunks.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    layer: MemoryLayer
    scope: MemoryScope = MemoryScope.GLOBAL
    content: str
    # Structured content (optional)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    # Provenance - ALWAYS required
    supporting_chunk_ids: list[str] = Field(default_factory=list)  # UUID strings
    # Source query that created this memory
    source_query: str | None = None
    # Confidence in this memory (0-1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Temporal validity
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    # Metadata
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""
    model_config = ConfigDict(extra="forbid")

    query_text: str
    layers: list[MemoryLayer] | None = None
    scope: MemoryScope | None = None
    limit: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional filters
    tags: list[str] = Field(default_factory=list)
    time_window_start: datetime | None = None
    time_window_end: datetime | None = None
    tags_match_all: bool = False


class MemorySearchResult(BaseModel):
    """Result of a memory search."""
    model_config = ConfigDict(extra="forbid")

    records: list[Any] = Field(default_factory=list)  # list[MemoryRecord]
    total_count: int = 0
    query: MemoryQuery | None = None


# =============================================================================
# Memory Store Interface (Phase 08 implements, Phase 02/08/09 depend on)
# =============================================================================

class MemoryStoreInterface(ABC):
    """Interface for the persistent memory store.

    Phase 08 implements this. Phase 02 (planner), Phase 08 (self-evolution),
    and Phase 09 (vault memory) depend on this interface.
    """

    @abstractmethod
    async def store(self, record: Any) -> None:  # MemoryRecord
        """Store a memory record."""
        ...

    @abstractmethod
    async def retrieve(self, query: Any) -> list[Any]:  # MemoryQuery -> list[MemoryRecord]
        """Retrieve relevant memories for a query."""
        ...

    @abstractmethod
    async def get_by_id(self, record_id: str) -> Any | None:  # MemoryRecord | None
        """Get a specific memory record by ID."""
        ...

    @abstractmethod
    async def update(self, record: Any) -> None:  # MemoryRecord
        """Update an existing memory record."""
        ...

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a memory record. Returns True if deleted."""
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        ...


class MemoryAwarePlannerInterface(ABC):
    """Interface for memory-aware planning.

    Phase 08 implements this. Phase 02's planner node can be
    extended to use this interface.
    """

    @abstractmethod
    async def enhance_plan_with_memory(
        self,
        plan: Any,  # ResearchPlan
        query: str,
        memory_store: Any,  # MemoryStoreInterface
    ) -> Any:  # Enhanced ResearchPlan
        """Enhance a research plan using relevant memories."""
        ...


# =============================================================================
# Vault Memory Layer (Phase 09 coordination)
# =============================================================================

class VaultMemoryRecord(BaseModel):
    """A vault memory record linking Obsidian notes to graph entities.

    Phase 09 uses this to coordinate vault memory with the evidence graph.
    """
    model_config = ConfigDict(extra="forbid")

    note_path: str  # Relative path in vault
    note_id: str  # UUID string
    # Graph entity links
    entity_ids: list[str] = Field(default_factory=list)  # UUID strings
    claim_ids: list[str] = Field(default_factory=list)
    # Classification
    note_type: str = "personal_context"  # Will be 7-class in Phase 09
    # Provenance
    chunk_ids: list[str] = Field(default_factory=list)
    # Timestamps
    ingested_at: datetime
    last_synced: datetime
    file_modified: datetime


class VaultMemoryInterface(ABC):
    """Interface for vault memory coordination.

    Phase 09 implements this. Phase 08 provides the base memory
    infrastructure; Phase 09 adds vault-specific coordination.
    """

    @abstractmethod
    async def sync_vault_memory(self, vault_path: str) -> dict[str, Any]:
        """Sync vault notes to memory layer. Returns sync stats."""
        ...

    @abstractmethod
    async def get_vault_entities(self, vault_path: str) -> list[Any]:
        """Get all entities extracted from vault notes."""
        ...

    @abstractmethod
    async def link_note_to_entities(
        self,
        note_path: str,
        entity_ids: list[str],
    ) -> None:
        """Link a vault note to graph entities."""
        ...

    @abstractmethod
    async def get_note_memory(self, note_path: str) -> Any | None:
        """Get memory record for a specific note."""
        ...


# =============================================================================
# Memory Factory (for dependency injection)
# =============================================================================

class MemoryFactoryInterface(ABC):
    """Factory for creating memory components.

    Phase 08 implements this. Other phases use this to get
    memory components without depending on concrete implementations.
    """

    @abstractmethod
    def create_memory_store(self) -> Any:  # MemoryStoreInterface
        """Create the memory store."""
        ...

    @abstractmethod
    def create_vault_memory(self) -> Any | None:  # VaultMemoryInterface
        """Create vault memory coordinator (Phase 09), or None."""
        ...


class DefaultMemoryFactory:
    """Default factory returning None (no memory)."""

    def create_memory_store(self) -> None:
        return None

    def create_vault_memory(self) -> None:
        return None


# Global factory instance
_memory_factory: Any = DefaultMemoryFactory()


def get_memory_factory() -> Any:
    return _memory_factory


def set_memory_factory(factory: Any) -> None:
    global _memory_factory
    _memory_factory = factory


__all__ = [
    # Factory
    "DefaultMemoryFactory",
    "MemoryAwarePlannerInterface",
    "MemoryFactoryInterface",
    # Enums
    "MemoryLayer",
    "MemoryQuery",
    # Models
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchResult",
    # Interfaces
    "MemoryStoreInterface",
    "VaultMemoryInterface",
    "VaultMemoryRecord",
    "get_memory_factory",
    "set_memory_factory",
]