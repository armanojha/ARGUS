"""Memory data models (Phase 08).

Extended models for the persistent multi-layer memory system.
All memory records maintain full provenance back to evidence chunks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.memory.interfaces import MemoryQuery, MemoryRecord


class MemorySearchResult(BaseModel):
    """Result of a memory search with layer breakdown."""

    model_config = ConfigDict(extra="forbid")

    records: list[MemoryRecord] = Field(default_factory=list)
    total_count: int = 0
    query: MemoryQuery | None = None
    layer_counts: dict[str, int] = Field(default_factory=dict)


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
    "MemorySearchResult",
    "MemoryStats",
]