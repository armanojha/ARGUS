"""ARGUS Brain API surface (control layer).

ARGUS Brain = ARGUS's persistent machine memory (``app.memory``). This is a
distinct layer from the user document corpus (Knowledge Base) and from the
Obsidian brain vault. It preserves provenance (supporting chunk IDs + source
query per memory) and selective promotion (``MemoryPromotionStatus``).

Thin, deterministic, read-only seams for the UI. When memory is disabled
(``memory_enabled=False``), endpoints report a graceful disabled status rather
than raising.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.memory import get_memory_factory_instance
from app.memory.interfaces import MemoryLayer, MemoryStoreInterface

router = APIRouter(prefix="/api/v1", tags=["argus-brain"])


class BrainStatus(BaseModel):
    """Status snapshot of ARGUS persistent memory."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    total_records: int
    layer_counts: dict[str, int]
    promotion_counts: dict[str, int]
    scope_counts: dict[str, int]
    avg_confidence: float
    db_size_bytes: int
    recent_records: list[dict[str, object]] = Field(default_factory=list)


def _get_memory_store() -> MemoryStoreInterface | None:
    """Resolve the active memory store, or None when memory is disabled."""
    factory = get_memory_factory_instance()
    if factory is None:
        return None
    try:
        store = factory.create_memory_store()
    except Exception:  # noqa: BLE001 - memory is optional; never crash the control plane
        store = None
    return store


@router.get("/brain/status", response_model=BrainStatus)
def get_brain_status() -> BrainStatus:
    """Report the state of ARGUS persistent memory."""
    store = _get_memory_store()
    if store is None:
        return BrainStatus(
            enabled=False,
            total_records=0,
            layer_counts={},
            promotion_counts={},
            scope_counts={},
            avg_confidence=0.0,
            db_size_bytes=0,
            recent_records=[],
        )

    try:
        # get_stats may be async on the interface; use asyncio.run when needed.
        stats = _await_get_stats(store)
    except Exception:  # noqa: BLE001 - graceful degradation
        stats = {}

    recent: list[dict[str, object]] = []
    try:
        recent = _list_recent(store)
    except Exception:  # noqa: BLE001 - graceful degradation
        recent = []

    return BrainStatus(
        enabled=True,
        total_records=int(stats.get("total_records", 0)),
        layer_counts=stats.get("layer_counts", {}),
        promotion_counts=stats.get("promotion_counts", {}),
        scope_counts=stats.get("scope_counts", {}),
        avg_confidence=float(stats.get("avg_confidence", 0.0)),
        db_size_bytes=int(stats.get("db_size_bytes", 0)),
        recent_records=recent,
    )


def _await_get_stats(store: MemoryStoreInterface) -> dict[str, Any]:
    import asyncio

    try:
        return asyncio.run(store.get_stats())
    except RuntimeError:
        # Already inside a running loop (e.g. tests) -> fall back to empty.
        return {}


def _list_recent(store: MemoryStoreInterface, limit: int = 20) -> list[dict[str, object]]:
    import asyncio

    from app.memory.interfaces import MemoryQuery

    query = MemoryQuery(query_text="", limit=limit, min_confidence=0.0)
    records = asyncio.run(store.retrieve(query))
    out: list[dict[str, object]] = []
    for rec in records:
        layer_any = getattr(rec, "layer", None)
        layer: object = layer_any.value if isinstance(layer_any, MemoryLayer) else str(layer_any or "")
        created_at_any = getattr(rec, "created_at", None)
        created_at = created_at_any.isoformat() if created_at_any else None
        out.append(
            {
                "id": str(getattr(rec, "id", "")),
                "layer": layer,
                "content": getattr(rec, "content", ""),
                "confidence": float(getattr(rec, "confidence", 0.0)),
                "promotion_status": str(getattr(rec, "promotion_status", "")),
                "source_query": getattr(rec, "source_query", None),
                "supporting_chunk_ids": list(getattr(rec, "supporting_chunk_ids", [])),
                "created_at": created_at,
            }
        )
    return out