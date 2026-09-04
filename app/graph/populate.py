"""Retroactive Evidence Graph population (Phase 03 bridge).

The standard ingestion pipeline populates the SQLite evidence store (chunks)
but historically never fed those chunks through LLM extraction into the
NetworkX evidence graph. This module provides the missing bridge: it reads all
chunks from the evidence store, runs them through :func:`extract_from_chunks`
in batches, and applies each :class:`ExtractionResult` to the graph store via
:meth:`EvidenceGraphStore.apply_extraction`.

This is deliberately additive and non-destructive. It does not alter the
ingestion pipeline, the retrieval path, or the orchestration loop. It reuses
the existing, fully-implemented extraction and graph-store functions.

Graph population is gated by ``settings.graph_extraction_enabled`` (default
true) exactly as the previously-unused ingestion flag intended.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.evidence.store import EvidenceStore, get_evidence_store
from app.graph.extraction import extract_from_chunks
from app.graph.store import EvidenceGraphStore, get_graph_store
from app.logging_config import get_logger

logger = get_logger("argus.graph.populate")


def _get_chunks(store: EvidenceStore) -> list[Any]:
    """Load every chunk from the evidence store."""
    chunk_ids = store.get_all_chunk_ids()
    if not chunk_ids:
        return []
    return store.get_chunks_by_ids(chunk_ids)


async def populate_graph(
    batch_size: int | None = None,
    store: EvidenceStore | None = None,
    graph_store: EvidenceGraphStore | None = None,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Run LLM extraction over all existing chunks and populate the graph.

    Args:
        batch_size: Chunks per extraction call (defaults to
            ``settings.graph_extraction_batch_size``).
        store: Evidence store (defaults to the process singleton).
        graph_store: Graph store (defaults to the process singleton).
        settings: App settings (defaults to the cached singleton).

    Returns:
        A summary dict with ``chunks``, ``batches``, ``entities``, ``claims``,
        ``events`` and ``edges`` added, plus ``processed`` chunk count.
    """
    from app.llm_gateway import get_router

    settings = settings or get_settings()
    store = store or get_evidence_store()
    graph_store = graph_store or get_graph_store()
    batch_size = batch_size or getattr(settings, "graph_extraction_batch_size", 10)

    chunks = _get_chunks(store)
    if not chunks:
        logger.info("graph_populate_no_chunks")
        return {"chunks": 0, "batches": 0, "entities": 0, "claims": 0, "events": 0, "edges": 0, "processed": 0}

    router = get_router()
    before = graph_store.stats()

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    for batch in batches:
        extraction = await extract_from_chunks(batch, router, settings, request_id="brain-populate")
        graph_store.apply_extraction(extraction)

    after = graph_store.stats()
    return {
        "chunks": len(chunks),
        "batches": len(batches),
        "entities": max(after.get("entities", 0) - before.get("entities", 0), 0),
        "claims": max(after.get("claims", 0) - before.get("claims", 0), 0),
        "events": max(after.get("events", 0) - before.get("events", 0), 0),
        "edges": max(after.get("edges", 0) - before.get("edges", 0), 0),
        "processed": len(chunks),
    }
