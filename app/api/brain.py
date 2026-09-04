"""ARGUS Brain API surface (control layer).

ARGUS Brain = ARGUS's persistent machine memory (``app.memory``). This is a
distinct layer from the user document corpus (Knowledge Base) and from the
Obsidian brain vault. It preserves provenance (supporting chunk IDs + source
query per memory) and selective promotion (``MemoryPromotionStatus``).

Thin, deterministic, read-only seams for the UI. When memory is disabled
(``memory_enabled=False``), endpoints report a graceful disabled status rather
than raising.

In addition to persistent memory, this router exposes the Evidence Graph
(``app.graph``) that powers the ARGUS Brain knowledge-space UI: a node/edge
export suitable for a force-directed WebGL/Canvas visualization.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
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


class BrainGraphNode(BaseModel):
    """A node in the ARGUS Brain knowledge graph (frontend-facing)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    node_type: str
    summary: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrainGraphEdge(BaseModel):
    """An edge in the ARGUS Brain knowledge graph (frontend-facing)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    edge_type: str
    confidence: float = 0.0
    directed: bool = True


class BrainGraphData(BaseModel):
    """Serialized knowledge graph for the Brain visualization."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[BrainGraphNode] = Field(default_factory=list)
    edges: list[BrainGraphEdge] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    source: str = "evidence-graph"


def _node_label(node_type: str, data: dict[str, Any]) -> str:
    """Produce a human-readable label for a graph node."""
    if node_type == "entity":
        return str(data.get("canonical_name") or "Unnamed entity")
    if node_type == "claim":
        return str(data.get("text") or "Untitled claim")
    if node_type == "event":
        return str(data.get("name") or "Untitled event")
    if node_type == "chunk":
        path = data.get("source_path") or "document"
        ordinal = data.get("ordinal")
        suffix = f" #{int(ordinal) + 1}" if isinstance(ordinal, (int, float)) else ""
        return f"{_basename(str(path))}{suffix}"
    if node_type in ("document", "source"):
        path = data.get("path")
        return _basename(str(path)) if path else str(data.get("id") or node_type)
    return str(data.get("name") or data.get("path") or data.get("id") or node_type)


def _basename(path: str) -> str:
    """Return the trailing file/folder name of a path."""
    normalized = (path or "").replace("\\", "/")
    return normalized.rstrip("/").split("/")[-1] if normalized else path


def _node_summary(node_type: str, data: dict[str, Any]) -> str | None:
    """Produce a short contextual summary for a graph node."""
    if node_type == "entity":
        return data.get("description")
    if node_type == "claim":
        text = data.get("text")
        return text[:300] + "…" if text and len(text) > 300 else text
    if node_type == "event":
        return data.get("name")
    if node_type == "chunk":
        text = data.get("text")
        return text[:240] + "…" if text and len(text) > 240 else text
    if node_type in ("document", "source"):
        return data.get("path")
    return None


@router.get("/brain/graph", response_model=BrainGraphData)
def get_brain_graph() -> BrainGraphData:
    """Export the ARGUS Evidence Graph as a frontend-friendly node/edge payload.

    The graph is derived from the NetworkX evidence graph (``app.graph``):
    entities, claims, events, chunks, sources and documents as nodes, with
    typed relationships (supports, contradicts, derives_from, mentions,
    relates_to, ...) as edges. Returns a flat, JSON-serializable structure the
    Brain UI can feed directly into a force-directed layout.
    """
    from app.evidence.store import get_evidence_store
    from app.graph.store import get_graph_store

    graph_store = get_graph_store()
    evidence = get_evidence_store()

    def _resolve_unknown(data: dict[str, Any], node_id: str) -> tuple[str, dict[str, Any]]:
        """Hydrate an auto-created endpoint node (chunk/source/document) from the evidence store.

        When ``add_edge`` runs on NetworkX, it auto-creates the endpoint node
        with no ``node_type``/``data`` attributes (the edge type implies whether
        the endpoint is a chunk, document or source). We reconstruct those from
        the evidence store so the full corpus scale shows up in the Brain.
        """
        try:
            uid = UUID(node_id)
        except (ValueError, TypeError):
            return "unknown", data
        chunk = evidence.get_chunk(uid)
        if chunk is not None:
            doc = evidence.get_document(chunk.document_id)
            source_path = None
            if doc is not None:
                src = evidence.get_source(doc.source_id)
                if src is not None:
                    source_path = src.path
            return "chunk", {
                "id": node_id,
                "document_id": str(chunk.document_id),
                "text": chunk.text,
                "token_count": chunk.token_count,
                "ordinal": chunk.ordinal,
                "page_start": chunk.page_start,
                "section_path": chunk.section_path,
                "source_path": source_path,
            }
        doc = evidence.get_document(uid)
        if doc is not None:
            src = evidence.get_source(doc.source_id) if doc.source_id else None
            return "document", {
                "id": node_id,
                "path": src.path if src else None,
                "chunking_strategy": doc.chunking_strategy,
            }
        src = evidence.get_source(uid)
        if src is not None:
            return "source", {"id": node_id, "path": src.path, "type": src.type.value}
        return "unknown", data

    nodes: list[BrainGraphNode] = []
    edges: list[BrainGraphEdge] = []
    seen: set[str] = set()
    canonical_ids: set[str] = set()
    # Category aliases produced by edge auto-creation that map back to a typed
    # category node already emitted (e.g. 'Concept:<uuid>' == 'entity:<uuid>').
    _PREFIX_TO_TYPE = {
        "Concept": "entity",
        "Entity": "entity",
        "Event": "event",
        "Claim": "claim",
    }

    def _node_id_from_key(key: str) -> str:
        """Extract the bare UUID from a NetworkX node key (`<prefix>:<uuid>`)."""
        if ":" in key:
            return key.split(":", 1)[1]
        return key

    def _type_from_key(key: str) -> str | None:
        prefix = key.split(":", 1)[0]
        return _PREFIX_TO_TYPE.get(prefix)

    # First pass: emit properly-typed nodes (entity/claim/event) and track them.
    for key in graph_store._graph.nodes:
        node_data = graph_store._graph.nodes[key]
        node_type = node_data.get("node_type")
        if node_type is None:
            continue
        node_id = str(node_data.get("node_id") or key)
        if node_id in seen:
            continue
        seen.add(node_id)
        canonical_ids.add(node_id)
        data: dict[str, Any] = {}
        try:
            data = json.loads(node_data.get("data", "{}"))
        except (TypeError, ValueError):
            data = {}
        nodes.append(
            BrainGraphNode(
                id=node_id,
                label=_node_label(node_type, data),
                node_type=node_type,
                summary=_node_summary(node_type, data),
                confidence=float(data.get("confidence", 0.0) or 0.0),
                metadata={
                    "aliases": data.get("aliases", []),
                    "entity_type": data.get("entity_type"),
                    "section_path": data.get("section_path"),
                    "page_start": data.get("page_start"),
                    "source_path": data.get("source_path") or data.get("path"),
                },
            )
        )

    # Second pass: resolve auto-created endpoint nodes (no node_type). These are
    # either chunk/source/document references (hydrate from the evidence store)
    # or duplicate category aliases (Concept:/Event:..) that point at a typed
    # node already emitted — skip the latter.
    for key in graph_store._graph.nodes:
        node_data = graph_store._graph.nodes[key]
        if node_data.get("node_type") is not None:
            continue
        node_id = _node_id_from_key(key)
        if node_id in seen:
            continue
        if node_id in canonical_ids:
            continue
        prefix_type = _type_from_key(key)
        # Duplicate alias of an already-emitted typed node under a different prefix.
        if prefix_type is not None:
            continue
        resolved_type, data = _resolve_unknown({}, node_id)
        if resolved_type == "unknown":
            continue
        seen.add(node_id)
        nodes.append(
            BrainGraphNode(
                id=node_id,
                label=_node_label(resolved_type, data),
                node_type=resolved_type,
                summary=_node_summary(resolved_type, data),
                confidence=float(data.get("confidence", 0.0) or 0.0),
                metadata={
                    "aliases": data.get("aliases", []),
                    "entity_type": data.get("entity_type"),
                    "section_path": data.get("section_path"),
                    "page_start": data.get("page_start"),
                    "source_path": data.get("source_path") or data.get("path"),
                },
            )
        )

    for _, _, _, edge_data in graph_store._graph.edges(keys=True, data=True):
        edge_type = str(edge_data.get("edge_type", "relates_to"))
        data: dict[str, Any] = {}
        try:
            data = json.loads(edge_data.get("data", "{}"))
        except (TypeError, ValueError):
            data = {}
        source_id = str(data.get("source_node_id") or "")
        target_id = str(data.get("target_node_id") or "")
        if not source_id or not target_id:
            continue
        if source_id not in seen or target_id not in seen:
            # Edge references an endpoint that was skipped (duplicate alias); drop it.
            continue
        edges.append(
            BrainGraphEdge(
                id=str(data.get("id") or "") or f"{source_id}:{target_id}",
                source=source_id,
                target=target_id,
                edge_type=edge_type,
                confidence=float(data.get("confidence", 0.0) or 0.0),
                directed=True,
            )
        )

    stats: dict[str, int] = {}
    try:
        stats = graph_store.stats()
    except Exception:  # noqa: BLE001 - stats are best-effort
        stats = {}

    return BrainGraphData(nodes=nodes, edges=edges, stats=stats, source="evidence-graph")


class GraphPopulateResult(BaseModel):
    """Result of populating the evidence graph from existing chunks."""

    model_config = ConfigDict(extra="forbid")

    triggered: bool
    message: str
    detail: dict[str, int] = Field(default_factory=dict)


@router.post("/brain/graph/populate", response_model=GraphPopulateResult)
async def populate_brain_graph() -> GraphPopulateResult:
    """Run LLM extraction over the existing corpus and build the evidence graph.

    Additive and non-destructive: reads all chunks, runs them through the
    existing `extract_from_chunks` in batches, and applies results to the
    NetworkX evidence graph. Returns immediately with the summary.
    """
    from app.graph.populate import populate_graph

    try:
        detail = await populate_graph()
    except Exception as exc:  # noqa: BLE001 - surface as a graceful message
        return GraphPopulateResult(triggered=False, message=f"Graph population failed: {exc}", detail={})
    return GraphPopulateResult(
        triggered=True,
        message="Evidence graph populated from the knowledge base.",
        detail=detail,
    )


class DocumentContentResult(BaseModel):
    """Resolved document/chunk content for the Brain's 'Open Document' action.

    Read-only, derived from the existing evidence store (no redesign): given a
    graph ``node_id`` (or an explicit ``document_id``/``source_id``), return the
    underlying text (single chunk, all chunks of a document, or the source
    path when only a path is known).
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: str
    title: str
    path: str | None = None
    section_path: str | None = None
    page_start: int | None = None
    content: str
    length: int


@router.get("/brain/document", response_model=DocumentContentResult)
def get_brain_document(
    node_id: str = Query(..., description="Graph node UUID (chunk, document or source)."),
) -> DocumentContentResult:
    """Resolve a Brain graph node to its underlying document text.

    ``node_id`` may be a chunk, document, or source UUID. Returns the text to
    display in the Brain UI's document viewer. 404 when the node cannot be
    resolved to known content.
    """
    from app.evidence.store import get_evidence_store

    evidence = get_evidence_store()
    node_uid = _parse_uuid(node_id)

    # 1) Chunk
    chunk = None
    if node_uid is not None:
        chunk = evidence.get_chunk(node_uid)
    if chunk is not None:
        doc = evidence.get_document(chunk.document_id) if chunk.document_id else None
        source_path = None
        if doc is not None:
            src = evidence.get_source(doc.source_id) if doc.source_id else None
            source_path = src.path if src is not None else None
        content = chunk.text or ""
        return DocumentContentResult(
            node_id=node_id,
            node_type="chunk",
            title=str(chunk.section_path or "Chunk #" + str((chunk.ordinal or 0) + 1)),
            path=source_path,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            content=content,
            length=len(content),
        )

    # 2) Document -> concatenate its chunks.
    if node_uid is not None:
        doc = evidence.get_document(node_uid)
        if doc is not None:
            src = evidence.get_source(doc.source_id) if doc.source_id else None
            source_path = src.path if src is not None else None
            chunks = evidence.get_chunks_by_document(doc.id)
            ordered = sorted(chunks, key=lambda c: (c.ordinal or 0))
            content = "\n\n".join(
                f"[{c.ordinal + 1}]{' ' + (c.section_path or '') if c.section_path else ''}\n{c.text or ''}"
                for c in ordered
            ) or "(empty document)"
            return DocumentContentResult(
                node_id=node_id,
                node_type="document",
                title=_basename(str(source_path or "")) or "Document #" + node_id[:8],
                path=source_path,
                section_path=None,
                page_start=None,
                content=content,
                length=len(content),
            )

    # 3) Source -> path only (no chunk text if no document was segmented).
    if node_uid is not None:
        src = evidence.get_source(node_uid)
        if src is not None:
            return DocumentContentResult(
                node_id=node_id,
                node_type="source",
                title=_basename(src.path or ""),
                path=src.path,
                section_path=None,
                page_start=None,
                content=src.path or "",
                length=len(src.path or ""),
            )

    raise HTTPException(status_code=404, detail="No document content found for node_id.")


def _parse_uuid(node_id: str) -> UUID | None:
    """Safely parse a node_id into a UUID, or None when it is not a valid UUID."""
    try:
        return UUID(str(node_id))
    except (ValueError, TypeError):
        return None


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