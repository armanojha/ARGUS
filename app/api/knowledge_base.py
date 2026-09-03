"""Knowledge Base API surface (control layer).

Thin HTTP seams over the existing ingestion pipeline and EvidenceStore so the
user-facing UI can operate the Knowledge Base without touching Python internals.

  * ``GET  /api/v1/knowledge-base/status`` — corpus status (path, counts,
    supported types, recent documents).
  * ``POST /api/v1/knowledge-base/ingest`` — re-sync the configured corpus
    directory into the EvidenceStore (recursive, idempotent).
  * ``POST /api/v1/knowledge-base/upload`` — upload file(s); saved into the
    same corpus directory and fed through the SAME ingestion pipeline.

No LLM calls are made on the control plane. Upload/ingest are deterministic
indexing operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.evidence.store import get_evidence_store
from app.ingestion.knowledge_base import (
    ingest_file,
    ingest_knowledge_base,
    supported_extensions,
)
from app.retrieval.hybrid import get_hybrid_retriever

router = APIRouter(prefix="/api/v1", tags=["knowledge-base"])


class KnowledgeBaseStatus(BaseModel):
    """Status snapshot of the user Knowledge Base."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_path: str
    exists: bool
    document_count: int
    source_count: int
    chunk_count: int
    supported_types: list[str]
    indexed: bool
    recently_ingested: list[dict[str, object]]


class IngestRequest(BaseModel):
    """Request body for a knowledge-base re-sync."""

    model_config = ConfigDict(extra="forbid")

    rebuild_indexes: bool = Field(default=True, description="Refresh BM25/FAISS indexes after ingestion.")


class IngestResponse(BaseModel):
    """Result of a knowledge-base re-sync."""

    model_config = ConfigDict(extra="forbid")

    ingested: int
    unchanged: int
    errors: int
    indexed: bool
    indexed_chunks: int
    knowledge_base_path: str
    documents_ingested: list[dict[str, object]]
    error_paths: list[str]
    duration_s: float


class UploadResponse(BaseModel):
    """Result of one or more file uploads."""

    model_config = ConfigDict(extra="forbid")

    uploaded: list[dict[str, object]]
    rejected: list[dict[str, object]]
    indexed: bool
    indexed_chunks: int


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
def get_knowledge_base_status(
    settings: Settings = Depends(get_settings),
) -> KnowledgeBaseStatus:
    """Report the current state of the user Knowledge Base."""
    store = get_evidence_store()
    root = settings.knowledge_base_path
    exists = root.exists() and root.is_dir()

    recent = store.list_documents(limit=10)
    recently_ingested: list[dict[str, object]] = []
    for doc in recent:
        source = store.get_source(doc.source_id)
        recently_ingested.append(
            {
                "document_id": str(doc.id),
                "source_id": str(doc.source_id),
                "source_path": source.path if source else None,
                "version": doc.version,
                "created_at": doc.created_at.isoformat(),
            }
        )

    indexed = (
        settings.bm25_index_path.exists()
        and settings.faiss_index_path.exists()
        and store.count_chunks() > 0
    )
    return KnowledgeBaseStatus(
        knowledge_base_path=str(root),
        exists=exists,
        document_count=store.count_documents(),
        source_count=store.count_sources(),
        chunk_count=store.count_chunks(),
        supported_types=sorted(supported_extensions()),
        indexed=indexed,
        recently_ingested=recently_ingested,
    )


@router.post("/knowledge-base/ingest", response_model=IngestResponse)
def ingest_endpoint(
    body: IngestRequest,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Re-sync the configured corpus directory into the EvidenceStore."""
    result = ingest_knowledge_base(
        root=settings.knowledge_base_path,
        rebuild_indexes=body.rebuild_indexes,
    )
    return IngestResponse(
        ingested=result.ingested,
        unchanged=result.unchanged,
        errors=result.errors,
        indexed=result.indexed,
        indexed_chunks=result.indexed_chunks,
        knowledge_base_path=result.knowledge_base_path,
        documents_ingested=result.documents_ingested,
        error_paths=result.error_paths,
        duration_s=result.duration_s,
    )


@router.post("/knowledge-base/upload", response_model=UploadResponse)
async def upload_endpoint(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Upload files into the Knowledge Base and ingest via the shared pipeline.

    Files are written into ``settings.knowledge_base_path`` (created if
    missing) and then ingested through the same ``IngestionPipeline`` used for
    directory placement, so uploaded documents become part of the same corpus.

    Unsupported file types and read/write failures are reported per-file and
    never abort the batch.
    """
    store = get_evidence_store()
    root = settings.knowledge_base_path
    root.mkdir(parents=True, exist_ok=True)

    exts = set(supported_extensions())
    pipeline = None
    started = datetime.now(UTC)

    uploaded: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for upload in files:
        filename = Path(upload.filename or "unnamed").name
        suffix = Path(filename).suffix.lower()

        if not filename or suffix not in exts:
            rejected.append(
                {
                    "filename": filename,
                    "reason": f"Unsupported type '{suffix or '?'}'",
                }
            )
            continue

        destination = root / filename
        try:
            content = await upload.read()
            destination.write_bytes(content)
        except Exception as exc:  # noqa: BLE001 - per-file upload must not abort batch
            rejected.append({"filename": filename, "reason": str(exc)})
            continue

        try:
            if pipeline is None:
                from app.ingestion.pipeline import IngestionPipeline

                pipeline = IngestionPipeline(store)
            doc, was_new = ingest_file(destination, pipeline, started)
            uploaded.append(
                {
                    "filename": filename,
                    "path": str(destination),
                    "document_id": str(doc.id),
                    "version": doc.version,
                    "was_new": was_new,
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append({"filename": filename, "reason": str(exc)})

    indexed = False
    indexed_chunks = 0
    if uploaded:
        try:
            retriever = get_hybrid_retriever()
            retriever.mark_dirty()
            retriever.ensure_indexes()
            indexed = True
            indexed_chunks = store.count_chunks()
        except Exception:  # noqa: BLE001 - index refresh must not fail upload
            indexed = False

    return UploadResponse(
        uploaded=uploaded,
        rejected=rejected,
        indexed=indexed,
        indexed_chunks=indexed_chunks,
    )