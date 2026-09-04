"""User Knowledge Base ingestion service (control plane).

This is the clean, managed entry point that turns the user's document corpus
(``settings.knowledge_base_path``) into ARGUS's EvidenceStore. It reuses the
existing ``IngestionPipeline`` (and its built-in content-checksum dedup) rather
than introducing a second ingestion architecture.

Design guarantees:
  * Deterministic, index-layer control plane: NO LLM calls are made here.
  * Recursive discovery of supported file types (PDF / TXT / Markdown /
    XLSX / XLS / XLSM / CSV).
  * Idempotent / incremental: unchanged files are skipped via the existing
    source/document checksum dedup (``source_exists`` / ``document_unchanged``),
    so repeated syncs avoid reprocessing.
  * Optional index refresh after ingestion via ``HybridRetriever.ensure_indexes``.
  * Per-file outcome reporting (ingested / unchanged / error) for the UI.

Serves both the CLI (``scripts/ingest_knowledge_base.py``) and the HTTP API
(``POST /api/v1/knowledge-base/ingest``); the UI upload path feeds the same
``IngestionPipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evidence.models import Document, SourceType
from app.evidence.store import EvidenceStore, get_evidence_store
from app.ingestion.pipeline import IngestionPipeline
from app.logging_config import get_logger
from app.retrieval.hybrid import get_hybrid_retriever

logger = get_logger("argus.ingestion.knowledge_base")

# Extensions always supported.
_ALWAYS_SUPPORTED = {".pdf", ".txt", ".md"}
# Spreadsheet extensions only when the multimodal spreadsheet feature is on.
_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv"}


@dataclass
class KnowledgeBaseIngestResult:
    """Outcome of a full or single-file knowledge-base ingestion."""

    ingested: int = 0
    unchanged: int = 0
    errors: int = 0
    documents_ingested: list[dict[str, Any]] = field(default_factory=list)
    error_paths: list[str] = field(default_factory=list)
    indexed: bool = False
    indexed_chunks: int = 0
    knowledge_base_path: str = ""
    started_at: str = ""
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return self.ingested + self.unchanged


def supported_extensions() -> list[str]:
    """Return the ordered list of supported file extensions at runtime."""
    settings = get_settings()
    exts = set(_ALWAYS_SUPPORTED)
    if settings.multimodal_spreadsheet_enabled:
        exts.update(_SPREADSHEET_EXTENSIONS)
    return sorted(exts)


def discover_files(root: Path) -> list[Path]:
    """Recursively discover supported files under ``root``.

    Returns files sorted by path for deterministic ordering.
    """
    settings = get_settings()
    exts = set(_ALWAYS_SUPPORTED)
    if settings.multimodal_spreadsheet_enabled:
        exts.update(_SPREADSHEET_EXTENSIONS)

    if not root.exists() or not root.is_dir():
        return []

    found: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in exts:
            found.append(file_path)
    return sorted(found, key=lambda p: str(p).lower())


def kind_of(file_path: Path) -> tuple[SourceType, str]:
    """Return (SourceType, suffix) for a discovered file path."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return SourceType.PDF, suffix
    if suffix in _SPREADSHEET_EXTENSIONS:
        return SourceType.SPREADSHEET, suffix
    if suffix == ".md":
        return SourceType.TEXT, suffix
    if suffix == ".txt":
        return SourceType.TEXT, suffix
    return SourceType.TEXT, suffix


def ingest_file(
    file_path: Path,
    pipeline: IngestionPipeline,
    started_at: datetime,
) -> tuple[Document, bool]:
    """Ingest a single file through the existing pipeline.

    Returns ``(document, was_new)`` where ``was_new`` is True when the file's
    document was actually created/updated during this run (versus unchanged).
    """
    source_type, suffix = kind_of(file_path)
    if suffix == ".pdf":
        doc = pipeline.ingest_pdf(file_path, source_type=source_type)
    elif suffix in _SPREADSHEET_EXTENSIONS:
        doc = pipeline.ingest_spreadsheet_file(file_path, source_type=source_type)
    else:
        doc = pipeline.ingest_text_file(file_path, source_type=source_type)
    # Unchanged documents carry their original created_at; new/updated documents
    # are stamped after this run began.
    was_new = doc.created_at >= started_at
    return doc, was_new


def ingest_knowledge_base(
    root: Path | None = None,
    store: EvidenceStore | None = None,
    *,
    rebuild_indexes: bool = True,
) -> KnowledgeBaseIngestResult:
    """Sync the configured knowledge-base directory into the EvidenceStore.

    If ``root`` is None, ``settings.knowledge_base_path`` is used.
    """
    started = datetime.now(UTC)
    settings = get_settings()
    root = root or settings.knowledge_base_path
    root = Path(root).resolve()
    store = store or get_evidence_store()
    pipeline = IngestionPipeline(store)

    result = KnowledgeBaseIngestResult(
        knowledge_base_path=str(root),
        started_at=started.isoformat(),
    )

    files = discover_files(root)
    logger.info("kb_discovery", root=str(root), files=len(files))

    for file_path in files:
        try:
            doc, was_new = ingest_file(file_path, pipeline, started)
        except (OSError, ValueError, RuntimeError) as exc:
            result.errors += 1
            result.error_paths.append(str(file_path))
            logger.warning("kb_file_error", path=str(file_path), error=str(exc))
            continue

        if was_new:
            result.ingested += 1
            result.documents_ingested.append(
                {
                    "document_id": str(doc.id),
                    "source_id": str(doc.source_id),
                    "filename": file_path.name,
                    "path": str(file_path),
                    "version": doc.version,
                }
            )
            logger.info("kb_ingested", path=str(file_path), document_id=str(doc.id))
        else:
            result.unchanged += 1
            logger.info("kb_unchanged", path=str(file_path), document_id=str(doc.id))

    if rebuild_indexes and result.ingested > 0:
        try:
            retriever = get_hybrid_retriever()
            retriever.mark_dirty()
            retriever.ensure_indexes()
            result.indexed = True
            result.indexed_chunks = store.count_chunks()
            logger.info("kb_indexes_rebuilt", chunks=result.indexed_chunks)
        except Exception as exc:  # noqa: BLE001 - index rebuild must not fail sync
            result.errors += 1
            logger.warning("kb_index_rebuild_failed", error=str(exc))

    result.duration_s = round((datetime.now(UTC) - started).total_seconds(), 3)
    logger.info(
        "kb_sync_complete",
        ingested=result.ingested,
        unchanged=result.unchanged,
        errors=result.errors,
        duration_s=result.duration_s,
    )
    return result