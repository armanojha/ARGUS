"""SQLite Evidence Store (Phase 01).

Provides deterministic, ACID-compliant storage for Sources, Documents,
and Chunks with full provenance tracking. Uses SQLite per D-003.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.evidence.models import Chunk, Document, EvidenceRef, Source, SourceType
from app.logging_config import get_logger

logger = get_logger("argus.evidence.store")

SCHEMA_SQL = """
-- Sources: original document origins
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Documents: versioned processed documents from sources
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    version INTEGER NOT NULL DEFAULT 1,
    checksum TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source_id, version)
);

-- Chunks: canonical text spans with provenance
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    section_path TEXT,
    embedding_index INTEGER,
    bm25_doc_id INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(document_id, ordinal)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_idx ON chunks(embedding_index);
CREATE INDEX IF NOT EXISTS idx_chunks_bm25_doc_id ON chunks(bm25_doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
"""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(text: str) -> Any:
    if not text:
        return {}
    return json.loads(text)


class EvidenceStore:
    """SQLite-backed evidence store with full provenance."""

    def __init__(
        self,
        db_path: Path | None = None,
        bm25_index_path: Path | None = None,
        faiss_index_path: Path | None = None,
    ):
        settings = get_settings()
        self.db_path = db_path or settings.evidence_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path = bm25_index_path or settings.bm25_index_path
        self.faiss_index_path = faiss_index_path or settings.faiss_index_path
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # -- Source operations -------------------------------------------------

    def upsert_source(self, source: Source) -> Source:
        """Insert or update a source. Returns the existing source on conflict, new source otherwise."""
        # First check if source with this checksum already exists
        existing = self.get_source_by_checksum(source.checksum)
        if existing:
            # Update the existing source
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE sources SET path=?, type=?, metadata=?, updated_at=?
                    WHERE checksum=?
                    """,
                    (
                        source.path,
                        source.type.value,
                        _json_dumps(source.metadata),
                        source.updated_at.isoformat(),
                        source.checksum,
                    ),
                )
                conn.commit()
            return existing

        # No existing source, insert new
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, type, path, checksum, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(source.id),
                    source.type.value,
                    source.path,
                    source.checksum,
                    _json_dumps(source.metadata),
                    source.created_at.isoformat(),
                    source.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return source

    def get_source_by_checksum(self, checksum: str) -> Source | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE checksum = ?", (checksum,)
            ).fetchone()
        return self._row_to_source(row) if row else None

    def get_source(self, source_id: UUID) -> Source | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (str(source_id),)).fetchone()
        return self._row_to_source(row) if row else None

    # -- Document operations -----------------------------------------------

    def insert_document(self, document: Document) -> Document:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source_id, version, checksum, chunking_strategy, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    str(document.source_id),
                    document.version,
                    document.checksum,
                    document.chunking_strategy,
                    _json_dumps(document.metadata),
                    document.created_at.isoformat(),
                ),
            )
            conn.commit()
        return document

    def get_document(self, document_id: UUID) -> Document | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (str(document_id),)).fetchone()
        return self._row_to_document(row) if row else None

    def get_latest_document_for_source(self, source_id: UUID) -> Document | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_id = ? ORDER BY version DESC LIMIT 1",
                (str(source_id),),
            ).fetchone()
        return self._row_to_document(row) if row else None

    # -- Chunk operations --------------------------------------------------

    def insert_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        with self._conn() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, ordinal, text, token_count,
                        page_start, page_end, char_start, char_end, section_path,
                        embedding_index, bm25_doc_id, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(chunk.id),
                        str(chunk.document_id),
                        chunk.ordinal,
                        chunk.text,
                        chunk.token_count,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section_path,
                        chunk.embedding_index,
                        chunk.bm25_doc_id,
                        _json_dumps(chunk.metadata),
                        chunk.created_at.isoformat(),
                    ),
                )
            conn.commit()
        return chunks

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (str(chunk_id),)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal",
                (str(document_id),),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                [str(cid) for cid in chunk_ids],
            ).fetchall()
        # Preserve input order
        chunk_map = {self._row_to_chunk(row).id: self._row_to_chunk(row) for row in rows}
        return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]

    def get_chunks_by_embedding_indices(self, indices: list[int]) -> list[Chunk]:
        if not indices:
            return []
        placeholders = ",".join("?" * len(indices))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE embedding_index IN ({placeholders})",
                indices,
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunks_by_bm25_ids(self, bm25_ids: list[int]) -> list[Chunk]:
        if not bm25_ids:
            return []
        placeholders = ",".join("?" * len(bm25_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE bm25_doc_id IN ({placeholders})",
                bm25_ids,
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # -- Bulk retrieval ----------------------------------------------------

    def get_all_chunk_ids(self) -> list[UUID]:
        """Return all chunk IDs ordered by document_id and ordinal."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks ORDER BY document_id, ordinal"
            ).fetchall()
        return [UUID(row["id"]) for row in rows]

    # -- Citation / EvidenceRef -------------------------------------------

    def get_evidence_refs(self, chunk_ids: list[UUID], scores: list[float]) -> list[EvidenceRef]:
        """Build EvidenceRef objects with full citation info for retrieved chunks."""
        if not chunk_ids:
            return []

        chunks = self.get_chunks_by_ids(chunk_ids)
        if not chunks:
            return []

        # Fetch source info for each chunk
        refs = []
        for rank, (chunk, score) in enumerate(zip(chunks, scores), 1):
            document = self.get_document(chunk.document_id)
            if not document:
                continue
            source = self.get_source(document.source_id)
            if not source:
                continue

            refs.append(EvidenceRef(
                chunk_id=chunk.id,
                document_id=document.id,
                source_id=source.id,
                source_path=source.path,
                source_type=source.type,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                score=score,
                rank=rank,
            ))
        return refs

    # -- Helpers -----------------------------------------------------------

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        return Source(
            id=UUID(row["id"]),
            type=SourceType(row["type"]),
            path=row["path"],
            checksum=row["checksum"],
            metadata=_json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=UUID(row["id"]),
            source_id=UUID(row["source_id"]),
            version=row["version"],
            checksum=row["checksum"],
            chunking_strategy=row["chunking_strategy"],
            metadata=_json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            ordinal=row["ordinal"],
            text=row["text"],
            token_count=row["token_count"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            section_path=row["section_path"],
            embedding_index=row["embedding_index"],
            bm25_doc_id=row["bm25_doc_id"],
            metadata=_json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def compute_checksum(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


# Singleton instance
_store: EvidenceStore | None = None


def get_evidence_store() -> EvidenceStore:
    global _store
    if _store is None:
        _store = EvidenceStore()
    return _store