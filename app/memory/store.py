"""SQLite-backed Memory Store (Phase 08).

Provides persistent multi-layer memory storage with full provenance.
Uses SQLite per D-003 (consistent with Evidence Store).

Also hosts the versioning schema (graph_deltas, delta_chain,
claim_versions) so that the memory store and the graph version
manager share a single database.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryLayer,
    MemoryPromotionStatus,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryStoreInterface,
)

logger = get_logger("argus.memory.store")


SCHEMA_SQL = """
-- Memory Records: the core memory storage
CREATE TABLE IF NOT EXISTS memory_records (
    id TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    supporting_chunk_ids TEXT NOT NULL DEFAULT '[]',
    source_query TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    valid_from TEXT,
    valid_to TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    promotion_status TEXT NOT NULL DEFAULT 'provisional',
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id TEXT,
    superseded_by_id TEXT
);

-- Indexes for memory records
CREATE INDEX IF NOT EXISTS idx_memory_layer ON memory_records(layer);
CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_records(scope);
CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory_records(confidence);
CREATE INDEX IF NOT EXISTS idx_memory_promotion ON memory_records(promotion_status);
CREATE INDEX IF NOT EXISTS idx_memory_source_query ON memory_records(source_query);
CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_records(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_supersedes ON memory_records(supersedes_id);

-- Memory-Tags association for efficient tag queries
CREATE TABLE IF NOT EXISTS memory_tags (
    memory_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag),
    FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
);

-- Memory-Chunk association for provenance queries
CREATE TABLE IF NOT EXISTS memory_chunks (
    memory_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, chunk_id),
    FOREIGN KEY (memory_id) REFERENCES memory_records(id) ON DELETE CASCADE
);

-- =============================================================================
-- Versioning tables (Phase 08.2) — shared with GraphVersionManager
-- =============================================================================

-- Graph deltas: versioned changes to the evidence graph
CREATE TABLE IF NOT EXISTS graph_deltas (
    id TEXT PRIMARY KEY,
    delta_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'provisional',
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    previous_state TEXT,  -- JSON
    new_state TEXT,       -- JSON
    supporting_chunk_ids TEXT NOT NULL DEFAULT '[]',
    source_query TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT,
    valid_to TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    promoted_at TEXT,
    promoted_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_delta_target ON graph_deltas(target_id, target_type);
CREATE INDEX IF NOT EXISTS idx_delta_status ON graph_deltas(status);
CREATE INDEX IF NOT EXISTS idx_delta_type ON graph_deltas(delta_type);
CREATE INDEX IF NOT EXISTS idx_delta_created_at ON graph_deltas(created_at);
CREATE INDEX IF NOT EXISTS idx_delta_confidence ON graph_deltas(confidence);

-- Delta chain: links deltas that modify the same target
CREATE TABLE IF NOT EXISTS delta_chain (
    id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    delta_id TEXT NOT NULL,
    sequence_num INTEGER NOT NULL,
    FOREIGN KEY (delta_id) REFERENCES graph_deltas(id)
);

CREATE INDEX IF NOT EXISTS idx_chain_target ON delta_chain(target_id, target_type);
CREATE INDEX IF NOT EXISTS idx_chain_sequence ON delta_chain(target_id, target_type, sequence_num);

-- Claim version history (for easy inspection of claim evolution)
CREATE TABLE IF NOT EXISTS claim_versions (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    delta_id TEXT NOT NULL,
    claim_state TEXT NOT NULL,  -- Full claim JSON at this version
    confidence REAL NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (delta_id) REFERENCES graph_deltas(id)
);

CREATE INDEX IF NOT EXISTS idx_claim_version_claim ON claim_versions(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_version_current ON claim_versions(claim_id, is_current);
"""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(text: str) -> Any:
    if not text:
        return {}
    return json.loads(text)


def _serialize_tags(tags: list[str]) -> str:
    return json.dumps(tags, ensure_ascii=False)


def _deserialize_tags(text: str) -> list[str]:
    if not text:
        return []
    return json.loads(text)


def _serialize_uuids(uuids: list[str]) -> str:
    return json.dumps(uuids, ensure_ascii=False)


def _deserialize_uuids(text: str) -> list[str]:
    if not text:
        return []
    return json.loads(text)


class MemoryStore(MemoryStoreInterface):
    """SQLite-backed persistent memory store with multi-layer support."""

    def __init__(self, db_path: Path | None = None, max_records_per_layer: int | None = None):
        settings = get_settings()
        self.db_path = db_path or settings.memory_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records_per_layer = max_records_per_layer or settings.memory_max_records_per_layer
        self._conn_pool: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection (pooled across calls).

        On success the caller is expected to commit explicitly.
        On exception the connection is rolled back so the pool
        stays in a clean state.
        """
        if self._conn_pool is None:
            self._conn_pool = sqlite3.connect(self.db_path)
            self._conn_pool.row_factory = sqlite3.Row
        try:
            yield self._conn_pool
        except Exception:
            try:
                self._conn_pool.rollback()
            except sqlite3.Error:
                logger.debug("rollback_failed", exc_info=True)
            raise

    def close(self) -> None:
        """Close the connection pool."""
        if self._conn_pool is not None:
            try:
                self._conn_pool.close()
            except sqlite3.Error:
                logger.debug("pool_close_failed", exc_info=True)
            self._conn_pool = None

    # -- MemoryRecord operations -------------------------------------------------

    async def store(self, record: MemoryRecord) -> None:
        """Store a memory record with full provenance."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memory_records (
                    id, layer, scope, content, subject, predicate, object,
                    supporting_chunk_ids, source_query, confidence,
                    valid_from, valid_to, tags, metadata,
                    created_at, updated_at, promotion_status,
                    version, supersedes_id, superseded_by_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    record.layer.value,
                    record.scope.value,
                    record.content,
                    record.subject,
                    record.predicate,
                    record.object,
                    _serialize_uuids(record.supporting_chunk_ids),
                    record.source_query,
                    record.confidence,
                    record.valid_from.isoformat() if record.valid_from else None,
                    record.valid_to.isoformat() if record.valid_to else None,
                    _serialize_tags(record.tags),
                    _json_dumps(record.metadata),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.promotion_status.value,
                    record.version,
                    str(record.supersedes_id) if record.supersedes_id else None,
                    str(record.superseded_by_id) if record.superseded_by_id else None,
                ),
            )
            # Update tag index
            for tag in record.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (str(record.id), tag),
                )
            # Update chunk index
            for chunk_id in record.supporting_chunk_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_chunks (memory_id, chunk_id) VALUES (?, ?)",
                    (str(record.id), chunk_id),
                )
            # If this supersedes another record, update the old record's superseded_by_id
            if record.supersedes_id:
                conn.execute(
                    "UPDATE memory_records SET superseded_by_id = ? WHERE id = ?",
                    (str(record.id), str(record.supersedes_id)),
                )
            # Fresh-evidence-wins: auto-supersede older contradicting/outdated records
            self._resolve_fresh_evidence(conn, record)
            # Enforce layer limits within the same transaction
            self._enforce_layer_limit_sync(conn, record.layer)
            conn.commit()
        logger.debug("memory_stored", memory_id=str(record.id), layer=record.layer.value)

    def _resolve_fresh_evidence(self, conn: sqlite3.Connection, record: MemoryRecord) -> None:
        """Apply fresh-evidence-wins semantics.

        If a new record contradicts or updates an existing record on the same
        ``(subject, predicate, object, layer)`` key with a different content, the
        older record is superseded (archived) rather than hard-deleted, preserving
        history while making the newer evidence authoritative.
        """
        if not record.subject or not record.predicate:
            return
        if record.supersedes_id:
            return  # Explicit supersession already set

        older = conn.execute(
            """
            SELECT * FROM memory_records
            WHERE layer = ? AND subject = ? AND predicate = ?
              AND id != ? AND content != ?
              AND (object IS ? OR object = ?)
              AND promotion_status IN (?, ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (
                record.layer.value,
                record.subject,
                record.predicate,
                str(record.id),
                record.content,
                record.object,
                record.object,
                MemoryPromotionStatus.PROVISIONAL.value,
                MemoryPromotionStatus.PROMOTED.value,
            ),
        ).fetchone()

        if older is None:
            return

        older_id = older["id"]
        older_conf: float = older["confidence"]
        older_created = datetime.fromisoformat(older["created_at"])
        record_created = record.created_at

        # Fresh-evidence-wins: prefer a strictly newer record; ties favour the
        # higher-confidence record, and equal-confidence falls back to newer.
        newer = record_created > older_created
        if not newer and record_created == older_created:
            newer = record.confidence >= older_conf
        elif not newer and record.confidence > older_conf:
            newer = True
        if not newer:
            return

        conn.execute(
            "UPDATE memory_records SET superseded_by_id = ?, promotion_status = ? WHERE id = ?",
            (str(record.id), MemoryPromotionStatus.ARCHIVED.value, older_id),
        )
        logger.info(
            "memory_fresh_evidence_superseded",
            older_id=older_id,
            newer_id=str(record.id),
            layer=record.layer.value,
        )

    def _enforce_layer_limit_sync(self, conn: sqlite3.Connection, layer: MemoryLayer) -> None:
        """Enforce max records per layer (synchronous, within existing transaction).

        Deletes the oldest lowest-confidence records that exceed the layer
        limit.  Must be called inside an open connection context so the
        DELETE is part of the same transaction as the INSERT.
        """
        count_row = conn.execute(
            "SELECT COUNT(*) as count FROM memory_records WHERE layer = ?",
            (layer.value,),
        ).fetchone()
        count = count_row["count"] if count_row else 0

        if count > self.max_records_per_layer:
            excess = count - self.max_records_per_layer
            conn.execute(
                """
                DELETE FROM memory_records
                WHERE layer = ?
                AND id IN (
                    SELECT id FROM memory_records
                    WHERE layer = ?
                    ORDER BY confidence ASC, created_at ASC
                    LIMIT ?
                )
                """,
                (layer.value, layer.value, excess),
            )
            logger.info("memory_layer_pruned", layer=layer.value, removed=excess)

    async def _enforce_layer_limit(self, layer: MemoryLayer) -> None:
        """Enforce max records per layer (async wrapper for backward compat)."""
        with self._conn() as conn:
            self._enforce_layer_limit_sync(conn, layer)
            conn.commit()

    async def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve relevant memories for a query."""
        # Build WHERE clause
        conditions = ["1=1"]
        params: list[Any] = []

        if query.layers:
            placeholders = ",".join("?" * len(query.layers))
            conditions.append(f"layer IN ({placeholders})")
            params.extend([layer.value for layer in query.layers])

        if query.scope:
            conditions.append("scope = ?")
            params.append(query.scope.value)

        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)

        if query.tags:
            if query.tags_match_all:
                # All tags must match - use subquery
                for tag in query.tags:
                    conditions.append("id IN (SELECT memory_id FROM memory_tags WHERE tag = ?)")
                    params.append(tag)
            else:
                # Any tag matches
                placeholders = ",".join("?" * len(query.tags))
                conditions.append(f"id IN (SELECT memory_id FROM memory_tags WHERE tag IN ({placeholders}))")
                params.extend(query.tags)

        if query.time_window_start:
            conditions.append("created_at >= ?")
            params.append(query.time_window_start.isoformat())

        if query.time_window_end:
            conditions.append("created_at <= ?")
            params.append(query.time_window_end.isoformat())

        # Text search - simple LIKE-based search
        if query.query_text:
            conditions.append("(content LIKE ? OR subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR source_query LIKE ?)")
            search_term = f"%{query.query_text}%"
            params.extend([search_term] * 5)

        where_clause = " AND ".join(conditions)

        # Order by confidence DESC, then by created_at DESC for recency
        sql = f"""
            SELECT * FROM memory_records
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
        """
        params.append(query.limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    async def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Get a specific memory record by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    async def update(self, record: MemoryRecord) -> None:
        """Update an existing memory record."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE memory_records SET
                    layer = ?, scope = ?, content = ?, subject = ?, predicate = ?, object = ?,
                    supporting_chunk_ids = ?, source_query = ?, confidence = ?,
                    valid_from = ?, valid_to = ?, tags = ?, metadata = ?,
                    updated_at = ?, promotion_status = ?, version = ?,
                    supersedes_id = ?, superseded_by_id = ?
                WHERE id = ?
                """,
                (
                    record.layer.value,
                    record.scope.value,
                    record.content,
                    record.subject,
                    record.predicate,
                    record.object,
                    _serialize_uuids(record.supporting_chunk_ids),
                    record.source_query,
                    record.confidence,
                    record.valid_from.isoformat() if record.valid_from else None,
                    record.valid_to.isoformat() if record.valid_to else None,
                    _serialize_tags(record.tags),
                    _json_dumps(record.metadata),
                    record.updated_at.isoformat(),
                    record.promotion_status.value,
                    record.version,
                    str(record.supersedes_id) if record.supersedes_id else None,
                    str(record.superseded_by_id) if record.superseded_by_id else None,
                    str(record.id),
                ),
            )
            # Update tag index
            conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (str(record.id),))
            for tag in record.tags:
                conn.execute(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (str(record.id), tag),
                )
            # Update chunk index
            conn.execute("DELETE FROM memory_chunks WHERE memory_id = ?", (str(record.id),))
            for chunk_id in record.supporting_chunk_ids:
                conn.execute(
                    "INSERT INTO memory_chunks (memory_id, chunk_id) VALUES (?, ?)",
                    (str(record.id), chunk_id),
                )
            conn.commit()
        logger.debug("memory_updated", memory_id=str(record.id))

    async def delete(self, record_id: str) -> bool:
        """Delete a memory record. Returns True if deleted."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM memory_records WHERE id = ?", (record_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("memory_deleted", memory_id=record_id)
        return deleted

    async def promote_memory(self, record_id: str, new_confidence: float, reason: str = "") -> bool:
        """Promote a memory record to higher confidence."""
        record = await self.get_by_id(record_id)
        if not record:
            return False

        if new_confidence <= record.confidence:
            return False

        dump = record.model_dump(exclude={"confidence", "promotion_status", "updated_at"})
        updated_record = MemoryRecord(
            **dump,
            confidence=new_confidence,
            promotion_status=MemoryPromotionStatus.PROMOTED,
            updated_at=datetime.now(UTC),
        )
        await self.update(updated_record)
        logger.info("memory_promoted", record_id=record_id, new_confidence=new_confidence, reason=reason)
        return True

    # -- Stats ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        with self._conn() as conn:
            # Total records
            total = conn.execute("SELECT COUNT(*) as c FROM memory_records").fetchone()["c"]

            # Layer counts
            layer_rows = conn.execute(
                "SELECT layer, COUNT(*) as c FROM memory_records GROUP BY layer"
            ).fetchall()
            layer_counts = {row["layer"]: row["c"] for row in layer_rows}

            # Promotion counts
            promo_rows = conn.execute(
                "SELECT promotion_status, COUNT(*) as c FROM memory_records GROUP BY promotion_status"
            ).fetchall()
            promotion_counts = {row["promotion_status"]: row["c"] for row in promo_rows}

            # Scope counts
            scope_rows = conn.execute(
                "SELECT scope, COUNT(*) as c FROM memory_records GROUP BY scope"
            ).fetchall()
            scope_counts = {row["scope"]: row["c"] for row in scope_rows}

            # Average confidence
            avg_conf = conn.execute(
                "SELECT AVG(confidence) as avg FROM memory_records"
            ).fetchone()["avg"] or 0.0

            # DB size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_records": total,
            "layer_counts": layer_counts,
            "promotion_counts": promotion_counts,
            "scope_counts": scope_counts,
            "avg_confidence": round(avg_conf, 4),
            "db_size_bytes": db_size,
        }

    # -- Helpers -----------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=UUID(row["id"]),
            layer=MemoryLayer(row["layer"]),
            scope=MemoryScope(row["scope"]),
            content=row["content"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            supporting_chunk_ids=_deserialize_uuids(row["supporting_chunk_ids"]),
            source_query=row["source_query"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            tags=_deserialize_tags(row["tags"]),
            metadata=_json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            promotion_status=MemoryPromotionStatus(row["promotion_status"]),
            version=row["version"],
            supersedes_id=UUID(row["supersedes_id"]) if row["supersedes_id"] else None,
            superseded_by_id=UUID(row["superseded_by_id"]) if row["superseded_by_id"] else None,
        )


# Singleton instance
_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the singleton memory store."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def close_memory_store() -> None:
    """Close the singleton memory store (for testing)."""
    global _store
    if _store is not None:
        _store.close()
    _store = None