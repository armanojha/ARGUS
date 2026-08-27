"""SQLite-backed Memory Store (Phase 08).

Provides persistent multi-layer memory storage with full provenance.
Uses SQLite per D-003 (consistent with Evidence Store).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryLayer,
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

-- Graph Deltas: versioned changes to the evidence graph
CREATE TABLE IF NOT EXISTS graph_deltas (
    id TEXT PRIMARY KEY,
    delta_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    new_data TEXT NOT NULL DEFAULT '{}',
    old_data TEXT,
    supporting_chunk_ids TEXT NOT NULL DEFAULT '[]',
    source_query TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    is_provisional INTEGER NOT NULL DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1,
    previous_delta_id TEXT,
    created_at TEXT NOT NULL
);

-- Indexes for graph deltas
CREATE INDEX IF NOT EXISTS idx_delta_target ON graph_deltas(target_id);
CREATE INDEX IF NOT EXISTS idx_delta_type ON graph_deltas(delta_type);
CREATE INDEX IF NOT EXISTS idx_delta_provisional ON graph_deltas(is_provisional);
CREATE INDEX IF NOT EXISTS idx_delta_created_at ON graph_deltas(created_at);

-- Graph Versions: versioned snapshots
CREATE TABLE IF NOT EXISTS graph_versions (
    id TEXT PRIMARY KEY,
    version_number INTEGER NOT NULL UNIQUE,
    description TEXT NOT NULL,
    delta_ids TEXT NOT NULL DEFAULT '[]',
    node_counts TEXT NOT NULL DEFAULT '{}',
    edge_count INTEGER NOT NULL DEFAULT 0,
    parent_version_id TEXT,
    created_by_query TEXT,
    created_at TEXT NOT NULL
);

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
"""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(text: str) -> Any:
    if not text:
        return {}
    return json.loads(text)


def _serialize_tags(tags: list[str]) -> str:
    return _json_dumps(tags)


def _deserialize_tags(text: str) -> list[str]:
    return _json_loads(text)


def _serialize_uuids(uuids: list[str]) -> str:
    return _json_dumps(uuids)


def _deserialize_uuids(text: str) -> list[str]:
    return _json_loads(text)


class MemoryStore(MemoryStoreInterface):
    """SQLite-backed persistent memory store with multi-layer support."""

    def __init__(self, db_path: Path | None = None, max_records_per_layer: int | None = None):
        settings = get_settings()
        self.db_path = db_path or settings.memory_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records_per_layer = max_records_per_layer or settings.memory_max_records_per_layer
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
            conn.commit()
        logger.debug("memory_stored", memory_id=str(record.id), layer=record.layer.value)

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
                    conditions.append(f"id IN (SELECT memory_id FROM memory_tags WHERE tag = ?)")
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

    # -- Graph Delta operations --------------------------------------------------

    async def store_delta(self, delta: GraphDelta) -> None:
        """Store a graph delta."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO graph_deltas (
                    id, delta_type, target_id, target_type, new_data, old_data,
                    supporting_chunk_ids, source_query, confidence,
                    is_provisional, version, previous_delta_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(delta.id),
                    delta.delta_type.value,
                    str(delta.target_id),
                    delta.target_type,
                    _json_dumps(delta.new_data),
                    _json_dumps(delta.old_data) if delta.old_data else None,
                    _serialize_uuids(delta.supporting_chunk_ids),
                    delta.source_query,
                    delta.confidence,
                    1 if delta.is_provisional else 0,
                    delta.version,
                    str(delta.previous_delta_id) if delta.previous_delta_id else None,
                    delta.created_at.isoformat(),
                ),
            )
            conn.commit()
        logger.debug("graph_delta_stored", delta_id=str(delta.id), type=delta.delta_type.value)

    async def get_deltas_for_target(self, target_id: UUID, target_type: str) -> list[GraphDelta]:
        """Get all deltas for a specific target, ordered by version."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM graph_deltas
                WHERE target_id = ? AND target_type = ?
                ORDER BY version ASC
                """,
                (str(target_id), target_type),
            ).fetchall()
        return [self._row_to_delta(row) for row in rows]

    async def get_provisional_deltas(self, target_type: str | None = None) -> list[GraphDelta]:
        """Get all provisional (low-confidence) deltas, optionally filtered by type."""
        with self._conn() as conn:
            if target_type:
                rows = conn.execute(
                    "SELECT * FROM graph_deltas WHERE is_provisional = 1 AND target_type = ? ORDER BY created_at DESC",
                    (target_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM graph_deltas WHERE is_provisional = 1 ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_delta(row) for row in rows]

    async def promote_delta(self, delta_id: UUID) -> bool:
        """Mark a delta as promoted (no longer provisional)."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE graph_deltas SET is_provisional = 0 WHERE id = ?",
                (str(delta_id),),
            )
            conn.commit()
            return cursor.rowcount > 0

    # -- Graph Version operations ------------------------------------------------

    async def store_version(self, version: GraphVersion) -> None:
        """Store a graph version snapshot."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO graph_versions (
                    id, version_number, description, delta_ids, node_counts,
                    edge_count, parent_version_id, created_by_query, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(version.id),
                    version.version_number,
                    version.description,
                    _serialize_uuids([str(d) for d in version.delta_ids]),
                    _json_dumps(version.node_counts),
                    version.edge_count,
                    str(version.parent_version_id) if version.parent_version_id else None,
                    version.created_by_query,
                    version.created_at.isoformat(),
                ),
            )
            conn.commit()
        logger.debug("graph_version_stored", version_id=str(version.id), number=version.version_number)

    async def get_latest_version(self) -> GraphVersion | None:
        """Get the latest graph version."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM graph_versions ORDER BY version_number DESC LIMIT 1"
            ).fetchone()
        return self._row_to_version(row) if row else None

    async def get_version(self, version_number: int) -> GraphVersion | None:
        """Get a specific graph version by number."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM graph_versions WHERE version_number = ?",
                (version_number,),
            ).fetchone()
        return self._row_to_version(row) if row else None

    # -- Stats ------------------------------------------------------------------

    async def get_stats(self) -> MemoryStats:
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

        return MemoryStats(
            total_records=total,
            layer_counts=layer_counts,
            promotion_counts=promotion_counts,
            scope_counts=scope_counts,
            avg_confidence=round(avg_conf, 4),
            db_size_bytes=db_size,
        )

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

    def _row_to_delta(self, row: sqlite3.Row) -> GraphDelta:
        return GraphDelta(
            id=UUID(row["id"]),
            delta_type=GraphDeltaType(row["delta_type"]),
            target_id=UUID(row["target_id"]),
            target_type=row["target_type"],
            new_data=_json_loads(row["new_data"]),
            old_data=_json_loads(row["old_data"]) if row["old_data"] else None,
            supporting_chunk_ids=_deserialize_uuids(row["supporting_chunk_ids"]),
            source_query=row["source_query"],
            confidence=row["confidence"],
            is_provisional=bool(row["is_provisional"]),
            version=row["version"],
            previous_delta_id=UUID(row["previous_delta_id"]) if row["previous_delta_id"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_version(self, row: sqlite3.Row) -> GraphVersion:
        return GraphVersion(
            id=UUID(row["id"]),
            version_number=row["version_number"],
            description=row["description"],
            delta_ids=[UUID(d) for d in _deserialize_uuids(row["delta_ids"])],
            node_counts=_json_loads(row["node_counts"]),
            edge_count=row["edge_count"],
            parent_version_id=UUID(row["parent_version_id"]) if row["parent_version_id"] else None,
            created_by_query=row["created_by_query"],
            created_at=datetime.fromisoformat(row["created_at"]),
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
    _store = None