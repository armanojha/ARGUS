"""SQLite-backed Persistent Memory Store (Phase 08).

Implements the MemoryStoreInterface with a multi-layer memory system:
- Working memory (short-term, per-query/session)
- Long-term knowledge (verified facts, high confidence)
- Research history (past queries, plans, results)
- Source memory (source reliability, bias tracking)
- User memory (user preferences, corrections)
- Vault memory (Obsidian vault pointers, Phase 09)

All memories maintain full provenance to evidence chunks.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryLayer,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStoreInterface,
)

logger = get_logger("argus.memory.store")

# Schema for the memory database
SCHEMA_SQL = """
-- Memory records table
CREATE TABLE IF NOT EXISTS memory_records (
    id TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    content TEXT NOT NULL,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    supporting_chunk_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array of UUID strings
    source_query TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    valid_from TEXT,
    valid_to TEXT,
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array
    metadata TEXT NOT NULL DEFAULT '{}',  -- JSON object
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memory_layer ON memory_records(layer);
CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_records(scope);
CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory_records(confidence);
CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_records(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_valid_from ON memory_records(valid_from);
CREATE INDEX IF NOT EXISTS idx_memory_valid_to ON memory_records(valid_to);

-- Memory access log for analytics/debugging
CREATE TABLE IF NOT EXISTS memory_access_log (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    query_text TEXT,
    layer TEXT,
    retrieved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_access_record ON memory_access_log(record_id);
CREATE INDEX IF NOT EXISTS idx_access_retrieved_at ON memory_access_log(retrieved_at);

-- Memory promotion log (tracking promotion from provisional to promoted)
CREATE TABLE IF NOT EXISTS memory_promotion_log (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    from_confidence REAL NOT NULL,
    to_confidence REAL NOT NULL,
    promoted_at TEXT NOT NULL DEFAULT (datetime('now')),
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_promotion_record ON memory_promotion_log(record_id);
"""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(text: str) -> Any:
    if not text:
        return {}
    return json.loads(text)


class SQLiteMemoryStore(MemoryStoreInterface):
    """SQLite-backed persistent memory store implementing MemoryStoreInterface."""

    def __init__(
        self,
        db_path: Path | None = None,
        max_records_per_layer: int | None = None,
    ):
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
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=UUID(row["id"]),
            layer=MemoryLayer(row["layer"]),
            scope=MemoryScope(row["scope"]),
            content=row["content"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            supporting_chunk_ids=_json_loads(row["supporting_chunk_ids"]),
            source_query=row["source_query"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            tags=_json_loads(row["tags"]),
            metadata=_json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def store(self, record: MemoryRecord) -> None:
        """Store a memory record."""
        # Enforce layer limits
        await self._enforce_layer_limit(record.layer)

        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_records (
                    id, layer, scope, content, subject, predicate, object,
                    supporting_chunk_ids, source_query, confidence,
                    valid_from, valid_to, tags, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    record.layer.value,
                    record.scope.value,
                    record.content,
                    record.subject,
                    record.predicate,
                    record.object,
                    _json_dumps(record.supporting_chunk_ids),
                    record.source_query,
                    record.confidence,
                    record.valid_from.isoformat() if record.valid_from else None,
                    record.valid_to.isoformat() if record.valid_to else None,
                    _json_dumps(record.tags),
                    _json_dumps(record.metadata),
                    record.created_at.isoformat(),
                    now,
                ),
            )
            conn.commit()

        logger.debug("memory_stored", record_id=str(record.id), layer=record.layer.value)

    async def retrieve(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve relevant memories for a query."""
        conditions = []
        params = []

        # Layer filter
        if query.layers:
            placeholders = ",".join("?" * len(query.layers))
            conditions.append(f"layer IN ({placeholders})")
            params.extend([layer.value for layer in query.layers])

        # Scope filter
        if query.scope:
            conditions.append("scope = ?")
            params.append(query.scope.value)

        # Confidence filter
        conditions.append("confidence >= ?")
        params.append(query.min_confidence)

        # Tags filter
        if query.tags:
            if query.tags_match_all:
                # All tags must match - use LIKE for each tag
                for tag in query.tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
            else:
                # Any tag matches
                tag_conditions = " OR ".join(["tags LIKE ?"] * len(query.tags))
                conditions.append(f"({tag_conditions})")
                params.extend([f"%{tag}%" for tag in query.tags])

        # Time window filter
        if query.time_window_start:
            conditions.append("(valid_to IS NULL OR valid_to >= ?)")
            params.append(query.time_window_start.isoformat())
        if query.time_window_end:
            conditions.append("(valid_from IS NULL OR valid_from <= ?)")
            params.append(query.time_window_end.isoformat())

        # Text search (simple LIKE-based for MVP, can be upgraded to FTS5)
        if query.query_text:
            conditions.append("(content LIKE ? OR subject LIKE ? OR predicate LIKE ? OR object LIKE ?)")
            search_term = f"%{query.query_text}%"
            params.extend([search_term] * 4)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT * FROM memory_records
            WHERE {where_clause}
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
        """
        params.append(query.limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_record(row) for row in rows]

        # Log access for analytics
        if records:
            await self._log_access(records, query.query_text)

        return records

    async def _log_access(self, records: list[MemoryRecord], query_text: str) -> None:
        """Log memory access for analytics."""
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            for record in records:
                conn.execute(
                    """
                    INSERT INTO memory_access_log (id, record_id, query_text, layer, retrieved_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid4()), str(record.id), query_text, record.layer.value, now),
                )
            conn.commit()

    async def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Get a specific memory record by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    async def update(self, record: MemoryRecord) -> None:
        """Update an existing memory record."""
        record = MemoryRecord(
            **record.model_dump(),
            updated_at=datetime.now(UTC),
        )
        await self.store(record)

    async def delete(self, record_id: str) -> bool:
        """Delete a memory record. Returns True if deleted."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM memory_records WHERE id = ?", (record_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("memory_deleted", record_id=record_id)
        return deleted

    async def _enforce_layer_limit(self, layer: MemoryLayer) -> None:
        """Enforce max records per layer by removing oldest lowest-confidence records."""
        with self._conn() as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) as count FROM memory_records WHERE layer = ?",
                (layer.value,),
            ).fetchone()
            count = count_row["count"] if count_row else 0

            if count >= self.max_records_per_layer:
                # Delete oldest lowest-confidence records to make room
                excess = count - self.max_records_per_layer + 100  # Delete in batches
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
                conn.commit()
                logger.info("memory_layer_pruned", layer=layer.value, removed=excess)

    async def promote_memory(self, record_id: str, new_confidence: float, reason: str = "") -> bool:
        """Promote a memory record to higher confidence (e.g., provisional -> verified)."""
        record = await self.get_by_id(record_id)
        if not record:
            return False

        old_confidence = record.confidence
        if new_confidence <= old_confidence:
            return False

        updated_record = MemoryRecord(
            **record.model_dump(),
            confidence=new_confidence,
            updated_at=datetime.now(UTC),
        )
        await self.store(updated_record)

        # Log promotion
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memory_promotion_log (id, record_id, from_confidence, to_confidence, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid4()), record_id, old_confidence, new_confidence, reason),
            )
            conn.commit()

        logger.info("memory_promoted", record_id=record_id, from_confidence=old_confidence, to_confidence=new_confidence)
        return True

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        with self._conn() as conn:
            # Total records
            total_row = conn.execute("SELECT COUNT(*) as count FROM memory_records").fetchone()
            total = total_row["count"] if total_row else 0

            # By layer
            layer_rows = conn.execute(
                "SELECT layer, COUNT(*) as count FROM memory_records GROUP BY layer"
            ).fetchall()
            by_layer = {row["layer"]: row["count"] for row in layer_rows}

            # By scope
            scope_rows = conn.execute(
                "SELECT scope, COUNT(*) as count FROM memory_records GROUP BY scope"
            ).fetchall()
            by_scope = {row["scope"]: row["count"] for row in scope_rows}

            # Average confidence
            conf_row = conn.execute("SELECT AVG(confidence) as avg_conf FROM memory_records").fetchone()
            avg_confidence = conf_row["avg_conf"] if conf_row and conf_row["avg_conf"] is not None else 0.0

            # Recent activity (last 24h)
            recent_row = conn.execute(
                "SELECT COUNT(*) as count FROM memory_records WHERE created_at >= datetime('now', '-1 day')"
            ).fetchone()
            recent_24h = recent_row["count"] if recent_row else 0

        return {
            "total_records": total,
            "by_layer": by_layer,
            "by_scope": by_scope,
            "average_confidence": round(avg_confidence, 3),
            "recent_24h": recent_24h,
            "max_records_per_layer": self.max_records_per_layer,
            "db_path": str(self.db_path),
        }

    async def get_layer_records(self, layer: MemoryLayer, limit: int = 100) -> list[MemoryRecord]:
        """Get all records for a specific layer (for inspection/debugging)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_records WHERE layer = ? ORDER BY confidence DESC, created_at DESC LIMIT ?",
                (layer.value, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def get_recent_research_history(self, limit: int = 10) -> list[MemoryRecord]:
        """Get recent research history records."""
        return await self.retrieve(
            MemoryQuery(
                query_text="",
                layers=[MemoryLayer.RESEARCH_HISTORY],
                limit=limit,
            )
        )


# Singleton instance
_memory_store: SQLiteMemoryStore | None = None


def get_memory_store() -> SQLiteMemoryStore:
    """Get or create the singleton memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = SQLiteMemoryStore()
    return _memory_store


def close_memory_store() -> None:
    """Close the singleton memory store (for testing)."""
    global _memory_store
    _memory_store = None