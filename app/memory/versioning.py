"""Graph Versioning and Deltas (Phase 08.2).

Implements versioned, non-destructive graph updates with:
- Delta tracking for all graph mutations
- Provisional vs promoted confidence handling
- Traceability of all prior claim states
- Automatic promotion of high-confidence updates
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("argus.memory.versioning")


class DeltaType(str, Enum):
    """Type of graph delta operation."""
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_MERGED = "entity_merged"
    CLAIM_CREATED = "claim_created"
    CLAIM_UPDATED = "claim_updated"
    CLAIM_CONTRADICTED = "claim_contradicted"
    CLAIM_SUPERSEDED = "claim_superseded"
    CLAIM_REVISED = "claim_revised"
    EVENT_CREATED = "event_created"
    EVENT_UPDATED = "event_updated"
    EDGE_ADDED = "edge_added"
    EDGE_UPDATED = "edge_updated"
    EDGE_REMOVED = "edge_removed"


class DeltaStatus(str, Enum):
    """Status of a delta (provisional vs promoted)."""
    PROVISIONAL = "provisional"      # Low confidence, needs verification
    PROMOTED = "promoted"            # High confidence, auto-promoted
    REJECTED = "rejected"            # Contradicted or invalidated
    SUPERSEDED = "superseded"        # Replaced by newer delta


@dataclass(frozen=True)
class GraphDelta:
    """A versioned delta representing a graph mutation."""
    id: UUID
    delta_type: DeltaType
    status: DeltaStatus
    # Target node/edge
    target_id: UUID
    target_type: str  # 'entity', 'claim', 'event', 'edge'
    # Change details
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    # Provenance
    supporting_chunk_ids: list[UUID] = field(default_factory=list)
    source_query: str | None = None
    confidence: float = 0.5
    # Temporal
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    promoted_by: str | None = None  # 'auto' or 'manual'

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "delta_type": self.delta_type.value,
            "status": self.status.value,
            "target_id": str(self.target_id),
            "target_type": self.target_type,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "supporting_chunk_ids": [str(cid) for cid in self.supporting_chunk_ids],
            "source_query": self.source_query,
            "confidence": self.confidence,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "promoted_by": self.promoted_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphDelta:
        return cls(
            id=UUID(data["id"]),
            delta_type=DeltaType(data["delta_type"]),
            status=DeltaStatus(data["status"]),
            target_id=UUID(data["target_id"]),
            target_type=data["target_type"],
            previous_state=data.get("previous_state"),
            new_state=data.get("new_state"),
            supporting_chunk_ids=[UUID(cid) for cid in data.get("supporting_chunk_ids", [])],
            source_query=data.get("source_query"),
            confidence=data.get("confidence", 0.5),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_to=datetime.fromisoformat(data["valid_to"]) if data.get("valid_to") else None,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            promoted_by=data.get("promoted_by"),
        )


# Schema for versioning database
VERSIONING_SCHEMA = """
-- Graph deltas table
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


class GraphVersionManager:
    """Manages versioned graph deltas with provisional/promoted handling."""

    def __init__(
        self,
        db_path: Path | None = None,
        confidence_threshold: float | None = None,
        retention_days: int | None = None,
    ):
        settings = get_settings()
        self.db_path = db_path or (settings.data_dir / "memory" / "graph_deltas.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold or settings.memory_confidence_threshold
        self.retention_days = retention_days or settings.graph_delta_retention_days
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(VERSIONING_SCHEMA)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _json_dumps(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    def _json_loads(self, text: str | None) -> Any:
        if not text:
            return None
        return json.loads(text)

    def record_delta(self, delta: GraphDelta) -> GraphDelta:
        """Record a new graph delta."""
        with self._conn() as conn:
            # Insert delta
            conn.execute(
                """
                INSERT INTO graph_deltas (
                    id, delta_type, status, target_id, target_type,
                    previous_state, new_state, supporting_chunk_ids,
                    source_query, confidence, valid_from, valid_to,
                    metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(delta.id),
                    delta.delta_type.value,
                    delta.status.value,
                    str(delta.target_id),
                    delta.target_type,
                    self._json_dumps(delta.previous_state),
                    self._json_dumps(delta.new_state),
                    self._json_dumps([str(cid) for cid in delta.supporting_chunk_ids]),
                    delta.source_query,
                    delta.confidence,
                    delta.valid_from.isoformat() if delta.valid_from else None,
                    delta.valid_to.isoformat() if delta.valid_to else None,
                    self._json_dumps(delta.metadata),
                    delta.created_at.isoformat(),
                ),
            )

            # Add to delta chain
            sequence_num = self._get_next_sequence(conn, delta.target_id, delta.target_type)
            conn.execute(
                """
                INSERT INTO delta_chain (id, target_id, target_type, delta_id, sequence_num)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid4()), str(delta.target_id), delta.target_type, str(delta.id), sequence_num),
            )

            # If this is a claim, record version
            if delta.target_type == "claim" and delta.new_state:
                conn.execute(
                    """
                    INSERT INTO claim_versions (id, claim_id, version, delta_id, claim_state, confidence, is_current, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        str(delta.target_id),
                        sequence_num,
                        str(delta.id),
                        self._json_dumps(delta.new_state),
                        delta.confidence,
                        1,  # is_current
                        delta.created_at.isoformat(),
                    ),
                )
                # Mark previous versions as not current
                conn.execute(
                    "UPDATE claim_versions SET is_current = 0 WHERE claim_id = ? AND version < ?",
                    (str(delta.target_id), sequence_num),
                )

            conn.commit()

        # Auto-promote if confidence is high enough
        if delta.confidence >= self.confidence_threshold and delta.status == DeltaStatus.PROVISIONAL:
            self.promote_delta(delta.id, "auto")

        logger.debug("delta_recorded", delta_id=str(delta.id), type=delta.delta_type.value, status=delta.status.value)
        return delta

    def _get_next_sequence(self, conn: sqlite3.Connection, target_id: UUID, target_type: str) -> int:
        row = conn.execute(
            "SELECT MAX(sequence_num) as max_seq FROM delta_chain WHERE target_id = ? AND target_type = ?",
            (str(target_id), target_type),
        ).fetchone()
        return (row["max_seq"] + 1) if row and row["max_seq"] is not None else 1

    def promote_delta(self, delta_id: UUID | str, promoted_by: str = "manual") -> bool:
        """Promote a provisional delta to promoted status."""
        delta_id_str = str(delta_id)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM graph_deltas WHERE id = ?", (delta_id_str,)
            ).fetchone()
            if not row:
                return False

            if row["status"] != DeltaStatus.PROVISIONAL.value:
                return False  # Already promoted/rejected

            conn.execute(
                """
                UPDATE graph_deltas
                SET status = ?, promoted_at = ?, promoted_by = ?
                WHERE id = ?
                """,
                (DeltaStatus.PROMOTED.value, datetime.now(UTC).isoformat(), promoted_by, delta_id_str),
            )
            conn.commit()

        logger.info("delta_promoted", delta_id=delta_id_str, promoted_by=promoted_by)
        return True

    def reject_delta(self, delta_id: UUID | str, reason: str = "") -> bool:
        """Reject a provisional delta."""
        delta_id_str = str(delta_id)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM graph_deltas WHERE id = ?", (delta_id_str,)
            ).fetchone()
            if not row:
                return False

            conn.execute(
                "UPDATE graph_deltas SET status = ? WHERE id = ?",
                (DeltaStatus.REJECTED.value, delta_id_str),
            )
            conn.commit()

        logger.info("delta_rejected", delta_id=delta_id_str, reason=reason)
        return True

    def get_delta(self, delta_id: UUID | str) -> GraphDelta | None:
        """Get a specific delta by ID."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM graph_deltas WHERE id = ?", (str(delta_id),)).fetchone()
        return self._row_to_delta(row) if row else None

    def get_deltas_for_target(
        self,
        target_id: UUID | str,
        target_type: str,
        include_provisional: bool = True,
    ) -> list[GraphDelta]:
        """Get all deltas for a specific graph target, ordered by sequence."""
        with self._conn() as conn:
            query = """
                SELECT d.* FROM graph_deltas d
                JOIN delta_chain c ON c.delta_id = d.id
                WHERE c.target_id = ? AND c.target_type = ?
            """
            params = [str(target_id), target_type]
            if not include_provisional:
                query += " AND d.status != ?"
                params.append(DeltaStatus.PROVISIONAL.value)
            query += " ORDER BY c.sequence_num"
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_delta(row) for row in rows]

    def get_claim_history(self, claim_id: UUID | str) -> list[dict[str, Any]]:
        """Get full version history for a claim."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM claim_versions
                WHERE claim_id = ?
                ORDER BY version
                """,
                (str(claim_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_provisional_deltas(self, limit: int = 100) -> list[GraphDelta]:
        """Get all provisional deltas pending promotion/rejection."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM graph_deltas
                WHERE status = ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (DeltaStatus.PROVISIONAL.value, limit),
            ).fetchall()
        return [self._row_to_delta(row) for row in rows]

    def auto_promote_eligible(self) -> int:
        """Auto-promote all provisional deltas above confidence threshold."""
        promoted = 0
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id FROM graph_deltas
                WHERE status = ? AND confidence >= ?
                """,
                (DeltaStatus.PROVISIONAL.value, self.confidence_threshold),
            ).fetchall()

            for row in rows:
                conn.execute(
                    """
                    UPDATE graph_deltas
                    SET status = ?, promoted_at = ?, promoted_by = ?
                    WHERE id = ?
                    """,
                    (DeltaStatus.PROMOTED.value, datetime.now(UTC).isoformat(), "auto", row["id"]),
                )
                promoted += 1
            conn.commit()

        if promoted:
            logger.info("auto_promoted_deltas", count=promoted)
        return promoted

    def cleanup_old_deltas(self) -> int:
        """Remove deltas older than retention period (keeping promoted ones)."""
        cutoff = datetime.now(UTC).isoformat()
        # SQLite date arithmetic
        with self._conn() as conn:
            cursor = conn.execute(
                """
                DELETE FROM graph_deltas
                WHERE created_at < datetime(?, ?)
                AND status != ?
                """,
                (cutoff, f"-{self.retention_days} days", DeltaStatus.PROMOTED.value),
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted:
            logger.info("old_deltas_cleaned", count=deleted, retention_days=self.retention_days)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get versioning statistics."""
        with self._conn() as conn:
            total_row = conn.execute("SELECT COUNT(*) as count FROM graph_deltas").fetchone()
            total = total_row["count"] if total_row else 0

            status_rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM graph_deltas GROUP BY status"
            ).fetchall()
            by_status = {row["status"]: row["count"] for row in status_rows}

            type_rows = conn.execute(
                "SELECT delta_type, COUNT(*) as count FROM graph_deltas GROUP BY delta_type"
            ).fetchall()
            by_type = {row["delta_type"]: row["count"] for row in type_rows}

            chain_row = conn.execute("SELECT COUNT(DISTINCT target_id) as count FROM delta_chain").fetchone()
            unique_targets = chain_row["count"] if chain_row else 0

        return {
            "total_deltas": total,
            "by_status": by_status,
            "by_type": by_type,
            "unique_targets_versioned": unique_targets,
            "confidence_threshold": self.confidence_threshold,
            "retention_days": self.retention_days,
            "db_path": str(self.db_path),
        }

    def _row_to_delta(self, row: sqlite3.Row) -> GraphDelta:
        return GraphDelta(
            id=UUID(row["id"]),
            delta_type=DeltaType(row["delta_type"]),
            status=DeltaStatus(row["status"]),
            target_id=UUID(row["target_id"]),
            target_type=row["target_type"],
            previous_state=self._json_loads(row["previous_state"]),
            new_state=self._json_loads(row["new_state"]),
            supporting_chunk_ids=[UUID(cid) for cid in self._json_loads(row["supporting_chunk_ids"])],
            source_query=row["source_query"],
            confidence=row["confidence"],
            valid_from=datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None,
            valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            metadata=self._json_loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            promoted_at=datetime.fromisoformat(row["promoted_at"]) if row["promoted_at"] else None,
            promoted_by=row["promoted_by"],
        )


# Singleton instance
_version_manager: GraphVersionManager | None = None


def get_version_manager() -> GraphVersionManager:
    """Get or create the singleton graph version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = GraphVersionManager()
    return _version_manager


def close_version_manager() -> None:
    """Close the singleton version manager (for testing)."""
    global _version_manager
    _version_manager = None