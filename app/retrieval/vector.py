"""FAISS Vector Store (Phase 01).

Provides dense vector retrieval using FAISS over chunk embeddings.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from app.config import get_settings
from app.evidence.store import EvidenceStore, get_evidence_store
from app.logging_config import get_logger

logger = get_logger("argus.retrieval.vector")


class FAISSVectorStore:
    """FAISS-based dense vector retriever over chunk embeddings."""

    def __init__(
        self,
        store: EvidenceStore | None = None,
        index_path: Path | None = None,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 dimension
    ):
        self.store = store or get_evidence_store()
        self.settings = get_settings()
        self.index_path = index_path or self.settings.faiss_index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        self._index: faiss.Index | None = None
        self._chunk_ids: list[str] = []  # chunk UUIDs as strings, aligned with index

    def build_index(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[UUID],
    ) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Array of shape (n_chunks, embedding_dim)
            chunk_ids: List of chunk UUIDs corresponding to embeddings
        """
        if len(embeddings) == 0:
            logger.warning("faiss_build_empty", message="No embeddings to index")
            return

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)

        # Create index (IndexFlatIP for cosine similarity via inner product)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embeddings.astype(np.float32))

        self._chunk_ids = [str(cid) for cid in chunk_ids]

        self.save_index()
        logger.info("faiss_index_built", chunk_count=len(chunk_ids), dim=self.embedding_dim)

    def save_index(self) -> None:
        """Persist FAISS index and chunk IDs to disk."""
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.index_path))
        # Save chunk IDs separately
        ids_path = self.index_path.with_suffix(".ids.pkl")
        with ids_path.open("wb") as f:
            pickle.dump(self._chunk_ids, f)
        logger.debug("faiss_index_saved", path=str(self.index_path))

    def load_index(self) -> bool:
        """Load FAISS index from disk. Returns True if successful."""
        if not self.index_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(self.index_path))
            ids_path = self.index_path.with_suffix(".ids.pkl")
            with ids_path.open("rb") as f:
                self._chunk_ids = pickle.load(f)
            logger.info("faiss_index_loaded", chunk_count=len(self._chunk_ids))
            return True
        except (OSError, pickle.PickleError, RuntimeError) as e:
            logger.error("faiss_index_load_failed", error=str(e))
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[UUID, float]]:
        """Search FAISS index.

        Returns list of (chunk_id, score) tuples sorted by score descending.
        Score is cosine similarity (inner product of normalized vectors).
        """
        if self._index is None and not self.load_index():
                logger.warning("faiss_search_no_index")
                return []

        assert self._index is not None, "FAISS index should be loaded"
        top_k = top_k or self.settings.retrieval_top_k

        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:
                chunk_id = UUID(self._chunk_ids[idx])
                results.append((chunk_id, float(score)))

        return results

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "indexed_chunks": len(self._chunk_ids),
            "embedding_dim": self.embedding_dim,
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
        }


def assign_embedding_indices(store: EvidenceStore) -> int:
    """Assign sequential embedding indices to chunks that don't have one.

    Returns the number of chunks updated.
    """
    with store._conn() as conn:
        # Find chunks without embedding_index
        rows = conn.execute(
            "SELECT id FROM chunks WHERE embedding_index IS NULL ORDER BY document_id, ordinal"
        ).fetchall()

        if not rows:
            return 0

        # Get max existing embedding_index
        max_row = conn.execute("SELECT MAX(embedding_index) FROM chunks").fetchone()
        next_idx = (max_row[0] or -1) + 1

        # Assign sequential indices
        for row in rows:
            conn.execute(
                "UPDATE chunks SET embedding_index = ? WHERE id = ?",
                (next_idx, row["id"]),
            )
            next_idx += 1

        conn.commit()

    logger.info("embedding_indices_assigned", count=len(rows))
    return len(rows)