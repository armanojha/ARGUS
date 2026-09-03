"""BM25 Lexical Retrieval (Phase 01).

Provides sparse vector retrieval using BM25 over chunk texts.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.evidence.models import Chunk
from app.evidence.store import EvidenceStore, get_evidence_store
from app.logging_config import get_logger

logger = get_logger("argus.retrieval.bm25")


class BM25Retriever:
    """BM25-based lexical retriever over chunk corpus."""

    def __init__(
        self,
        store: EvidenceStore | None = None,
        index_path: Path | None = None,
    ):
        self.store = store or get_evidence_store()
        self.settings = get_settings()
        self.index_path = index_path or self.settings.bm25_index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []  # chunk UUIDs as strings
        self._corpus_tokens: list[list[str]] = []

    def build_index(self, chunks: list[Chunk] | None = None) -> None:
        """Build BM25 index from chunks.

        If chunks not provided, fetches all chunks from store.
        """
        if chunks is None:
            # Fetch all chunks with bm25_doc_id assigned via the public read API.
            chunks = self.store.iter_chunks()

        if not chunks:
            logger.warning("bm25_build_empty", message="No chunks to index")
            self._bm25 = None
            self._chunk_ids = []
            self._corpus_tokens = []
            return

        # Tokenize corpus
        self._corpus_tokens = [self._tokenize(chunk.text) for chunk in chunks]
        self._chunk_ids = [str(chunk.id) for chunk in chunks]

        # Build BM25 index
        self._bm25 = BM25Okapi(self._corpus_tokens)

        # Save index
        self.save_index()

        logger.info("bm25_index_built", chunk_count=len(chunks))

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def save_index(self) -> None:
        """Persist BM25 index to disk."""
        if self._bm25 is None:
            return
        data = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "corpus_tokens": self._corpus_tokens,
        }
        with self.index_path.open("wb") as f:
            pickle.dump(data, f)
        logger.debug("bm25_index_saved", path=str(self.index_path))

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if not self.index_path.exists():
            return False
        try:
            with self.index_path.open("rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            self._corpus_tokens = data["corpus_tokens"]
            logger.info("bm25_index_loaded", chunk_count=len(self._chunk_ids))
            return True
        except (OSError, pickle.PickleError, KeyError) as e:
            logger.error("bm25_index_load_failed", error=str(e))
            return False

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[UUID, float]]:
        """Search BM25 index.

        Returns list of (chunk_id, score) tuples sorted by score descending.
        """
        if self._bm25 is None and not self.load_index():
            logger.warning("bm25_search_no_index")
            return []

        if self._bm25 is None:
            raise RuntimeError("BM25 index not loaded")
        top_k = top_k or self.settings.retrieval_top_k
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = UUID(self._chunk_ids[idx])
                results.append((chunk_id, float(scores[idx])))

        return results

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "indexed_chunks": len(self._chunk_ids),
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
        }


def assign_bm25_doc_ids(store: EvidenceStore) -> int:
    """Assign sequential BM25 doc IDs to chunks that don't have one.

    Returns the number of chunks updated.
    """
    with store._conn() as conn:
        # Find chunks without bm25_doc_id
        rows = conn.execute(
            "SELECT id FROM chunks WHERE bm25_doc_id IS NULL ORDER BY document_id, ordinal"
        ).fetchall()

        if not rows:
            return 0

        # Get max existing bm25_doc_id
        max_row = conn.execute("SELECT MAX(bm25_doc_id) FROM chunks").fetchone()
        next_id = (max_row[0] or 0) + 1

        # Assign sequential IDs
        for row in rows:
            conn.execute(
                "UPDATE chunks SET bm25_doc_id = ? WHERE id = ?",
                (next_id, row["id"]),
            )
            next_id += 1

        conn.commit()

    logger.info("bm25_doc_ids_assigned", count=len(rows))
    return len(rows)