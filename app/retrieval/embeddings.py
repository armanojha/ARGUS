"""Embedding generation (Phase 01).

Generates dense vector embeddings using local sentence-transformers model.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.evidence.models import Chunk
from app.evidence.store import EvidenceStore, get_evidence_store
from app.logging_config import get_logger

logger = get_logger("argus.retrieval.embeddings")


class EmbeddingGenerator:
    """Generates embeddings using a local sentence-transformers model."""

    _model: SentenceTransformer | None = None

    # Bounded per-instance cache for query strings -> embedding vectors. The
    # agentic loop re-retrieves with the same query across sufficiency
    # assessment iterations; caching avoids re-running the (CPU) forward pass
    # for an identical query. Cache hits are result-identical to a fresh
    # encode() of the same text. LRU-bounded so memory stays flat.
    _max_cache_entries = 64

    def __init__(self, model_name: str | None = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self._lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        with self._lock:
            if EmbeddingGenerator._model is None:
                logger.info("loading_embedding_model", model=self.model_name)
                EmbeddingGenerator._model = SentenceTransformer(self.model_name)
            return EmbeddingGenerator._model

    def cache_clear(self) -> None:
        """Clear the query-embedding cache (used when evidence is re-ingested)."""
        with self._lock:
            self._query_cache.clear()

    def _cache_get(self, key: str) -> np.ndarray | None:
        with self._lock:
            entry = self._query_cache.get(key)
            if entry is not None:
                self._query_cache.move_to_end(key)
            return entry

    def _cache_put(self, key: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._query_cache[key] = embedding
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._max_cache_entries:
                self._query_cache.popitem(last=False)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Single-element queries are memoized keyed on the exact text so the
        agentic loop's repeated retrieval of the same query skips a redundant
        forward pass. Returns array of shape (n_texts, embedding_dim).
        """
        if len(texts) == 1:
            cached = self._cache_get(texts[0])
            if cached is not None:
                return cached

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We'll normalize in FAISS
        )

        if len(texts) == 1:
            self._cache_put(texts[0], embeddings)
        return embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Generate embeddings for a list of chunks."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)


def generate_and_store_embeddings(
    store: EvidenceStore | None = None,
    generator: EmbeddingGenerator | None = None,
) -> int:
    """Generate embeddings for all chunks that don't have embedding_index.

    Returns the number of chunks embedded.
    """
    store = store or get_evidence_store()
    generator = generator or EmbeddingGenerator()

    # Get chunks without embeddings
    chunks = []
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE embedding_index IS NULL ORDER BY document_id, ordinal"
        ).fetchall()

    if not rows:
        logger.info("no_chunks_to_embed")
        return 0

    chunk_ids = [UUID(row["id"]) for row in rows]
    chunks = store.get_chunks_by_ids(chunk_ids)

    # Generate embeddings
    embeddings = generator.embed_chunks(chunks)

    # Build dict mapping chunk_id to embedding before any re-fetch
    embedding_map = {cid: emb for cid, emb in zip(chunk_ids, embeddings)}

    # Assign embedding indices
    from app.retrieval.vector import assign_embedding_indices
    assign_embedding_indices(store)

    # Re-fetch to get the assigned indices
    chunks = store.get_chunks_by_ids(chunk_ids)

    # Build FAISS index
    from app.retrieval.vector import FAISSVectorStore
    vector_store = FAISSVectorStore(store)
    ordered_embeddings = np.array([embedding_map[cid] for cid in chunk_ids])
    vector_store.build_index(ordered_embeddings, chunk_ids)

    logger.info("embeddings_generated", count=len(chunks))
    return len(chunks)