"""Embedding generation (Phase 01).

Generates dense vector embeddings using local sentence-transformers model.
"""

from __future__ import annotations

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

    def __init__(self, model_name: str | None = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if EmbeddingGenerator._model is None:
            logger.info("loading_embedding_model", model=self.model_name)
            EmbeddingGenerator._model = SentenceTransformer(self.model_name)
        return EmbeddingGenerator._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Returns array of shape (n_texts, embedding_dim).
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We'll normalize in FAISS
        )
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

    # Assign embedding indices
    from app.retrieval.vector import assign_embedding_indices
    assign_embedding_indices(store)

    # Update chunks with embedding_index (already assigned by the function above)
    # Re-fetch to get the assigned indices
    chunks = store.get_chunks_by_ids(chunk_ids)

    # Build FAISS index
    from app.retrieval.vector import FAISSVectorStore
    vector_store = FAISSVectorStore(store)
    vector_store.build_index(embeddings, chunk_ids)

    logger.info("embeddings_generated", count=len(chunks))
    return len(chunks)