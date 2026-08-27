"""Hybrid Retrieval Orchestrator (Phase 01).

Combines BM25 lexical retrieval with FAISS dense vector retrieval.
"""

from __future__ import annotations

from app.config import get_settings
from app.evidence.models import EvidenceRef
from app.evidence.store import EvidenceStore, get_evidence_store
from app.logging_config import get_logger
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices

logger = get_logger("argus.retrieval.hybrid")


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense vector search."""

    def __init__(
        self,
        store: EvidenceStore | None = None,
        bm25: BM25Retriever | None = None,
        vector: FAISSVectorStore | None = None,
        embedder: EmbeddingGenerator | None = None,
    ):
        self.store = store or get_evidence_store()
        self.settings = get_settings()
        self.bm25 = bm25 or BM25Retriever(self.store)
        self.vector = vector or FAISSVectorStore(self.store)
        self.embedder = embedder or EmbeddingGenerator()
        self._dirty = True

    def mark_dirty(self) -> None:
        """Mark indexes as needing rebuild on next search."""
        self._dirty = True

    def ensure_indexes(self) -> None:
        """Ensure both BM25 and FAISS indexes are built and up to date with current store data."""
        if not self._dirty:
            return

        # Assign BM25 doc IDs if needed
        assign_bm25_doc_ids(self.store)

        # Assign embedding indices if needed
        assign_embedding_indices(self.store)

        # Get all chunk IDs from store via public API
        chunk_ids = self.store.get_all_chunk_ids()
        if not chunk_ids:
            logger.warning("no_chunks_to_index")
            return

        chunks = self.store.get_chunks_by_ids(chunk_ids)

        # Build BM25 index (force rebuild)
        self.bm25.build_index(chunks)

        # Generate embeddings for all chunks
        embeddings = self.embedder.embed_chunks(chunks)

        # Build FAISS index (force rebuild)
        self.vector.build_index(embeddings, chunk_ids)

        self._dirty = False
        logger.info("indexes_ensured", chunk_count=len(chunks))

    def search(
        self,
        query: str,
        top_k: int | None = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[EvidenceRef]:
        """Hybrid search combining BM25 and vector scores.

        Args:
            query: Search query string
            top_k: Number of results to return (after fusion)
            bm25_weight: Weight for BM25 scores in fusion
            vector_weight: Weight for vector scores in fusion

        Returns:
            List of EvidenceRef with fused scores, sorted by fused score.
        """
        top_k = top_k or self.settings.retrieval_top_k

        # Validate and normalize weights
        total_weight = bm25_weight + vector_weight
        if abs(total_weight - 1.0) > 1e-6:
            if total_weight == 0:
                bm25_weight, vector_weight = 0.5, 0.5
            else:
                bm25_weight /= total_weight
                vector_weight /= total_weight

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        bm25_scores = {cid: score for cid, score in bm25_results}
        logger.debug("hybrid_search_bm25", query=query, bm25_results=len(bm25_results))

        # Vector search
        query_embedding = self.embedder.embed_texts([query])[0]
        vector_results = self.vector.search(query_embedding, top_k=top_k * 2)
        vector_scores = {cid: score for cid, score in vector_results}
        logger.debug("hybrid_search_vector", query=query, vector_results=len(vector_results))

        # Fuse scores (reciprocal rank fusion or weighted sum)
        all_chunk_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        logger.debug("hybrid_search_fusion", all_chunk_ids=len(all_chunk_ids))

        fused_results = []
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            vector_score = vector_scores.get(chunk_id, 0.0)

            # Normalize scores to [0, 1] range (rough approximation)
            # BM25 scores can be unbounded, vector scores are cosine similarity [0, 1]
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            norm_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            norm_vector = vector_score  # already [0, 1]

            fused_score = bm25_weight * norm_bm25 + vector_weight * norm_vector
            fused_results.append((chunk_id, fused_score, bm25_score, vector_score))

        # Sort by fused score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        top_results = fused_results[:top_k]

        logger.debug("hybrid_search_fused", top_results=len(top_results))

        # Build EvidenceRefs
        chunk_ids = [cid for cid, _, _, _ in top_results]
        fused_scores = [score for _, score, _, _ in top_results]

        evidence_refs = self.store.get_evidence_refs(chunk_ids, fused_scores)

        # Add component scores to metadata
        for i, ref in enumerate(evidence_refs):
            _, _, bm25_s, vec_s = top_results[i]
            ref.metadata["bm25_score"] = bm25_s
            ref.metadata["vector_score"] = vec_s
            ref.metadata["fused_score"] = fused_scores[i]

        logger.info("hybrid_search", query=query[:50], results=len(evidence_refs))
        return evidence_refs

    def search_bm25_only(self, query: str, top_k: int | None = None) -> list[EvidenceRef]:
        """Search using BM25 only."""
        top_k = top_k or self.settings.retrieval_top_k
        results = self.bm25.search(query, top_k=top_k)
        chunk_ids = [cid for cid, _ in results]
        scores = [score for _, score in results]
        return self.store.get_evidence_refs(chunk_ids, scores)

    def search_vector_only(self, query: str, top_k: int | None = None) -> list[EvidenceRef]:
        """Search using vector only."""
        top_k = top_k or self.settings.retrieval_top_k
        query_embedding = self.embedder.embed_texts([query])[0]
        results = self.vector.search(query_embedding, top_k=top_k)
        chunk_ids = [cid for cid, _ in results]
        scores = [score for _, score in results]
        return self.store.get_evidence_refs(chunk_ids, scores)


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the singleton hybrid retriever."""
    return HybridRetriever()