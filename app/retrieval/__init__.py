"""Retrieval exports (Phase 01 / Phase 06 policy)."""

from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.embeddings import EmbeddingGenerator, generate_and_store_embeddings
from app.retrieval.hybrid import HybridRetriever, get_hybrid_retriever
from app.retrieval.router import (
    RetrievalPolicyRouter,
    get_retrieval_policy_router,
    load_retrieval_policy,
)
from app.retrieval.seeking import (
    AdaptiveEvidenceGapDetector,
    ObsidianHypothesisSeeker,
    get_adaptive_gap_detector,
)
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices

__all__ = [
    "AdaptiveEvidenceGapDetector",
    "BM25Retriever",
    "EmbeddingGenerator",
    "FAISSVectorStore",
    "HybridRetriever",
    "ObsidianHypothesisSeeker",
    "RetrievalPolicyRouter",
    "assign_bm25_doc_ids",
    "assign_embedding_indices",
    "generate_and_store_embeddings",
    "get_adaptive_gap_detector",
    "get_hybrid_retriever",
    "get_retrieval_policy_router",
    "load_retrieval_policy",
]