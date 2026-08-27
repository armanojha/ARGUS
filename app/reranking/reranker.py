"""Reranking (Phase 01).

Provides cross-encoder reranking over fused retrieval candidates.
"""

from __future__ import annotations

import threading
from typing import Any

from app.config import get_settings
from app.evidence.models import EvidenceRef
from app.logging_config import get_logger

logger = get_logger("argus.reranking")


class Reranker:
    """Cross-encoder reranker for retrieval candidates.

    Uses a local cross-encoder model to rerank candidates based on
    query-passage relevance.
    """

    _model: Any | None = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.settings = get_settings()
        self._lock = threading.Lock()

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        with self._lock:
            if Reranker._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                    logger.info("loading_reranker_model", model=self.model_name)
                    Reranker._model = CrossEncoder(self.model_name)
                except ImportError:
                    logger.warning("cross_encoder_not_available", model=self.model_name)
                    return None
            return Reranker._model

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceRef],
        top_k: int | None = None,
    ) -> list[EvidenceRef]:
        """Rerank candidates using cross-encoder.

        Args:
            query: The search query
            candidates: List of EvidenceRef to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of EvidenceRef with updated scores
        """
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            logger.warning("reranker_unavailable", fallback="original_order")
            return candidates[:top_k] if top_k else candidates

        top_k = top_k or self.settings.rerank_top_k

        # Prepare query-passage pairs
        pairs = [(query, cand.text) for cand in candidates]

        # Get cross-encoder scores
        scores = model.predict(pairs, batch_size=16, show_progress_bar=False)

        # Combine with original candidates
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (cand, score) in enumerate(scored_candidates[:top_k], 1):
            # Create new EvidenceRef with updated score and rank
            reranked.append(EvidenceRef(
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                source_id=cand.source_id,
                source_path=cand.source_path,
                source_type=cand.source_type,
                text=cand.text,
                page_start=cand.page_start,
                page_end=cand.page_end,
                section_path=cand.section_path,
                score=float(score),
                rank=rank,
                metadata={
                    **cand.metadata,
                    "rerank_score": float(score),
                    "original_score": cand.score,
                    "original_rank": cand.rank,
                },
            ))

        logger.info("reranked", query=query[:50], candidates=len(candidates), returned=len(reranked))
        return reranked


class NoOpReranker:
    """No-op reranker for when cross-encoder is not available."""

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceRef],
        top_k: int | None = None,
    ) -> list[EvidenceRef]:
        top_k = top_k or get_settings().rerank_top_k
        return candidates[:top_k]


def get_reranker() -> Reranker | NoOpReranker:
    """Get reranker instance, falling back to no-op if unavailable."""
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return Reranker()
    except ImportError:
        logger.warning("cross_encoder_not_installed", using="noop_reranker")
        return NoOpReranker()