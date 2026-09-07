"""Evidence Selector — post-accumulation context optimization (Phase 20).

Selects a minimal but sufficient evidence set from the full accumulated
evidence pool before sending it to the LLM. Optimizes for:

    evidence utility =
        relevance
      + source diversity
      - semantic redundancy
      - unnecessary context

The selector is applied AFTER all retrieval iterations complete, right
before assess_node or synthesize_node formats evidence for the LLM.
The full evidence pool is preserved in state["evidence"]) for internal
use (graph reasoning, debugging, evaluation). Only the selected subset
reaches the LLM context.

Architecture:
    state["evidence"]  (full accumulated pool)
            │
            ▼
    EvidenceSelector.select()
            │
            ▼
    selected_evidence  (minimal, high-coverage subset)
            │
            ▼
    build_assessment_messages() / build_synthesis_messages()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.evidence.models import EvidenceRef
from app.logging_config import get_logger

logger = get_logger("argus.retrieval.evidence_selector")

_SENTINEL = object()  # sentinel for "use instance default"


@dataclass
class SelectionMetrics:
    """Observability metrics for evidence selection."""

    candidate_count: int = 0
    selected_count: int = 0
    redundancy_removed: int = 0
    source_count: int = 0
    estimated_tokens: int = 0
    selection_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "redundancy_removed": self.redundancy_removed,
            "source_count": self.source_count,
            "estimated_tokens": self.estimated_tokens,
            "selection_latency_ms": round(self.selection_latency_ms, 2),
        }


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Same as nodes._estimate_tokens."""
    return max(1, len(text) // 4)


class EvidenceSelector:
    """Selects a minimal, high-coverage evidence set for LLM context.

    Configuration:
        similarity_threshold: cosine similarity above which two chunks are
            near-duplicates (default 0.85). 0 disables dedup.
        max_chunks: hard cap on evidence chunks sent to LLM (default 8).
        max_tokens: hard cap on estimated tokens (default 6000).
        min_chunks: always keep at least this many chunks (default 2).
        min_sources: always represent at least this many distinct documents.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        max_chunks: int = 8,
        max_tokens: int = 6000,
        min_chunks: int = 2,
        min_sources: int = 2,
        vector_store: Any | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
        self.min_chunks = min_chunks
        self.min_sources = min_sources
        self.vector_store = vector_store

    def select(
        self,
        evidence: list[EvidenceRef],
        vector_store: Any | None = _SENTINEL,
        metrics: SelectionMetrics | None = None,
    ) -> list[EvidenceRef]:
        """Select a minimal, high-coverage evidence subset.

        Args:
            evidence: Full accumulated evidence (score-sorted, deduped by chunk_id).
            vector_store: FAISSVectorStore for embedding lookups. If _SENTINEL,
                uses the instance's vector_store. If None, semantic dedup is skipped.
            metrics: Optional mutable metrics object to populate.

        Returns:
            Selected evidence list, score-sorted, with re-assigned ranks.
        """
        t0 = time.perf_counter()

        if not evidence:
            if metrics:
                metrics.selection_latency_ms = (time.perf_counter() - t0) * 1000
            return []

        # Resolve vector store: parameter overrides instance default
        vs = self.vector_store if vector_store is _SENTINEL else vector_store

        # --- Pass 1: semantic dedup ---
        deduped = self._semantic_dedup(evidence, vs)
        redundancy_removed = len(evidence) - len(deduped)

        # --- Pass 2: source-diversity-aware greedy selection ---
        selected = self._diversity_select(deduped)

        # --- Pass 3: enforce token budget ---
        selected = self._enforce_token_budget(selected)

        # --- Re-rank and re-assign ranks ---
        selected.sort(key=lambda r: r.score, reverse=True)
        selected = [
            ref.model_copy(update={"rank": i + 1})
            for i, ref in enumerate(selected)
        ]

        estimated_tokens = sum(_estimate_tokens(r.text) for r in selected)
        source_count = len({r.document_id for r in selected})

        if metrics:
            metrics.candidate_count = len(evidence)
            metrics.selected_count = len(selected)
            metrics.redundancy_removed = redundancy_removed
            metrics.source_count = source_count
            metrics.estimated_tokens = estimated_tokens
            metrics.selection_latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "evidence_selection",
            candidates=len(evidence),
            selected=len(selected),
            redundancy_removed=redundancy_removed,
            sources=source_count,
            est_tokens=estimated_tokens,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

        return selected

    # ------------------------------------------------------------------
    # Internal: semantic dedup
    # ------------------------------------------------------------------

    def _semantic_dedup(
        self,
        evidence: list[EvidenceRef],
        vector_store: Any | None,
    ) -> list[EvidenceRef]:
        """Remove near-duplicate chunks using embedding similarity.

        Greedy: iterate score-sorted evidence, keep a chunk only if its
        cosine similarity to all already-selected chunks is below threshold.
        Falls back to returning original list if embeddings are unavailable.
        """
        if not evidence or self.similarity_threshold <= 0:
            return evidence

        if vector_store is None:
            return evidence

        # Gather embeddings
        embeddings: list[np.ndarray | None] = []
        for ref in evidence:
            emb = vector_store.get_embedding(ref.chunk_id)
            embeddings.append(emb)

        # If any chunk lacks an embedding, fall back (safe default)
        if any(e is None for e in embeddings):
            return evidence

        selected_indices: list[int] = []
        selected_embs: list[np.ndarray] = []

        for i, emb in enumerate(embeddings):
            is_dup = False
            for sel_emb in selected_embs:
                sim = float(np.dot(emb, sel_emb))  # normalized → cosine
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                selected_indices.append(i)
                selected_embs.append(emb)

        return [evidence[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Internal: diversity-aware greedy selection
    # ------------------------------------------------------------------

    def _diversity_select(
        self,
        evidence: list[EvidenceRef],
    ) -> list[EvidenceRef]:
        """Greedy selection that balances relevance, diversity, and budget.

        Algorithm:
        1. Pass 1: pick top-scoring chunk from each distinct source (ensures
           minimum source diversity).
        2. Pass 2: fill remaining slots by score, skipping chunks that would
           exceed max_chunks.
        3. If we have fewer than min_sources represented, add from any source.
        """
        if not evidence:
            return []

        target = min(self.max_chunks, len(evidence))

        # Group by source document
        by_source: dict[str, list[EvidenceRef]] = {}
        for ref in evidence:
            key = str(ref.document_id)
            by_source.setdefault(key, []).append(ref)

        selected: list[EvidenceRef] = []
        selected_ids: set = set()

        # Pass 1: top chunk from each source (up to target)
        for doc_id, refs in by_source.items():
            if len(selected) >= target:
                break
            if refs:
                selected.append(refs[0])
                selected_ids.add(refs[0].chunk_id)

        # Pass 2: fill remaining by score
        remaining = [r for r in evidence if r.chunk_id not in selected_ids]
        remaining.sort(key=lambda r: r.score, reverse=True)
        for ref in remaining:
            if len(selected) >= target:
                break
            selected.append(ref)
            selected_ids.add(ref.chunk_id)

        # Pass 3: ensure minimum sources
        if len({r.document_id for r in selected}) < self.min_sources:
            for ref in evidence:
                if len(selected) >= target:
                    break
                if ref.chunk_id not in selected_ids:
                    selected.append(ref)
                    selected_ids.add(ref.chunk_id)

        return selected

    # ------------------------------------------------------------------
    # Internal: token budget enforcement
    # ------------------------------------------------------------------

    def _enforce_token_budget(
        self,
        evidence: list[EvidenceRef],
    ) -> list[EvidenceRef]:
        """Trim evidence to fit within the token budget.

        Always keeps at least min_chunks (if available). Removes lowest-scored
        chunks first when over budget.
        """
        if not evidence:
            return []

        total = sum(_estimate_tokens(r.text) for r in evidence)
        if total <= self.max_tokens:
            return evidence

        # Sort by score ascending (lowest first) for removal
        by_score = sorted(evidence, key=lambda r: r.score)
        result = list(evidence)
        removed = 0

        for ref in by_score:
            if len(result) - removed <= self.min_chunks:
                break
            total -= _estimate_tokens(ref.text)
            removed += 1
            if total <= self.max_tokens:
                break

        if removed:
            # Remove the lowest-scored chunks we identified
            remove_ids = {r.chunk_id for r in by_score[:removed]}
            result = [r for r in evidence if r.chunk_id not in remove_ids]

        # Warn if min_chunks still exceeds budget (unavoidable with current evidence)
        final_tokens = sum(_estimate_tokens(r.text) for r in result)
        if final_tokens > self.max_tokens:
            logger.warning(
                "evidence_token_budget_exceeded",
                final_tokens=final_tokens,
                max_tokens=self.max_tokens,
                min_chunks=self.min_chunks,
            )

        return result
