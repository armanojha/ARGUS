"""Multi-Query Evidence Retrieval (Phase 15/16).

Takes a QueryPlan with multiple EvidenceNeeds, runs targeted retrieval
for each need, then merges and selects the best evidence set.

Phase 16 adds targeted second-stage evidence recovery when initial
retrieval does not adequately cover evidence needs.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import (
    ClaimType,
    EvidenceNeed,
    NeedPriority,
    QueryPlan,
)
from app.retrieval.recovery import (
    CoverageAnalysis,
    RecoveryBudget,
    RecoveryResult,
    RecoveryType,
    TargetedRecovery,
)

if TYPE_CHECKING:
    from app.retrieval.router import RetrievalPolicyRouter

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result of multi-query evidence retrieval."""
    query_plan: QueryPlan
    candidates: list[EvidenceRef] = field(default_factory=list)
    selected: list[EvidenceRef] = field(default_factory=list)
    need_coverage: dict[str, float] = field(default_factory=dict)
    total_retrieval_calls: int = 0
    total_candidates: int = 0
    source_diversity: int = 0
    planner_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    # Phase 16 recovery fields
    recovery_type: RecoveryType = RecoveryType.NONE
    recovery_activated: bool = False
    recovery_attempts: int = 0
    recovery_candidates_added: int = 0
    coverage_before_recovery: float = 0.0
    coverage_after_recovery: float = 0.0


class MultiQueryRetriever:
    """Executes multi-query retrieval and coverage-aware selection.

    Uses the existing RetrievalPolicyRouter and HybridRetriever for
    each individual search, then merges results across queries.

    Phase 16 adds targeted second-stage evidence recovery when initial
    retrieval does not adequately cover evidence needs.
    """

    def __init__(
        self,
        router: RetrievalPolicyRouter,
        retriever: HybridRetriever,
        top_k: int = 10,
        recovery_budget: RecoveryBudget | None = None,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.top_k = top_k
        self.recovery = TargetedRecovery(budget=recovery_budget)

    def retrieve(self, plan: QueryPlan) -> RetrievalResult:
        """Execute multi-query retrieval for a planned query.

        Args:
            plan: QueryPlan with evidence needs and search variants.

        Returns:
            RetrievalResult with merged candidates and selected evidence.
        """
        import time

        result = RetrievalResult(query_plan=plan)

        if not plan.is_planned:
            # Pass-through: single query, no planning
            start = time.perf_counter()
            refs = self.retriever.search(plan.original_query, top_k=self.top_k)
            result.retrieval_latency_ms = (time.perf_counter() - start) * 1000
            result.candidates = refs
            result.selected = refs[:self.top_k]
            result.total_retrieval_calls = 1
            result.total_candidates = len(refs)
            result.source_diversity = len({r.document_id for r in refs})
            return result

        # Multi-query: run each evidence need's search
        all_candidates: list[EvidenceRef] = []
        seen_chunk_ids: set[uuid.UUID] = set()
        call_count = 0

        start = time.perf_counter()

        for need in plan.evidence_needs:
            query = need.search_query
            if not query:
                continue

            # Classify the sub-query and get appropriate mix
            pattern = self.router.classify_question(query)

            # Use existing router for each sub-query
            refs = self._search_single(query, pattern, need)
            call_count += 1

            # Tag each result with its source need
            for ref in refs:
                if ref.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(ref.chunk_id)
                    ref.metadata["evidence_need_id"] = need.id
                    ref.metadata["evidence_need_claim"] = need.claim_type.value
                    ref.metadata["retrieval_wave"] = call_count
                    all_candidates.append(ref)
                else:
                    # Dedup: boost score for chunks found by multiple needs
                    for existing in all_candidates:
                        if existing.chunk_id == ref.chunk_id:
                            existing.score = max(existing.score, ref.score)
                            existing.metadata["multi_need_hit"] = True
                            break

        result.retrieval_latency_ms = (time.perf_counter() - start) * 1000
        result.total_retrieval_calls = call_count
        result.candidates = all_candidates
        result.total_candidates = len(all_candidates)

        # Coverage-aware selection
        result.selected = self._coverage_select(
            all_candidates, plan.evidence_needs, self.top_k
        )

        # Compute need coverage
        result.need_coverage = self._compute_need_coverage(
            result.selected, plan.evidence_needs
        )

        result.source_diversity = len({r.document_id for r in result.selected})

        # Phase 16: Targeted recovery if coverage is incomplete
        coverage_ratio = sum(result.need_coverage.values()) / len(result.need_coverage) if result.need_coverage else 1.0
        result.coverage_before_recovery = coverage_ratio

        if coverage_ratio < 1.0 and plan.pattern in ("conflict", "complex_research", "multi_hop"):
            recovery_start = time.perf_counter()

            # Analyze coverage
            analysis = self.recovery.analyze_coverage(plan, result.selected)

            # Execute recovery
            import asyncio
            store = self.retriever.store
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a new task
                    recovered = []
                else:
                    recovered = loop.run_until_complete(
                        self.recovery.recover(
                            plan, analysis, result.selected,
                            self.retriever, store,
                        )
                    )
            except RuntimeError:
                # No event loop, create one
                recovered = asyncio.run(
                    self.recovery.recover(
                        plan, analysis, result.selected,
                        self.retriever, store,
                    )
                )

            # Merge recovered candidates
            if recovered:
                result.recovery_activated = True
                result.recovery_type = analysis.recovery_type
                result.recovery_attempts = len(recovered)
                result.recovery_candidates_added = len(recovered)

                # Add recovered to candidates and re-select
                for ref in recovered:
                    if ref.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(ref.chunk_id)
                        ref.metadata["recovery_source"] = analysis.recovery_type.value
                        all_candidates.append(ref)

                # Re-run coverage-aware selection with recovered candidates
                result.selected = self._coverage_select(
                    all_candidates, plan.evidence_needs, self.top_k
                )

                # Re-compute coverage
                result.need_coverage = self._compute_need_coverage(
                    result.selected, plan.evidence_needs
                )
                result.source_diversity = len({r.document_id for r in result.selected})

            result.retrieval_latency_ms += (time.perf_counter() - recovery_start) * 1000

        result.coverage_after_recovery = sum(result.need_coverage.values()) / len(result.need_coverage) if result.need_coverage else 1.0

        logger.info(
            "multi_query_retrieval",
            pattern=plan.pattern,
            needs=plan.need_count,
            calls=result.total_retrieval_calls,
            candidates=result.total_candidates,
            selected=len(result.selected),
            sources=result.source_diversity,
            recovery_activated=result.recovery_activated,
            recovery_type=result.recovery_type.value if result.recovery_activated else "none",
            coverage_before=result.coverage_before_recovery,
            coverage_after=result.coverage_after_recovery,
            latency_ms=round(result.retrieval_latency_ms, 1),
        )

        return result

    def _search_single(
        self,
        query: str,
        pattern: Any,
        need: EvidenceNeed,
    ) -> list[EvidenceRef]:
        """Run a single search for one evidence need.

        Uses the router's dispatch for pattern-appropriate retrieval.
        """
        from app.retrieval.policy import QuestionPattern

        # Map string pattern to enum
        try:
            pat_enum = QuestionPattern(pattern.value if hasattr(pattern, 'value') else pattern)
        except (ValueError, AttributeError):
            pat_enum = QuestionPattern.CONCEPTUAL

        # Get the retrieval mix for this pattern
        mix = self.router.get_retrieval_mix(pat_enum)

        # Determine search depth based on need priority
        if need.priority == NeedPriority.HIGH:
            top_k = min(20, self.top_k * 2)
        elif need.priority == NeedPriority.MEDIUM:
            top_k = min(15, int(self.top_k * 1.5))
        else:
            top_k = self.top_k

        # Run hybrid search directly (most efficient path)
        refs = self.retriever.search(
            query,
            top_k=top_k,
            bm25_weight=mix.bm25_weight,
            vector_weight=mix.vector_weight,
        )

        return refs

    def _coverage_select(
        self,
        candidates: list[EvidenceRef],
        needs: list[EvidenceNeed],
        top_k: int,
    ) -> list[EvidenceRef]:
        """Select evidence that maximizes coverage of evidence needs.

        Greedy algorithm:
        1. Score each candidate by relevance + need coverage
        2. Select candidates that cover uncovered needs
        3. Avoid redundancy (don't select 8 chunks covering the same need)
        """
        if not candidates or not needs:
            return candidates[:top_k]

        # Track which needs are covered
        covered_needs: set[str] = set()
        selected: list[EvidenceRef] = []

        # Pre-compute need IDs
        need_ids = {need.id for need in needs}
        high_priority_ids = {need.id for need in needs if need.priority == NeedPriority.HIGH}

        # Sort candidates by score (descending)
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)

        for candidate in sorted_candidates:
            if len(selected) >= top_k:
                break

            candidate_need_id = candidate.metadata.get("evidence_need_id", "")
            is_multi_need = candidate.metadata.get("multi_need_hit", False)

            # Compute selection score
            selection_score = candidate.score

            # Bonus for covering a new need
            if candidate_need_id and candidate_need_id not in covered_needs:
                selection_score *= 1.3
                covered_needs.add(candidate_need_id)

            # Bonus for high-priority needs
            if candidate_need_id in high_priority_ids:
                selection_score *= 1.2

            # Bonus for multi-need hits (found by multiple searches)
            if is_multi_need:
                selection_score *= 1.15

            # Penalty if we already have enough from this source
            source_count = sum(
                1 for s in selected
                if s.document_id == candidate.document_id
            )
            if source_count >= 3:
                selection_score *= 0.7

            # Only add if the score is reasonable
            if selection_score > 0:
                selected.append(candidate)

        # If we haven't filled top_k, add remaining by score
        if len(selected) < top_k:
            selected_ids = {s.chunk_id for s in selected}
            for candidate in sorted_candidates:
                if len(selected) >= top_k:
                    break
                if candidate.chunk_id not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate.chunk_id)

        # Re-rank by score and assign final ranks
        selected.sort(key=lambda r: r.score, reverse=True)
        final = selected[:top_k]
        for rank, ref in enumerate(final, 1):
            final[rank - 1] = ref.model_copy(update={"rank": rank})

        return final

    def _compute_need_coverage(
        self,
        selected: list[EvidenceRef],
        needs: list[EvidenceNeed],
    ) -> dict[str, float]:
        """Compute how well selected evidence covers each need."""
        coverage: dict[str, float] = {}

        # Group selected by need
        need_hits: dict[str, int] = defaultdict(int)
        for ref in selected:
            need_id = ref.metadata.get("evidence_need_id", "")
            if need_id:
                need_hits[need_id] += 1

        for need in needs:
            hits = need_hits.get(need.id, 0)
            # Coverage is binary: 0 if no evidence, 1+ if any evidence
            coverage[need.id] = min(1.0, hits / 2.0)  # 2 hits = full coverage

        return coverage
