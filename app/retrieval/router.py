"""Retrieval Policy Router (Phase 06).

Implements the `RetrievalPolicyInterface` shared contract from
`app/retrieval/policy.py`: classifies a question into a `QuestionPattern`,
selects the matching retrieval mix from the V2 §5.2 / V3 §5 policy table,
and executes the mix against the Phase 01 hybrid retriever (plus the
Phase 03 graph retriever when available).

Design rules honored here:
- Deterministic question classification (keyword heuristics), so a failing
  LLM never blocks the policy — model/provider assignment is out of scope
  (Phase 07), and this phase never fabricates evidence.
- Every dispatched method returns real `EvidenceRef` objects from the
  evidence store; the `web` method has no provider in the current
  architecture and degrades to hybrid retrieval with an explicit
  `policy_fallback` marker instead of silently inventing sources.
- Weights across methods are normalized so fused scores stay in [0, 1]
  and remain comparable across phases.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import REPO_ROOT, Settings, get_settings
from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.planner import EvidenceNeedPlanner
from app.retrieval.policy import (
    QuestionPattern,
    RetrievalMethod,
    RetrievalMix,
    RetrievalPolicy,
    RetrievalPolicyEntry,
    RetrievalPolicyInterface,
)

logger = get_logger("argus.retrieval.policy_router")

_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b", re.IGNORECASE)

_MULTIMODAL_WORDS = {"figure", "diagram", "chart", "table", "image", "graph", "illustration", "photo", "screenshot"}
_HISTORICAL_WORDS = {"history", "historical", "era", "century", "origins", "founded", "during", "before", "after"}
_FRESH_WORDS = {"latest", "current", "recent", "newest", "update", "status", "as of", "this week", "this year", "this month"}
_LONG_WORDS = {"overview", "comprehensive", "report", "summary", "review", "survey", "everything", "state of", "analysis"}
_EXACT_WORDS = {"define", "definition", "meaning", "term", "stands for"}
_RELATION_WORDS = {"relationship", "relation", "relate", "associated", "correlat", "influence", "impact", "cause", "lead to", "between", "connected", "linked", "based on"}
_CONCEPT_WORDS = {"explain", "how does", "how do", "why does", "concept", "conceptual", "principle", "theory", "mechanism", "idea"}
_COMPARATIVE_WORDS = {"compare", "comparison", "difference", "differences", "versus", "vs", "contrast", "similarities", "similar to", "differ from", "better", "worse", "advantage", "disadvantage", "pros", "cons", "tradeoffs"}
_CAUSAL_WORDS = {"cause", "caused", "reason for", "because", "leads to", "result in", "effect of", "impact of", "consequence", "trigger", "root cause", "underlying"}
_PROCEDURAL_WORDS = {"how to", "steps to", "guide", "tutorial", "instructions", "walkthrough", "procedure", "process for", "method to", "approach to", "best practice", "workflow", "recipe"}

# Phase 17: Canonical classification word lists
_CONFLICT_WORDS = {
    "conflict", "contradict", "contradiction", "disagree", "disagreement",
    "however", "although", "despite", "on the other hand",
    "legacy.*superseded", "superseded.*current", "authoritative",
    "which.*correct", "which.*accurate",
}
_MULTI_HOP_WORDS = {
    "downstream", "upstream", "indirectly", "depends on", "via",
    "through.*chain", "propagat", "affects.*then",
    "integrates with.*depends", "connects to.*which",
}
_COMPLEX_RESEARCH_WORDS = {
    "assess", "evaluate", "synthesize", "comprehensive analysis",
    "compare.*and.*explain", "analyze.*relationship",
    "how.*fit together", "what it depends on",
}
_ABSENT_WORDS = {
    "which quarter.*best sales", "what.*not covered",
    "what.*missing", "what is not",
}
_ADVERSARIAL_WORDS = {
    "fabricated", "hallucinated", "made up", "not audited",
    "illustrative only", "these figures are not",
}

_STOPWORDS = {
    "what", "which", "when", "where", "who", "how", "why", "the", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "is", "are", "was", "were", "does", "do", "did", "not",
    "this", "that", "these", "those", "from", "by", "at", "it", "its", "about", "between",
}


def _query_terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]{4,}", text.lower()) if t not in _STOPWORDS}


_DOUBLE_QUOTE_RE = re.compile(r'["\u201c\u201d]')
_SINGLE_QUOTE_PAIR_RE = re.compile(r"'(?:[^']{1,60})'")


def _has_balanced_quotes(text: str) -> bool:
    """Detect a quoted phrase (pair of quote chars), not lone apostrophes.

    Possessive/contraction apostrophes ("fox's", "France's") must not trick
    the classifier into treating a normal sentence as an exact-term lookup.
    """
    if _DOUBLE_QUOTE_RE.search(text):
        return True
    return bool(_SINGLE_QUOTE_PAIR_RE.search(text))


class RetrievalPolicyRouter(RetrievalPolicyInterface):
    """Concrete adaptive retrieval policy router (Phase 06)."""

    def __init__(
        self,
        policy: RetrievalPolicy | None = None,
        graph_retriever: Any | None = None,
        settings: Settings | None = None,
        bge_m3_retriever: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._policy = policy or load_retrieval_policy(self.settings)
        # Optional Phase 03 graph retriever. When absent, GRAPH/TEMPORAL
        # dispatch degrades to hybrid retrieval (never fabricates output).
        self.graph_retriever = graph_retriever
        # Optional BGE-M3 experimental backend. Lazy-loaded on first use.
        self._bge_m3_retriever = bge_m3_retriever
        # Phase 15: Evidence Need Planner (lazy-initialized)
        self._planner: EvidenceNeedPlanner | None = None
        self._multi_query_retriever: MultiQueryRetriever | None = None

    # -- policy access -------------------------------------------------------

    def get_policy(self) -> RetrievalPolicy:
        return self._policy

    def get_retrieval_mix(self, pattern: QuestionPattern) -> RetrievalMix:
        return self._policy.get_mix_for_pattern(pattern)

    # -- question classification --------------------------------------------

    def classify_question(self, query: str, context: dict[str, Any] | None = None) -> QuestionPattern:
        """Deterministic question-pattern classification (V2 §5.2 / V3 §5).

        Heuristic, ordered most-specific first. No LLM dependency: the
        policy must keep working when the gateway is unavailable.

        This is the CANONICAL classifier. All code paths (production,
        benchmark, diagnostics, telemetry) MUST use this or convert
        through QuestionPattern.from_eval_class().
        """
        q = query.strip()
        low = q.lower()

        has_quoted = _has_balanced_quotes(q)
        year_matches = _YEAR_RE.findall(q)

        # --- Standard patterns (most-specific first) ---

        if any(w in low for w in _MULTIMODAL_WORDS) and re.search(r"(show|compare|in the .* (figure|chart|table|diagram))", low):
            return QuestionPattern.MULTIMODAL

        if year_matches and any(w in low for w in ("during", "era", "century", "origins", "history", "founded", "before", "after")):
            return QuestionPattern.HISTORICAL
        if any(w in low for w in _HISTORICAL_WORDS):
            return QuestionPattern.HISTORICAL

        if any(w in low for w in _FRESH_WORDS):
            return QuestionPattern.FRESH_MISSING

        if any(w in low for w in ("overview", "comprehensive", "state of")):
            return QuestionPattern.LONG_REPORT
        if any(w in low for w in _LONG_WORDS) and len(_query_terms(q)) >= 3:
            return QuestionPattern.LONG_REPORT

        if any(w in low for w in _RELATION_WORDS) and ("and" in low or "between" in low):
            return QuestionPattern.ENTITY_RELATIONSHIP

        if any(w in low for w in _COMPARATIVE_WORDS):
            return QuestionPattern.COMPARATIVE

        if any(w in low for w in _CAUSAL_WORDS):
            return QuestionPattern.CAUSAL

        if any(w in low for w in _PROCEDURAL_WORDS):
            return QuestionPattern.PROCEDURAL

        if has_quoted or any(w in low for w in _EXACT_WORDS):
            return QuestionPattern.EXACT_TERM

        # --- Phase 17: Plannable patterns (after standard patterns) ---

        # CONFLICT: Detect conflict/contradiction keywords
        # Must be more specific to avoid false positives on "compare" queries
        if any(w in low for w in _CONFLICT_WORDS) and any(w in low for w in ("conflict", "contradict", "disagree", "however", "although", "legacy", "superseded", "authoritative")):
            return QuestionPattern.CONFLICT

        # MULTI_HOP: Detect multi-hop reasoning patterns
        if any(w in low for w in _MULTI_HOP_WORDS):
            return QuestionPattern.MULTI_HOP

        # COMPLEX_RESEARCH: Detect complex research/analysis queries
        if any(w in low for w in _COMPLEX_RESEARCH_WORDS):
            return QuestionPattern.COMPLEX_RESEARCH

        # --- Absent/adversarial detection ---

        # ABSENT_INFO: Detect queries about absent/non-existent information
        if any(w in low for w in _ABSENT_WORDS):
            return QuestionPattern.ABSENT_INFO

        # ADVERSARIAL: Detect adversarial/misleading queries
        if any(w in low for w in _ADVERSARIAL_WORDS):
            return QuestionPattern.ADVERSARIAL

        # --- Fallback patterns ---

        # NUMERICAL: Detect numerical/quantitative queries (only if very specific)
        if any(w in low for w in ("how many", "what percentage", "what fraction", "total output", "sum of")):
            return QuestionPattern.NUMERICAL

        if any(w in low for w in _CONCEPT_WORDS):
            return QuestionPattern.CONCEPTUAL

        return QuestionPattern.CONCEPTUAL

    # -- retrieval dispatch --------------------------------------------------

    async def execute_retrieval(
        self,
        query: str,
        pattern: QuestionPattern,
        retriever: HybridRetriever,
        top_k: int | None = None,
        reranker: Any | None = None,  # Reranker | NoOpReranker
    ) -> list[EvidenceRef]:
        """Execute the retrieval mix for a classified pattern.

        Dispatches each method in the mix against the Phase 01 retriever
        (and Phase 03 graph retriever for graph/temporal methods), fuses
        results by normalized weighted score, dedups by chunk_id, and
        optionally reranks. Returns real EvidenceRefs only.
        """
        top_k = top_k or self.settings.retrieval_top_k
        mix = self.get_retrieval_mix(pattern)

        # Phase 07f: dispatch the independent retrieval methods concurrently.
        # Each method (BM25 / VECTOR / HYBRID / GRAPH / TEMPORAL / METADATA_FILTER)
        # is an isolated read over the (read-only) indexes producing its own
        # EvidenceRef list — there is no data dependency between methods, so they
        # can overlap. The synchronous searches (local BM25 index + local/remote
        # embeddings) are CPU/I-O bound and GIL-releasing (numpy/torch inference),
        # so `asyncio.to_thread` genuinely overlaps them. Determinism is preserved:
        # results are collected per-method in the original mix order, and the
        # downstream `_fuse` dedups by chunk_id and sorts by weighted score — its
        # output is identical regardless of completion order. A bounded semaphore
        # caps fan-out so a pathological method list can never spawn unbounded tasks.
        import asyncio

        per_method: list[tuple[RetrievalMethod, list[EvidenceRef]]] = []
        sem = asyncio.Semaphore(min(4, max(1, len(mix.methods))))

        async def _run_method(method: RetrievalMethod) -> tuple[RetrievalMethod, list[EvidenceRef]]:
            async with sem:
                try:
                    refs = await asyncio.to_thread(
                        self._dispatch, method, mix, query, retriever, mix.max_results_per_method
                    )
                except Exception as exc:  # noqa: BLE001 - a single method must not fail the whole search
                    logger.warning("policy_method_failed", method=method.value, error=str(exc))
                    refs = []
            return method, refs

        collected = await asyncio.gather(*(_run_method(m) for m in mix.methods))
        per_method = [(m, refs) for m, refs in collected if refs]

        if not per_method:
            # Deterministic fallback to a plain hybrid pass (never empty-handed).
            logger.warning("policy_mix_empty", pattern=pattern.value, fallback="hybrid")
            per_method = [(RetrievalMethod.HYBRID, retriever.search(query, top_k=top_k))]

        fused = self._fuse(per_method, mix.weights)
        fused = self._apply_metadata_filters(fused, mix.metadata_filters)

        # Parent context expansion for long-document patterns
        if pattern in (QuestionPattern.LONG_REPORT, QuestionPattern.COMPARATIVE):
            fused = retriever.expand_with_parent_context(fused, max_context_chunks=3)

        # Source-diversified selection for multi-source patterns
        if (self.settings.retrieval_source_diversification
                and pattern in (QuestionPattern.COMPARATIVE, QuestionPattern.LONG_REPORT)):
            fused = self._conditional_diversify(fused, top_k, top_k,
                                               self.settings.retrieval_diversification_min_sources)
        else:
            fused = fused[:top_k]

        if fused and reranker is not None:
            fused = reranker.rerank(query, fused, top_k=top_k)
        else:
            for rank, ref in enumerate(fused, 1):
                fused[rank - 1] = ref.model_copy(update={"rank": rank})

        # Confidence fallback: if the narrow-path mix scored poorly on the top
        # result, retry with a full hybrid pass.  Only activates when the
        # classified pattern is NOT already a full hybrid (avoids infinite loop).
        fallback_threshold = self.settings.retrieval_policy_fallback_threshold
        already_hybrid = mix.methods == [RetrievalMethod.HYBRID]
        if fused and fused[0].score < fallback_threshold and not already_hybrid:
            logger.info(
                "policy_low_confidence_fallback",
                pattern=pattern.value,
                top_score=fused[0].score,
                threshold=fallback_threshold,
            )
            hybrid_refs = retriever.search(query, top_k=top_k)
            if hybrid_refs:
                if reranker is not None:
                    hybrid_refs = reranker.rerank(query, hybrid_refs, top_k=top_k)
                hybrid_refs = [
                    ref.model_copy(update={"metadata": {**ref.metadata, "policy_fallback": True}})
                    for ref in hybrid_refs
                ]
                if not fused or hybrid_refs[0].score > fused[0].score:
                    fused = hybrid_refs

        logger.info(
            "policy_retrieval_executed",
            pattern=pattern.value,
            methods=[m.value for m, _ in per_method],
            results=len(fused),
        )
        return fused

    async def execute_planned_retrieval(
        self,
        query: str,
        pattern: QuestionPattern,
        retriever: HybridRetriever,
        top_k: int | None = None,
        reranker: Any | None = None,
    ) -> list[EvidenceRef]:
        """Execute retrieval with evidence need planning (Phase 15).

        For complex patterns (CONFLICT, COMPLEX_RESEARCH, MULTI_HOP),
        decomposes the query into evidence needs and runs multi-query
        retrieval. For simple patterns, falls through to execute_retrieval.

        This is the entry point for the Query Intelligence layer.
        """
        top_k = top_k or self.settings.retrieval_top_k
        pattern_value = pattern.value if hasattr(pattern, 'value') else pattern

        # Only plan for complex patterns
        if pattern_value not in EvidenceNeedPlanner.PLANNABLE_PATTERNS:
            return await self.execute_retrieval(query, pattern, retriever, top_k, reranker)

        # Lazy-init planner and multi-query retriever
        if self._planner is None:
            self._planner = EvidenceNeedPlanner()
        if self._multi_query_retriever is None:
            self._multi_query_retriever = MultiQueryRetriever(
                router=self, retriever=retriever, top_k=top_k,
            )

        # Create query plan
        start = time.perf_counter()
        plan = self._planner.plan(query, pattern_value)
        planner_latency = (time.perf_counter() - start) * 1000

        if not plan.is_planned:
            # Planner decided not to decompose; use standard path
            return await self.execute_retrieval(query, pattern, retriever, top_k, reranker)

        logger.info(
            "planner_activated",
            pattern=pattern_value,
            needs=plan.need_count,
            queries=plan.query_count,
            planner_latency_ms=round(planner_latency, 1),
        )

        # Execute multi-query retrieval
        result = await self._multi_query_retriever.retrieve(plan)

        # Apply optional reranking
        if result.selected and reranker is not None:
            result.selected = reranker.rerank(query, result.selected, top_k=top_k)

        # Ensure ranks are correct
        for rank, ref in enumerate(result.selected, 1):
            result.selected[rank - 1] = ref.model_copy(update={"rank": rank})

        logger.info(
            "planned_retrieval_complete",
            pattern=pattern_value,
            selected=len(result.selected),
            coverage=result.need_coverage,
            sources=result.source_diversity,
            retrieval_latency_ms=round(result.retrieval_latency_ms, 1),
        )

        return result.selected

    def _dispatch(
        self,
        method: RetrievalMethod,
        mix: RetrievalMix,
        query: str,
        retriever: HybridRetriever,
        upper_k: int,
    ) -> list[EvidenceRef]:
        """Run a single retrieval method and return its EvidenceRefs."""
        if method == RetrievalMethod.BM25:
            return retriever.search_bm25_only(query, top_k=upper_k)
        if method == RetrievalMethod.VECTOR:
            return retriever.search_vector_only(query, top_k=upper_k)
        if method == RetrievalMethod.HYBRID:
            # HARDEN-06.5.5: a hybrid pass should only run the lexical and/or
            # dense mechanisms the rest of the mix does NOT already cover with a
            # dedicated single-mechanism method. Concretely:
            #   * an explicit VECTOR method already provides dense coverage, so
            #     the hybrid only contributes BM25 (no duplicate embedding);
            #   * a mix with an explicit BM25 method and no vector source
            #     (e.g. EXACT_TERM) is lexical-only, so the hybrid runs BM25 only
            #     and skips the embedding round-trip entirely.
            # A mix with no single-mechanism method (LONG_REPORT -> HYBRID only)
            # still runs the full hybrid.
            needed = self._needed_mechanisms(mix, upper_k)
            if needed is not None:
                if not needed:
                    logger.debug("policy_hybrid_skipped")
                    return []
                logger.debug("policy_hybrid_limited", mechanisms=sorted(needed))
                return retriever.search(
                    query,
                    top_k=upper_k,
                    bm25_weight=mix.bm25_weight,
                    vector_weight=mix.vector_weight,
                    mechanisms=needed,
                )
            return retriever.search(
                query, top_k=upper_k, bm25_weight=mix.bm25_weight, vector_weight=mix.vector_weight
            )
        if method in (RetrievalMethod.GRAPH, RetrievalMethod.GRAPH_BM25, RetrievalMethod.GRAPH_VECTOR):
            return self._graph_dispatch(method, mix, query, retriever, upper_k)
        if method == RetrievalMethod.TEMPORAL:
            return self._temporal_dispatch(mix, query, retriever, upper_k)
        if method == RetrievalMethod.METADATA_FILTER:
            return retriever.search(query, top_k=upper_k)
        if method == RetrievalMethod.WEB:
            # No web provider in the current architecture. Degrade to hybrid,
            # explicitly marked so downstream code never mistakes this for a
            # fresh web result (prompt-injection posture: never fabricate).
            refs = retriever.search(query, top_k=upper_k)
            for ref in refs:
                ref.metadata["policy_fallback"] = "web->hybrid"
            return refs
        if method in (
            RetrievalMethod.BGE_M3_DENSE,
            RetrievalMethod.BGE_M3_SPARSE,
            RetrievalMethod.BGE_M3_HYBRID,
        ):
            return self._bge_m3_dispatch(method, query, upper_k)
        return retriever.search(query, top_k=upper_k)

    @staticmethod
    def _needed_mechanisms(
        mix: RetrievalMix, upper_k: int
    ) -> set[str] | None:
        """Mechanisms a HYBRID method should run, or None to run a full hybrid.

        Returns None (full hybrid) when no dedicated BM25/VECTOR sibling exists.
        Otherwise the hybrid only supplies the mechanism NOT covered by the
        single-mechanism siblings:
          * an explicit VECTOR method covers dense -> hybrid contributes {"bm25"};
          * a lexical-only mix (BM25 + HYBRID, no vector source) is best served
            purely lexically -> hybrid contributes {"bm25"}, skipping the expensive
            query embedding entirely;
          * both BM25 and VECTOR are explicit -> hybrid adds nothing -> {}.
        """
        has_bm25 = RetrievalMethod.BM25 in mix.methods
        has_vector = RetrievalMethod.VECTOR in mix.methods
        if has_bm25 and has_vector:
            # Both mechanisms already dispatched by dedicated methods.
            return set()
        if has_vector:
            # VECTOR covers dense; hybrid only adds the lexical pass.
            return {"bm25"}
        if has_bm25:
            # Only BM25 + HYBRID -> lexical-only mix; no embedding needed.
            return {"bm25"}
        return None

    def _graph_dispatch(
        self,
        method: RetrievalMethod,
        mix: RetrievalMix,
        query: str,
        retriever: HybridRetriever,
        upper_k: int,
    ) -> list[EvidenceRef]:
        if self.graph_retriever is None:
            return retriever.search(query, top_k=upper_k)
        graph_refs = self.graph_retriever.search(query, top_k=upper_k, max_hops=mix.graph_max_hops)
        if method == RetrievalMethod.GRAPH:
            return graph_refs
        # GRAPH_BM25 / GRAPH_VECTOR: union graph traversal with the lexical
        # or dense pass, mirroring the policy table's multi-method intent.
        if method == RetrievalMethod.GRAPH_BM25:
            return graph_refs + retriever.search_bm25_only(query, top_k=upper_k)
        return graph_refs + retriever.search_vector_only(query, top_k=upper_k)

    def _temporal_dispatch(
        self,
        mix: RetrievalMix,
        query: str,
        retriever: HybridRetriever,
        upper_k: int,
    ) -> list[EvidenceRef]:
        from datetime import UTC, datetime, timedelta

        if self.graph_retriever is None:
            return retriever.search(query, top_k=upper_k)
        window = mix.temporal_window_days or 365
        end = datetime.now(UTC)
        start = end - timedelta(days=window)
        return self.graph_retriever.search_temporal(
            query,
            time_start=start.isoformat(),
            time_end=end.isoformat(),
            top_k=upper_k,
        )

    def _conditional_diversify(
        self,
        candidates: list[EvidenceRef],
        expanded_top: int,
        final_top: int,
        min_sources: int,
    ) -> list[EvidenceRef]:
        """Conditionally diversify: only when one source dominates top results.

        If the top results already cover >= min_sources distinct documents,
        return them unchanged.  Otherwise, run round-robin diversification to
        ensure minimum source coverage, keeping expanded_top candidates.
        """
        if not candidates or min_sources <= 0:
            return candidates[:final_top]

        top = candidates[:expanded_top]
        sources_in_top = len({r.document_id for r in top})

        if sources_in_top >= min_sources:
            # Already diverse enough — just trim to final_top
            result = candidates[:final_top]
        else:
            # One source dominates — diversify
            logger.debug("conditional_diversify_triggered",
                         sources_in_top=sources_in_top, min_sources=min_sources)
            result = self._diversify_by_source(candidates, expanded_top, min_sources)
            result = result[:final_top]

        for rank, ref in enumerate(result, 1):
            result[rank - 1] = ref.model_copy(update={"rank": rank})
        return result

    def _diversify_by_source(
        self,
        candidates: list[EvidenceRef],
        top_k: int,
        min_sources: int,
    ) -> list[EvidenceRef]:
        """Select candidates ensuring source document diversity.

        Greedy round-robin: first pass picks the top-scoring chunk from each
        distinct source, then fills remaining slots by score.  This prevents
        one dominant document from monopolizing all top-k slots when the query
        spans multiple documents.
        """
        if not candidates or min_sources <= 0:
            return candidates[:top_k]

        # Group by source document
        by_source: dict[UUID, list[EvidenceRef]] = {}
        for ref in candidates:
            by_source.setdefault(ref.document_id, []).append(ref)

        selected: list[EvidenceRef] = []
        selected_ids: set[UUID] = set()
        remaining: list[EvidenceRef] = []

        # Pass 1: pick top chunk from each source
        for doc_id, refs in by_source.items():
            if refs:
                selected.append(refs[0])
                selected_ids.add(doc_id)
                remaining.extend(refs[1:])

        # Pass 2: fill remaining slots by score from what's left
        remaining.sort(key=lambda r: r.score, reverse=True)
        for ref in remaining:
            if len(selected) >= top_k:
                break
            if ref.chunk_id not in {s.chunk_id for s in selected}:
                selected.append(ref)

        # If we still haven't hit min_sources, add from any source
        if len(selected) < min_sources:
            for ref in candidates:
                if len(selected) >= top_k:
                    break
                if ref.chunk_id not in {s.chunk_id for s in selected}:
                    selected.append(ref)

        # Re-rank by score
        selected.sort(key=lambda r: r.score, reverse=True)
        result = selected[:top_k]
        for rank, ref in enumerate(result, 1):
            result[rank - 1] = ref.model_copy(update={"rank": rank})

        logger.debug(
            "diversify_by_source",
            input_count=len(candidates),
            output_count=len(result),
            sources_covered=len({r.document_id for r in result}),
        )
        return result

    def _bge_m3_dispatch(
        self,
        method: RetrievalMethod,
        query: str,
        upper_k: int,
    ) -> list[EvidenceRef]:
        """Dispatch BGE-M3 retrieval methods.

        Lazy-loads the BGEM3Retriever on first use. Degrades to baseline
        hybrid if BGE-M3 is unavailable or fails.
        """
        if self._bge_m3_retriever is None:
            try:
                from app.retrieval.bge_m3 import BGEM3Retriever

                self._bge_m3_retriever = BGEM3Retriever()
            except Exception as exc:  # noqa: BLE001
                logger.warning("bge_m3_unavailable", error=str(exc))
                return []

        bge = self._bge_m3_retriever
        if method == RetrievalMethod.BGE_M3_DENSE:
            return bge.search_as_refs(query, top_k=upper_k, mode="dense")
        if method == RetrievalMethod.BGE_M3_SPARSE:
            return bge.search_as_refs(query, top_k=upper_k, mode="sparse")
        return bge.search_as_refs(query, top_k=upper_k, mode="hybrid")

    # -- fusion helpers ------------------------------------------------------

    @staticmethod
    def _fuse(
        per_method: list[tuple[RetrievalMethod, list[EvidenceRef]]],
        weights: dict[RetrievalMethod, float] | None,
    ) -> list[EvidenceRef]:
        weights = weights or {}
        if not weights:
            weights = {method: 1.0 / len(per_method) for method, _ in per_method}

        best: dict[Any, tuple[float, EvidenceRef]] = {}
        for method, refs in per_method:
            if not refs:
                continue
            w = weights.get(method, 1.0 / len(per_method))
            method_max = max(r.score for r in refs) or 1.0
            for ref in refs:
                weighted = (ref.score / method_max) * w
                prior = best.get(ref.chunk_id)
                if prior is None or weighted > prior[0]:
                    best[ref.chunk_id] = (weighted, ref)

        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        out: list[EvidenceRef] = []
        for rank, (scored, ref) in enumerate(ranked, 1):
            out.append(
                ref.model_copy(
                    update={
                        "score": scored,
                        "rank": rank,
                        "metadata": {**ref.metadata, "policy_score": round(scored, 4)},
                    }
                )
            )
        return out

    @staticmethod
    def _apply_metadata_filters(
        refs: list[EvidenceRef],
        filters: dict[str, Any] | None,
    ) -> list[EvidenceRef]:
        if not filters:
            return refs
        filtered: list[EvidenceRef] = []
        for ref in refs:
            match = True
            for key, expected in filters.items():
                if key == "source_type":
                    match = ref.source_type.value == expected
                elif key in ref.metadata:
                    match = ref.metadata[key] == expected
                elif hasattr(ref, key):
                    match = getattr(ref, key) == expected
                else:
                    match = False
                if not match:
                    break
            if match:
                filtered.append(ref)
        return filtered


# =============================================================================
# Policy loading
# =============================================================================

_METHOD_BY_NAME = {m.value: m for m in RetrievalMethod}
_PATTERN_BY_NAME = {p.value: p for p in QuestionPattern}


def _mix_from_dict(data: dict[str, Any]) -> RetrievalMix:
    methods = [RetrievalMethod(name) for name in data.get("methods", ["hybrid"])]
    raw_weights = data.get("weights") or {}
    weights = {RetrievalMethod(k): float(v) for k, v in raw_weights.items()}
    return RetrievalMix(
        methods=methods,
        weights=weights,
        bm25_weight=float(data.get("bm25_weight", 0.5)),
        vector_weight=float(data.get("vector_weight", 0.5)),
        max_results_per_method=int(data.get("max_results_per_method", 20)),
        rerank_top_k=int(data.get("rerank_top_k", 10)),
        graph_max_hops=int(data.get("graph_max_hops", 2)),
        temporal_window_days=data.get("temporal_window_days"),
        metadata_filters=data.get("metadata_filters") or {},
    )


def _entry_from_dict(data: dict[str, Any]) -> RetrievalPolicyEntry:
    pattern = QuestionPattern(data["pattern"])
    mix = _mix_from_dict(data.get("retrieval_mix") or {})
    return RetrievalPolicyEntry(
        pattern=pattern,
        retrieval_mix=mix,
        required_entities=data.get("required_entities", []),
        min_confidence=float(data.get("min_confidence", 0.0)),
        priority=int(data.get("priority", 0)),
    )


def load_retrieval_policy(settings: Settings | None = None) -> RetrievalPolicy:
    """Load the retrieval policy from YAML, falling back to the default table.

    The YAML file is a human-editable mirror of the V2 §5.2 / V3 §5 table.
    Missing or malformed files degrade to `DEFAULT_RETRIEVAL_POLICY` from
    `app/retrieval/policy.py` (deterministic, never an error at startup).
    """
    from app.retrieval.policy import DEFAULT_RETRIEVAL_POLICY, get_default_retrieval_policy

    settings = settings or get_settings()
    raw_path = settings.retrieval_policy_config_path
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        logger.info("retrieval_policy_config_missing", path=str(path), fallback="default")
        return get_default_retrieval_policy()

    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [_entry_from_dict(e) for e in data.get("entries", [])]
        if not entries:
            logger.warning("retrieval_policy_config_empty", path=str(path), fallback="default")
            return DEFAULT_RETRIEVAL_POLICY
        return RetrievalPolicy(entries=entries)
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieval_policy_config_invalid", path=str(path), error=str(exc), fallback="default")
        return DEFAULT_RETRIEVAL_POLICY


def get_retrieval_policy_router(
    settings: Settings | None = None,
    graph_retriever: Any | None = None,
    bge_m3_retriever: Any | None = None,
) -> RetrievalPolicyRouter:
    """Create a policy router bound to the settings' policy table."""
    return RetrievalPolicyRouter(
        policy=None,
        graph_retriever=graph_retriever,
        settings=settings,
        bge_m3_retriever=bge_m3_retriever,
    )


__all__ = [
    "RetrievalPolicyRouter",
    "get_retrieval_policy_router",
    "load_retrieval_policy",
]