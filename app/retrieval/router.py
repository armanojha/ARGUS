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
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT, Settings, get_settings
from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.hybrid import HybridRetriever
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
    ) -> None:
        self.settings = settings or get_settings()
        self._policy = policy or load_retrieval_policy(self.settings)
        # Optional Phase 03 graph retriever. When absent, GRAPH/TEMPORAL
        # dispatch degrades to hybrid retrieval (never fabricates output).
        self.graph_retriever = graph_retriever

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
        """
        q = query.strip()
        low = q.lower()

        has_quoted = _has_balanced_quotes(q)
        year_matches = _YEAR_RE.findall(q)

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

        if has_quoted or any(w in low for w in _EXACT_WORDS):
            return QuestionPattern.EXACT_TERM

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

        per_method: list[tuple[RetrievalMethod, list[EvidenceRef]]] = []
        for method in mix.methods:
            try:
                refs = self._dispatch(method, mix, query, retriever, upper_k=mix.max_results_per_method)
            except Exception as exc:  # noqa: BLE001 - a single method must not fail the whole search
                logger.warning("policy_method_failed", method=method.value, error=str(exc))
                refs = []
            if refs:
                per_method.append((method, refs))

        if not per_method:
            # Deterministic fallback to a plain hybrid pass (never empty-handed).
            logger.warning("policy_mix_empty", pattern=pattern.value, fallback="hybrid")
            per_method = [(RetrievalMethod.HYBRID, retriever.search(query, top_k=top_k))]

        fused = self._fuse(per_method, mix.weights)
        fused = self._apply_metadata_filters(fused, mix.metadata_filters)

        if fused and reranker is not None:
            fused = reranker.rerank(query, fused, top_k=top_k)
        else:
            fused = fused[:top_k]
            for rank, ref in enumerate(fused, 1):
                fused[rank - 1] = ref.model_copy(update={"rank": rank})

        logger.info(
            "policy_retrieval_executed",
            pattern=pattern.value,
            methods=[m.value for m, _ in per_method],
            results=len(fused),
        )
        return fused

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
        return retriever.search(query, top_k=upper_k)

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
) -> RetrievalPolicyRouter:
    """Create a policy router bound to the settings' policy table."""
    return RetrievalPolicyRouter(policy=None, graph_retriever=graph_retriever, settings=settings)


__all__ = [
    "RetrievalPolicyRouter",
    "get_retrieval_policy_router",
    "load_retrieval_policy",
]