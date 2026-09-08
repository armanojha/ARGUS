"""Graph node implementations for the Agentic RAG loop (Phase 02).

Each `make_*_node` factory closes over the dependencies it needs (the
LLM router, the Phase 01 hybrid retriever, the reranker, and settings)
and returns an async callable matching LangGraph's `(state) -> dict`
node signature. Dependencies are injected rather than imported as
module-level singletons so tests can substitute a `MockProvider` /
in-memory store without monkeypatching.

Failure handling: every LLM call is wrapped so a provider error or a
malformed structured response degrades to a deterministic fallback
instead of raising out of the graph. Evidence already accumulated is
never discarded on an LLM failure — synthesis always runs over
whatever evidence exists, per the vault's evidence-first rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.evidence.models import EvidenceRef
from app.llm_gateway.providers.exceptions import LLMProviderError
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.models import (
    ComplexityLevel,
    EvidenceAssessment,
    QueryAnalysis,
    ResearchPlan,
    StopReason,
)
from app.orchestration.prompts import (
    build_analysis_messages,
    build_assessment_messages,
    build_planning_messages,
    build_synthesis_messages,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.stopping import stop_condition_to_reason
from app.reranking.reranker import NoOpReranker, Reranker
from app.retrieval.evidence_selector import EvidenceSelector
from app.retrieval.hybrid import HybridRetriever

# ---------------------------------------------------------------------------
# Deterministic contradiction detection (Phase 39)
# ---------------------------------------------------------------------------

# Negation patterns that signal factual opposition
_NEGATION_PAIRS = [
    ("is", "is not"), ("is", "isn't"),
    ("was", "was not"), ("was", "wasn't"),
    ("are", "are not"), ("are", "aren't"),
    ("can", "can not"), ("can", "can't"),
    ("will", "will not"), ("will", "won't"),
    ("does", "does not"), ("does", "doesn't"),
    ("did", "did not"), ("did", "didn't"),
    ("has", "has not"), ("has", "hasn't"),
    ("have", "have not"), ("have", "haven't"),
    ("no ", "yes "), ("not ", ""),
    ("true", "false"), ("confirmed", "denied"),
    ("possible", "impossible"), ("safe", "unsafe"),
    ("increase", "decrease"), ("rise", "fall"),
    ("higher", "lower"), ("more", "less"),
    ("supports", "contradicts"), ("contains", "does not contain"),
]

_METRIC_KEYWORDS = frozenset({
    "revenue", "employees", "utilization", "growth", "output",
    "production", "units", "profit", "income", "capacity",
    "rate", "percent", "sales", "cost", "price", "margin",
})

# Temporal patterns: years, quarters, "as of", etc.
_YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_AS_OF_RE = re.compile(r"\bas of\b.*?\b(20[0-9]{2})\b")

# Unit normalization: map unit strings to a canonical form
_UNIT_NORMALIZE = {
    "billion": "B", "bn": "B", "b": "B",
    "million": "M", "mn": "M", "m": "M",
    "thousand": "K", "k": "K",
    "percent": "%", "pct": "%", "%": "%",
}


def _extract_years(text: str) -> set[int]:
    """Extract mentioned years from text."""
    years = set()
    for m in _YEAR_RE.finditer(text):
        years.add(int(m.group(1)))
    return years


def _extract_quarters(text: str) -> set[str]:
    """Extract mentioned quarters from text."""
    quarters = set()
    for m in _QUARTER_RE.finditer(text):
        quarters.add(f"Q{m.group(1)}")
    return quarters


def _normalize_number_with_unit(raw: str) -> tuple[float, str]:
    """Normalize a number string like '$3.1 billion' to (3.1, 'B').

    Returns (numeric_value, canonical_unit).
    """
    s = raw.strip().lower()
    # Detect unit suffix
    unit = ""
    for unit_str, canonical in _UNIT_NORMALIZE.items():
        if s.endswith(unit_str):
            unit = canonical
            s = s[: -len(unit_str)].strip()
            break
    # Strip currency symbols and commas
    s = re.sub(r"[$,]", "", s)
    try:
        val = float(s)
    except ValueError:
        return (0.0, "")
    return (val, unit)


def _normalize_value(val: float, unit: str) -> float:
    """Normalize a value to a common scale (billions) for comparison.

    This allows comparing '$3.1 billion' with '$3100 million' as equal.
    """
    multipliers = {"B": 1.0, "M": 0.001, "K": 0.000001, "%": 1.0, "": 1.0}
    return val * multipliers.get(unit, 1.0)


def _extract_entity_keywords(text: str) -> set[str]:
    """Extract likely entity names (capitalized words, proper nouns)."""
    # Simple heuristic: words that are capitalized and 3+ chars
    words = set()
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        w = m.group(1).lower()
        if w not in {"the", "and", "for", "with", "from", "this", "that", "report", "annual", "fiscal", "total", "combined"}:
            words.add(w)
    return words


def _detect_temporal_context(text: str) -> dict:
    """Extract temporal context from text."""
    years = _extract_years(text)
    quarters = _extract_quarters(text)
    # Check for "as of" pattern
    as_of_match = _AS_OF_RE.search(text)
    as_of_year = int(as_of_match.group(1)) if as_of_match else None
    return {
        "years": years,
        "quarters": quarters,
        "as_of_year": as_of_year,
        "has_temporal": bool(years or quarters),
    }


def _compute_confidence(
    shared_metrics: set[str],
    years_i: set[int],
    years_j: set[int],
    entities_i: set[str],
    entities_j: set[str],
    severity: float,
) -> str:
    """Compute conflict confidence: HIGH, MEDIUM, or LOW.

    HIGH: Same entity, same metric, same timeframe, different values
    MEDIUM: Same metric, similar context, different values
    LOW: Same metric but different context or weak signals
    """
    same_entity = bool(entities_i & entities_j)
    same_timeframe = bool(years_i & years_j) or (not years_i and not years_j)

    if same_entity and same_timeframe and shared_metrics and severity >= 0.8:
        return "HIGH"
    elif same_entity and shared_metrics and severity >= 0.5:
        return "MEDIUM"
    elif shared_metrics:
        return "LOW"
    return "LOW"


def _classify_conflict_type(
    years_i: set[int],
    years_j: set[int],
    entities_i: set[str],
    entities_j: set[str],
    metrics_i: set[str],
    metrics_j: set[str],
) -> str:
    """Classify the type of conflict."""
    same_entity = bool(entities_i & entities_j)
    same_timeframe = bool(years_i & years_j)
    same_metrics = bool(metrics_i & metrics_j)
    both_have_years = bool(years_i or years_j)

    if same_entity and same_timeframe and same_metrics:
        return "GENUINE_CONTRADICTION"
    elif same_entity and both_have_years and not same_timeframe:
        return "DIFFERENT_TIMEFRAME"
    elif not same_entity and same_metrics:
        return "DIFFERENT_SOURCE"
    elif same_entity and same_metrics:
        return "POSSIBLE_CONTRADICTION"
    else:
        return "IRRELEVANT_DIFFERENCE"


# ---------------------------------------------------------------------------
# Query relevance gate (deterministic, no LLM)
# ---------------------------------------------------------------------------

# Stopwords excluded from relevance overlap calculations
_STOPWORDS = frozenset({
    "what", "which", "when", "where", "does", "that", "have", "been",
    "from", "about", "how", "many", "was", "were", "this", "with", "their", "supports", "claim", "evidence",
    "information", "data", "system", "using", "based", "provide",
})


def _compute_query_relevance(
    evidence_text: str,
    query: str,
) -> float:
    """Compute a deterministic relevance score between evidence and query.

    Uses lexical overlap (Jaccard-like) weighted by content-word density.
    Returns a float in [0.0, 1.0].  No LLM call.

    A score below 0.15 indicates the evidence is almost certainly irrelevant
    to the query.
    """
    query_lower = query.lower()
    evidence_lower = evidence_text.lower()

    query_words = set(re.findall(r"\b\w{4,}\b", query_lower)) - _STOPWORDS
    evidence_words = set(re.findall(r"\b\w{4,}\b", evidence_lower)) - _STOPWORDS

    if not query_words or not evidence_words:
        return 0.0

    overlap = query_words & evidence_words
    # Jaccard-like: intersection / union, boosted by overlap count
    union = query_words | evidence_words
    jaccard = len(overlap) / len(union) if union else 0.0

    # Coverage: what fraction of query words appear in evidence
    coverage = len(overlap) / len(query_words) if query_words else 0.0

    # Weighted combination: coverage matters more than Jaccard for relevance
    return 0.4 * jaccard + 0.6 * coverage


def _is_topic_coherent(
    text_i: str,
    text_j: str,
    min_shared_significant: int = 3,
) -> bool:
    """Check whether two evidence chunks discuss the same topic.

    Requires at least ``min_shared_significant`` meaningful words (4+ chars,
    excluding stopwords) to overlap.  This prevents flagging contradictions
    between documents that merely share generic vocabulary like "analytics"
    or "database".
    """
    words_i = set(re.findall(r"\b\w{4,}\b", text_i.lower())) - _STOPWORDS
    words_j = set(re.findall(r"\b\w{4,}\b", text_j.lower())) - _STOPWORDS
    shared = words_i & words_j
    return len(shared) >= min_shared_significant


def _filter_evidence_by_relevance(
    evidence: list[EvidenceRef],
    query: str,
    relevance_threshold: float = 0.12,
) -> tuple[list[EvidenceRef], list[dict]]:
    """Filter evidence by query relevance. Returns (relevant, excluded).

    Each excluded item gets metadata explaining why it was filtered.
    This is deterministic and adds no LLM calls.
    """
    relevant: list[EvidenceRef] = []
    excluded: list[dict] = []

    for i, ev in enumerate(evidence):
        score = _compute_query_relevance(ev.text, query)
        if score >= relevance_threshold:
            relevant.append(ev)
        else:
            excluded.append({
                "index": i,
                "reason": "insufficient_relevance",
                "relevance_score": round(score, 3),
                "text_preview": ev.text[:120],
            })

    return relevant, excluded


def _detect_contradictions(
    evidence: list[EvidenceRef],
    query: str = "",
) -> list[dict]:
    """Deterministically detect pairwise contradictions in evidence chunks.

    Context-aware detection that distinguishes:
    - GENUINE_CONTRADICTION: same entity, same metric, same timeframe, different values
    - DIFFERENT_TIMEFRAME: same entity/metric but different years (not a contradiction)
    - DIFFERENT_SOURCE: different sources with different values
    - IRRELEVANT_DIFFERENCE: unrelated numbers in different contexts

    Returns a list of contradiction signal dicts suitable for
    ``state["contradiction_signals"]``.
    """
    if len(evidence) < 2:
        return []

    contradictions: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            text_i = evidence[i].text.lower()
            text_j = evidence[j].text.lower()

            # Extract temporal context
            ctx_i = _detect_temporal_context(evidence[i].text)
            ctx_j = _detect_temporal_context(evidence[j].text)

            # Extract entity keywords
            entities_i = _extract_entity_keywords(evidence[i].text)
            entities_j = _extract_entity_keywords(evidence[j].text)

            # Extract metric keywords present in both chunks
            words_i = set(re.findall(r"\b\w{4,}\b", text_i))
            words_j = set(re.findall(r"\b\w{4,}\b", text_j))
            shared_metrics = words_i & words_j & _METRIC_KEYWORDS

            # ── Check 1: Direct negation pairs ──
            # SAFETY: Require topic coherence AND entity overlap before
            # flagging a negation-based contradiction.  Shared generic
            # words alone (e.g. "analytics", "database") are NOT sufficient.
            for neg_a, neg_b in _NEGATION_PAIRS:
                if (neg_a in text_i and neg_b in text_j) or (neg_b in text_i and neg_a in text_j):
                    # Topic coherence: chunks must share meaningful vocabulary
                    if not _is_topic_coherent(evidence[i].text, evidence[j].text, min_shared_significant=3):
                        continue
                    shared_words = words_i & words_j
                    shared_words -= {"this", "that", "with", "from", "have", "been", "were", "their", "which", "about"}
                    # Require entity overlap for negation to be meaningful
                    entity_overlap = entities_i & entities_j
                    if len(shared_words) >= 3 or (len(shared_words) >= 2 and entity_overlap):
                        pair = (min(i, j), max(i, j))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            conflict_type = _classify_conflict_type(
                                ctx_i["years"], ctx_j["years"],
                                entities_i, entities_j,
                                shared_metrics, shared_metrics,
                            )
                            confidence = _compute_confidence(
                                shared_metrics, ctx_i["years"], ctx_j["years"],
                                entities_i, entities_j, 1.0,
                            )
                            contradictions.append({
                                "severity": 1.0,
                                "confidence": confidence,
                                "conflict_type": conflict_type,
                                "description": (
                                    f"Evidence [{i+1}] and [{j+1}] contain opposing claims: "
                                    f"shared terms: {', '.join(sorted(shared_words)[:5])}"
                                ),
                                "evidence_indices": [i + 1, j + 1],
                                "entity_overlap": sorted(entities_i & entities_j)[:3],
                                "metric_overlap": sorted(shared_metrics)[:3],
                                "timeframe_i": sorted(ctx_i["years"]),
                                "timeframe_j": sorted(ctx_j["years"]),
                                "resolved": False,
                                "critical": True,
                            })
                            # SAFETY: If this is a negation pair with entity overlap
                            # but no shared metrics, upgrade to POSSIBLE_CONTRADICTION.
                            # IRRELEVANT_DIFFERENCE is wrong for direct opposition
                            # on the same entity.
                            if (conflict_type == "IRRELEVANT_DIFFERENCE"
                                    and entities_i & entities_j):
                                contradictions[-1]["conflict_type"] = "POSSIBLE_CONTRADICTION"
                        break  # One contradiction per pair is enough

            # ── Check 2: Numerical discrepancies with context ──
            # SAFETY: Require topic coherence before checking numerical discrepancies.
            # Documents about unrelated topics sharing a generic metric keyword
            # (e.g. "rate", "growth") must NOT be flagged.
            if shared_metrics and _is_topic_coherent(evidence[i].text, evidence[j].text, min_shared_significant=3):
                # For each shared metric, extract numbers near it in each chunk
                metric_nums_i: set[str] = set()
                metric_nums_j: set[str] = set()
                raw_nums_i: list[tuple[float, str]] = []
                raw_nums_j: list[tuple[float, str]] = []

                for metric in shared_metrics:
                    for text, num_set, raw_list in [
                        (text_i, metric_nums_i, raw_nums_i),
                        (text_j, metric_nums_j, raw_nums_j),
                    ]:
                        for m in re.finditer(r"\b" + re.escape(metric) + r"\b", text):
                            start = max(0, m.start() - 50)
                            end = min(len(text), m.end() + 50)
                            window = text[start:end]
                            for n in re.findall(r"\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|%)?", window):
                                cleaned = re.sub(r"[,$]", "", n.strip())
                                if cleaned:
                                    num_set.add(cleaned)
                                    raw_list.append(_normalize_number_with_unit(n))

                if metric_nums_i and metric_nums_j and metric_nums_i != metric_nums_j:
                    pair = (min(i, j), max(i, j))
                    if pair not in seen_pairs:
                        # Check if values are actually equivalent after normalization
                        norm_i = {_normalize_value(v, u) for v, u in raw_nums_i if v > 0}
                        norm_j = {_normalize_value(v, u) for v, u in raw_nums_j if v > 0}

                        # If normalized values are the same, it's not a contradiction
                        if norm_i and norm_j and norm_i == norm_j:
                            continue

                        seen_pairs.add(pair)
                        conflict_type = _classify_conflict_type(
                            ctx_i["years"], ctx_j["years"],
                            entities_i, entities_j,
                            shared_metrics, shared_metrics,
                        )
                        confidence = _compute_confidence(
                            shared_metrics, ctx_i["years"], ctx_j["years"],
                            entities_i, entities_j, 0.8,
                        )
                        contradictions.append({
                            "severity": 0.8,
                            "confidence": confidence,
                            "conflict_type": conflict_type,
                            "description": (
                                f"Evidence [{i+1}] and [{j+1}] present different values "
                                f"for metric(s) {sorted(shared_metrics)}: "
                                f"{sorted(metric_nums_i)[:3]} vs {sorted(metric_nums_j)[:3]}"
                            ),
                            "evidence_indices": [i + 1, j + 1],
                            "entity_overlap": sorted(entities_i & entities_j)[:3],
                            "metric_overlap": sorted(shared_metrics)[:3],
                            "timeframe_i": sorted(ctx_i["years"]),
                            "timeframe_j": sorted(ctx_j["years"]),
                            "resolved": False,
                            "critical": True,
                        })

    return contradictions


async def _semantic_contradiction_check(
    contradictions: list[dict],
    evidence: list[EvidenceRef],
    router: LLMRouter,
    settings: object,
    request_id: str = "",
) -> list[dict]:
    """Stage 3: LLM semantic verification of deterministic contradiction candidates.

    Takes pairs flagged by the deterministic layer and asks an LLM to determine
    whether they are genuinely CONTRADICTED, ENTAILED, NEUTRAL, or
    INSUFFICIENT_CONTEXT.

    Semantics:
      - LLM success: only pairs with a CONTRADICTED verdict are kept
        (marked with ``semantic_verified=True``).
      - LLM failure: the deterministic candidate is retained WITHOUT the
        ``semantic_verified`` flag (safe default — never silently drop a
        deterministic signal because the LLM call errored).

    This prevents false positives from heuristic matching (e.g., "Revenue
    increased to $3B" vs "Revenue reached $2.7B" being different time periods).
    """
    if not contradictions:
        return contradictions

    verified = []
    for sig in contradictions:
        idx_i = sig.get("evidence_indices", [0, 0])[0] - 1
        idx_j = sig.get("evidence_indices", [0, 0])[1] - 1
        if idx_i < 0 or idx_i >= len(evidence) or idx_j < 0 or idx_j >= len(evidence):
            sig["semantic_verified"] = False
            verified.append(sig)
            continue

        text_i = evidence[idx_i].text[:500]
        text_j = evidence[idx_j].text[:500]

        messages = [
            {"role": "system", "content": (
                "You are a contradiction detection verifier. Given two text "
                "passages, determine whether they CONTRADICT each other, are "
                "ENTAILED (one implies the other), are NEUTRAL (unrelated), or "
                "have INSUFFICIENT_CONTEXT to determine. Respond with exactly "
                "one label: CONTRADICTED, ENTAILED, NEUTRAL, or INSUFFICIENT_CONTEXT."
            )},
            {"role": "user", "content": (
                f"Passage A: {text_i}\n\n"
                f"Passage B: {text_j}\n\n"
                f"Deterministic flag: {sig.get('conflict_type', 'unknown')} "
                f"(confidence: {sig.get('confidence', 'LOW')})\n\n"
                "Verdict:"
            )},
        ]

        try:
            from pydantic import BaseModel

            class ContradictionVerdict(BaseModel):
                verdict: str  # CONTRADICTED, ENTAILED, NEUTRAL, INSUFFICIENT_CONTEXT

            result = await _safe_structured_call(
                router,
                messages=messages,
                response_model=ContradictionVerdict,
                call_type="verification",
                settings=settings,
                request_id=request_id,
            )
            model, error = result[0], result[1]
            if model is None:
                # LLM call failed (provider error, malformed JSON, schema
                # error). Retain the deterministic candidate WITHOUT the
                # semantic_verified flag — never silently drop a deterministic
                # signal because the LLM call errored.
                logger.debug(
                    "semantic_contradiction_llm_error",
                    conflict_type=sig.get("conflict_type"),
                    error=error,
                    request_id=request_id,
                )
                sig["semantic_verified"] = False
                verified.append(sig)
            elif model.verdict.upper() == "CONTRADICTED":
                sig["semantic_verified"] = True
                verified.append(sig)
            else:
                logger.debug(
                    "semantic_contradiction_rejected",
                    conflict_type=sig.get("conflict_type"),
                    verdict=model.verdict,
                    request_id=request_id,
                )
        except Exception:  # noqa: BLE001 - fail-safe: keep deterministic signal on LLM error
            # If LLM check fails, keep the deterministic signal WITHOUT the
            # semantic_verified flag (safe default — never silently drop a
            # deterministic signal because the LLM call errored).
            sig["semantic_verified"] = False
            verified.append(sig)

    return verified


def filter_contradictions_by_query(
    contradictions: list[dict],
    evidence: list[EvidenceRef],
    query: str,
) -> list[dict]:
    """Filter contradiction signals to only those relevant to the user's query.

    A contradiction is relevant if:
    1. Its metric overlap includes a metric the query is asking about
    2. Its conflict type is not DIFFERENT_TIMEFRAME (unless query asks about history)
    3. Its confidence is not LOW (unless query explicitly asks about conflicts)

    This prevents false positives where historical data (2023 vs 2025) is
    incorrectly flagged as contradictory.
    """
    if not contradictions:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r"\b\w{4,}\b", query_lower))
    query_words -= {"what", "which", "when", "where", "does", "that", "have", "been", "from", "about", "how", "many", "was", "were"}

    # Extract the query's intended metric
    query_metrics = query_words & _METRIC_KEYWORDS

    # Check if query explicitly asks about conflicts/history
    conflict_intent = any(w in query_lower for w in [
        "conflict", "contradict", "discrepan", "differ",
        "disagree", "inconsistent", "versus", "vs",
    ])
    historical_intent = any(w in query_lower for w in [
        "history", "historical", "trend", "over time", "change",
        "compare", "comparison", "before", "after",
    ])

    filtered = []
    for sig in contradictions:
        conflict_type = sig.get("conflict_type", "UNKNOWN")
        confidence = sig.get("confidence", "LOW")
        entity_overlap = set(sig.get("entity_overlap", []))
        metric_overlap = set(sig.get("metric_overlap", []))
        set(sig.get("timeframe_i", []))
        set(sig.get("timeframe_j", []))

        # ── Rule 1: DIFFERENT_TIMEFRAME is filtered unless query asks about history
        # OR the query asks about the same metric (user needs both values) ──
        if (conflict_type == "DIFFERENT_TIMEFRAME"
                and not historical_intent and not (metric_overlap & query_metrics)):
            continue

        # ── Rule 2: LOW confidence signals are filtered unless query asks about conflicts ──
        if confidence == "LOW" and not conflict_intent:
            continue

        # ── Rule 3: Check metric relevance to query ──
        # The conflict is relevant only if its metric matches what the query asks about
        if query_metrics:
            # Query asks about a specific metric — conflict must match
            if not (metric_overlap & query_metrics):
                continue
        else:
            # Query doesn't mention a specific metric — don't show conflicts
            # unless query explicitly asks about conflicts or history
            if not conflict_intent and not historical_intent:
                continue

        # ── Rule 4: Check entity relevance ──
        # If query mentions a specific entity, conflict should involve that entity
        query_entities = query_words - _METRIC_KEYWORDS
        if (query_entities and not (entity_overlap & query_entities)
                and not (metric_overlap & query_metrics)):
            continue

        filtered.append(sig)

    return filtered


def _is_evidence_absent(
    evidence: list[EvidenceRef],
    query: str,
    *,
    score_threshold: float = 0.15,
) -> bool:
    """Deterministically detect if retrieved evidence contains no relevant content.

    Returns True when evidence was retrieved but is irrelevant to the query
    (all scores below threshold and low keyword overlap). This signals that
    the corpus genuinely lacks the requested information.
    """
    if not evidence:
        return True

    # Check if all evidence scores are very low
    low_score_count = sum(1 for e in evidence if e.score > 0 and e.score < score_threshold)
    if low_score_count == len(evidence):
        # All scored evidence is low — likely irrelevant
        return True

    # Check keyword overlap between query and evidence
    query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
    query_words -= {"what", "which", "when", "where", "does", "that", "have", "been", "from", "about", "what's"}
    if not query_words:
        return False

    evidence_text = " ".join(e.text.lower() for e in evidence)
    evidence_words = set(re.findall(r"\b\w{4,}\b", evidence_text))

    overlap = query_words & evidence_words
    overlap_ratio = len(overlap) / len(query_words) if query_words else 0

    # Very low overlap with the query suggests irrelevant evidence
    return overlap_ratio < 0.2


logger = get_logger("argus.orchestration.nodes")

NodeFn = Callable[[OrchestrationState], Coroutine[Any, Any, dict]]

_MAX_CONSECUTIVE_EMPTY_RETRIEVALS = 2


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budget accounting.

    Not a substitute for a real tokenizer — good enough to bound loop
    growth without pulling in a model-specific tokenizer dependency.
    """
    return max(1, len(text) // 4)


async def _safe_structured_call(
    router: LLMRouter,
    *,
    messages: list,
    response_model: type[BaseModel],
    call_type: str,
    settings: Settings,
    request_id: str | None,
    query: str | None = None,
) -> tuple[Any | None, str | None]:
    """Run a structured LLM call, returning (parsed_model_or_None, error_message_or_None).

    ``query`` (optional) is forwarded to the router so it can auto-classify
    task complexity for tier-adaptive routing (HARDEN-06.5.2) without an
    extra LLM round-trip.

    Never raises: provider errors, malformed JSON, and schema validation
    failures are all normalized into an error string so callers can
    apply a deterministic fallback and keep the loop moving.
    """
    try:
        response = await router.complete(
            messages,
            response_format=response_model,
            timeout=settings.orchestration_llm_timeout,
            call_type=call_type,
            request_id=request_id,
            query=query,
        )
    except LLMProviderError as exc:
        logger.warning("orchestration_llm_call_failed", call_type=call_type, error=str(exc))
        return None, f"{call_type} call failed: {exc}"

    if not response.content:
        logger.warning("orchestration_llm_empty_response", call_type=call_type)
        return None, f"{call_type} call returned no content"

    try:
        parsed = response_model.model_validate(json.loads(response.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("orchestration_llm_malformed_response", call_type=call_type, error=str(exc))
        return None, f"{call_type} response did not match schema: {exc}"

    return parsed, None


def make_analyze_node(router: LLMRouter, settings: Settings) -> NodeFn:
    async def analyze_node(state: OrchestrationState) -> dict:
        messages = build_analysis_messages(state["query"])
        analysis, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=QueryAnalysis,
            call_type="query_analysis",
            settings=settings,
            request_id=state["request_id"],
            query=state["query"],
        )
        warnings = list(state["warnings"])
        if analysis is None:
            warnings.append(f"query_analysis_fallback: {error}")
            analysis = QueryAnalysis(
                complexity=ComplexityLevel.MODERATE,
                reasoning="Fallback: query analysis LLM call failed.",
                suggested_subquestion_count=settings.orchestration_default_subquestions,
            )
        return {"query_analysis": analysis, "warnings": warnings}

    return analyze_node


def make_plan_node(router: LLMRouter, settings: Settings) -> NodeFn:
    async def plan_node(state: OrchestrationState) -> dict:
        analysis = state["query_analysis"] or QueryAnalysis(
            complexity=ComplexityLevel.MODERATE,
            reasoning="No analysis available.",
            suggested_subquestion_count=settings.orchestration_default_subquestions,
        )
        messages = build_planning_messages(state["query"], analysis)
        plan, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=ResearchPlan,
            call_type="research_planning",
            settings=settings,
            request_id=state["request_id"],
            query=state["query"],
        )
        warnings = list(state["warnings"])
        if plan is None:
            warnings.append(f"research_plan_fallback: {error}")
            plan = ResearchPlan(
                objective=state["query"],
                subquestions=[state["query"]],
                token_budget=settings.orchestration_token_budget,
                iteration_budget=settings.orchestration_max_iterations,
                stopping_condition="Fallback single-pass plan: planner LLM call failed.",
            )

        # The planner LLM may propose budgets; the orchestrator has the
        # final word. Clamp to configured hard ceilings (never allow the
        # plan to request more than ARGUS is configured to spend).
        clamped_iteration_budget = min(plan.iteration_budget, settings.orchestration_max_iterations)
        clamped_token_budget = min(plan.token_budget, settings.orchestration_token_budget)
        plan = plan.model_copy(
            update={
                "iteration_budget": max(1, clamped_iteration_budget),
                "token_budget": max(1, clamped_token_budget),
            }
        )

        pending = list(plan.subquestions) if plan.subquestions else [plan.objective or state["query"]]

        return {
            "plan": plan,
            "pending_subquestions": pending,
            "max_iterations": plan.iteration_budget,
            "token_budget": plan.token_budget,
            "warnings": warnings,
        }

    return plan_node


def _merge_evidence(existing: list[EvidenceRef], new: list[EvidenceRef]) -> tuple[list[EvidenceRef], int]:
    """Dedup by chunk_id, keeping the highest score seen. Returns (merged, new_count)."""
    by_id: dict[UUID, EvidenceRef] = {ref.chunk_id: ref for ref in existing}
    new_count = 0
    for ref in new:
        prior = by_id.get(ref.chunk_id)
        if prior is None:
            new_count += 1
            by_id[ref.chunk_id] = ref
        elif ref.score > prior.score:
            by_id[ref.chunk_id] = ref
    merged = sorted(by_id.values(), key=lambda r: r.score, reverse=True)
    return merged, new_count


def make_retrieve_node(
    retriever: HybridRetriever,
    reranker: Reranker | NoOpReranker,
    settings: Settings,
    policy_router: Any | None = None,
) -> NodeFn:
    async def retrieve_node(state: OrchestrationState) -> dict:
        pending = list(state["pending_subquestions"])
        subquery = pending.pop(0) if pending else state["query"]

        question_pattern = state.get("question_pattern")

        # Strategy-mutated depth: when the adaptive policy escalated retrieval
        # after a gain stall, honor its top_k; otherwise use the configured
        # default. None (policy off) always falls back to settings.
        top_k = state.get("strategy_top_k") or settings.orchestration_retrieval_top_k

        try:
            if policy_router is not None and settings.retrieval_policy_enabled:
                pattern = policy_router.classify_question(subquery)
                if pattern is not None:
                    question_pattern = pattern.value
                results = await policy_router.execute_retrieval(
                    subquery,
                    pattern,
                    retriever,
                    top_k=top_k,
                    reranker=reranker,
                )
                if not results:
                    # Deterministic fallback: plain hybrid pass. Never fabricate.
                    results = await retriever.search_async(subquery, top_k=top_k)
                    if results:
                        results = reranker.rerank(subquery, results, top_k=top_k)
            else:
                # Phase 07f: the async hybrid overlaps the independent BM25 and
                # vector passes (deterministic fusion, unaffected by ordering).
                results = await retriever.search_async(subquery, top_k=top_k)
                if results:
                    results = reranker.rerank(subquery, results, top_k=top_k)
        except Exception as exc:
            logger.exception("orchestration_retrieval_critical", subquery=subquery, error=str(exc))
            results = []

        merged_evidence, new_count = _merge_evidence(state["evidence"], results)

        issued = list(state["issued_subqueries"]) + [subquery]

        # Token accounting: only count tokens from NEW evidence (not already accumulated)
        new_evidence_ids = {ref.chunk_id for ref in merged_evidence} - {ref.chunk_id for ref in state["evidence"]}
        new_evidence_tokens = sum(_estimate_tokens(r.text) for r in results if r.chunk_id in new_evidence_ids)
        tokens_used = state["tokens_used"] + new_evidence_tokens

        consecutive_empty = state["consecutive_empty_retrievals"] + 1 if new_count == 0 else 0

        prev_total = len(state["evidence"])
        gain = (new_count / prev_total) if prev_total else (1.0 if new_count else 0.0)
        gain_history = list(state.get("retrieval_gain_history") or []) + [round(gain, 4)]

        logger.info(
            "orchestration_retrieve_iteration",
            request_id=state["request_id"],
            subquery=subquery[:80],
            new_evidence=new_count,
            total_evidence=len(merged_evidence),
            iteration=state["iteration"] + 1,
            pattern=question_pattern,
        )

        return {
            "pending_subquestions": pending,
            "issued_subqueries": issued,
            "evidence": merged_evidence,
            "tokens_used": tokens_used,
            "iteration": state["iteration"] + 1,
            "consecutive_empty_retrievals": consecutive_empty,
            "question_pattern": question_pattern,
            "retrieval_gain_history": gain_history,
        }

    return retrieve_node


def make_assess_node(
    router: LLMRouter,
    settings: Settings,
    gap_detector: Any | None = None,
    evidence_selector: EvidenceSelector | None = None,
    adaptive_research_policy: Any | None = None,
) -> NodeFn:
    async def assess_node(state: OrchestrationState) -> dict:
        plan = state["plan"]
        if plan is None:
            return {}

        # Deterministic short-circuits: never spend an LLM call once a
        # hard bound is already exceeded.
        if state["consecutive_empty_retrievals"] >= _MAX_CONSECUTIVE_EMPTY_RETRIEVALS:
            return {"sufficient": True, "stop_reason": StopReason.NO_NEW_EVIDENCE.value}

        if state["iteration"] >= state["max_iterations"] or state["tokens_used"] >= state["token_budget"]:
            return {"sufficient": True, "stop_reason": StopReason.BUDGET_EXHAUSTED.value}

        # Phase 39: Deterministic contradiction detection.
        # Detect pairwise contradictions in evidence and populate
        # contradiction_signals so synthesis can acknowledge conflicts.
        #
        # Phase 43: Relevance gate — only run contradiction detection on
        # evidence that is relevant to the query.  This prevents false
        # contradictions between unrelated documents sharing generic vocabulary.
        evidence = state["evidence"]
        contradiction_signals = list(state.get("contradiction_signals") or [])
        if evidence and not contradiction_signals:
            # Relevance gate: filter out clearly irrelevant evidence first
            relevant_evidence, excluded_evidence = _filter_evidence_by_relevance(
                evidence, state["query"], relevance_threshold=0.12,
            )
            if excluded_evidence:
                logger.info(
                    "evidence_relevance_filtered",
                    total=len(evidence),
                    relevant=len(relevant_evidence),
                    excluded=len(excluded_evidence),
                    request_id=state["request_id"],
                )
            # Run contradiction detection only on relevant evidence
            detected = _detect_contradictions(
                relevant_evidence if relevant_evidence else evidence,
                query=state["query"],
            )
            if detected:
                # Stage 3: LLM semantic verification (feature-flagged)
                semantic_check = getattr(settings, "conflict_semantic_check_enabled", False)
                if semantic_check and detected and router is not None:
                    detected = await _semantic_contradiction_check(
                        detected,
                        relevant_evidence if relevant_evidence else evidence,
                        router,
                        settings,
                        request_id=state["request_id"],
                    )
                # Phase 41: Query-aware filtering (feature-flagged)
                conflict_filtering = getattr(settings, "conflict_filtering_enabled", False)
                if conflict_filtering:
                    filtered = filter_contradictions_by_query(
                        detected, evidence, state["query"]
                    )
                else:
                    filtered = detected
                contradiction_signals = filtered
                logger.info(
                    "contradictions_detected",
                    count=len(detected),
                    filtered_count=len(filtered),
                    filtering_enabled=conflict_filtering,
                    request_id=state["request_id"],
                )

        # Phase 39: Deterministic absent-info detection.
        # When evidence exists but is all irrelevant (low scores, low overlap),
        # signal that information is genuinely absent from the corpus.
        if evidence and not state.get("sufficient") and _is_evidence_absent(evidence, state["query"]):
            logger.info(
                "absent_info_deterministic",
                evidence_count=len(evidence),
                request_id=state["request_id"],
            )
            return {
                    "sufficient": True,
                    "stop_reason": StopReason.NO_NEW_EVIDENCE.value,
                    "contradiction_signals": contradiction_signals,
                    "warnings": list(state["warnings"]) + ["absent_info_deterministic"],
                }

        # Phase 24.1: Adaptive research policy pre-check.
        # When enabled, the deterministic policy can short-circuit the LLM
        # assess call if evidence is clearly sufficient or clearly needs
        # investigation. This preserves the existing LLM-based assessment
        # as a fallback for ambiguous cases.
        if adaptive_research_policy is not None:
            decision = adaptive_research_policy.should_continue_retrieval(
                evidence=state["evidence"],
                need_coverage={},  # computed from plan needs below
                gain_history=state.get("retrieval_gain_history") or [],
                iteration=state["iteration"],
                max_iterations=state["max_iterations"],
                pattern=state.get("question_pattern") or "",
                contradictions_detected=len(state.get("contradiction_signals") or []),
                pending_subquestions=state.get("pending_subquestions") or [],
            )
            if decision.action == "synthesize":
                logger.info(
                    "adaptive_synthesize",
                    reason=decision.reason,
                    level=decision.sufficiency_level,
                    request_id=state["request_id"],
                )
                return {
                    "sufficient": True,
                    "stop_reason": StopReason.SUFFICIENT_EVIDENCE.value,
                    "contradiction_signals": contradiction_signals,
                    "warnings": list(state["warnings"]) + [f"adaptive_synthesize: {decision.reason}"],
                }

        # Evidence selection: pick minimal high-coverage subset for LLM context
        evidence_for_llm = state["evidence"]
        if evidence_selector and evidence_for_llm:
            evidence_for_llm = evidence_selector.select(evidence_for_llm)

        messages = build_assessment_messages(
            plan, evidence_for_llm, state["issued_subqueries"], state["pending_subquestions"]
        )
        assessment, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=EvidenceAssessment,
            call_type="evidence_extraction",
            settings=settings,
            request_id=state["request_id"],
            query=state["query"],
        )

        warnings = list(state["warnings"])
        if assessment is None:
            warnings.append(f"assessment_fallback: {error}")
            # Fail safe: stop rather than loop forever on repeated LLM errors.
            return {
                "sufficient": True,
                "stop_reason": StopReason.ASSESSMENT_ERROR.value,
                "contradiction_signals": contradiction_signals,
                "warnings": warnings,
            }

        # Evidence-aware complexity tier adjustment (Phase 18+):
        # When evidence is already strong, downgrade the tier so subsequent
        # calls (synthesis, verification) use cheaper models.
        updated_tier = state.get("complexity_tier")
        evidence = state["evidence"]
        if evidence and updated_tier and updated_tier != "fast":
            from app.llm_gateway.routing.complexity import (
                ComplexityTier,
                adjust_tier_for_evidence,
            )
            scores = [ref.score for ref in evidence if ref.score > 0]
            current = ComplexityTier(updated_tier)
            adjusted = adjust_tier_for_evidence(current, scores)
            if adjusted != current:
                updated_tier = adjusted.value
                logger.info(
                    "complexity_tier_downgraded",
                    from_tier=current.value,
                    to_tier=adjusted.value,
                    avg_score=round(sum(scores[:3]) / min(len(scores), 3), 3) if scores else 0,
                    request_id=state["request_id"],
                )

        # Active evidence seeking (Phase 06.2): formulate targeted retrieval
        # actions whenever the assessment concludes the evidence does not
        # answer the question. Deterministic — never an LLM call.
        gaps: list[dict[str, Any]] = []
        if gap_detector is not None and not assessment.sufficient:
            gaps = gap_detector.detect_gaps(state, plan, state["evidence"])
            evidence_tasks = list(state.get("evidence_tasks") or []) + list(gaps)
        else:
            evidence_tasks = list(state.get("evidence_tasks") or [])

        # Adaptive strategy mutation: let this iteration's discoveries change
        # how the NEXT iteration retrieves. Gated by the adaptive research
        # policy (default off, so base_pending/top_k/mutation stay neutral
        # and behavior is unchanged unless the policy is enabled).
        strategy_base_pending = list(state["pending_subquestions"])
        strategy_top_k: int | None = None
        strategy_mutation: str | None = None
        if adaptive_research_policy is not None and not assessment.sufficient:
            try:
                unresolved = any(
                    bool(s.get("critical")) and not bool(s.get("resolved"))
                    for s in contradiction_signals if isinstance(s, dict)
                )
                mutation = adaptive_research_policy.mutate_strategy(
                    current_queries=list(state["pending_subquestions"]),
                    gain_history=list(state.get("retrieval_gain_history") or []),
                    contradictions_unresolved=unresolved,
                    gaps=gaps,
                    iteration=int(state.get("iteration", 0)),
                    max_iterations=int(state.get("max_iterations", 0)),
                    base_top_k=int(getattr(settings, "orchestration_retrieval_top_k", 8)),
                    query=state["query"],
                )
            except Exception:  # noqa: BLE001 - mutation must never break assessment
                mutation = None
            if mutation is not None:
                if mutation.reordered_queries is not None:
                    strategy_base_pending = list(mutation.reordered_queries)
                strategy_top_k = mutation.top_k_override
                strategy_mutation = mutation.mutation

        if assessment.sufficient or not assessment.next_subquery:
            if not assessment.sufficient and gap_detector is not None and gap_detector.should_re_retrieve(gaps):
                # The assessor ran out of ideas but the policy sees a real
                # evidence gap: queue its highest-priority targeted action.
                top_gap = max(gaps, key=lambda g: g.get("priority", 0.0))
                next_q = (top_gap.get("suggested_query") or "").strip()
                already_issued = {q.strip().lower() for q in state["issued_subqueries"]}
                pending = list(strategy_base_pending)
                if next_q and next_q.lower() not in already_issued and next_q not in pending:
                    pending.append(next_q)
                    return {
                        "sufficient": False,
                        "pending_subquestions": pending,
                        "evidence_tasks": evidence_tasks,
                        "contradiction_signals": contradiction_signals,
                        "warnings": warnings,
                        "strategy_top_k": strategy_top_k,
                        "strategy_mutation": strategy_mutation,
                    }

            stop_reason = (
                StopReason.SUFFICIENT_EVIDENCE.value
                if assessment.sufficient
                else StopReason.NO_SUBQUESTIONS.value
            )
            return {
                "sufficient": True,
                "stop_reason": stop_reason,
                "contradiction_signals": contradiction_signals,
                "warnings": warnings,
                "evidence_tasks": evidence_tasks,
                "complexity_tier": updated_tier,
            }

        # Not sufficient, and a next query was proposed: queue it unless
        # it's a near-duplicate of one we've already issued.
        already_issued = {q.strip().lower() for q in state["issued_subqueries"]}
        pending = list(strategy_base_pending)
        next_q = assessment.next_subquery.strip()
        if next_q and next_q.lower() not in already_issued and next_q not in pending:
            pending.append(next_q)
        elif not pending:
            # Nothing new to try and nothing queued: stop rather than loop.
            return {
                "sufficient": True,
                "stop_reason": StopReason.NO_SUBQUESTIONS.value,
                "contradiction_signals": contradiction_signals,
                "warnings": warnings,
                "evidence_tasks": evidence_tasks,
                "complexity_tier": updated_tier,
            }

        return {
            "sufficient": False,
            "pending_subquestions": pending,
            "contradiction_signals": contradiction_signals,
            "warnings": warnings,
            "evidence_tasks": evidence_tasks,
            "complexity_tier": updated_tier,
            "strategy_top_k": strategy_top_k,
            "strategy_mutation": strategy_mutation,
        }

    return assess_node


def make_stop_check_node(stopping_logic: Any | None) -> NodeFn:
    """Evaluate the Phase 06 stopping logic after each assessment (V2 §5.4).

    When no stopping logic is wired, the node is a no-op and the router
    keeps the Phase 02 assess-based behavior unchanged.
    """

    async def stop_check_node(state: OrchestrationState) -> dict:
        if stopping_logic is None:
            return {}
        decision = await stopping_logic.should_stop(state)
        checked_all = list((decision.metadata or {}).get("checked", []))
        fired = decision.condition.value if decision.condition is not None else None
        if decision.should_stop:
            reason = stop_condition_to_reason(decision.condition)
            return {
                "sufficient": True,
                "stop_reason": reason.value if reason else state.get("stop_reason"),
                "stop_conditions_checked": checked_all,
                "stop_condition_fired": fired,
            }
        return {
            "stop_conditions_checked": checked_all,
            "stop_condition_fired": fired,
        }

    return stop_check_node


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

# Oriental/full-width variants models sometimes emit, e.g. 【1】 / ０１.
# Normalize to ASCII so the bracket-index matcher actually sees what the model
# cited instead of silently missing it (HARDEN-07d.3).
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _normalize_citation_markers(answer: str) -> str:
    """Map full-width citation punctuation/digits to ASCII for marker extraction."""
    if "【" in answer or "】" in answer:
        answer = answer.replace("【", "[").replace("】", "]")
    if any("０" <= c <= "９" for c in answer):
        answer = answer.translate(_FULLWIDTH_DIGITS)
    return answer


def make_synthesize_node(
    router: LLMRouter,
    settings: Settings,
    evidence_selector: EvidenceSelector | None = None,
) -> NodeFn:
    async def synthesize_node(state: OrchestrationState) -> dict:
        plan = state["plan"]
        if plan is None:
            return {}
        evidence = state["evidence"]
        warnings = list(state["warnings"])

        if not evidence:
            answer = (
                "No supporting evidence was retrieved for this question. "
                "I can't produce a cited answer without evidence to draw on."
            )
            return {"answer": answer, "warnings": warnings}

        # Evidence selection: pick minimal high-coverage subset for LLM context
        evidence_for_llm = evidence
        if evidence_selector:
            evidence_for_llm = evidence_selector.select(evidence)

        contradiction_signals = state.get("contradiction_signals") or []
        messages = build_synthesis_messages(
            plan, evidence_for_llm, contradiction_signals=contradiction_signals
        )
        try:
            response = await router.complete(
                messages,
                temperature=0.2,
                timeout=settings.orchestration_llm_timeout,
                call_type="synthesis",
                request_id=state["request_id"],
                query=state["query"],
                tier="strong",
            )
            answer = response.content or ""
        except LLMProviderError as exc:
            logger.warning("orchestration_synthesis_failed", error=str(exc))
            warnings.append(f"synthesis_fallback: {exc}")
            answer = ""

        if not answer.strip():
            # Degrade to a clean, still-evidence-grounded statement (Phase 07e):
            # prefer a truthful "could not synthesize" + the grounded evidence
            # with citations, NOT a naked raw-evidence dump (07c §15.1). The
            # citation markers [[i]] still map to the evidence list below, and
            # _derive_outcome labels this ANSWERED_DEGRADED.
            #
            # Phase 43: Safe synthesis fallback.
            # 1. Filter evidence by query relevance before presenting
            # 2. Only include relevant conflict signals
            # 3. Say "insufficient evidence" when nothing is relevant
            # 4. Never dump unrelated evidence as if it supports the query

            query_text = state.get("query", "")
            relevant_evidence, excluded_evidence = _filter_evidence_by_relevance(
                evidence, query_text, relevance_threshold=0.12,
            )

            # Also filter contradiction signals by relevance to query
            relevant_conflicts = contradiction_signals
            if contradiction_signals and query_text:
                relevant_conflicts = filter_contradictions_by_query(
                    contradiction_signals, evidence, query_text,
                )

            total_evidence = len(evidence)
            relevant_count = len(relevant_evidence)
            excluded_count = len(excluded_evidence)

            if relevant_count == 0:
                # No relevant evidence at all — honest "insufficient" response
                answer = (
                    "Synthesis unavailable and the retrieved evidence does not "
                    "contain sufficient relevant information to support this "
                    f"claim.\n\n"
                    f"Evidence status: Insufficient\n"
                    f"Retrieved: {total_evidence} items\n"
                    f"Relevant: 0 / {total_evidence}\n\n"
                    "The retrieved sources did not provide enough directly "
                    "relevant information. No conclusion was generated."
                )
                warnings.append("synthesis_fallback_insufficient_evidence")
            else:
                # Some relevant evidence exists — present it clearly labeled
                top = relevant_evidence[: min(5, len(relevant_evidence))]
                bullets = "\n".join(
                    f"- {r.text.strip()[:300]} [{idx+1}]"
                    for idx, r in enumerate(top)
                )

                excluded_note = ""
                if excluded_count > 0:
                    excluded_note = (
                        f"\n\n{excluded_count} retrieved items excluded as "
                        "insufficiently relevant to this query."
                    )

                conflict_note = ""
                if relevant_conflicts:
                    conflict_lines = []
                    for sig in relevant_conflicts:
                        desc = sig.get("description", "Unknown conflict")
                        conflict_lines.append(f"  - {desc}")
                    conflict_note = (
                        "\n\nConflicts detected (not fully resolved):\n"
                        + "\n".join(conflict_lines)
                    )

                answer = (
                    "Synthesis unavailable — this is unsynthesized evidence, "
                    "not a final answer.\n\n"
                    f"Evidence status: {relevant_count} relevant / "
                    f"{total_evidence} retrieved\n\n"
                    f"Relevant evidence:\n{bullets}"
                    f"{excluded_note}"
                    f"{conflict_note}"
                )
                warnings.append("synthesis_degraded_to_evidence_summary")

        # Phase 25: deterministic claim grounding check
        grounding_warnings = check_claim_grounding(answer, len(evidence_for_llm))
        warnings.extend(grounding_warnings)

        return {"answer": answer, "warnings": warnings}

    return synthesize_node


def extract_cited_indices(answer: str, evidence_count: int) -> list[int]:
    """Parse bracket citation markers like `[2]` / `【2】` out of the answer.

    Full-width markers and digits are normalized first (HARDEN-07d.3). Returns
    1-based indices that are within range, in first-seen order, deduplicated.
    Invalid/out-of-range/malformed markers are dropped (never presented as
    valid), which is what the graph's explicit top-evidence fallback reacts to.
    """
    normalized = _normalize_citation_markers(answer)
    seen: list[int] = []
    for match in _CITATION_MARKER_RE.finditer(normalized):
        idx = int(match.group(1))
        if 1 <= idx <= evidence_count and idx not in seen:
            seen.append(idx)
    return seen


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def check_claim_grounding(
    answer: str,
    evidence_count: int,
) -> list[str]:
    """Deterministic post-synthesis check: verify each answer sentence has a citation.

    Returns a list of warning strings for sentences that lack any bracket citation.
    Empty list means all sentences are grounded. This is a lightweight check — it
    does not verify semantic alignment, only structural citation presence.
    """
    if not answer.strip() or evidence_count == 0:
        return []

    normalized = _normalize_citation_markers(answer)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    warnings = []
    for sentence in sentences:
        has_citation = bool(_CITATION_MARKER_RE.search(sentence))
        if not has_citation:
            preview = sentence[:120] + ("..." if len(sentence) > 120 else "")
            warnings.append(f"unsupported_claim: {preview}")
    return warnings
