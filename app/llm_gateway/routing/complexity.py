"""Query complexity classification for task-adaptive routing (HARDEN-06.5.2).

Deterministic, zero-LLM-cost classifier that maps a user query to a routing
tier. The tier selects which explicit model chain the router uses for a call
type, so simple lookups use fast/cheap models and hard reasoning uses strong
models — without giving the model fabric free rein over model choice (the
chains themselves are still explicit configuration).

Tiers:
* ``FAST``     — simple, low-risk, extractive (lookups, short factual).
* ``BALANCED`` — typical research question (default tier).
* ``STRONG``   — comparative/causal/abstract/long-form reasoning.

Heuristics are intentionally cheap and rule-based. They are NOT a substitute
for the orchestration layer's richer ``QuestionPattern`` classifier; this is
a self-contained approximation used at the routing seam so that adding a tier
to an LLM call never requires an extra LLM round-trip.
"""

from __future__ import annotations

import re
from enum import Enum


class ComplexityTier(str, Enum):
    """Routing tier for a query."""

    FAST = "fast"
    BALANCED = "balanced"
    STRONG = "strong"


# ── Signal patterns ──────────────────────────────────────────────────

# Comparative / causal / abstract / relationship signals → STRONG
_COMPARE_RE = re.compile(
    r"\b(compare|comparison|difference|differ|versus|vs\.?|vs\b|better|worst)\b",
    re.IGNORECASE,
)
_CASUAL_RE = re.compile(
    r"\b(cause|caused by|because of|result of|impact|effect|leads? to|"
    r"influence|affect|why|consequence)\b",
    re.IGNORECASE,
)
_ABSTRACT_RE = re.compile(
    r"\b(implicat|implication|synthesiz|synthesis|argument|theor|"
    r"what if|evaluate|assess|critique|policy|tension|trade-?off)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_RE = re.compile(
    r"\b(trace|relationship|relate|connected|between|how does|how are|"
    r"chain|pathway)\b",
    re.IGNORECASE,
)

# Temporal signals → STRONG (time-sensitive queries need careful handling)
_TEMPORAL_RE = re.compile(
    r"\b(before|after|during|since|until|between \d{4}|from \d{4}|"
    r"in \d{4}|timeline|chronolog|historical|evolution|over time)\b",
    re.IGNORECASE,
)

# Contradiction / conflict signals → STRONG (need strongest model)
_CONTRADICTION_RE = re.compile(
    r"\b(however|but|contradict|conflict|disagree|contrary|"
    r"on the other hand|despite|although|nevertheless|offset)\b",
    re.IGNORECASE,
)

# Entity density signal — many proper nouns = multi-entity reasoning
_ENTITY_DENSITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Simple lookup signals → FAST
_HOW_WHEN_RE = re.compile(
    r"\b(how (do|can|to)|when (did|was)|define|what is|who is|when did|"
    r"where is)\b",
    re.IGNORECASE,
)

# Length threshold above which a query is treated as a long-form/deep request.
_LONG_THRESHOLD = 160

# Entity count threshold — queries mentioning 3+ named entities are harder
_HIGH_ENTITY_THRESHOLD = 3


def classify_complexity(query: str) -> ComplexityTier:
    """Classify a query into a routing tier using cheap heuristics.

    Priority: strongest signals win; short/simple queries fall to FAST.
    """
    text = (query or "").strip()
    if not text:
        return ComplexityTier.BALANCED

    # Length / enumeration of many facets indicates a deep request.
    if len(text) > _LONG_THRESHOLD:
        return ComplexityTier.STRONG

    # High entity density — multi-entity reasoning needs a strong model.
    entity_count = len(_ENTITY_DENSITY_RE.findall(text))
    if entity_count >= _HIGH_ENTITY_THRESHOLD:
        return ComplexityTier.STRONG

    # Contradiction / conflict detection — needs careful verification.
    if _CONTRADICTION_RE.search(text):
        return ComplexityTier.STRONG

    # Temporal queries — time-sensitive, need careful handling.
    if _TEMPORAL_RE.search(text):
        return ComplexityTier.STRONG

    # Comparative, causal, abstract, or relationship-heavy queries need a
    # strong (higher-quality) model for sound synthesis.
    strong_signals = (
        bool(_COMPARE_RE.search(text))
        or bool(_CASUAL_RE.search(text))
        or bool(_ABSTRACT_RE.search(text))
        or bool(_RELATIONSHIP_RE.search(text))
    )
    if strong_signals:
        return ComplexityTier.STRONG

    # Short, concrete lookups (what/who/when/where/how-to) are fast.
    if _HOW_WHEN_RE.search(text) and len(text) < 60:
        return ComplexityTier.FAST

    return ComplexityTier.BALANCED


def adjust_tier_for_evidence(
    current_tier: ComplexityTier,
    evidence_scores: list[float],
    min_scores: int = 3,
) -> ComplexityTier:
    """Dynamically adjust complexity tier based on evidence quality.

    When evidence is already strong (high scores), subsequent LLM calls
    can use cheaper models since the grounding is solid. This enables
    cost savings on assessment and synthesis when retrieval succeeded early.

    Args:
        current_tier: The current routing tier.
        evidence_scores: Similarity scores of top evidence chunks.
        min_scores: Minimum number of evidence scores needed to adjust.

    Returns:
        The adjusted tier (never upgrades, only downgrades or stays same).
    """
    if current_tier == ComplexityTier.FAST:
        return current_tier  # Already at cheapest tier

    if len(evidence_scores) < min_scores:
        return current_tier  # Not enough evidence to judge

    avg_score = sum(evidence_scores[:min_scores]) / min_scores

    # Very strong evidence (avg >= 0.75) → can use FAST tier for remaining calls
    if avg_score >= 0.75 and current_tier in (ComplexityTier.STRONG, ComplexityTier.BALANCED):
        return ComplexityTier.FAST

    # Strong evidence (avg >= 0.6) → can safely use BALANCED tier
    if avg_score >= 0.6 and current_tier == ComplexityTier.STRONG:
        return ComplexityTier.BALANCED

    return current_tier