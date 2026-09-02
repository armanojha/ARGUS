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


# Language signals that push a query toward harder reasoning.
_COMPARE_RE = re.compile(
    r"\b(compare|comparison|difference|differ|versus|vs\.?|vs\b|better|worst)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
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
_HOW_WHEN_RE = re.compile(
    r"\b(how (do|can|to)|when (did|was)|define|what is|who is|when did|"
    r"where is)\b",
    re.IGNORECASE,
)
# NOTE/TODO: heuristics only — no state, no token budget concerns.

# Length threshold above which a query is treated as a long-form/deep request.
_LONG_THRESHOLD = 160


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

    # Comparative, causal, abstract, or relationship-heavy queries need a
    # strong (higher-quality) model for sound synthesis.
    strong_signals = (
        bool(_COMPARE_RE.search(text))
        or bool(_CAUSAL_RE.search(text))
        or bool(_ABSTRACT_RE.search(text))
        or bool(_RELATIONSHIP_RE.search(text))
    )
    if strong_signals:
        return ComplexityTier.STRONG

    # Short, concrete lookups (what/who/when/where/how-to) are fast.
    if _HOW_WHEN_RE.search(text) and len(text) < 60:
        return ComplexityTier.FAST

    return ComplexityTier.BALANCED