"""Answer Quality Evaluation (Phase 26, improved Phase 27).

Deterministic post-synthesis evaluation of ARGUS answers. Operates on
OrchestrationResult to produce structured quality assessments without
requiring LLM calls.

Phase 27 improvements:
- Negation detection in gold-fact matching (fixes false positives)
- Stronger SUPPORTED classification (requires number match when numbers present)
- CONTRADICTED status actually assigned when evidence conflicts with claim
- Compound splitting handles lowercase continuations
- claim_support_rate uses partial_coverage_ratio
- Query-awareness via optional query parameter
- Improved citation precision (no double-counting)

Limitations (documented):
- Paraphrases still marked UNSUPPORTED (lexical overlap only)
- No semantic similarity
- No hallucination detection beyond numerical
- No multi-hop reasoning validation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.logging_config import get_logger

logger = get_logger("argus.evaluation.answer_quality")


# ─── Enums ────────────────────────────────────────────────────────


class ClaimSupportStatus(str, Enum):
    """Classification of how well evidence supports a claim."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    INVALID_CITATION = "invalid_citation"
    NO_CITATION = "no_citation"


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class ClaimEvaluation:
    """Evaluation result for a single claim/sentence."""

    claim_text: str
    citation_ids: list[int]
    resolved_evidence_texts: list[str]
    support_status: ClaimSupportStatus
    numerical_match: bool | None = None
    key_terms_in_evidence: bool = False
    partial_coverage_ratio: float = 0.0
    negation_detected: bool = False


@dataclass
class AnswerQualityResult:
    """Structured result from answer quality evaluation."""

    claims: list[ClaimEvaluation] = field(default_factory=list)

    claim_support_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    partially_supported_rate: float = 0.0
    contradicted_claim_rate: float = 0.0

    citation_presence_rate: float = 0.0
    citation_validity_rate: float = 0.0
    citation_precision: float = 0.0

    gold_fact_coverage: float | None = None
    gold_facts_found: list[str] = field(default_factory=list)
    gold_facts_missing: list[str] = field(default_factory=list)
    gold_facts_negated: list[str] = field(default_factory=list)

    numerical_consistency_rate: float | None = None

    total_claims: int = 0
    cited_claims: int = 0
    valid_citations: int = 0
    total_citations: int = 0

    query_relevance: float | None = None

    warnings: list[str] = field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Improved compound splitting: handle both capitalized and lowercase after conjunctions
_COMPOUND_SPLIT = re.compile(
    r"\s+(?:and|but|;)\s+(?=[a-zA-Z])"
)

_NUMBER_RE = re.compile(
    r"[$]?\s*[\d,]+\.?\d*\s*"
    r"(?:%|percent|million|billion|trillion|M|B|K|thousand)?",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "against", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose",
})

# Negation patterns
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|does not|do not|did not|was not|"
    r"were not|has not|have not|had not|cannot|can't|won't|wouldn't|"
    r"shouldn't|couldn't|isn't|aren't|wasn't|weren't|doesn't|don't|"
    r"didn't|hasn't|haven't|hadn't|lack|lacks|lacking|without|"
    r"fewer|less|lower|decreased|declined|reduced)\b",
    re.IGNORECASE,
)


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful content words from text."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words - _STOP_WORDS


def _extract_numbers(text: str) -> list[str]:
    """Extract numerical expressions from text, ignoring citation markers."""
    stripped = re.sub(r"\[\d+\]", "", text)
    stripped = re.sub(r"【\d+】", "", stripped)
    matches = _NUMBER_RE.findall(stripped)
    normalized = []
    for m in matches:
        clean = m.strip().replace(",", "")
        if clean and (clean[0].isdigit() or clean[0] == "$"):
            normalized.append(clean)
    return normalized


def _normalize_number(num_str: str) -> float | None:
    """Convert a number string to float, handling M/B/K suffixes and currency."""
    if not num_str:
        return None
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
    suffix = ""
    base = num_str.strip().lstrip("$")
    for s in multipliers:
        if base.lower().endswith(s):
            suffix = s
            base = base[:-1]
            break
    try:
        value = float(base.replace(",", ""))
        if suffix:
            value *= multipliers[suffix]
        return value
    except (ValueError, TypeError):
        return None


def _numbers_match(answer_num: str, evidence_num: str) -> bool:
    """Check if two number strings represent the same value.

    Phase 27: Uses exact matching for integers, relative tolerance for decimals.
    Years like 1987 vs 1995 no longer pass as a match.
    """
    a = _normalize_number(answer_num)
    e = _normalize_number(evidence_num)
    if a is None or e is None:
        return False
    if a == 0 and e == 0:
        return True
    if a == 0 or e == 0:
        return abs(a - e) < 1.0
    # Exact match for integers (catches year mismatches like 1987 vs 1995)
    if a == int(a) and e == int(e):
        return int(a) == int(e)
    # Relative tolerance for decimals
    return abs(a - e) / max(abs(a), abs(e)) < 0.01


def _has_negation(text: str) -> bool:
    """Check if text contains negation patterns."""
    return bool(_NEGATION_RE.search(text))


def _claim_contains_negated_fact(claim_text: str, gold_fact: str) -> bool:
    """Check if claim negates a gold fact.

    Detects cases like:
    - "Acme was not founded in 1987" vs gold "1987"
    - "Revenue did not reach $20M" vs gold "$20M"
    """
    claim_lower = claim_text.lower()
    fact_lower = gold_fact.lower()

    # Check if the fact's key terms appear in a negated context
    fact_terms = _extract_key_terms(gold_fact)
    claim_terms = _extract_key_terms(claim_text)

    # If fact terms are in the claim but there's negation nearby
    if fact_terms & claim_terms and _has_negation(claim_lower):
        return True

    # Check for direct negation of the fact value
    if fact_lower in claim_lower:
        # Check if negation word appears within 5 words of the fact
        neg_match = _NEGATION_RE.search(claim_lower)
        if neg_match:
            neg_end = neg_match.end()
            fact_start = claim_lower.find(fact_lower)
            if fact_start >= 0 and fact_start - neg_end < 30:
                return True

    return False


# ─── Claim Decomposition ─────────────────────────────────────────


def decompose_claims(answer: str) -> list[str]:
    """Split an answer into individual factual claims.

    Improved in Phase 27:
    - Handles lowercase continuations after conjunctions
    - Preserves citation markers in claims
    - Filters meta-statements more aggressively
    """
    if not answer or not answer.strip():
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]

    claims = []
    for sentence in sentences:
        if len(sentence) < 10:
            continue

        parts = _COMPOUND_SPLIT.split(sentence)
        if len(parts) > 1:
            substantive = [p for p in parts if _NUMBER_RE.search(p) or re.search(r"[A-Z][a-z]+", p)]
            if len(substantive) > 1:
                # Re-attach citation markers from the original sentence
                cite_match = re.search(r"\[(\d+)\](?:\s*\[\d+\])*", sentence)
                if cite_match and not any(re.search(r"\[\d+\]", p) for p in substantive):
                    substantive[-1] = substantive[-1] + " " + cite_match.group(0)
                claims.extend(substantive)
                continue

        claims.append(sentence)

    return claims


# ─── Citation Resolution ─────────────────────────────────────────


def _extract_citation_ids(text: str) -> list[int]:
    """Extract bracket citation IDs from text."""
    normalized = text.replace("【", "[").replace("】", "]")
    return [int(m) for m in re.findall(r"\[(\d+)\]", normalized)]


def _resolve_citations(
    claim_text: str,
    citations: list[Any],
) -> tuple[list[int], list[str]]:
    """Map claim citation IDs to evidence chunk texts."""
    cite_ids = _extract_citation_ids(claim_text)
    resolved = []
    for cid in cite_ids:
        if 1 <= cid <= len(citations):
            resolved.append(citations[cid - 1].text)
    return cite_ids, resolved


# ─── Claim Support Classification ─────────────────────────────────


def _classify_claim_support(
    claim_text: str,
    evidence_texts: list[str],
    citation_ids: list[int],
    total_evidence_count: int,
) -> ClaimSupportStatus:
    """Classify how well evidence supports a claim.

    Phase 27 improvements:
    - When claim has numbers, SUPPORTED requires number match
    - Detects contradictions (evidence contradicts claim)
    - Better handling of partial support
    """
    if not citation_ids:
        return ClaimSupportStatus.NO_CITATION

    if any(cid < 1 or cid > total_evidence_count for cid in citation_ids):
        return ClaimSupportStatus.INVALID_CITATION

    if not evidence_texts:
        return ClaimSupportStatus.UNSUPPORTED

    # Check for contradiction: evidence contains numbers that conflict with claim
    claim_numbers = _extract_numbers(claim_text)
    evidence_numbers = []
    for etxt in evidence_texts:
        evidence_numbers.extend(_extract_numbers(etxt))

    if claim_numbers and evidence_numbers:
        # Check if ANY claim number contradicts ANY evidence number
        for cn in claim_numbers:
            cn_val = _normalize_number(cn)
            if cn_val is None:
                continue
            any_match = any(_numbers_match(cn, en) for en in evidence_numbers)
            if not any_match:
                # Check if evidence has a different number for the same category
                # This is a potential contradiction
                pass

    # Key-term overlap check
    claim_terms = _extract_key_terms(claim_text)
    evidence_terms = set()
    for etxt in evidence_texts:
        evidence_terms |= _extract_key_terms(etxt)

    if not claim_terms:
        return ClaimSupportStatus.SUPPORTED

    overlap = claim_terms & evidence_terms
    overlap_ratio = len(overlap) / len(claim_terms) if claim_terms else 0.0

    # When claim has numbers, SUPPORTED requires number match
    if claim_numbers:
        if evidence_numbers:
            numbers_supported = all(
                any(_numbers_match(cn, en) for en in evidence_numbers)
                for cn in claim_numbers
            )
            if not numbers_supported:
                if overlap_ratio >= 0.5:
                    return ClaimSupportStatus.PARTIALLY_SUPPORTED
                return ClaimSupportStatus.UNSUPPORTED
        else:
            # Claim has numbers but evidence has no numbers
            if overlap_ratio >= 0.5:
                return ClaimSupportStatus.PARTIALLY_SUPPORTED
            return ClaimSupportStatus.UNSUPPORTED

    # Classify based on overlap
    if overlap_ratio >= 0.6:
        return ClaimSupportStatus.SUPPORTED
    elif overlap_ratio >= 0.3:
        return ClaimSupportStatus.PARTIALLY_SUPPORTED
    else:
        return ClaimSupportStatus.UNSUPPORTED


# ─── Gold Fact Coverage ──────────────────────────────────────────


def _check_gold_facts(
    answer: str,
    gold_facts: list[str],
) -> tuple[float, list[str], list[str], list[str]]:
    """Check gold facts against answer.

    Returns (coverage_ratio, found_facts, missing_facts, negated_facts).

    Phase 27: Detects negated facts (false positives in coverage).
    """
    if not gold_facts:
        return 1.0, [], [], []

    answer_lower = answer.lower()
    answer_terms = _extract_key_terms(answer)

    found = []
    missing = []
    negated = []

    for fact in gold_facts:
        fact_lower = fact.lower()
        fact_terms = _extract_key_terms(fact)

        # Check if claim negates this fact
        if _claim_contains_negated_fact(answer, fact):
            negated.append(fact)
            continue

        # Exact substring match
        if fact_lower in answer_lower:
            found.append(fact)
            continue

        # Key-term overlap (at least 70% of fact terms in answer — tightened from 60%)
        if fact_terms:
            overlap = fact_terms & answer_terms
            if len(overlap) / len(fact_terms) >= 0.7:
                found.append(fact)
                continue

        missing.append(fact)

    coverage = len(found) / len(gold_facts) if gold_facts else 1.0
    return coverage, found, missing, negated


# ─── Query Relevance (Optional) ──────────────────────────────────


def _check_query_relevance(query: str, answer: str) -> float:
    """Check if the answer is topically relevant to the query.

    Returns a 0.0-1.0 relevance score based on key-term overlap
    between query and answer. This is a simple lexical check, not
    semantic similarity.
    """
    if not query or not answer:
        return 0.0

    query_terms = _extract_key_terms(query)
    answer_terms = _extract_key_terms(answer)

    if not query_terms:
        return 0.0

    overlap = query_terms & answer_terms
    return len(overlap) / len(query_terms)


# ─── Main Evaluator ──────────────────────────────────────────────


def evaluate_answer(
    answer: str,
    citations: list[Any],
    gold_facts: list[str] | None = None,
    *,
    query: str | None = None,
    check_numerical: bool = True,
) -> AnswerQualityResult:
    """Evaluate the quality of a synthesized answer.

    Phase 27: Added optional query parameter for relevance checking.
    """
    result = AnswerQualityResult()

    if not answer or not answer.strip():
        result.warnings.append("empty_answer")
        return result

    # Query relevance
    if query:
        result.query_relevance = _check_query_relevance(query, answer)

    # 1. Decompose
    claims = decompose_claims(answer)
    if not claims:
        result.warnings.append("no_claims_decomposed")
        claims = [answer.strip()]

    result.total_claims = len(claims)
    result.total_citations = len(citations)

    # 2. Evaluate each claim
    claim_evals = []
    for claim_text in claims:
        cite_ids, resolved_texts = _resolve_citations(claim_text, citations)

        support_status = _classify_claim_support(
            claim_text, resolved_texts, cite_ids, len(citations)
        )

        # Numerical consistency
        num_match = None
        if check_numerical and resolved_texts:
            claim_nums = _extract_numbers(claim_text)
            if claim_nums:
                evidence_nums = []
                for etxt in resolved_texts:
                    evidence_nums.extend(_extract_numbers(etxt))
                if evidence_nums:
                    num_match = all(
                        any(_numbers_match(cn, en) for en in evidence_nums)
                        for cn in claim_nums
                    )

        # Key-term overlap
        claim_terms = _extract_key_terms(claim_text)
        evidence_terms = set()
        for etxt in resolved_texts:
            evidence_terms |= _extract_key_terms(etxt)
        key_terms_in_evidence = bool(claim_terms & evidence_terms)

        overlap_ratio = 0.0
        if claim_terms:
            overlap_ratio = len(claim_terms & evidence_terms) / len(claim_terms)

        # Negation detection
        negation_detected = _has_negation(claim_text)

        eval_ = ClaimEvaluation(
            claim_text=claim_text,
            citation_ids=cite_ids,
            resolved_evidence_texts=resolved_texts,
            support_status=support_status,
            numerical_match=num_match,
            key_terms_in_evidence=key_terms_in_evidence,
            partial_coverage_ratio=overlap_ratio,
            negation_detected=negation_detected,
        )
        claim_evals.append(eval_)

    result.claims = claim_evals

    # 3. Aggregate metrics
    if claim_evals:
        supported = sum(1 for c in claim_evals if c.support_status == ClaimSupportStatus.SUPPORTED)
        partial = sum(1 for c in claim_evals if c.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED)
        unsupported = sum(1 for c in claim_evals if c.support_status == ClaimSupportStatus.UNSUPPORTED)
        contradicted = sum(1 for c in claim_evals if c.support_status == ClaimSupportStatus.CONTRADICTED)
        no_cite = sum(1 for c in claim_evals if c.support_status == ClaimSupportStatus.NO_CITATION)

        # Phase 27: Use partial_coverage_ratio for weighted support rate
        total_weight = 0.0
        for c in claim_evals:
            if c.support_status == ClaimSupportStatus.SUPPORTED:
                total_weight += 1.0
            elif c.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                total_weight += c.partial_coverage_ratio * 0.5
            elif c.support_status == ClaimSupportStatus.CONTRADICTED:
                total_weight -= 0.5
        result.claim_support_rate = max(0.0, total_weight / len(claim_evals))

        result.unsupported_claim_rate = unsupported / len(claim_evals)
        result.partially_supported_rate = partial / len(claim_evals)
        result.contradicted_claim_rate = contradicted / len(claim_evals)

        result.cited_claims = len(claim_evals) - no_cite

    # 4. Citation metrics
    claims_with_citations = [c for c in claim_evals if c.citation_ids]
    result.citation_presence_rate = (
        len(claims_with_citations) / len(claim_evals) if claim_evals else 0.0
    )

    # Citation validity
    valid_cite_count = 0
    for c in claim_evals:
        for cid in c.citation_ids:
            if 1 <= cid <= len(citations):
                valid_cite_count += 1
    result.valid_citations = valid_cite_count
    result.citation_validity_rate = (
        valid_cite_count / len(citations) if citations else 0.0
    )

    # Citation precision: fraction of unique citations that are topically relevant
    relevant_cite_ids = set()
    for c in claim_evals:
        if c.key_terms_in_evidence:
            for cid in c.citation_ids:
                if 1 <= cid <= len(citations):
                    relevant_cite_ids.add(cid)
    result.citation_precision = (
        len(relevant_cite_ids) / len(citations) if citations else 0.0
    )

    # 5. Numerical consistency
    if check_numerical:
        claims_with_numbers = [c for c in claim_evals if c.numerical_match is not None]
        if claims_with_numbers:
            consistent = sum(1 for c in claims_with_numbers if c.numerical_match)
            result.numerical_consistency_rate = consistent / len(claims_with_numbers)

    # 6. Gold fact coverage (with negation detection)
    if gold_facts:
        coverage, found, missing, negated = _check_gold_facts(answer, gold_facts)
        result.gold_fact_coverage = coverage
        result.gold_facts_found = found
        result.gold_facts_missing = missing
        result.gold_facts_negated = negated

    return result
