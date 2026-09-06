"""Deterministic Evidence Verification (Phase 22).

Lightweight, LLM-free verification of evidence coverage and contradictions.
Operates on existing retrieved EvidenceRef objects and QueryPlan structures.

This module provides:
1. EvidenceNeedCoverageVerifier — checks if retrieved chunks actually cover each need
2. TextContradictionDetector — detects contradictions in retrieved text
3. MultiHopChainVerifier — verifies hop-by-hop chain completeness
4. EvidenceConfidenceScorer — categorical confidence from existing signals
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.evidence.models import EvidenceRef
from app.logging_config import get_logger
from app.retrieval.planner import EvidenceNeed, ClaimType, QueryPlan
from app.verification.models import ContradictionDetail, ContradictionType

logger = get_logger("argus.verification.deterministic")


# ---------------------------------------------------------------------------
# Evidence Coverage Verification
# ---------------------------------------------------------------------------

class CoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    CONTRADICTED = "contradicted"
    NO_EVIDENCE_EXISTS = "no_evidence_exists"


@dataclass
class NeedVerificationResult:
    need_id: str
    need_topic: str
    status: CoverageStatus
    supporting_chunks: list[str] = field(default_factory=list)
    contradicting_chunks: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    reasoning: str = ""


class EvidenceNeedCoverageVerifier:
    """Verifies whether retrieved evidence actually covers each EvidenceNeed."""

    def verify_need(
        self,
        need: EvidenceNeed,
        selected_chunks: list[EvidenceRef],
        all_retrieved_chunks: list[EvidenceRef] | None = None,
    ) -> NeedVerificationResult:
        """Check if selected chunks cover this specific evidence need."""
        if not selected_chunks:
            return NeedVerificationResult(
                need_id=need.id,
                need_topic=need.topic,
                status=CoverageStatus.NO_EVIDENCE_EXISTS,
                reasoning="No evidence was retrieved for this need",
            )

        topic_lower = need.topic.lower()
        entities_lower = [e.lower() for e in need.entities if len(e) > 2]

        supporting = []
        contradicting = []
        partial_matches = []

        for ref in selected_chunks:
            text_lower = ref.text.lower()

            # Check topic match
            topic_present = topic_lower in text_lower
            entity_present = any(e in text_lower for e in entities_lower)

            if need.claim_type == ClaimType.PRIMARY:
                # PRIMARY: needs topic + entity presence
                if topic_present and entity_present:
                    supporting.append(str(ref.chunk_id))
                elif topic_present or entity_present:
                    partial_matches.append(str(ref.chunk_id))

            elif need.claim_type == ClaimType.CONTRADICTORY:
                # CONTRADICTORY: looking for conflict indicators
                conflict_indicators = ["conflict", "disagree", "however", "although",
                                       "contrary", "different", "revised", "updated"]
                if any(ind in text_lower for ind in conflict_indicators):
                    supporting.append(str(ref.chunk_id))
                elif topic_present:
                    partial_matches.append(str(ref.chunk_id))

            elif need.claim_type == ClaimType.SUPPORTING:
                # SUPPORTING: topic or entity mention is sufficient
                if topic_present or entity_present:
                    supporting.append(str(ref.chunk_id))

            elif need.claim_type == ClaimType.CAVEAT:
                # CAVEAT: needs disclaimer/limitation patterns
                caveat_patterns = ["note", "however", "limitation", "disclaimer",
                                   "caveat", "except", "exclusion", "not include"]
                if any(p in text_lower for p in caveat_patterns):
                    supporting.append(str(ref.chunk_id))
                elif topic_present:
                    partial_matches.append(str(ref.chunk_id))

            else:
                # CONTEXTUAL: topic mention is sufficient
                if topic_present:
                    supporting.append(str(ref.chunk_id))

        # Determine status
        if supporting:
            status = CoverageStatus.COVERED
            score = 1.0
            reasoning = f"{len(supporting)} chunks directly support this need"
        elif partial_matches:
            status = CoverageStatus.PARTIALLY_COVERED
            score = 0.5
            reasoning = f"{len(partial_matches)} chunks partially match (topic or entity present)"
        else:
            status = CoverageStatus.NOT_COVERED
            score = 0.0
            reasoning = "No retrieved chunks match this need's topic or entities"

        return NeedVerificationResult(
            need_id=need.id,
            need_topic=need.topic,
            status=status,
            supporting_chunks=supporting,
            contradicting_chunks=contradicting,
            coverage_score=score,
            reasoning=reasoning,
        )

    def verify_plan(
        self,
        plan: QueryPlan,
        selected_chunks: list[EvidenceRef],
    ) -> list[NeedVerificationResult]:
        """Verify all evidence needs in a plan."""
        if not plan.is_planned or not plan.evidence_needs:
            return []

        return [
            self.verify_need(need, selected_chunks)
            for need in plan.evidence_needs
        ]


# ---------------------------------------------------------------------------
# Text Contradiction Detection
# ---------------------------------------------------------------------------

@dataclass
class TextContradiction:
    contradiction_type: str
    description: str
    chunk_a_id: str
    chunk_b_id: str
    source_a: str
    source_b: str
    value_a: str
    value_b: str
    entity_context: str
    severity: float


class TextContradictionDetector:
    """Detects contradictions in retrieved text without LLM calls."""

    # Number extraction patterns
    NUMERIC_PATTERNS = [
        # Revenue/profit with units
        r'\$?([\d,]+\.?\d*)\s*(billion|million|thousand|B|M|K)',
        # Percentages
        r'([\d,]+\.?\d*)\s*%',
        # Standalone large numbers
        r'\b([\d,]{4,})\b',
        # Years
        r'\b(20[0-2]\d)\b',
    ]

    # Entity-attribute patterns (order matters: more specific first)
    ATTRIBUTE_PATTERNS = [
        # Revenue: "$X billion in revenue" / "revenue of $X billion" / "revenue was $X"
        (r'(?:revenue|sales|income)\s*(?:of|was|is|were|totaled?|reached|amounted?)\s*\$?([\d,]+\.?\d*)\s*(billion|million|thousand|%)?',
         "financial"),
        (r'\$?([\d,]+\.?\d*)\s*(billion|million|thousand)\s+in\s+(revenue|sales|income)',
         "financial"),
        # Employees: "X employees" / "employee count of X" / "workforce of X"
        (r'(?:employee|worker|staff|headcount)(?:s)?\s*(?:of|was|is|were|count\s+of)\s*([\d,]+)',
         "workforce"),
        (r'([\d,]+)\s+(?:employees?|workers?|staff)',
         "workforce"),
        # Utilization: "utilization of X%" / "X% utilization" / "utilization was X%"
        (r'(?:utilization|capacity|rate)\s*(?:of|was|is|were|at|reached)\s*([\d,]+\.?\d*)\s*(%|percent)?',
         "operational"),
        (r'([\d,]+\.?\d*)\s*%\s+(?:utilization|capacity)',
         "operational"),
        # Output: "output of X" / "produced X units"
        (r'(?:output|production|produced)\s*(?:of|was|is|were|at|reached)?\s*([\d,]+\.?\d*)\s*(million|thousand|units)?',
         "operational"),
        # Founded: "founded in YYYY"
        (r'(?:founded|established|incorporated)\s+(?:in|on)?\s*(\d{4})',
         "founding_date"),
        # Generic percentage: "X%" — only compare within same document context
        # (Removed from general comparison to avoid false positives)
    ]

    def detect_contradictions(
        self,
        chunks: list[EvidenceRef],
        min_severity: float = 0.3,
    ) -> list[TextContradiction]:
        """Detect potential contradictions between chunks."""
        if len(chunks) < 2:
            return []

        # Extract claims from each chunk
        chunk_claims = []
        for ref in chunks:
            claims = self._extract_claims(ref)
            if claims:
                chunk_claims.append({
                    "chunk_id": str(ref.chunk_id),
                    "doc_id": str(ref.document_id),
                    "source": ref.source_path,
                    "text": ref.text,
                    "claims": claims,
                })

        contradictions = []

        # Pairwise comparison
        for i, ca in enumerate(chunk_claims):
            for cb in chunk_claims[i+1:]:
                # Skip same-document comparisons (different docs can conflict)
                if ca["doc_id"] == cb["doc_id"]:
                    continue

                for claim_a in ca["claims"]:
                    for claim_b in cb["claims"]:
                        contra = self._compare_claims(claim_a, claim_b, ca, cb)
                        if contra and contra.severity >= min_severity:
                            contradictions.append(contra)

        return contradictions

    def _extract_claims(self, ref: EvidenceRef) -> list[dict]:
        """Extract numerical/factual claims from a chunk."""
        claims = []
        text = ref.text

        for pat, category in self.ATTRIBUTE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                groups = m.groups()
                # Pattern-dependent extraction:
                # Patterns starting with (category_words) have groups: (category, value, unit)
                # Patterns starting with \$?([\d...]) have groups: (value, unit, ...)
                # The "percentage" pattern has groups: (value)
                first_group = groups[0] if groups else ""

                # Detect if first group is a number or a category word
                is_number = bool(re.match(r'^[\d,]+\.?\d*$', first_group)) if first_group else False

                if is_number:
                    value = first_group
                    unit = groups[1] if len(groups) > 1 else ""
                else:
                    # First group is category text (e.g., "revenue"), value is second group
                    value = groups[1] if len(groups) > 1 else ""
                    unit = groups[2] if len(groups) > 2 else ""

                claims.append({
                    "category": category,
                    "attribute": first_group if not is_number else category,
                    "value": value,
                    "unit": unit or "",
                    "text": m.group(0),
                    "span": (m.start(), m.end()),
                })

        return claims

    def _compare_claims(
        self,
        claim_a: dict,
        claim_b: dict,
        chunk_a: dict,
        chunk_b: dict,
    ) -> TextContradiction | None:
        """Compare two claims for contradiction.

        Only flags contradictions when:
        1. Same category
        2. Different values
        3. Compatible units (or both unitless)
        """
        # Same attribute category required
        if claim_a["category"] != claim_b["category"]:
            return None

        # Same value -> not a contradiction
        if claim_a["value"] == claim_b["value"]:
            return None

        # Both must have a non-empty value
        if not claim_a["value"] or not claim_b["value"]:
            return None

        # Unit-aware check: skip if units are incompatible
        unit_a = (claim_a.get("unit") or "").lower().strip()
        unit_b = (claim_b.get("unit") or "").lower().strip()

        # Define incompatible unit groups
        financial_units = {"billion", "million", "thousand", "$", "k", "m", "b"}
        percentage_units = {"%", "percent"}
        count_units = {"", "employees", "workers", "staff"}  # unitless = count

        def unit_group(u: str) -> str:
            if u in percentage_units:
                return "percentage"
            if u in financial_units:
                return "financial"
            return "count"

        group_a = unit_group(unit_a)
        group_b = unit_group(unit_b)

        if group_a != group_b:
            return None  # Incompatible units, not a contradiction

        # Same category + same unit group + different values -> contradiction
        return TextContradiction(
            contradiction_type="numerical_conflict",
            description=f"Different {claim_a['category']} values: "
                       f"{claim_a['value']} {unit_a} vs {claim_b['value']} {unit_b}",
            chunk_a_id=chunk_a["chunk_id"],
            chunk_b_id=chunk_b["chunk_id"],
            source_a=chunk_a["doc_id"],
            source_b=chunk_b["doc_id"],
            value_a=claim_a["value"],
            value_b=claim_b["value"],
            entity_context=claim_a["category"],
            severity=0.7,
        )


# ---------------------------------------------------------------------------
# Multi-Hop Chain Verification
# ---------------------------------------------------------------------------

@dataclass
class HopVerification:
    hop_index: int
    description: str
    status: CoverageStatus
    supporting_chunk_id: str | None = None
    reasoning: str = ""


@dataclass
class ChainVerificationResult:
    chain_status: CoverageStatus
    hops: list[HopVerification]
    weakest_hop: int = -1
    reasoning: str = ""


class MultiHopChainVerifier:
    """Verifies multi-hop reasoning chains hop-by-hop."""

    def verify_chain(
        self,
        plan: QueryPlan,
        selected_chunks: list[EvidenceRef],
    ) -> ChainVerificationResult:
        """Verify each hop in a multi-hop chain."""
        if not plan.is_planned or not plan.evidence_needs:
            return ChainVerificationResult(
                chain_status=CoverageStatus.NOT_COVERED,
                hops=[],
                reasoning="No planned evidence needs to verify",
            )

        hops = []
        for i, need in enumerate(plan.evidence_needs):
            topic_lower = need.topic.lower()
            entities_lower = [e.lower() for e in need.entities if len(e) > 2]

            # Check if any chunk covers this hop
            supporting_chunk = None
            for ref in selected_chunks:
                text_lower = ref.text.lower()
                if (topic_lower in text_lower or
                    any(e in text_lower for e in entities_lower)):
                    supporting_chunk = str(ref.chunk_id)
                    break

            if supporting_chunk:
                hops.append(HopVerification(
                    hop_index=i,
                    description=need.original_need or need.topic,
                    status=CoverageStatus.COVERED,
                    supporting_chunk_id=supporting_chunk,
                    reasoning=f"Hop {i+1} covered by chunk {supporting_chunk[:8]}",
                ))
            else:
                hops.append(HopVerification(
                    hop_index=i,
                    description=need.original_need or need.topic,
                    status=CoverageStatus.NOT_COVERED,
                    reasoning=f"Hop {i+1} has no supporting evidence",
                ))

        # Find weakest hop
        weakest = -1
        for i, hop in enumerate(hops):
            if hop.status == CoverageStatus.NOT_COVERED:
                weakest = i
                break

        # Overall chain status
        covered_count = sum(1 for h in hops if h.status == CoverageStatus.COVERED)
        total = len(hops)

        if covered_count == total:
            chain_status = CoverageStatus.COVERED
            reasoning = f"All {total} hops covered"
        elif covered_count > 0:
            chain_status = CoverageStatus.PARTIALLY_COVERED
            reasoning = f"{covered_count}/{total} hops covered, weakest: hop {weakest+1}"
        else:
            chain_status = CoverageStatus.NOT_COVERED
            reasoning = "No hops covered"

        return ChainVerificationResult(
            chain_status=chain_status,
            hops=hops,
            weakest_hop=weakest,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Evidence Confidence Scorer (Categorical)
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CONFLICTED = "conflicted"


@dataclass
class EvidenceConfidence:
    level: ConfidenceLevel
    coverage_score: float
    source_agreement: float
    contradiction_count: int
    chain_completeness: float
    reasoning: str


class EvidenceConfidenceScorer:
    """Produces categorical confidence from existing verification signals."""

    def score(
        self,
        need_results: list[NeedVerificationResult],
        contradictions: list[TextContradiction],
        chain_result: ChainVerificationResult | None = None,
    ) -> EvidenceConfidence:
        """Compute categorical confidence from verification results."""
        if not need_results:
            return EvidenceConfidence(
                level=ConfidenceLevel.LOW,
                coverage_score=0.0,
                source_agreement=1.0,
                contradiction_count=0,
                chain_completeness=0.0,
                reasoning="No evidence needs to evaluate",
            )

        # Coverage score
        covered = sum(1 for r in need_results if r.status == CoverageStatus.COVERED)
        total = len(need_results)
        coverage_score = covered / total if total > 0 else 0.0

        # Source agreement (inverse of contradiction count)
        source_agreement = max(0.0, 1.0 - len(contradictions) * 0.3)

        # Chain completeness
        chain_completeness = 0.0
        if chain_result and chain_result.hops:
            chain_completeness = sum(
                1 for h in chain_result.hops if h.status == CoverageStatus.COVERED
            ) / len(chain_result.hops)

        # Determine level
        if len(contradictions) > 0:
            level = ConfidenceLevel.CONFLICTED
            reasoning = f"{len(contradictions)} contradictions detected"
        elif coverage_score >= 0.8 and source_agreement >= 0.7:
            level = ConfidenceLevel.HIGH
            reasoning = f"Strong coverage ({coverage_score:.0%}) with source agreement"
        elif coverage_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
            reasoning = f"Moderate coverage ({coverage_score:.0%})"
        else:
            level = ConfidenceLevel.LOW
            reasoning = f"Low coverage ({coverage_score:.0%})"

        return EvidenceConfidence(
            level=level,
            coverage_score=coverage_score,
            source_agreement=source_agreement,
            contradiction_count=len(contradictions),
            chain_completeness=chain_completeness,
            reasoning=reasoning,
        )
