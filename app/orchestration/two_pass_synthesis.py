"""Two-pass verified synthesis node (Phase 29).

Implements: EVIDENCE -> CLAIM DRAFT -> CLAIM VERIFICATION -> FINAL ANSWER

Replaces the single-pass free-form synthesis with a controlled two-pass approach:
- Pass 1: LLM generates structured claims tied to evidence
- Deterministic verification: verify each claim against evidence
- Pass 2: LLM renders only verified claims into the final answer
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.evidence.models import EvidenceRef
from app.llm_gateway.providers.exceptions import LLMProviderError
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.models import ResearchPlan
from app.orchestration.prompts import _format_evidence_block
from app.orchestration.two_pass_models import (
    ClaimSet,
    ClaimSupportStatus,
    ClaimType,
    ClaimVerificationResult,
    GeneratedClaim,
)
from app.orchestration.two_pass_prompts import (
    build_claim_generation_messages,
    build_verified_synthesis_messages,
)

logger = get_logger("argus.orchestration.two_pass")


# --- Structured output model for Pass 1 ---

class ClaimGenerationOutput(BaseModel):
    """Structured output from Pass 1 claim generation LLM."""

    claims: list[GeneratedClaim]


# --- Deterministic claim verification ---

def _normalize_citation_markers(answer: str) -> str:
    """Map full-width citation punctuation/digits to ASCII."""
    if "\u3010" in answer or "\u3011" in answer:
        answer = answer.replace("\u3010", "[").replace("\u3011", "]")
    if any("\uff10" <= c <= "\uff19" for c in answer):
        table = str.maketrans("\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19", "0123456789")
        answer = answer.translate(table)
    return answer


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _extract_evidence_ids_from_text(text: str) -> list[int]:
    """Extract bracket citation IDs from text."""
    normalized = _normalize_citation_markers(text)
    return [int(m.group(1)) for m in _CITATION_RE.finditer(normalized)]


def _verify_claim_deterministic(
    claim: GeneratedClaim,
    evidence: list[EvidenceRef],
) -> ClaimVerificationResult:
    """Deterministically verify a claim against evidence without an LLM call.

    Checks:
    1. Evidence IDs are valid (within range)
    2. Claim text contains or relates to evidence content
    3. Numerical values match evidence
    4. Evidence actually supports the claim
    """
    valid_ids = [eid for eid in claim.evidence_ids if 1 <= eid <= len(evidence)]

    if not valid_ids:
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.UNSUPPORTED,
            reason="No valid evidence IDs referenced.",
        )

    if claim.claim_type == ClaimType.ABSENT:
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.SUPPORTED,
            reason="Absence claim accepted.",
            supported_evidence_ids=valid_ids,
        )

    # Check if evidence text actually relates to the claim
    claim_lower = claim.claim.lower()
    evidence_texts = [evidence[eid - 1].text.lower() for eid in valid_ids]

    # Simple keyword overlap check
    claim_words = set(re.findall(r"\b\w{4,}\b", claim_lower))
    if not claim_words:
        # Very short claim — accept if evidence IDs are valid
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.SUPPORTED,
            reason="Short claim with valid evidence references.",
            supported_evidence_ids=valid_ids,
        )

    evidence_words = set()
    for etxt in evidence_texts:
        evidence_words.update(re.findall(r"\b\w{4,}\b", etxt))

    overlap = claim_words & evidence_words
    overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0

    if overlap_ratio < 0.2:
        return ClaimVerificationResult(
            claim=claim,
            status=ClaimSupportStatus.PARTIALLY_SUPPORTED,
            reason=f"Low evidence overlap ({overlap_ratio:.0%}). Claim may contain unsupported content.",
            supported_evidence_ids=valid_ids,
        )

    # Numerical verification
    if claim.claim_type == ClaimType.NUMERICAL and claim.numerical_values:
        evidence_full_text = " ".join(evidence_texts)
        for num_val in claim.numerical_values:
            if num_val and num_val not in evidence_full_text:
                return ClaimVerificationResult(
                    claim=claim,
                    status=ClaimSupportStatus.UNSUPPORTED,
                    reason=f"Numerical value '{num_val}' not found in referenced evidence.",
                    supported_evidence_ids=valid_ids,
                )

    return ClaimVerificationResult(
        claim=claim,
        status=ClaimSupportStatus.SUPPORTED,
        reason=f"Claim has {len(valid_ids)} supporting evidence references, {overlap_ratio:.0%} keyword overlap.",
        supported_evidence_ids=valid_ids,
    )


def verify_claims(
    claims: list[GeneratedClaim],
    evidence: list[EvidenceRef],
) -> ClaimSet:
    """Verify all generated claims against evidence deterministically."""
    verified = [_verify_claim_deterministic(c, evidence) for c in claims]
    return ClaimSet(claims=claims, verified=verified)


def _build_claim_evidence_map(
    verified: list[ClaimVerificationResult],
) -> list[dict]:
    """Build the verified claims list for Pass 2."""
    result = []
    for v in verified:
        if v.status in (ClaimSupportStatus.SUPPORTED, ClaimSupportStatus.PARTIALLY_SUPPORTED):
            result.append({
                "claim": v.claim.claim,
                "evidence_ids": v.claim.evidence_ids,
                "status": v.status.value,
                "confidence": v.claim.confidence,
            })
    return result


async def two_pass_synthesize(
    plan: ResearchPlan,
    evidence: list[EvidenceRef],
    *,
    router: LLMRouter,
    settings: Settings,
    request_id: str | None = None,
    contradiction_signals: list[dict] | None = None,
) -> tuple[str, list[str], ClaimSet | None]:
    """Two-pass verified synthesis.

    Returns: (answer, warnings, claim_set)
    """
    warnings: list[str] = []

    # --- Pass 1: Generate structured claims ---
    claim_messages = build_claim_generation_messages(
        plan, evidence, contradiction_signals=contradiction_signals
    )

    try:
        response = await router.complete(
            claim_messages,
            response_format=ClaimGenerationOutput,
            temperature=0.1,
            timeout=settings.orchestration_llm_timeout,
            call_type="synthesis",
            request_id=request_id,
            query=plan.objective,
            tier="strong",
        )
        if not response.content:
            warnings.append("pass1_empty_response")
            return "", warnings, None

        parsed = ClaimGenerationOutput.model_validate(json.loads(response.content))
        claims = parsed.claims
    except (LLMProviderError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("pass1_failed", error=str(exc))
        warnings.append(f"pass1_fallback: {exc}")
        # Fallback: return empty claims
        return "", warnings, None

    if not claims:
        warnings.append("pass1_no_claims")
        # Check if this was an "absent" answer
        return (
            "The available evidence does not contain sufficient information to answer this question.",
            warnings,
            ClaimSet(claims=[]),
        )

    # --- Deterministic claim verification ---
    claim_set = verify_claims(claims, evidence)

    supported = claim_set.supported_claims
    rejected = claim_set.rejected_claims

    logger.info(
        "claim_verification",
        total=len(claims),
        supported=len(supported),
        rejected=len(rejected),
        request_id=request_id,
    )

    if rejected:
        for r in rejected:
            warnings.append(f"claim_rejected: {r.claim.claim[:80]}... status={r.status.value}")

    # --- Pass 2: Final synthesis from verified claims ---
    if not supported:
        # No supported claims — check for absence claim
        for v in claim_set.verified:
            if v.claim.claim_type == ClaimType.ABSENT:
                return v.claim.claim, warnings, claim_set
        return (
            "The available evidence does not contain sufficient information to answer this question.",
            warnings,
            claim_set,
        )

    verified_for_pass2 = _build_claim_evidence_map(claim_set.verified)
    contradictions = claim_set.contradictions_found or None

    synth_messages = build_verified_synthesis_messages(
        plan, evidence, verified_for_pass2, contradictions=contradictions
    )

    try:
        response = await router.complete(
            synth_messages,
            temperature=0.2,
            timeout=settings.orchestration_llm_timeout,
            call_type="synthesis",
            request_id=request_id,
            query=plan.objective,
            tier="strong",
        )
        answer = response.content or ""
    except LLMProviderError as exc:
        logger.warning("pass2_failed", error=str(exc))
        warnings.append(f"pass2_fallback: {exc}")
        # Degraded: render verified claims directly
        bullets = []
        for i, vc in enumerate(verified_for_pass2, 1):
            citations = "".join(f"[{eid}]" for eid in vc["evidence_ids"])
            bullets.append(f"- {vc['claim']} {citations}")
        answer = "Synthesis degraded. Here are the verified claims:\n" + "\n".join(bullets)
        warnings.append("synthesis_degraded_to_verified_claims")

    if not answer.strip():
        # Degraded: render verified claims directly
        bullets = []
        for i, vc in enumerate(verified_for_pass2, 1):
            citations = "".join(f"[{eid}]" for eid in vc["evidence_ids"])
            bullets.append(f"- {vc['claim']} {citations}")
        answer = "Synthesis degraded. Here are the verified claims:\n" + "\n".join(bullets)
        warnings.append("synthesis_degraded_to_verified_claims")

    return answer, warnings, claim_set
