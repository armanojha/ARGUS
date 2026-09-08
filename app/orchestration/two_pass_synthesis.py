"""Two-pass verified synthesis node (Phase 29).

Implements: EVIDENCE -> CLAIM DRAFT -> CLAIM VERIFICATION -> FINAL ANSWER

Replaces the single-pass free-form synthesis with a controlled two-pass approach:
- Pass 1: LLM generates structured claims tied to evidence
- Deterministic verification: verify each claim against evidence
- Pass 2: LLM renders only verified claims into the final answer
"""

from __future__ import annotations

import dataclasses
import json
import re
import time

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.evidence.models import EvidenceRef
from app.llm_gateway.providers.exceptions import LLMProviderError
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.models import ResearchPlan
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
) -> tuple[str, list[str], ClaimSet | None, SynthesisMetrics]:
    """Two-pass verified synthesis.

    Returns: (answer, warnings, claim_set, metrics)
    """
    warnings: list[str] = []
    metrics = SynthesisMetrics()
    t_total = time.time()

    # --- Pass 1: Generate structured claims ---
    claim_messages = build_claim_generation_messages(
        plan, evidence, contradiction_signals=contradiction_signals
    )

    t_pass1 = time.time()
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
        metrics.pass1_latency_ms = (time.time() - t_pass1) * 1000
        if hasattr(response, "usage") and response.usage:
            metrics.pass1_tokens_in = getattr(response.usage, "prompt_tokens", 0) or 0
            metrics.pass1_tokens_out = getattr(response.usage, "completion_tokens", 0) or 0
        if not response.content:
            warnings.append("pass1_empty_response")
            metrics.total_latency_ms = (time.time() - t_total) * 1000
            return "", warnings, None, metrics

        parsed = ClaimGenerationOutput.model_validate(json.loads(response.content))
        claims = parsed.claims
    except (LLMProviderError, json.JSONDecodeError, ValidationError) as exc:
        metrics.pass1_latency_ms = (time.time() - t_pass1) * 1000
        logger.warning("pass1_failed", error=str(exc))
        warnings.append(f"pass1_fallback: {exc}")
        metrics.total_latency_ms = (time.time() - t_total) * 1000
        return "", warnings, None, metrics

    metrics.pass1_claims_generated = len(claims)

    if not claims:
        warnings.append("pass1_no_claims")
        metrics.pass2_method = "absent"
        metrics.total_latency_ms = (time.time() - t_total) * 1000
        return (
            "The available evidence does not contain sufficient information to answer this question.",
            warnings,
            ClaimSet(claims=[]),
            metrics,
        )

    # --- Deterministic claim verification ---
    t_verify = time.time()
    claim_set = verify_claims(claims, evidence)
    metrics.verification_latency_ms = (time.time() - t_verify) * 1000

    supported = claim_set.supported_claims
    rejected = claim_set.rejected_claims
    metrics.verified_claims_count = len(supported)
    metrics.rejected_claims_count = len(rejected)

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
        for v in claim_set.verified:
            if v.claim.claim_type == ClaimType.ABSENT:
                metrics.pass2_method = "absent"
                metrics.total_latency_ms = (time.time() - t_total) * 1000
                return v.claim.claim, warnings, claim_set, metrics
        metrics.pass2_method = "no_support"
        metrics.total_latency_ms = (time.time() - t_total) * 1000
        return (
            "The available evidence does not contain sufficient information to answer this question.",
            warnings,
            claim_set,
            metrics,
        )

    verified_for_pass2 = _build_claim_evidence_map(claim_set.verified)
    metrics.pass2_claims_rendered = len(verified_for_pass2)
    contradictions = claim_set.contradictions_found or None

    # Detect pattern for routing
    pattern = _detect_question_pattern(plan.objective)
    metrics.pattern = pattern

    # Check early exit
    if _should_early_exit(claim_set, pattern):
        t_pass2 = time.time()
        answer = _render_verified_claims(claim_set.verified, plan.objective)
        metrics.pass2_latency_ms = (time.time() - t_pass2) * 1000
        metrics.pass2_method = f"deterministic_{pattern}"
        metrics.citations_in_answer = len(_extract_evidence_ids_from_text(answer))
        metrics.total_latency_ms = (time.time() - t_total) * 1000
        return answer, warnings, claim_set, metrics

    # Full LLM Pass 2
    synth_messages = build_verified_synthesis_messages(
        plan, evidence, verified_for_pass2, contradictions=contradictions
    )

    t_pass2 = time.time()
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
        metrics.pass2_latency_ms = (time.time() - t_pass2) * 1000
        if hasattr(response, "usage") and response.usage:
            metrics.pass2_tokens_in = getattr(response.usage, "prompt_tokens", 0) or 0
            metrics.pass2_tokens_out = getattr(response.usage, "completion_tokens", 0) or 0
    except LLMProviderError as exc:
        metrics.pass2_latency_ms = (time.time() - t_pass2) * 1000
        logger.warning("pass2_failed", error=str(exc))
        warnings.append(f"pass2_fallback: {exc}")
        bullets = []
        for i, vc in enumerate(verified_for_pass2, 1):
            citations = "".join(f"[{eid}]" for eid in vc["evidence_ids"])
            bullets.append(f"- {vc['claim']} {citations}")
        answer = "Synthesis degraded. Here are the verified claims:\n" + "\n".join(bullets)
        warnings.append("synthesis_degraded_to_verified_claims")
        metrics.pass2_method = "deterministic_fallback"
    else:
        metrics.pass2_method = "llm_restricted"

    if not answer.strip():
        bullets = []
        for i, vc in enumerate(verified_for_pass2, 1):
            citations = "".join(f"[{eid}]" for eid in vc["evidence_ids"])
            bullets.append(f"- {vc['claim']} {citations}")
        answer = "Synthesis degraded. Here are the verified claims:\n" + "\n".join(bullets)
        warnings.append("synthesis_degraded_to_verified_claims")
        metrics.pass2_method = "deterministic_fallback"

    metrics.citations_in_answer = len(_extract_evidence_ids_from_text(answer))
    metrics.total_latency_ms = (time.time() - t_total) * 1000
    return answer, warnings, claim_set, metrics


# --- Deterministic renderer (no LLM call) ---

def _render_verified_claims(
    verified: list[ClaimVerificationResult],
    objective: str,
) -> str:
    """Render verified claims into a natural answer without an LLM call.

    Groups claims by type, preserves citations, handles contradictions.
    Never introduces new facts.
    """
    supported = [v for v in verified if v.status in (ClaimSupportStatus.SUPPORTED, ClaimSupportStatus.PARTIALLY_SUPPORTED)]
    contradictions = [v for v in verified if v.status == ClaimSupportStatus.CONTRADICTED]

    if not supported and not contradictions:
        return "The available evidence does not contain sufficient information to answer this question."

    parts = []

    # Handle contradictions first
    if contradictions:
        parts.append("The evidence contains conflicting information:")
        for v in contradictions:
            citations = "".join(f"[{eid}]" for eid in v.claim.evidence_ids if 1 <= eid <= 99)
            parts.append(f"- {v.claim.claim} {citations}")
        parts.append("")

    # Group supported claims by type
    factual = [v for v in supported if v.claim.claim_type == ClaimType.FACTUAL]
    numerical = [v for v in supported if v.claim.claim_type == ClaimType.NUMERICAL]
    comparison = [v for v in supported if v.claim.claim_type == ClaimType.COMPARISON]
    relationship = [v for v in supported if v.claim.claim_type == ClaimType.RELATIONSHIP]
    absent = [v for v in supported if v.claim.claim_type == ClaimType.ABSENT]

    if absent:
        return absent[0].claim.claim

    for group in [factual, numerical, comparison, relationship]:
        for v in group:
            citations = "".join(f"[{eid}]" for eid in v.claim.evidence_ids if 1 <= eid <= 99)
            parts.append(f"{v.claim.claim} {citations}")

    if not parts:
        return "The available evidence does not contain sufficient information to answer this question."

    return " ".join(parts)


def _detect_question_pattern(query: str) -> str:
    """Simple heuristic question-pattern detection for early-exit routing."""
    q = query.lower()
    if any(w in q for w in ["where is", "what is the name", "who is", "when was", "how many employees"]):
        return "simple_lookup"
    if any(w in q for w in ["latency", "p99", "revenue growth rate", "percentage", "ratio"]):
        return "numerical"
    if any(w in q for w in ["conflict", "different sources", "disagreement", "discrepancies"]):
        return "conflict"
    if any(w in q for w in ["not contain", "not available", "no information", "undisclosed"]):
        return "absent_info"
    return "general"


def _should_early_exit(
    claim_set: ClaimSet,
    pattern: str,
    max_claims: int = 10,
) -> bool:
    """Determine if we can skip LLM Pass 2 and use deterministic rendering."""
    supported = claim_set.supported_claims
    rejected = claim_set.rejected_claims

    if rejected:
        return False

    if len(supported) > max_claims:
        return False

    if pattern in ("simple_lookup", "numerical", "absent_info"):
        return True

    return bool(pattern == "conflict" and len(supported) <= 5)


# --- Instrumented two-pass synthesis ---

@dataclasses.dataclass
class SynthesisMetrics:
    """Detailed metrics for the two-pass synthesis pipeline."""
    pass1_latency_ms: float = 0.0
    pass1_tokens_in: int = 0
    pass1_tokens_out: int = 0
    verification_latency_ms: float = 0.0
    pass2_latency_ms: float = 0.0
    pass2_tokens_in: int = 0
    pass2_tokens_out: int = 0
    total_latency_ms: float = 0.0
    pass1_claims_generated: int = 0
    verified_claims_count: int = 0
    rejected_claims_count: int = 0
    pass2_method: str = ""  # "llm" | "deterministic" | "early_exit" | "absent"
    pass2_filler_sentences: int = 0
    pass2_claims_rendered: int = 0
    citations_in_answer: int = 0
    pattern: str = ""
