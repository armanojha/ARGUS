"""Two-pass verified synthesis prompts (Phase 29).

Pass 1: Claim generation - forces the LLM to produce structured claims tied to evidence.
Pass 2: Final synthesis - renders only verified claims into a coherent answer.
"""

from __future__ import annotations

from app.evidence.models import EvidenceRef
from app.llm_gateway.providers.models import Message, MessageRole
from app.orchestration.models import ResearchPlan
from app.orchestration.prompts import _UNTRUSTED_NOTICE, _format_evidence_block


def build_claim_generation_messages(
    plan: ResearchPlan,
    evidence: list[EvidenceRef],
    *,
    contradiction_signals: list[dict] | None = None,
) -> list[Message]:
    """Pass 1: Generate structured claims tied to evidence.

    The LLM must output JSON with a list of claims, each referencing
    specific evidence IDs. No evidence ID = invalid claim.
    """
    system = (
        "You are the claim-generation stage of a research assistant. "
        "Given the research objective and numbered evidence passages, "
        "extract every factual claim the evidence supports.\n\n"
        "RULES:\n"
        "1. EVERY claim MUST cite at least one evidence passage using its [N] number.\n"
        "2. Claims with NO evidence ID are INVALID and must not be generated.\n"
        "3. For numerical claims, include the exact numbers from evidence.\n"
        "4. For comparison claims, cite evidence for EACH side.\n"
        "5. For contradictions, create SEPARATE claims for each conflicting fact.\n"
        "6. If the evidence does NOT contain information to answer the question, "
        "generate a single claim of type 'absent' explaining this.\n"
        "7. Do NOT use external knowledge. Only extract what is IN the evidence.\n"
        "8. Each claim should be ONE factual statement, not a compound paragraph.\n\n"
        "OUTPUT FORMAT: JSON with a 'claims' array. Each claim has:\n"
        '- "claim": the factual statement\n'
        '- "evidence_ids": list of 1-based evidence indices supporting it\n'
        '- "confidence": 0.0-1.0\n'
        '- "claim_type": "factual" | "numerical" | "comparison" | "relationship" | "absent"\n'
        '- "numerical_values": list of specific numbers referenced (if any)\n\n'
        f"{_UNTRUSTED_NOTICE}"
    )

    contradiction_section = ""
    if contradiction_signals:
        items = []
        for sig in contradiction_signals:
            severity = sig.get("severity", "unknown")
            desc = sig.get("description", "")
            items.append(f"- Severity {severity}: {desc}" if desc else f"- Severity {severity}")
        contradiction_section = (
            "\n--- CONTRADICTION ALERT ---\n"
            "The evidence contains contradictions. Create SEPARATE claims for each "
            "conflicting fact, citing their respective evidence sources.\n"
            + "\n".join(items)
            + "\n--- END CONTRADICTION ALERT ---\n"
        )

    user = (
        f"Objective: {plan.objective}\n"
        f"{contradiction_section}\n"
        f"--- NUMBERED EVIDENCE ---\n"
        f"{_format_evidence_block(evidence, include_scores=True)}\n"
        f"--- END EVIDENCE ---\n\n"
        "Extract all factual claims from the evidence. Output JSON with 'claims' array."
    )

    return [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]


def build_verified_synthesis_messages(
    plan: ResearchPlan,
    evidence: list[EvidenceRef],
    verified_claims: list[dict],
    contradictions: list[dict] | None = None,
) -> list[Message]:
    """Pass 2: Synthesize final answer from VERIFIED claims only.

    CRITICAL: Pass 2 does NOT receive raw evidence. It receives only
    the verified claims with their evidence IDs. This prevents the LLM
    from bypassing verification and generating new facts from evidence.
    """
    system = (
        "You are a renderer, not a researcher.\n\n"
        "You are given a user question and a set of VERIFIED CLAIMS.\n\n"
        "You may ONLY express information contained in the VERIFIED CLAIMS.\n\n"
        "RULES:\n"
        "1. Do NOT use outside knowledge.\n"
        "2. Do NOT infer new facts.\n"
        "3. Do NOT introduce new numbers.\n"
        "4. Do NOT introduce new entities.\n"
        "5. Do NOT add explanations unless directly represented by verified claims.\n"
        "6. Every factual sentence must contain the citation IDs belonging to the claim(s) it expresses.\n"
        "7. If no verified claim supports a requested fact, explicitly state that the evidence does not establish it.\n"
        "8. Do NOT add introductions, conclusions, generic context, disclaimers, or filler.\n"
        "9. If claims conflict, present BOTH sides with their sources.\n"
        "10. For numerical answers, use the exact values from verified claims.\n"
        "11. Answer only the question asked.\n\n"
        "The evidence IDs (e.g. [1], [2]) in each claim refer to source passages "
        "used during verification. Preserve them as citations in your answer."
    )

    claims_text = ""
    for i, vc in enumerate(verified_claims, 1):
        claim_text = vc.get("claim", "")
        evidence_ids = vc.get("evidence_ids", [])
        status = vc.get("status", "supported")
        citations = "".join(f"[{eid}]" for eid in evidence_ids)
        claims_text += f"\nClaim {i} ({status}): {claim_text} {citations}"

    contradiction_section = ""
    if contradictions:
        contradiction_section = "\n--- CONFLICTING CLAIMS ---\n"
        for c in contradictions:
            contradiction_section += f"- {c.get('description', 'Conflict detected')}\n"
        contradiction_section += "Present both sides with their sources.\n--- END CONFLICTING CLAIMS ---\n"

    user = (
        f"Question: {plan.objective}\n\n"
        f"--- VERIFIED CLAIMS ---\n{claims_text}\n--- END VERIFIED CLAIMS ---\n"
        f"{contradiction_section}\n"
        "Write the answer using ONLY the verified claims above, with bracket citations."
    )

    return [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]
