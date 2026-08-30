"""Verification API endpoint (Phase 12.1).

UI/control seam over the Phase 04 verification engine. Exposes claim
verification as a thin endpoint so the Phase 12.1 evidence-explorer UI can
display verifier result, confidence, and contradictions for a live query
without importing business logic into the UI.

Verification itself remains fully in `app/verification`; this router only
adapts HTTP to the existing engine and mirrors its graceful-degradation
philosophy (an unavailable verifier yields a `VerificationStatus.ERROR`
result instead of a 500).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.evidence.store import get_evidence_store
from app.graph.store import get_graph_store
from app.llm_gateway import get_router
from app.llm_gateway.providers.exceptions import LLMProviderError
from app.verification.engine import verify_claim
from app.verification.models import VerificationRequest, VerificationResult, VerificationStatus

router = APIRouter(prefix="/api/v1", tags=["verification"])


class VerifyRequest(BaseModel):
    """Request to verify a claim against evidence (presentation view of `VerificationRequest`)."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The claim to verify (typically the synthesized answer).",
    )
    supporting_chunk_ids: list[UUID] = Field(
        default_factory=list,
        description="Evidence chunk IDs that support the claim (from orchestration citations).",
    )
    contradicting_chunk_ids: list[UUID] = Field(
        default_factory=list,
        description="Evidence chunk IDs that may contradict the claim.",
    )
    related_claim_ids: list[UUID] = Field(
        default_factory=list,
        description="Related claim IDs for contradiction detection.",
    )
    entity_names: list[str] = Field(
        default_factory=list,
        description="Named entities in the claim (context for the verifier).",
    )
    temporal_context: str | None = Field(default=None, description="Relevant time window, if any.")
    max_evidence_items: int = Field(default=20, ge=1, le=50)
    require_cross_source: bool = Field(default=True)


@router.post("/verify", response_model=VerificationResult)
async def verify(
    request: VerifyRequest,
    http_request: Request,
    settings: Settings = Depends(get_settings),
) -> VerificationResult:
    """Verify a claim against the evidence store using the existing Phase 04 engine."""
    verification_request = VerificationRequest(
        claim_id=uuid4(),
        claim_text=request.claim_text,
        supporting_chunk_ids=request.supporting_chunk_ids,
        contradicting_chunk_ids=request.contradicting_chunk_ids,
        related_claim_ids=request.related_claim_ids,
        entity_names=request.entity_names,
        temporal_context=request.temporal_context,
        max_evidence_items=request.max_evidence_items,
        require_cross_source=request.require_cross_source,
    )
    request_id = getattr(http_request.state, "request_id", None)
    try:
        return await verify_claim(
            verification_request,
            # MultiModelRouter is structurally compatible with the engine's
            # `LLMRouter` annotation (Phase 07 widened the gateway return type).
            router=get_router(),  # type: ignore[arg-type]
            evidence_store=get_evidence_store(),
            graph_store=get_graph_store(),
            settings=settings,
            request_id=request_id,
        )
    except LLMProviderError as exc:
        # Mirror the engine's graceful degradation when the verifier is
        # unavailable (e.g. no configured provider) — the UI can render
        # an ERROR result instead of a 500.
        return VerificationResult(
            claim_id=verification_request.claim_id,
            claim_text=verification_request.claim_text,
            status=VerificationStatus.ERROR,
            confidence=0.0,
            reasoning=f"Verifier unavailable: {exc}",
            metadata={"error": str(exc)},
        )