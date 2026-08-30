"""Agentic RAG query API endpoint (Phase 02).

Exposes the orchestration loop (`app.orchestration.run_query`) as
`POST /api/v1/query`: question in, research-plan-driven cited answer
out. Wraps Phase 01 retrieval; does not implement its own retrieval
path.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.llm_gateway.telemetry import end_run_telemetry, start_run_telemetry
from app.orchestration.graph import run_query
from app.orchestration.models import OrchestrationResult

router = APIRouter(prefix="/api/v1", tags=["orchestration"])


class QueryRequest(BaseModel):
    """Request for the agentic query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000, description="The research question to answer.")
    user_early_stop: bool = Field(
        default=False,
        description="Phase 06: request an explicit stop. No further retrieval/LLM work; "
        "the loop synthesizes from the evidence gathered so far.",
    )


@router.post("/query", response_model=OrchestrationResult)
async def query(request: QueryRequest, http_request: Request) -> OrchestrationResult:
    """Answer a question via the Agentic RAG loop: plan, retrieve, synthesize, cite.

    Runs Phase 02's bounded orchestration loop on top of the Phase 01
    hybrid retriever and the Phase 00.3 LLM gateway. Iteration and
    token budgets are enforced regardless of what the planner proposes.
    With Phase 06 enabled, retrieval dispatch follows the adaptive
    policy and stopping follows the full V2 §5.4 condition set.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    request_id = getattr(http_request.state, "request_id", None)
    run_id = request_id or "ui"  # trace attributable to the UI query when present
    settings = get_settings()
    call_ceiling = settings.multimodel_call_ceiling
    try:
        start_run_telemetry(
            call_ceiling=call_ceiling,
            call_ceiling_warn=max(12, call_ceiling - 4),
            run_id=run_id[:64],
        )
        result = await run_query(
            request.query,
            request_id=request_id,
            user_early_stop=request.user_early_stop,
        )
    finally:
        summary = end_run_telemetry()
    if summary is not None:
        result = result.model_copy(update={"telemetry": summary})
    return result
