"""Agentic RAG query API endpoint (Phase 02).

Exposes the orchestration loop (`app.orchestration.run_query`) as
`POST /api/v1/query`: question in, research-plan-driven cited answer
out. Wraps Phase 01 retrieval; does not implement its own retrieval
path.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.orchestration.graph import run_query
from app.orchestration.models import OrchestrationResult

router = APIRouter(prefix="/api/v1", tags=["orchestration"])


class QueryRequest(BaseModel):
    """Request for the agentic query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000, description="The research question to answer.")


@router.post("/query", response_model=OrchestrationResult)
async def query(request: QueryRequest, http_request: Request) -> OrchestrationResult:
    """Answer a question via the Agentic RAG loop: plan, retrieve, synthesize, cite.

    Runs Phase 02's bounded orchestration loop on top of the Phase 01
    hybrid retriever and the Phase 00.3 LLM gateway. Iteration and
    token budgets are enforced regardless of what the planner proposes.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    request_id = getattr(http_request.state, "request_id", None)
    return await run_query(request.query, request_id=request_id)
