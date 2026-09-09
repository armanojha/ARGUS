"""SSE streaming for query progress (P1 showcase polish).

Provides `POST /api/v1/query/stream` which returns Server-Sent Events
as the orchestration loop progresses through each phase. The UI can
subscribe to this stream to show real-time progress updates.

Events:
  - phase: The current orchestration phase (plan, retrieve, assess, etc.)
  - progress: Iteration count and evidence count
  - result: The final OrchestrationResult
  - error: An error occurred
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.llm_gateway.telemetry import end_run_telemetry, start_run_telemetry
from app.orchestration.graph import run_query
from app.orchestration.models import sanitize_result_for_user

router = APIRouter(prefix="/api/v1", tags=["orchestration-stream"])


class StreamQueryRequest(BaseModel):
    """Request for the streaming query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000, description="The research question to answer.")
    user_early_stop: bool = Field(default=False)


def _sse_event(event: str, data: dict[str, Any] | str) -> str:
    """Format a single SSE event."""
    if isinstance(data, dict):
        payload = json.dumps(data, default=str)
    else:
        payload = data
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/query/stream")
async def query_stream(request: StreamQueryRequest, http_request: Request) -> StreamingResponse:
    """Stream query progress via Server-Sent Events.

    Events:
      - started: Query processing started
      - phase: Node phase starting/completing (plan, retrieve, assess, etc.)
      - result: Final OrchestrationResult
      - error: Error occurred
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    request_id = getattr(http_request.state, "request_id", None)
    run_id = request_id or "ui"
    settings = get_settings()
    call_ceiling = settings.multimodel_call_ceiling

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Signal start
            yield _sse_event("started", {
                "query": request.query[:100],
                "request_id": request_id,
            })

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

            result = sanitize_result_for_user(result)

            # Emit progress events from node traces
            for trace in result.node_traces:
                yield _sse_event("phase", {
                    "node": trace.get("node", "unknown"),
                    "status": trace.get("status", "completed"),
                    "description": trace.get("why", ""),
                    "latency_ms": trace.get("latency_ms", 0),
                    "iteration": trace.get("iteration", 0),
                    "evidence_count": trace.get("evidence_count", 0),
                    "tokens_used": trace.get("tokens_used", 0),
                })

            # Emit final result
            yield _sse_event("result", result.model_dump(mode="json"))

        except Exception as e:  # noqa: BLE001 - fail-safe: always emit error event
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
