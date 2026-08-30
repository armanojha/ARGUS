"""Run-trace telemetry API (Phase 12.2).

Exposes the Phase 07 telemetry fabric as read-only endpoints so the UI can
surface latency, tokens, provider/model, call counts, and failures per run.
Data comes from the in-process registry plus optional JSONL persistence
(``settings.data_dir / "telemetry"``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.llm_gateway.telemetry import get_run, list_runs

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


@router.get("")
async def list_run_summaries(limit: int = 50) -> list[dict[str, Any]]:
    """List completed run summaries, newest first (run_id, totals, provider/model)."""
    bounded = max(1, min(limit, 200))
    return list_runs(limit=bounded)


@router.get("/{run_id}")
async def run_summary(run_id: str) -> dict[str, Any]:
    """Return the full run trace (all routing decisions) for one run."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Telemetry run not found: {run_id}")
    return run