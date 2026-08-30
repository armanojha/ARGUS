"""Telemetry/observability for the Multi-Model Fabric (Phase 07).

Logs routing decisions, provider/model selection, latency, token usage,
and call counts per run. Designed to be lightweight and not leak secrets.
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.logging_config import get_logger

logger = get_logger("argus.telemetry")

# In-process registry of completed runs (newest first), plus an optional
# JSONL persistence directory so run traces survive app restarts. Persistence
# is opt-in (set via `set_telemetry_persistence_dir`).
_completed_runs: list[dict[str, Any]] = []
_completed_runs_limit = 500
_persistence_dir: Path | None = None

# Context variable for the current run's telemetry
_current_run_telemetry: ContextVar[RunTelemetry | None] = ContextVar(
    "current_run_telemetry", default=None
)


@dataclass
class RoutingDecision:
    """A single routing decision record."""

    call_type: str
    provider: str
    model: str
    is_fallback: bool
    fallback_reason: str | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    success: bool
    error_code: str | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunTelemetry:
    """Telemetry for a single research run."""

    run_id: str = field(default_factory=lambda: str(uuid4())[:8])
    start_time: float = field(default_factory=time.time)
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    call_ceiling: int = 16
    call_ceiling_warn: int = 12

    def record_routing_decision(self, decision: RoutingDecision) -> None:
        self.routing_decisions.append(decision)
        self.total_calls += 1
        if decision.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        if decision.prompt_tokens:
            self.total_prompt_tokens += decision.prompt_tokens
        if decision.completion_tokens:
            self.total_completion_tokens += decision.completion_tokens

    def is_at_ceiling(self) -> bool:
        return self.total_calls >= self.call_ceiling

    def is_at_warn_threshold(self) -> bool:
        return self.total_calls >= self.call_ceiling_warn

    def get_summary(self) -> dict[str, Any]:
        duration_ms = int((time.time() - self.start_time) * 1000)
        return {
            "run_id": self.run_id,
            "duration_ms": duration_ms,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "call_ceiling": self.call_ceiling,
            "call_ceiling_warn": self.call_ceiling_warn,
            "at_ceiling": self.is_at_ceiling(),
            "at_warn_threshold": self.is_at_warn_threshold(),
            "routing_decisions": [
                {
                    "call_type": d.call_type,
                    "provider": d.provider,
                    "model": d.model,
                    "is_fallback": d.is_fallback,
                    "fallback_reason": d.fallback_reason,
                    "latency_ms": d.latency_ms,
                    "prompt_tokens": d.prompt_tokens,
                    "completion_tokens": d.completion_tokens,
                    "total_tokens": d.total_tokens,
                    "success": d.success,
                    "error_code": d.error_code,
                }
                for d in self.routing_decisions
            ],
        }


def start_run_telemetry(
    call_ceiling: int = 16,
    call_ceiling_warn: int = 12,
    *,
    run_id: str | None = None,
) -> RunTelemetry:
    """Start a new telemetry run and bind it to the current context.

    ``run_id`` is optional: callers may supply a stable identifier (e.g. a
    benchmark item id) so the trace is attributable; otherwise a short UUID
    is generated.
    """
    telemetry = RunTelemetry(
        run_id=run_id or str(uuid4())[:8],
        call_ceiling=call_ceiling,
        call_ceiling_warn=call_ceiling_warn,
    )
    _current_run_telemetry.set(telemetry)
    logger.info("telemetry_run_started", run_id=telemetry.run_id, call_ceiling=call_ceiling)
    return telemetry


def get_current_telemetry() -> RunTelemetry | None:
    """Get the current run's telemetry, if any."""
    return _current_run_telemetry.get()


def set_telemetry_persistence_dir(directory: str | Path | None) -> None:
    """Enable optional JSONL persistence of completed run summaries.

    Pass a directory (typically ``settings.data_dir / "telemetry"``) or
    ``None`` to disable. Writing is append-only; readers dedupe by run_id.
    """
    global _persistence_dir
    _persistence_dir = Path(directory) if directory is not None else None
    if _persistence_dir is not None:
        _persistence_dir.mkdir(parents=True, exist_ok=True)


def _persisted_records() -> list[dict[str, Any]]:
    if _persistence_dir is None:
        return []
    path = _persistence_dir / "runs.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def _persist_summary(summary: dict[str, Any]) -> None:
    if _persistence_dir is None:
        return
    path = _persistence_dir / "runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, default=str) + "\n")


def end_run_telemetry() -> dict[str, Any] | None:
    """End the current run, return its summary, and record it."""
    telemetry = _current_run_telemetry.get()
    if telemetry is None:
        return None
    summary = telemetry.get_summary()
    logger.info("telemetry_run_completed", **summary)
    _current_run_telemetry.set(None)
    _completed_runs.insert(0, summary)
    del _completed_runs[_completed_runs_limit:]
    _persist_summary(summary)
    return summary


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent completed run summaries, newest first.

    Merges persisted records (disk, survives restarts) with the in-process
    registry; in-memory records win for the same run_id.
    """
    merged: dict[str, dict[str, Any]] = {}
    for rec in _persisted_records():
        if rec.get("run_id"):
            merged.setdefault(rec["run_id"], rec)
    for rec in _completed_runs:
        if rec.get("run_id"):
            merged[rec["run_id"]] = rec
    ordered = list(merged.values())
    return ordered[-limit:][::-1]


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return a single completed run summary by run_id (memory then disk)."""
    for rec in _completed_runs:
        if rec.get("run_id") == run_id:
            return rec
    for rec in _persisted_records():
        if rec.get("run_id") == run_id:
            return rec
    return None


def record_routing_decision(
    call_type: str,
    provider: str,
    model: str,
    is_fallback: bool = False,
    fallback_reason: str | None = None,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    success: bool = True,
    error_code: str | None = None,
) -> None:
    """Record a routing decision in the current telemetry run."""
    telemetry = _current_run_telemetry.get()
    if telemetry is None:
        return  # No active run - skip (e.g., standalone calls)

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    decision = RoutingDecision(
        call_type=call_type,
        provider=provider,
        model=model,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        success=success,
        error_code=error_code,
    )
    telemetry.record_routing_decision(decision)

    # Log the routing decision
    logger.info(
        "routing_decision",
        run_id=telemetry.run_id,
        call_type=call_type,
        provider=provider,
        model=model,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        success=success,
        error_code=error_code,
        total_calls=telemetry.total_calls,
        call_ceiling=telemetry.call_ceiling,
    )

    # Warn if approaching ceiling
    if telemetry.is_at_warn_threshold() and not telemetry.is_at_ceiling():
        logger.warning(
            "call_ceiling_warning",
            run_id=telemetry.run_id,
            total_calls=telemetry.total_calls,
            call_ceiling=telemetry.call_ceiling,
            call_ceiling_warn=telemetry.call_ceiling_warn,
        )


def check_call_ceiling() -> bool:
    """Check if the current run has exceeded the call ceiling.

    Returns True if over ceiling, False otherwise.
    """
    telemetry = _current_run_telemetry.get()
    if telemetry is None:
        return False
    return telemetry.is_at_ceiling()