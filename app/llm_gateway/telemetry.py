"""Telemetry/observability for the Multi-Model Fabric (Phase 07).

Logs routing decisions, provider/model selection, latency, token usage,
and call counts per run. Designed to be lightweight and not leak secrets.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.logging_config import get_logger

logger = get_logger("argus.telemetry")

# Context variable for the current run's telemetry
_current_run_telemetry: ContextVar["RunTelemetry | None"] = ContextVar(
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


def start_run_telemetry(call_ceiling: int = 16, call_ceiling_warn: int = 12) -> RunTelemetry:
    """Start a new telemetry run and bind it to the current context."""
    telemetry = RunTelemetry(
        call_ceiling=call_ceiling,
        call_ceiling_warn=call_ceiling_warn,
    )
    _current_run_telemetry.set(telemetry)
    logger.info("telemetry_run_started", run_id=telemetry.run_id, call_ceiling=call_ceiling)
    return telemetry


def get_current_telemetry() -> RunTelemetry | None:
    """Get the current run's telemetry, if any."""
    return _current_run_telemetry.get()


def end_run_telemetry() -> dict[str, Any] | None:
    """End the current run and return its summary."""
    telemetry = _current_run_telemetry.get()
    if telemetry is None:
        return None
    summary = telemetry.get_summary()
    logger.info("telemetry_run_completed", **summary)
    _current_run_telemetry.set(None)
    return summary


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