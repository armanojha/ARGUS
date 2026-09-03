"""Provider health / cooldown tracking (Phase 07 — Resilience).

A lightweight, persistent, thread-safe health tracker consulted by the
``MultiModelRouter`` so that routing decisions can *avoid* unhealthy or
exhausted providers instead of repeatedly paying the cost of a known-dead
provider.

This is deliberately NOT a distributed health system nor a full circuit
breaker (Phase 07.17: do not over-engineer). It is a small state map with a
cooldown clock that answers one question the existing ``QuotaTracker`` cannot:
"has this provider/model recently failed in a way that warrants *not* retrying
it for a short window?" The quota tracker answers the budget question
(request/token windows); this tracker answers the *health* question
(failure class + cooldown).

State is scoped (mirrors the router's exclude_models/exclude_providers rule):
  * A **model-scoped** failure (e.g. one model's rate limit / model-not-found)
    marks only ``provider/model`` so an intentional intra-provider fallback
    (an adjacent model on the same provider) stays reachable.
  * A **provider-scoped** failure (auth / whole-endpoint down / timeout/
    configuration) marks the whole provider so every model on it is skipped
    during cooldown.

States (Phase 07.3):
  HEALTHY         — no recent failure; normal routing.
  DEGRADED        — intermittent/unclassified failures; short cooldown.
  RATE_LIMITED    — a rate-limit (429) was observed on the entity; short cooldown.
  QUOTA_EXHAUSTED — quota/window exhausted; avoid until cooldown/reset.
  UNAVAILABLE     — auth / provider down / timeout / config; cooldown.

Failure classification maps the existing ARGUS error codes to these states so
ARGUS does not treat all failures identically (07.3). ``CALL_CEILING_EXCEEDED``
is a per-run budget signal, not a provider condition, so it is never recorded.

Cooldowns are monotonic (`time.monotonic`) and independent per entity.
Never holds secrets. Never logs keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

from app.config import Settings, get_settings

# Error codes that are *permanent* / not worth re-trying even after a cooldown
# under normal operation (until the operator addresses the root cause). The
# registry already discards providers that fail to build; this covers runtime
# auth/config faults without an error storm.
_HARD_FAILURE_CODES = {
    "AUTHENTICATION_ERROR",
    "CONFIGURATION_ERROR",
    "CAPABILITY_NOT_SUPPORTED",
}

# Per-run budget signals that are NOT provider health conditions.
_NON_HEALTH_CODES = {
    "CALL_CEILING_EXCEEDED",
}

# Transient/connectivity failures mapped to an UNAVAILABLE-style cooldown.
_UNAVAILABLE_CODES = {
    "TIMEOUT_OR_NETWORK_ERROR",
    "PROVIDER_UNAVAILABLE",
}

# Cooldown windows (seconds, monotonic-clock based). Modest and deliberately
# short: long enough to stop re-hitting a known-dead endpoint in a single run,
# short enough that a transient blip does not quarantine a provider for long.
# Auth/config faults get a longer window since they are unlikely to clear on
# their own within a short window.
_COOLDOWN_SECONDS = {
    "RATE_LIMITED": 10.0,
    "QUOTA_EXHAUSTED": 30.0,
    "UNAVAILABLE": 15.0,
    "UNAVAILABLE_HARD": 30.0,
    "DEGRADED": 5.0,
}


class HealthStatus(str, Enum):
    """Health state of a provider or provider/model (persistent across requests)."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    UNAVAILABLE = "unavailable"


@dataclass
class EntityHealth:
    """Health state for a single entity (a provider OR a provider/model)."""

    key: str
    scope: str  # "provider" | "model"
    name: str  # provider name
    model: str | None = None
    status: HealthStatus = HealthStatus.HEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_error_code: str | None = None
    last_error: str | None = None
    cooldown_until: float = 0.0  # monotonic timestamp; 0 = no cooldown
    failures: dict[str, int] = field(default_factory=dict)

    def in_cooldown(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.cooldown_until > now

    def blocks_routing(self, now: float | None = None) -> bool:
        """Whether routing should skip this entity right now."""
        if self.status == HealthStatus.HEALTHY:
            return False
        return self.in_cooldown(now)


class ProviderHealthTracker:
    """Central health tracker for all entities (provider and provider/model).

    Thread-safe. Consulted by the router before routing and updated from the
    outcome of each routing attempt. Model-scoped failures never quarantine the
    entire provider, preserving intentional intra-provider fallbacks.
    """

    # Prefix separator between provider and model in the entity key.
    MODEL_SEP = "/"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._health: dict[str, EntityHealth] = {}
        self._lock = Lock()

    @staticmethod
    def _provider_key(provider: str) -> str:
        return provider

    @classmethod
    def _model_key(cls, provider: str, model: str) -> str:
        return f"{provider}{cls.MODEL_SEP}{model}"

    def _entry(self, key: str) -> EntityHealth:
        entry = self._health.get(key)
        if entry is None:
            scope = "model" if key.count(self.MODEL_SEP) > 0 else "provider"
            name, _, model = key.partition(self.MODEL_SEP)
            entry = EntityHealth(key=key, scope=scope, name=name, model=model or None)
            self._health[key] = entry
        return entry

    def get_status(self, provider: str, model: str | None = None) -> HealthStatus:
        key = self._model_key(provider, model) if model else self._provider_key(provider)
        with self._lock:
            entry = self._health.get(key)
            if entry is None:
                return HealthStatus.HEALTHY
            # Cooldown-expired transient states recover to HEALTHY.
            if not entry.in_cooldown():
                entry.status = HealthStatus.HEALTHY
            return entry.status

    def _resolve_skip_reason(self, entry: EntityHealth | None) -> str | None:
        if entry is None:
            return None
        if not entry.blocks_routing():
            return None
        remaining = max(0, entry.cooldown_until - time.monotonic())
        if entry.status == HealthStatus.QUOTA_EXHAUSTED:
            return f"health_quota_exhausted(cooldown={remaining:.1f}s)"
        if entry.status == HealthStatus.RATE_LIMITED:
            return f"health_rate_limited(cooldown={remaining:.1f}s)"
        return f"health_unavailable(cooldown={remaining:.1f}s)"

    def skip_reason(self, provider: str, model: str | None = None) -> str | None:
        """Return a non-None reason string if routing should skip this entity.

        When ``model`` is given, both the provider-wide and the model-scoped
        states are consulted: a provider-wide failure skips every model, while a
        model-scoped failure skips only that model.
        """
        with self._lock:
            reason = self._resolve_skip_reason(self._health.get(self._provider_key(provider)))
            if reason is not None:
                return reason
            if model is not None:
                reason = self._resolve_skip_reason(self._health.get(self._model_key(provider, model)))
            return reason

    def can_make_request(self, provider: str, model: str | None = None) -> bool:
        """Whether the router may attempt this entity (health-aware)."""
        return self.skip_reason(provider, model) is None

    def _apply_failure(self, entry: EntityHealth, error_code: str, error: str | None) -> HealthStatus:
        entry.consecutive_failures += 1
        entry.consecutive_successes = 0
        entry.last_error_code = error_code
        entry.last_error = (error or "")[:300]
        if error_code:
            entry.failures[error_code] = entry.failures.get(error_code, 0) + 1

        now = time.monotonic()
        if error_code in _HARD_FAILURE_CODES:
            entry.status = HealthStatus.UNAVAILABLE
            entry.cooldown_until = now + _COOLDOWN_SECONDS["UNAVAILABLE_HARD"]
        elif error_code == "RATE_LIMIT_ERROR":
            entry.status = HealthStatus.RATE_LIMITED
            entry.cooldown_until = now + _COOLDOWN_SECONDS["RATE_LIMITED"]
        elif error_code in _UNAVAILABLE_CODES:
            entry.status = HealthStatus.UNAVAILABLE
            entry.cooldown_until = now + _COOLDOWN_SECONDS["UNAVAILABLE"]
        else:
            entry.status = HealthStatus.DEGRADED
            entry.cooldown_until = now + _COOLDOWN_SECONDS["DEGRADED"]
        return entry.status

    def record_failure(
        self,
        provider: str,
        error_code: str,
        error: str | None = None,
        *,
        model: str | None = None,
        scope: str = "model",
    ) -> HealthStatus:
        """Record a failure.

        ``scope`` selects the quarantine extent (Phase 07.3):
          * ``"model"`` (default) — only ``provider/model`` is marked, so an
            intentional intra-provider fallback (another model on the same
            provider) stays reachable.
          * ``"provider"`` — the whole provider is marked; every model on it is
            skipped during cooldown.

        ``CALL_CEILING_EXCEEDED`` is never recorded (per-run budget, not a
        provider condition).
        """
        if error_code in _NON_HEALTH_CODES:
            return HealthStatus.HEALTHY
        key = (
            self._model_key(provider, model) if (scope == "model" and model) else self._provider_key(provider)
        )
        with self._lock:
            entry = self._entry(key)
            return self._apply_failure(entry, error_code, error)

    def record_success(
        self,
        provider: str,
        *,
        model: str | None = None,
        scope: str = "model",
    ) -> HealthStatus:
        """Record a successful call; clears the matching entity's cooldown.

        ``scope="provider"`` clears the provider-wide state; the default
        ``"model"`` clears only the specific provider/model entry.
        """
        if scope == "provider" or model is None:
            key = self._provider_key(provider)
        else:
            key = self._model_key(provider, model)
        with self._lock:
            entry = self._health.get(key)
            if entry is None:
                return HealthStatus.HEALTHY
            entry.consecutive_successes += 1
            entry.consecutive_failures = 0
            entry.last_error_code = None
            entry.last_error = None
            entry.cooldown_until = 0.0
            entry.status = HealthStatus.HEALTHY
            return entry.status

    def recovery_candidate(
        self,
        specs: list[tuple[str, str | None]],
        probe_grace: float = 2.0,
    ) -> tuple[str, str | None, float] | None:
        """Pick the provider/model most-likely to have recovered from a transient
        failure, for a single bounded in-session recovery probe (Phase 07e).

        ``specs`` is an ordered list of ``(provider, model)`` candidates. This
        returns the candidate currently in cooldown whose cooldown is closest
        to expiry AND within ``probe_grace`` seconds of clearing (or already
        elapsed but not yet lazily re-selected) — the one most likely to serve
        again under easing burst pressure. Returns ``None`` when no candidate is
        near recovery, so a still-mid-cooldown / still-failing provider is NOT
        probed (no wasted calls; 0-repeat).

        This is deliberately a *bounded, health-backed* selection:
          * It proposes at most ONE candidate, which the caller probes once.
          * It only fires when the caller has no other eligible provider, so it
            adds no work to the healthy path.
          * A failed probe re-records the entity in cooldown, so a still-down
            provider is never repeatedly hammered.

        Returns ``(provider, model, cooldown_remaining_s)`` where a positive
        value indicates the cooldown is still active but about to clear.
        """
        now = time.monotonic()
        best: tuple[str, str | None, float] | None = None
        with self._lock:
            for provider, model in specs:
                key = self._model_key(provider, model) if model else self._provider_key(provider)
                entry = self._health.get(key)
                if entry is None or not entry.in_cooldown(now):
                    continue
                remaining = entry.cooldown_until - now
                if remaining > probe_grace:
                    # Still comfortably mid-cooldown: probing would just re-hit
                    # a known-failing provider for no value.
                    continue
                if best is None or remaining < best[2]:
                    best = (entry.name, entry.model, remaining)
        return best

    def get_all_status(self) -> dict[str, dict]:
        with self._lock:
            return {key: self._describe(entry) for key, entry in self._health.items()}

    @staticmethod
    def _describe(entry: EntityHealth) -> dict:
        now = time.monotonic()
        return {
            "scope": entry.scope,
            "provider": entry.name,
            "model": entry.model,
            "status": entry.status.value,
            "consecutive_failures": entry.consecutive_failures,
            "cooldown_active": entry.in_cooldown(now),
            "cooldown_remaining_s": round(max(0, entry.cooldown_until - now), 1),
            "last_error_code": entry.last_error_code,
        }

    def reset(self) -> None:
        with self._lock:
            self._health.clear()


# Global health tracker instance (mirrors quota.py singleton pattern).
_health_tracker: ProviderHealthTracker | None = None


def get_provider_health_tracker(settings: Settings | None = None) -> ProviderHealthTracker:
    global _health_tracker
    if _health_tracker is None:
        _health_tracker = ProviderHealthTracker(settings)
    return _health_tracker


def reset_provider_health_tracker() -> None:
    """Reset the global health tracker (test isolation)."""
    global _health_tracker
    _health_tracker = None


async def close_provider_health_tracker() -> None:
    """No-op teardown for runtime lifecycle parity with quota/memory close."""
    global _health_tracker
    _health_tracker = None