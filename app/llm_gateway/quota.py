"""Quota tracking for free-tier providers (Phase 07).

Deterministic, configuration-driven quota awareness. Tracks requests
and tokens per provider per time window. Does NOT implement complex
quota optimization — only basic availability checks as required by
the Phase 07 specification.

Quota limits come from `configs/model_policy.yaml` (the `quota` section).
Runtime values are updated from provider response headers (e.g.,
`x-ratelimit-remaining-requests`, `x-ratelimit-reset`) when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from app.config import Settings, get_settings, load_yaml_config
from app.llm_gateway.providers.models import CompletionResponse


@dataclass
class QuotaWindow:
    """A single quota window (e.g., per-minute, per-day)."""

    limit: int
    used: int = 0
    window_start: float = field(default_factory=time.time)
    window_seconds: int = 60

    def is_expired(self) -> bool:
        return time.time() - self.window_start >= self.window_seconds

    def reset_if_expired(self) -> None:
        if self.is_expired():
            self.used = 0
            self.window_start = time.time()

    def remaining(self) -> int:
        self.reset_if_expired()
        return max(0, self.limit - self.used)

    def consume(self, amount: int = 1) -> bool:
        """Try to consume quota. Returns True if successful, False if exhausted."""
        self.reset_if_expired()
        if self.used + amount > self.limit:
            return False
        self.used += amount
        return True


@dataclass
class ProviderQuota:
    """Quota tracking for a single provider."""

    name: str
    requests_per_minute: QuotaWindow
    requests_per_day: QuotaWindow
    tokens_per_minute: QuotaWindow
    tokens_per_day: QuotaWindow
    enabled: bool = True

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> ProviderQuota:
        return cls(
            name=name,
            requests_per_minute=QuotaWindow(
                limit=config.get("requests_per_minute", 30),
                window_seconds=60,
            ),
            requests_per_day=QuotaWindow(
                limit=config.get("requests_per_day", 1000),
                window_seconds=86400,
            ),
            tokens_per_minute=QuotaWindow(
                limit=config.get("tokens_per_minute", 8000),
                window_seconds=60,
            ),
            tokens_per_day=QuotaWindow(
                limit=config.get("tokens_per_day", 200000),
                window_seconds=86400,
            ),
            enabled=config.get("enabled", True),
        )

    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if a request with estimated_tokens can be made."""
        if not self.enabled:
            return True
        # Check request quotas
        if self.requests_per_minute.remaining() <= 0:
            return False
        if self.requests_per_day.remaining() <= 0:
            return False
        # Check token quotas (if estimated)
        if estimated_tokens > 0:
            if self.tokens_per_minute.remaining() < estimated_tokens:
                return False
            if self.tokens_per_day.remaining() < estimated_tokens:
                return False
        return True

    def record_request(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record a completed request."""
        if not self.enabled:
            return
        total_tokens = prompt_tokens + completion_tokens
        self.requests_per_minute.consume(1)
        self.requests_per_day.consume(1)
        if total_tokens > 0:
            self.tokens_per_minute.consume(total_tokens)
            self.tokens_per_day.consume(total_tokens)

    def get_status(self) -> dict[str, Any]:
        """Get current quota status for telemetry."""
        return {
            "provider": self.name,
            "enabled": self.enabled,
            "requests_per_minute": {
                "limit": self.requests_per_minute.limit,
                "used": self.requests_per_minute.used,
                "remaining": self.requests_per_minute.remaining(),
                "reset_seconds": max(0, int(self.requests_per_minute.window_seconds - (time.time() - self.requests_per_minute.window_start))),
            },
            "requests_per_day": {
                "limit": self.requests_per_day.limit,
                "used": self.requests_per_day.used,
                "remaining": self.requests_per_day.remaining(),
                "reset_seconds": max(0, int(self.requests_per_day.window_seconds - (time.time() - self.requests_per_day.window_start))),
            },
            "tokens_per_minute": {
                "limit": self.tokens_per_minute.limit,
                "used": self.tokens_per_minute.used,
                "remaining": self.tokens_per_minute.remaining(),
                "reset_seconds": max(0, int(self.tokens_per_minute.window_seconds - (time.time() - self.tokens_per_minute.window_start))),
            },
            "tokens_per_day": {
                "limit": self.tokens_per_day.limit,
                "used": self.tokens_per_day.used,
                "remaining": self.tokens_per_day.remaining(),
                "reset_seconds": max(0, int(self.tokens_per_day.window_seconds - (time.time() - self.tokens_per_day.window_start))),
            },
        }


class QuotaTracker:
    """Central quota tracker for all providers.

    Thread-safe. Updated from provider responses and consulted
    before routing decisions.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._quotas: dict[str, ProviderQuota] = {}
        self._lock = Lock()
        self._load_config()

    def _load_config(self) -> None:
        policy_path = self._settings.config_dir / "model_policy.yaml"
        data = load_yaml_config(policy_path)
        quota_config = data.get("quota", {})
        for provider_name, config in quota_config.items():
            self._quotas[provider_name] = ProviderQuota.from_config(provider_name, config)

    def get_quota(self, provider_name: str) -> ProviderQuota | None:
        with self._lock:
            return self._quotas.get(provider_name)

    def can_make_request(self, provider_name: str, estimated_tokens: int = 0) -> bool:
        quota = self.get_quota(provider_name)
        if quota is None:
            return True  # No quota tracking = unlimited
        return quota.can_make_request(estimated_tokens)

    def record_request(
        self,
        provider_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        quota = self.get_quota(provider_name)
        if quota is not None:
            with self._lock:
                quota.record_request(prompt_tokens, completion_tokens)

    def update_from_headers(self, provider_name: str, headers: dict[str, str]) -> None:
        """Update quota from provider response headers.

        Expected headers (provider-specific):
        - x-ratelimit-remaining-requests
        - x-ratelimit-remaining-tokens
        - x-ratelimit-reset (seconds until reset)
        """
        quota = self.get_quota(provider_name)
        if quota is None:
            return

        with self._lock:
            # Generic header parsing - providers vary
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
            reset_seconds = headers.get("x-ratelimit-reset")

            if remaining_requests is not None:
                try:
                    quota.requests_per_minute.used = quota.requests_per_minute.limit - int(remaining_requests)
                except ValueError:
                    pass
            if remaining_tokens is not None:
                try:
                    quota.tokens_per_minute.used = quota.tokens_per_minute.limit - int(remaining_tokens)
                except ValueError:
                    pass
            if reset_seconds is not None:
                try:
                    rs = int(reset_seconds)
                    quota.requests_per_minute.window_start = time.time() - (quota.requests_per_minute.window_seconds - rs)
                    quota.tokens_per_minute.window_start = time.time() - (quota.tokens_per_minute.window_seconds - rs)
                except ValueError:
                    pass

    def get_all_status(self) -> dict[str, Any]:
        with self._lock:
            return {name: quota.get_status() for name, quota in self._quotas.items()}


# Global quota tracker instance
_quota_tracker: QuotaTracker | None = None


def get_quota_tracker(settings: Settings | None = None) -> QuotaTracker:
    global _quota_tracker
    if _quota_tracker is None:
        _quota_tracker = QuotaTracker(settings)
    return _quota_tracker


async def close_quota_tracker() -> None:
    global _quota_tracker
    _quota_tracker = None


def update_quota_from_response(provider_name: str, response: CompletionResponse) -> None:
    """Update quota tracker from a completion response.

    This is a convenience function that extracts usage info from the
    response and records it in the quota tracker.
    """
    if response.usage is None:
        return
    quota_tracker = get_quota_tracker()
    quota_tracker.record_request(
        provider_name=provider_name,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )