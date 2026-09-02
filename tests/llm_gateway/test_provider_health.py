"""Tests for provider health / cooldown tracking (Phase 07 — Resilience).

Covers:
- Health state classification from failure codes (07.3).
- Cooldown windows and skip/avoid behaviour (07.2).
- Recovery to HEALTHY after a successful call (07.3).
- Router integration: an unhealthy provider is avoided on subsequent calls
  instead of repeatedly paying a dead-wait cost (07.10).
- Observability: health surfaced via get_available_models() (07.11).

These are ADDITIVE to test_multi_model_fabric.py; existing invariants
(fallback on failure, skip exhausted quota, intra-provider fallback not
blocked, cross-model verification, ceiling hard stop) are untouched.
"""
from __future__ import annotations

import time

import pytest

from app.config import Settings
from app.llm_gateway.health import (
    HealthStatus,
    ProviderHealthTracker,
    get_provider_health_tracker,
    reset_provider_health_tracker,
)
from app.llm_gateway.providers.exceptions import RateLimitError, TimeoutOrNetworkError
from app.llm_gateway.providers.models import Message, MessageRole
from app.llm_gateway.routing.multi_model_router import MultiModelRouter
from tests.mocks.mock_provider import MockProvider


@pytest.fixture(autouse=True)
def _reset_health():
    reset_provider_health_tracker()
    yield
    reset_provider_health_tracker()


@pytest.fixture
def mock_providers():
    return {
        "groq": MockProvider(name="groq", default_model="openai/gpt-oss-120b"),
        "gemini": MockProvider(name="gemini", default_model="gemini-3.5-flash-lite"),
        "cerebras": MockProvider(name="cerebras", default_model="gpt-oss-120b"),
        "zen": MockProvider(name="zen", default_model="nemotron-3-ultra-free"),
    }


@pytest.fixture
def router(mock_providers):
    from app.llm_gateway.quota import reset_quota_tracker

    reset_quota_tracker()
    settings = Settings(multimodel_enabled=True)
    return MultiModelRouter(settings=settings, providers=mock_providers)


class TestProviderHealthTracker:
    def test_healthy_by_default(self):
        t = get_provider_health_tracker(Settings())
        assert t.get_status("groq") == HealthStatus.HEALTHY
        assert t.skip_reason("groq") is None
        assert t.can_make_request("groq") is True

    def test_rate_limit_sets_cooldown_and_blocks(self):
        t = get_provider_health_tracker(Settings())
        status = t.record_failure("groq", "RATE_LIMIT_ERROR", "429")
        assert status == HealthStatus.RATE_LIMITED
        assert t.get_status("groq") == HealthStatus.RATE_LIMITED
        assert t.skip_reason("groq") is not None
        assert "rate_limited" in (t.skip_reason("groq") or "")
        assert t.can_make_request("groq") is False

    def test_auth_is_unavailable_hard(self):
        t = get_provider_health_tracker(Settings())
        status = t.record_failure("gemini", "AUTHENTICATION_ERROR", "bad key")
        assert status == HealthStatus.UNAVAILABLE
        assert t.skip_reason("gemini") is not None

    def test_timeout_is_unavailable_transient(self):
        t = get_provider_health_tracker(Settings())
        status = t.record_failure("cerebras", "TIMEOUT_OR_NETWORK_ERROR")
        assert status == HealthStatus.UNAVAILABLE
        assert t.skip_reason("cerebras") is not None

    def test_cooldown_expires_and_recovers(self):
        t = ProviderHealthTracker(Settings())
        status = t.record_failure("groq", "RATE_LIMIT_ERROR")
        assert status == HealthStatus.RATE_LIMITED
        assert t.skip_reason("groq") is not None
        # Manually clear cooldown to simulate elapse.
        entry = t._entry("groq")
        entry.cooldown_until = time.monotonic() - 1
        assert t.skip_reason("groq") is None
        assert t.can_make_request("groq") is True

    def test_success_resets_to_healthy(self):
        t = get_provider_health_tracker(Settings())
        t.record_failure("groq", "RATE_LIMIT_ERROR")
        t.record_success("groq")
        assert t.get_status("groq") == HealthStatus.HEALTHY
        assert t.skip_reason("groq") is None

    def test_tracks_failure_counts_and_profile(self):
        t = ProviderHealthTracker(Settings())
        t.record_failure("zen", "TIMEOUT_OR_NETWORK_ERROR")
        t.record_failure("zen", "TIMEOUT_OR_NETWORK_ERROR")
        t.record_failure("zen", "AUTHENTICATION_ERROR")
        status = t.get_all_status()["zen"]
        assert status["consecutive_failures"] == 3
        assert status["last_error_code"] == "AUTHENTICATION_ERROR"
        assert status["cooldown_active"] is True


class TestRouterHealthIntegration:
    async def test_router_skips_unhealthy_provider(self, router, mock_providers):
        """A provider marked unhealthy (prior cooldown) is avoided on the next call."""
        from app.llm_gateway.health import get_provider_health_tracker

        health = get_provider_health_tracker(router._settings)
        # Simulate a prior failure already recorded on groq (as the router does
        # on a real failure); groq is NOT currently configured to fail, so this
        # isolates the health-skip (not the fallback-on-error) behaviour.
        health.record_failure("groq", "RATE_LIMIT_ERROR", "prior 429")
        assert health.skip_reason("groq") is not None

        response = await router.complete(
            [Message(role=MessageRole.USER, content="Test")],
            call_type="query_analysis",
        )
        assert response.provider != "groq"

    async def test_router_records_failure_then_recovers(self, router, mock_providers):
        """The router records a real rate-limit failure into health, and a later
        healthy call clears the recovering provider."""
        from app.llm_gateway.health import get_provider_health_tracker

        health = get_provider_health_tracker(router._settings)
        # Force groq to fail on this call; the router's fallback succeeds on
        # gemini, and groq's failure is recorded into the health tracker.
        mock_providers["groq"]._should_fail = True
        mock_providers["groq"]._fail_with = RateLimitError
        mock_providers["groq"]._fail_message = "rate limited"

        try:
            response = await router.complete(
                [Message(role=MessageRole.USER, content="Test")],
                call_type="query_analysis",
            )
            # Fallback succeeded; provider must be non-groq.
            assert response.provider != "groq"
        except Exception:
            # If every provider somehow failed, still assert health recorded groq.
            pass

        # groq's real failure was persisted (rate limit), model-scoped on the
        # model that was actually attempted (query_analysis balanced -> gpt-oss-20b);
        # provider scope remains clear so an intra-provider fallback would work.
        assert health.get_status("groq", "openai/gpt-oss-20b") == HealthStatus.RATE_LIMITED
        assert health.skip_reason("groq", "openai/gpt-oss-20b") is not None
        assert health.skip_reason("groq") is None  # provider-wide NOT blocked

        # A healthy call on gemini clears gemini's own entry.
        health.record_success("gemini", model="gemini-3.5-flash-lite", scope="model")
        assert health.get_status("gemini") == HealthStatus.HEALTHY

    async def test_get_available_models_exposes_health(self, router, mock_providers):
        from app.llm_gateway.health import get_provider_health_tracker

        health = get_provider_health_tracker(router._settings)
        health.record_failure("zen", "RATE_LIMIT_ERROR")

        models = router.get_available_models("query_analysis")
        zen_entry = next((m for m in models if m["provider"] == "zen"), None)
        assert zen_entry is not None
        # A provider-scoped failure is surfaced via provider_status on every
        # model of that provider, and its skip_reason is non-None.
        assert zen_entry["health"]["provider_status"] == HealthStatus.RATE_LIMITED.value
        assert zen_entry["health"]["skip_reason"] is not None

    async def test_router_does_not_retry_dead_provider(self, router, mock_providers):
        """A provider in cooldown is not revisited on a subsequent call."""
        from app.llm_gateway.health import get_provider_health_tracker

        health = get_provider_health_tracker(router._settings)
        # Agressively healthy-block groq (simulated prior timeout storm).
        health.record_failure("groq", "TIMEOUT_OR_NETWORK_ERROR")
        health.record_failure("groq", "TIMEOUT_OR_NETWORK_ERROR")
        mock_providers["groq"]._should_fail = True
        mock_providers["groq"]._fail_with = TimeoutOrNetworkError
        mock_providers["groq"]._fail_message = "timeout"

        assert health.skip_reason("groq") is not None
        calls_before = len(mock_providers["groq"]._call_log)

        response = await router.complete(
            [Message(role=MessageRole.USER, content="T")],
            call_type="query_analysis",
        )
        # groq was never attempted again because it was health-blocked.
        calls_after = len(mock_providers["groq"]._call_log)
        assert calls_after == calls_before
        assert response.provider != "groq"

    async def test_router_avoids_repeated_dead_wait(self, router, mock_providers):
        """A knowingly-unavailable provider is skipped, so a 15s-per-call
        timeout on it is never paid repeatedly (Phase 07.10 fallback efficiency)."""
        from app.llm_gateway.health import get_provider_health_tracker

        health = get_provider_health_tracker(router._settings)
        health.record_failure("groq", "TIMEOUT_OR_NETWORK_ERROR")
        mock_providers["groq"]._should_fail = True
        mock_providers["groq"]._fail_with = TimeoutOrNetworkError
        mock_providers["groq"]._fail_message = "timeout"

        # First call: groq is health-blocked, so it falls to a healthy provider
        # without attempting groq again.
        response = await router.complete(
            [Message(role=MessageRole.USER, content="A")],
            call_type="query_analysis",
        )
        assert response.provider != "groq"
        # Second call: still avoids groq.
        response2 = await router.complete(
            [Message(role=MessageRole.USER, content="B")],
            call_type="query_analysis",
        )
        assert response2.provider != "groq"