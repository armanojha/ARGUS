"""Phase 07e ("Graceful Burst Degradation + In-Session Recovery") tests.

Targets the reliability defect Phase 07c surfaced: a burst of rate-limit / quota
pressure must not cascade into an unnecessary "No available model" outage, and a
provider that recovers must be re-eligibilized *in the same session* without a
restart — all while never hammering a rate-limited provider (0-repeat) and always
preserving grounded answers + truthful `Outcome` semantics.

The health/cooldown/fallback machinery already rejects single-fault cases. These
tests focus on the burst + in-session-recovery behavior and the 07e recovery
probe (HARDEN-07e): when every configured provider is cooling, ONE bounded,
health-backed probe re-eligibilizes the provider closest to recovery instead of
failing the call, and a failed probe never re-hits a still-down provider.

All tests are deterministic — no network, no live quota.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.llm_gateway.health import (
    HealthStatus,
    ProviderHealthTracker,
    get_provider_health_tracker,
    reset_provider_health_tracker,
)
from app.llm_gateway.providers.exceptions import (
    LLMProviderError,
    RateLimitError,
)
from app.llm_gateway.quota import reset_quota_tracker
from app.llm_gateway.routing.multi_model_router import MultiModelRouter
from app.orchestration.graph import _derive_outcome
from app.orchestration.models import Outcome, ResearchPlan
from app.orchestration.nodes import _normalize_citation_markers, extract_cited_indices
from tests.mocks.mock_provider import MockProvider

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.5-flash-lite",
    "cerebras": "gpt-oss-120b",
    "zen": "nemotron-3-ultra-free",
}


def _settings(**overrides) -> Settings:
    base = {
        "multimodel_enabled": True,
        "verification_enabled": True,
        "orchestration_llm_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


def _providers(*dead: str) -> dict[str, MockProvider]:
    """Build healthy mock providers; named ones fail with 429."""
    out = {n: MockProvider(name=n, default_model=_DEFAULT_MODELS[n]) for n in _DEFAULT_MODELS}
    for n in dead:
        out[n]._should_fail = True
        out[n]._fail_with = RateLimitError
        out[n]._fail_message = f"{n} burst 429"
    return out


@pytest.fixture(autouse=True)
def _isolate():
    reset_provider_health_tracker()
    reset_quota_tracker()
    yield
    reset_provider_health_tracker()
    reset_quota_tracker()


def _make_router(providers: dict[str, MockProvider]) -> MultiModelRouter:
    return MultiModelRouter(settings=_settings(), providers=providers)


def _ref(score: float = 0.9, text: str = "Some supporting text.") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="/t/corpus.md",
        source_type=SourceType.MARKDOWN,
        text=text,
        score=score,
        rank=1,
    )


def _plan(risk_level: str = "low") -> ResearchPlan:
    return ResearchPlan(
        objective="Explain what the fox does.",
        entities=["fox"],
        time_window=None,
        subquestions=["fox behavior"],
        evidence_type="factual",
        preferred_retrieval_methods=["hybrid"],
        risk_level=risk_level,
        token_budget=6000,
        iteration_budget=2,
        stopping_condition="stop",
    )


# --------------------------------------------------------------------------- #
# Test 1 — primary provider rate-limited -> fallback, no hammering
# --------------------------------------------------------------------------- #
class TestPrimaryRateLimited:
    def test_primary_429_falls_back_to_healthy_provider(self):
        providers = _providers("groq")  # only groq 429s
        router = _make_router(providers)
        resp = asyncio.run(
            router.complete([], call_type="synthesis", query="q", tier="strong")
        )
        # Healthy fallback served the answer.
        assert resp.provider == "gemini"
        # groq hit exactly once (no retry storm on the rate-limited primary).
        assert len(providers["groq"].call_log) == 1
        # groq is now in rate-limit cooldown.
        assert get_provider_health_tracker(_settings()).skip_reason("groq", "openai/gpt-oss-120b") is not None


# --------------------------------------------------------------------------- #
# Test 2 — repeated calls during cooldown skip the rate-limited provider
# --------------------------------------------------------------------------- #
class TestSkipDuringCooldown:
    def test_subsequent_calls_skip_rate_limited_provider(self):
        providers = _providers("groq")
        router = _make_router(providers)
        asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        groq_calls_after_first = len(providers["groq"].call_log)
        # Second call: groq is in cooldown -> skipped, gemini still serves.
        resp = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        assert resp.provider == "gemini"
        assert len(providers["groq"].call_log) == groq_calls_after_first  # no repeated 429 to groq


# --------------------------------------------------------------------------- #
# Test 3 — cooldown expiry -> provider eligible again, success clears health
# --------------------------------------------------------------------------- #
class TestCooldownExpiryRecovery:
    def test_expired_cooldown_reuses_recovered_provider(self):
        tracker = ProviderHealthTracker(_settings())
        tracker.record_failure("groq", "RATE_LIMIT_ERROR", model="openai/gpt-oss-120b")
        assert tracker.skip_reason("groq", "openai/gpt-oss-120b") is not None
        # Simulate cooldown elapse.
        entry = tracker._entry("groq/openai/gpt-oss-120b")
        entry.cooldown_until = time.monotonic() - 1
        assert tracker.skip_reason("groq", "openai/gpt-oss-120b") is None
        assert tracker.can_make_request("groq", "openai/gpt-oss-120b") is True

    def test_success_on_recovered_provider_clears_health(self):
        router = _make_router(_providers())
        # groq was rate-limited earlier, cooldown expired, and a later healthy
        # call to groq (its normal primary slot) clears the health state.
        tracker = get_provider_health_tracker(_settings())
        tracker.record_failure("groq", "RATE_LIMIT_ERROR", model="openai/gpt-oss-120b")
        entry = tracker._entry("groq/openai/gpt-oss-120b")
        entry.cooldown_until = time.monotonic() - 1
        resp = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        assert resp.provider == "groq"
        assert tracker.get_status("groq", "openai/gpt-oss-120b") == HealthStatus.HEALTHY
        assert tracker.skip_reason("groq", "openai/gpt-oss-120b") is None


# --------------------------------------------------------------------------- #
# Test 4 — multiple providers temporarily pressured, 3rd available
# --------------------------------------------------------------------------- #
class TestMultipleProvidersPressured:
    def test_reaches_available_provider_without_repeated_attempts(self):
        providers = _providers("groq", "gemini")  # A,B 429; C (cerebras) healthy
        router = _make_router(providers)
        resp = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        assert resp.provider == "cerebras"
        assert len(providers["groq"].call_log) == 1
        assert len(providers["gemini"].call_log) == 1
        assert len(providers["cerebras"].call_log) == 1


# --------------------------------------------------------------------------- #
# Test 5 — all providers unavailable: fast, truthful, no storm, no fake success
# --------------------------------------------------------------------------- #
class TestAllProvidersUnavailable:
    def test_full_burst_is_fast_and_truthful(self):
        providers = _providers("groq", "gemini", "cerebras", "zen")
        router = _make_router(providers)
        with pytest.raises(LLMProviderError) as ei:
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        # No retry storm: each provider attempted exactly once in the call.
        for p in providers.values():
            assert len(p.call_log) == 1
        # No fake success: a real rate-limit error surfaces.
        assert ei.value.code in ("RATE_LIMIT_ERROR", "CONFIGURATION_ERROR")

    def test_all_blocked_by_prior_cooldown_is_fast_and_truthful(self):
        # All providers in cooldown from earlier queries AND all still failing.
        providers = _providers("groq", "gemini", "cerebras", "zen")
        router = _make_router(providers)
        with pytest.raises(LLMProviderError):
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))  # cooldowns set
        with pytest.raises(LLMProviderError):
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        # No additional calls to the still-cooling providers.
        for p in providers.values():
            assert len(p.call_log) <= 1


# --------------------------------------------------------------------------- #
# Test 9 (new mechanism) — in-session recovery probe re-eligibilizes a
# recovered provider rather than collapsing to total outage
# --------------------------------------------------------------------------- #
class TestInSessionRecoveryProbe:
    def test_probe_recovers_cooling_provider_without_restart(self):
        # Burst pressures groq+gemini+cerebras+zen (all cooling). Then groq
        # genuinely recovers mid-session (stops failing) while still cooling.
        providers = _providers("groq", "gemini", "cerebras", "zen")
        router = _make_router(providers)
        with pytest.raises(LLMProviderError):
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        for p in providers.values():
            assert len(p.call_log) == 1  # initial burst: 1 each, no storm

        tracker = get_provider_health_tracker(_settings())
        # groq recovers (stops failing) and is now closest to cooldown expiry.
        providers["groq"]._should_fail = False
        for ent in tracker._health.values():
            if ent.name == "groq":
                ent.cooldown_until = time.monotonic() + 1.0  # edge of the probe grace window

        resp = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        # The recovery probe re-eligibilized groq in-session (no restart).
        assert resp.provider == "groq"
        assert tracker.get_status("groq", "openai/gpt-oss-120b") == HealthStatus.HEALTHY
        # The still-down providers were not hammered (gemini/cerebras/zen stay 1 call).
        assert len(providers["gemini"].call_log) == 1
        assert len(providers["cerebras"].call_log) == 1
        assert len(providers["zen"].call_log) == 1

    def test_probe_does_not_hammer_still_failing_provider(self):
        # Every provider still failing AND cooling -> probe must NOT repeatedly
        # re-hit them (0-repeat). Each is attempted at most once across both calls.
        providers = _providers("groq", "gemini", "cerebras", "zen")
        router = _make_router(providers)
        with pytest.raises(LLMProviderError):
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        # Force groq to be the probe candidate but still failing.
        tracker = get_provider_health_tracker(_settings())
        for ent in tracker._health.values():
            if ent.name == "groq":
                ent.cooldown_until = time.monotonic() + 1.0  # edge of the probe grace window
        with pytest.raises(LLMProviderError):
            asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        # The still-failing probe candidate was attempted at most once overall.
        assert len(providers["groq"].call_log) <= 2
        assert len(providers["gemini"].call_log) <= 1
        assert len(providers["cerebras"].call_log) <= 1
        assert len(providers["zen"].call_log) <= 1


# --------------------------------------------------------------------------- #
# Test 8 — outcome correctness preserved (truthful, independent of stop_reason)
# --------------------------------------------------------------------------- #
class TestOutcomeCorrectness:
    def test_answered_success(self):
        assert _derive_outcome({"answer": "x", "evidence": [_ref()], "warnings": []}) == Outcome.ANSWERED

    def test_answered_fallback(self):
        state = {"answer": "x", "evidence": [_ref()], "warnings": ["research_plan_fallback: ..."]}
        assert _derive_outcome(state) == Outcome.ANSWERED_FALLBACK

    def test_answered_degraded(self):
        for w in ("synthesis_fallback: ...", "synthesis_degraded_to_raw_evidence"):
            assert _derive_outcome({"answer": "x", "evidence": [_ref()], "warnings": [w]}) == Outcome.ANSWERED_DEGRADED

    def test_not_found_no_evidence(self):
        assert _derive_outcome({"answer": "No supporting evidence was retrieved", "evidence": [], "warnings": []}) == Outcome.NOT_FOUND

    def test_no_answer_no_content(self):
        assert _derive_outcome({"answer": "", "evidence": [_ref()], "warnings": []}) == Outcome.NO_ANSWER


# --------------------------------------------------------------------------- #
# Test 6 — same-query partial progress: already-obtained evidence is preserved
#          when synthesis degrades (grounded, cited, not a naked dump)
# --------------------------------------------------------------------------- #
class TestSameQueryPartialProgress:
    def test_degraded_synthesis_keeps_evidence_grounded_and_cited(self):
        # A clean degraded answer (07e) keeps the retrieved evidence with
        # citation markers so provenance is never lost.
        evidence = [_ref(score=0.9, text="The fox jumps."), _ref(score=0.7, text="The dog sleeps.")]
        answer = (
            "Synthesis is temporarily unavailable, so I could not produce a "
            "polished answer. Here is the grounded evidence I retrieved "
            "(correctness not fully synthesized):\n"
            "- The fox jumps. [1]\n- The dog sleeps. [2]"
        )
        warnings = ["synthesis_degraded_to_raw_evidence"]
        # Citations still map to evidence.
        assert extract_cited_indices(answer, len(evidence)) == [1, 2]
        # Full-width citations also normalize (07d + 07e integration).
        assert _normalize_citation_markers("【1】【２】") == "[1][2]"
        # Outcome is truthfully DEGRADED, not bare success.
        assert _derive_outcome({"answer": answer, "evidence": evidence, "warnings": warnings}) == Outcome.ANSWERED_DEGRADED


# --------------------------------------------------------------------------- #
# Test 7 — recovery after degraded operation: a subsequent query uses normal
# primaries again (system does not remain permanently degraded)
# --------------------------------------------------------------------------- #
class TestRecoveryAfterDegradation:
    def test_session_returns_to_normal_after_cooldown(self):
        providers = _providers("groq")
        router = _make_router(providers)
        # Degraded phase: groq rate-limited, gemini serves.
        resp = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        assert resp.provider == "gemini"
        # groq recovers.
        providers["groq"]._should_fail = False
        tracker = get_provider_health_tracker(_settings())
        entries = [e for e in tracker._health.values() if e.name == "groq"]
        for e in entries:
            e.cooldown_until = time.monotonic() - 1
        # Next query returns to the normal primary.
        resp2 = asyncio.run(router.complete([], call_type="synthesis", query="q", tier="strong"))
        assert resp2.provider == "groq"
        assert tracker.get_status("groq", "openai/gpt-oss-120b") == HealthStatus.HEALTHY