"""Phase 07d ("Verification & Outcome Hardening") tests.

Covers the four concrete defects found in the Phase 07c evaluation:

Part 1 (HARDEN-07d.1) — Verification-gate calibration:
  * gate reads the *grounding* score (top evidence), not the diluted top-K mean
  * simple/low-risk, well-grounded, non-conflicting evidence still skips
  * high-risk / conflicting / weakly-grounded evidence is still verified
    (quality is never un-protected)

Part 2 (HARDEN-07d.2) — Truthful outcome signal:
  * `Outcome` is derived from what was actually delivered and is independent of
    the control-flow `stop_reason`
  * success / degraded / fallback / not-found / no-answer are distinguishable
  * a healthy run that stopped for the budget is still "answered"

Part 3 (HARDEN-07d.3) — Citation normalization:
  * full-width markers 【N】 and full-width digits are normalized to ASCII
  * duplicates collapse, invalid/out-of-range markers are not presented as valid

Part 4 (HARDEN-07d.4) — Fast-fail on rate-limit:
  * HTTP 429 raises immediately (no redundant backoff retry on the same
    endpoint) so the router's health-cooldown + fallback acts promptly.

All tests are deterministic — no network, no live quota.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest

from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.llm_gateway.providers.exceptions import ProviderUnavailableError, RateLimitError
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider
from app.orchestration.agents.coordinator import should_skip_verification
from app.orchestration.graph import _derive_outcome
from app.orchestration.models import Outcome, ResearchPlan, StopReason
from app.orchestration.nodes import _normalize_citation_markers, extract_cited_indices


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ref(score: float, text: str = "Some supporting text.") -> EvidenceRef:
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


def _settings(**overrides) -> Settings:
    base = {
        "multimodel_enabled": True,
        "verification_enabled": True,
        "orchestration_llm_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# Part 1 — verification-gate calibration (HARDEN-07d.1)
# --------------------------------------------------------------------------- #
class TestVerificationGateCalibration:
    def test_empty_evidence_never_skips(self):
        assert should_skip_verification(_plan("low"), [], _settings()) is False

    def test_high_risk_plan_never_skips_even_with_high_scores(self):
        evidence = [_ref(0.95), _ref(0.93), _ref(0.94)]
        assert should_skip_verification(_plan("high"), evidence, _settings()) is False

    def test_low_grounding_score_verifies_even_if_mean_looks_high(self):
        # A diluted high mean must not win: the answer isn't well-grounded on
        # its top chunk, so verification fires. (Old gate would have mis-skipped.)
        evidence = [_ref(0.2), _ref(0.99), _ref(0.98), _ref(0.97)]
        assert should_skip_verification(_plan("low"), evidence, _settings()) is False

    def test_low_risk_well_grounded_nonconflicting_skips(self):
        # The genuine "simple / low-risk, well-grounded" case the gate exists
        # to fast-path: one clear, strong grounding chunk, low score spread.
        evidence = [_ref(0.95), _ref(0.93), _ref(0.94)]
        assert should_skip_verification(_plan("low"), evidence, _settings()) is True

    def test_conflicting_spread_verifies_despite_high_top1(self):
        # A high grounding score alone must NOT un-protect a conflicting base.
        evidence = [_ref(0.95), _ref(0.2), _ref(0.1), _ref(0.3)]
        assert should_skip_verification(_plan("low"), evidence, _settings()) is False

    def test_verify_threshold_is_on_grounding_scale(self):
        # Top-1 >= threshold but below it -> verify; the top score is compared
        # against an absolute [0,1] confidence threshold (the ground-truth).
        evidence = [_ref(0.79), _ref(0.78), _ref(0.77)]  # top1 0.79 < 0.8
        assert should_skip_verification(_plan("low"), evidence, _settings()) is False
        evidence_ok = [_ref(0.82), _ref(0.81), _ref(0.83)]
        assert should_skip_verification(_plan("low"), evidence_ok, _settings()) is True


# --------------------------------------------------------------------------- #
# Part 2 — truthful outcome signal (HARDEN-07d.2)
# --------------------------------------------------------------------------- #
def _state(*, evidence=("_unset",), answer="Foxes jump over dogs [1].", warnings=None):
    # Default: a non-empty evidence base. Explicit `[]` stays empty.
    evidence = [_ref(0.9)] if evidence == ("_unset",) else evidence
    return {
        "query": "q",
        "request_id": "r",
        "plan": _plan(),
        "evidence": evidence,
        "answer": answer,
        "stop_reason": "sufficient_evidence",
        "iteration": 1,
        "issued_subqueries": [],
        "tokens_used": 5,
        "warnings": warnings or [],
    }


class TestTruthfulOutcome:
    def test_normal_answer_is_answered(self):
        assert _derive_outcome(_state()) is Outcome.ANSWERED

    def test_healthy_run_stopped_by_budget_is_still_answered(self):
        # stop_reason is control-flow; a budget-stopped run that delivered a
        # grounded answer is a SUCCESS, not a failure (07c mislabeled A1).
        st = _state()
        st["stop_reason"] = StopReason.BUDGET_EXHAUSTED.value
        assert _derive_outcome(st) is Outcome.ANSWERED

    def test_no_evidence_no_answer_is_not_found(self):
        st = _state(evidence=[], answer="No supporting evidence was retrieved for this question.")
        assert _derive_outcome(st) is Outcome.NOT_FOUND

    def test_evidence_present_but_empty_answer_is_no_answer(self):
        st = _state(answer="")
        assert _derive_outcome(st) is Outcome.NO_ANSWER

    def test_synthesis_degraded_is_answered_degraded(self):
        st = _state(warnings=["synthesis_degraded_to_raw_evidence"])
        assert _derive_outcome(st) is Outcome.ANSWERED_DEGRADED

    def test_synthesis_fallback_is_answered_degraded(self):
        st = _state(warnings=["synthesis_fallback: boom"])
        assert _derive_outcome(st) is Outcome.ANSWERED_DEGRADED

    def test_plan_fallback_is_answered_fallback_not_failure(self):
        # Fallback success still counts as success (07d.2): an answer was
        # grounded and cited even though the primary plan path degraded.
        st = _state(warnings=["research_plan_fallback: boom"])
        assert _derive_outcome(st) is Outcome.ANSWERED_FALLBACK

    def test_hard_degraded_zero_call_still_reported_truthfully(self):
        # 07c: G1/I1/J1 hard-degraded 0-call queries wrongly looked like success
        # under stop_reason="no_unresolved_contradiction". With no answer, the
        # outcome must be truthful (NO_ANSWER / NOT_FOUND), never "answered".
        st = _state(
            evidence=[],
            answer="",
            warnings=["research_plan_fallback: boom"],
        )
        obj = _derive_outcome(st)
        assert obj is not Outcome.ANSWERED
        assert obj in (Outcome.NO_ANSWER, Outcome.NOT_FOUND)


# --------------------------------------------------------------------------- #
# Part 3 — citation normalization (HARDEN-07d.3)
# --------------------------------------------------------------------------- #
class TestCitationNormalization:
    def test_ascii_markers_unchanged(self):
        assert extract_cited_indices("x [1] y [2]", evidence_count=3) == [1, 2]

    def test_fullwidth_brackets_are_normalized(self):
        # The 07c defect: 【N】 markers silently unmatched -> top-3 fallback.
        assert extract_cited_indices("x 【1】 y 【2】", evidence_count=3) == [1, 2]

    def test_fullwidth_digits_are_normalized(self):
        assert extract_cited_indices("x [１] y [２]", evidence_count=3) == [1, 2]

    def test_mixed_ascii_and_fullwidth(self):
        assert extract_cited_indices("[1] and 【2】", evidence_count=3) == [1, 2]

    def test_duplicates_collapse_first_seen(self):
        assert extract_cited_indices("[2] then again [2]", evidence_count=3) == [2]
        assert extract_cited_indices("【3】【1】【3】", evidence_count=3) == [3, 1]

    def test_out_of_range_and_invalid_markers_not_presented_as_valid(self):
        # Invalid markers are dropped, never surfaced as valid citations.
        result = extract_cited_indices("[1] [99] [0] [-1]", evidence_count=3)
        assert result == [1]

    def test_no_markers_returns_empty(self):
        assert extract_cited_indices("no citations at all", evidence_count=3) == []

    def test_normalization_helper(self):
        assert _normalize_citation_markers("【２】【3】") == "[2][3]"
        assert _normalize_citation_markers("plain text [1]") == "plain text [1]"


# --------------------------------------------------------------------------- #
# Part 4 — fast-fail on rate-limit (HARDEN-07d.4)
# --------------------------------------------------------------------------- #
class DummyProvider(OpenAICompatibleProvider):
    pass


class TestFastFailOnRateLimit:
    def test_429_raises_immediately_without_backoff_retry(self):
        """A 429 must surface promptly so the router's cooldown + fallback can
        act, instead of burning time on a redundant same-endpoint backoff."""

        calls = {"n": 0}

        def _serve(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                status_code=429,
                json={"error": {"message": "rate limited"}},
                request=request,
            )

        provider = DummyProvider(
            name="dummy",
            base_url="https://example.invalid/v1",
            default_model="test-model",
            api_key="sk-test-not-a-real-key",
        )
        provider._client = httpx.AsyncClient(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(_serve),
        )
        with pytest.raises(RateLimitError):
            asyncio.run(provider._request_with_retry({}, timeout=5.0))
        # Exactly one attempt — the redundant backoff retry is gone.
        assert calls["n"] == 1, f"expected a single attempt, got {calls['n']}"

    def test_500_still_retries(self):
        """5xx server errors keep the Phase 07 bounded-retry behavior."""
        calls = {"n": 0}

        def _serve(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status_code=503, json={}, request=request)

        provider = DummyProvider(
            name="dummy",
            base_url="https://example.invalid/v1",
            default_model="test-model",
            api_key="sk-test-not-a-real-key",
            max_retries=1,
        )
        provider._client = httpx.AsyncClient(
            base_url="https://example.invalid/v1",
            transport=httpx.MockTransport(_serve),
        )
        with pytest.raises(ProviderUnavailableError):
            asyncio.run(provider._request_with_retry({}, timeout=5.0))
        assert calls["n"] > 1, "5xx errors should still be retried"