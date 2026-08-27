"""Full Stopping Logic (Phase 06).

Implements the `StoppingLogicInterface` shared contract from
`app/orchestration/contracts.py`: all 5 stop conditions from V2 §5.4,
each independently evaluable, composed in priority order by
`AdaptiveStoppingLogic`.

Conditions (V2 §5.4):
1. CLAIMS_SUPPORTED          — claims supported above threshold.
2. NO_UNRESOLVED_CONTRADICTION — no unresolved critical contradiction remains.
3. BUDGET_EXHAUSTED          — iteration/token ceiling reached (hard limits).
4. NEGLIGIBLE_EVIDENCE_GAIN  — new-evidence gain is negligible.
5. USER_EARLY_STOP           — explicit caller-requested stop.

Every checker is deterministic — no LLM call — so a gateway failure can
never hang the loop, and each decision is traceable to state (evidence
scores, budget fields, gain history, contradiction signals).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.logging_config import get_logger
from app.orchestration.contracts import (
    StopCondition,
    StopConditionChecker,
    StopDecision,
    StoppingLogicInterface,
)
from app.orchestration.models import StopReason
from app.orchestration.state import OrchestrationState

logger = get_logger("argus.orchestration.stopping")

# Condition evaluation order: explicit/hard signals first, then positive
# completion signals, then diminishing-returns. The composed logic stops as
# soon as ANY checker fires and reports that condition.
_DEFAULT_ORDER = [
    StopCondition.USER_EARLY_STOP,
    StopCondition.BUDGET_EXHAUSTED,
    StopCondition.NEGLIGIBLE_EVIDENCE_GAIN,
    StopCondition.CLAIMS_SUPPORTED,
    StopCondition.NO_UNRESOLVED_CONTRADICTION,
]

# Positive-completion conditions may only CONFIRM that the loop is done:
# they are gated behind the assessor's verdict. If the last assessment said
# more evidence is needed, these must not fire (the assessor is the arbiter
# of sufficiency; claims support / contradiction status alone cannot close
# a run the arbiter explicitly kept open). The gating lives inside each
# checker (require_assessor_agreement=True) so a checker stays safe even
# when used standalone, and the composition remains a plain priority loop.

class ClaimsSupportedChecker(StopConditionChecker):
    """Stop when claims are supported above the configured threshold.

    Support is estimated deterministically from the evidence already
    accumulated (top evidence scores) — no LLM. Callers may inject a
    richer `support_provider` (e.g. a verifier score feed) via the
    constructor.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        support_provider: Callable[[OrchestrationState], float] | None = None,
        require_assessor_agreement: bool = True,
    ) -> None:
        self.threshold = threshold
        self._support_provider = support_provider or self._default_support
        self.require_assessor_agreement = require_assessor_agreement

    @property
    def condition(self) -> StopCondition:
        return StopCondition.CLAIMS_SUPPORTED

    async def check(self, state: OrchestrationState) -> StopDecision:
        if self.require_assessor_agreement and state.get("sufficient") is not True:
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason="Deferred: assessor reports more evidence is needed.",
                metadata={
                    "support": round(self._support_provider(state), 4),
                    "threshold": self.threshold,
                    "deferred_by": "assessor_open",
                },
            )
        support = self._support_provider(state)
        if not state["evidence"]:
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason="No evidence accumulated yet; cannot confirm claim support.",
                metadata={"support": round(support, 4), "threshold": self.threshold},
            )
        if support >= self.threshold:
            return StopDecision(
                should_stop=True,
                condition=self.condition,
                reason=f"Claims supported above threshold ({support:.2f} >= {self.threshold}).",
                metadata={"support": round(support, 4), "threshold": self.threshold},
            )
        return StopDecision(
            should_stop=False,
            condition=self.condition,
            reason=f"Claims not yet supported ({support:.2f} < {self.threshold}).",
            metadata={"support": round(support, 4), "threshold": self.threshold},
        )

    @staticmethod
    def _default_support(state: OrchestrationState) -> float:
        evidence = state["evidence"]
        if not evidence:
            return 0.0
        top = sorted(evidence, key=lambda r: r.score, reverse=True)[:3]
        return min(1.0, sum(r.score for r in top) / len(top))


class NoUnresolvedContradictionChecker(StopConditionChecker):
    """Stop when no unresolved critical contradiction remains.

    Contradiction signals come from `state["contradiction_signals"]` by
    default (a list of dicts with `severity`/`resolved`/`critical`), or
    from an injected provider. Absence of contradiction data deterministically
    means "no unresolved contradiction" — the condition holds.
    """

    def __init__(
        self,
        contradictions_provider: Callable[[OrchestrationState], list[dict[str, Any]]] | None = None,
        critical_min_severity: float = 0.0,
        require_assessor_agreement: bool = True,
    ) -> None:
        self._provider = contradictions_provider or self._state_signals
        self.critical_min_severity = critical_min_severity
        self.require_assessor_agreement = require_assessor_agreement

    @property
    def condition(self) -> StopCondition:
        return StopCondition.NO_UNRESOLVED_CONTRADICTION

    async def check(self, state: OrchestrationState) -> StopDecision:
        if self.require_assessor_agreement and state.get("sufficient") is not True:
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason="Deferred: assessor reports more evidence is needed.",
                metadata={
                    "unresolved_contradictions": len(self._provider(state)),
                    "deferred_by": "assessor_open",
                },
            )
        signals = self._provider(state)
        unresolved = [
            s for s in signals
            if not s.get("resolved", False)
            and s.get("critical", True)
            and float(s.get("severity", 0.0)) >= self.critical_min_severity
        ]
        if unresolved:
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason=f"{len(unresolved)} unresolved critical contradiction(s) remain.",
                metadata={"unresolved_contradictions": len(unresolved), "signals": signals},
            )
        return StopDecision(
            should_stop=True,
            condition=self.condition,
            reason="No unresolved critical contradictions.",
            metadata={"unresolved_contradictions": 0, "signals_available": bool(signals)},
        )

    @staticmethod
    def _state_signals(state: OrchestrationState) -> list[dict[str, Any]]:
        return list(state.get("contradiction_signals") or [])


class BudgetExhaustedChecker(StopConditionChecker):
    """Stop when the hard iteration or token ceiling is reached.

    Mirrors the Phase 02 deterministic short-circuit so the policy always
    respects the configured ceilings regardless of planner proposals.
    """

    @property
    def condition(self) -> StopCondition:
        return StopCondition.BUDGET_EXHAUSTED

    async def check(self, state: OrchestrationState) -> StopDecision:
        iteration_limit = state["iteration"] >= state["max_iterations"]
        token_limit = state["tokens_used"] >= state["token_budget"]
        if iteration_limit or token_limit:
            return StopDecision(
                should_stop=True,
                condition=self.condition,
                reason=(
                    f"Budget exhausted (iterations {state['iteration']}/{state['max_iterations']}, "
                    f"tokens {state['tokens_used']}/{state['token_budget']})."
                ),
                metadata={"iterations": state["iteration"], "tokens": state["tokens_used"]},
            )
        return StopDecision(
            should_stop=False,
            condition=self.condition,
            reason="Within budget.",
            metadata={"iterations": state["iteration"], "tokens": state["tokens_used"]},
        )


class NegligibleEvidenceGainChecker(StopConditionChecker):
    """Stop when recent retrieval iterations produce negligible new evidence.

    Uses the per-iteration gain history recorded by the retrieve node
    (new evidence / prior total). Requires a couple of iterations to have
    run so a single flat retrieval is not misread as diminishing returns.
    """

    def __init__(self, threshold: float = 0.05, min_window: int = 2) -> None:
        self.threshold = threshold
        self.min_window = min_window

    @property
    def condition(self) -> StopCondition:
        return StopCondition.NEGLIGIBLE_EVIDENCE_GAIN

    async def check(self, state: OrchestrationState) -> StopDecision:
        history = list(state.get("retrieval_gain_history") or [])
        metadata = {"gain_history": history, "threshold": self.threshold}
        if state.get("pending_subquestions"):
            # A query is already queued (a fresh subquery or a targeted
            # evidence-seeking action): run it before declaring diminishing
            # returns, otherwise active re-retrieval could never execute.
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason=f"{len(state['pending_subquestions'])} queued query(ies) not yet tried.",
                metadata={**metadata, "pending": len(state["pending_subquestions"])},
            )
        if len(history) < self.min_window:
            return StopDecision(
                should_stop=False,
                condition=self.condition,
                reason=f"Gain history too short ({len(history)} < {self.min_window}).",
                metadata=metadata,
            )
        last_gain = history[-1]
        if last_gain <= self.threshold:
            return StopDecision(
                should_stop=True,
                condition=self.condition,
                reason=f"Negligible evidence gain ({last_gain} <= {self.threshold}).",
                metadata={**metadata, "last_gain": last_gain},
            )
        return StopDecision(
            should_stop=False,
            condition=self.condition,
            reason=f"Evidence gain still meaningful ({last_gain} > {self.threshold}).",
            metadata={**metadata, "last_gain": last_gain},
        )


class UserEarlyStopChecker(StopConditionChecker):
    """Stop when the caller explicitly requests an early stop."""

    @property
    def condition(self) -> StopCondition:
        return StopCondition.USER_EARLY_STOP

    async def check(self, state: OrchestrationState) -> StopDecision:
        if bool(state.get("user_early_stop", False)):
            return StopDecision(
                should_stop=True,
                condition=self.condition,
                reason="User requested an early stop.",
                metadata={},
            )
        return StopDecision(
            should_stop=False,
            condition=self.condition,
            reason="No user early-stop requested.",
            metadata={},
        )


class AdaptiveStoppingLogic(StoppingLogicInterface):
    """Composed stopping logic over the five V2 §5.4 condition checkers."""

    def __init__(self, checkers: list[StopConditionChecker] | None = None) -> None:
        # Evaluation order: given, else the documented default priority order.
        if checkers:
            self._checkers = checkers
        else:
            by_condition = {c.condition: c for c in self._default_checkers()}
            self._checkers = [by_condition[cond] for cond in _DEFAULT_ORDER if cond in by_condition]

    @staticmethod
    def _default_checkers() -> list[StopConditionChecker]:
        return [
            ClaimsSupportedChecker(),
            NoUnresolvedContradictionChecker(),
            BudgetExhaustedChecker(),
            NegligibleEvidenceGainChecker(),
            UserEarlyStopChecker(),
        ]

    def get_checkers(self) -> list[StopConditionChecker]:
        return list(self._checkers)

    async def should_stop(self, state: OrchestrationState) -> StopDecision:
        checked: list[dict[str, Any]] = []
        for checker in self._checkers:
            decision = await checker.check(state)
            checked.append({
                "condition": checker.condition.value,
                "evaluated": True,
                "should_stop": decision.should_stop,
                "reason": decision.reason,
                "metadata": decision.metadata,
            })
            if decision.should_stop:
                logger.info(
                    "stopping_condition_fired",
                    condition=checker.condition.value,
                    request_id=state.get("request_id"),
                )
                return StopDecision(
                    should_stop=True,
                    condition=checker.condition,
                    reason=decision.reason,
                    metadata={**decision.metadata, "checked": checked},
                )
        return StopDecision(
            should_stop=False,
            reason="No stop condition fired.",
            metadata={"checked": checked},
        )


def stop_condition_to_reason(condition: StopCondition | None) -> StopReason | None:
    """Map a Phase 06 stop condition to the Phase 02 `StopReason` enum."""
    if condition is None:
        return None
    mapping = {
        StopCondition.CLAIMS_SUPPORTED: StopReason.CLAIMS_SUPPORTED,
        StopCondition.NO_UNRESOLVED_CONTRADICTION: StopReason.NO_UNRESOLVED_CONTRADICTION,
        StopCondition.BUDGET_EXHAUSTED: StopReason.BUDGET_EXHAUSTED,
        StopCondition.NEGLIGIBLE_EVIDENCE_GAIN: StopReason.NEGLIGIBLE_EVIDENCE_GAIN,
        StopCondition.USER_EARLY_STOP: StopReason.USER_EARLY_STOP,
    }
    return mapping.get(condition)


def build_stopping_logic(
    settings: Settings,
    contradictions_provider: Callable[[OrchestrationState], list[dict[str, Any]]] | None = None,
) -> AdaptiveStoppingLogic:
    """Build the Phase 06 stopping logic from settings thresholds."""
    checkers = [
        UserEarlyStopChecker(),
        BudgetExhaustedChecker(),
        NegligibleEvidenceGainChecker(threshold=settings.stopping_evidence_gain_threshold),
        ClaimsSupportedChecker(threshold=settings.stopping_claim_support_threshold),
        NoUnresolvedContradictionChecker(contradictions_provider=contradictions_provider),
    ]
    return AdaptiveStoppingLogic(checkers)


__all__ = [
    "AdaptiveStoppingLogic",
    "BudgetExhaustedChecker",
    "ClaimsSupportedChecker",
    "NegligibleEvidenceGainChecker",
    "NoUnresolvedContradictionChecker",
    "UserEarlyStopChecker",
    "build_stopping_logic",
    "stop_condition_to_reason",
]