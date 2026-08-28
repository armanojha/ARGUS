"""Orchestration Extension Contracts (Phase 06, 10).

Defines extension points for the orchestration layer that later phases
extend. Phase 02 provides the base orchestration loop; Phases 06 and 10
extend it with stopping logic and multi-agent support respectively.

These contracts define the interfaces without implementing the logic.
(Phase 08 memory integration is defined canonically in `app/memory/interfaces.py`.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from app.orchestration.state import OrchestrationState

# =============================================================================
# Phase 06: Stopping Logic Contract
# =============================================================================

class StopCondition(str, Enum):
    """Standardized stop conditions for the orchestration loop.

    Phase 06 implements all 5 conditions from V2 §5.4.
    """
    CLAIMS_SUPPORTED = "claims_supported"           # Claims supported above threshold
    NO_UNRESOLVED_CONTRADICTION = "no_unresolved_contradiction"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NEGLIGIBLE_EVIDENCE_GAIN = "negligible_evidence_gain"
    USER_EARLY_STOP = "user_early_stop"


@dataclass(frozen=True)
class StopDecision:
    """Result of a stop condition check."""
    should_stop: bool
    condition: StopCondition | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StopConditionChecker(ABC):
    """Interface for individual stop condition checkers.

    Phase 06 implements all 5 checkers. The orchestration loop
    composes them and stops if ANY returns should_stop=True.
    """

    @property
    @abstractmethod
    def condition(self) -> StopCondition:
        """The stop condition this checker evaluates."""
        ...

    @abstractmethod
    async def check(self, state: OrchestrationState) -> StopDecision:
        """Evaluate the stop condition against current state."""
        ...


class StoppingLogicInterface(ABC):
    """Interface for the complete stopping logic.

    Phase 06 implements this. The orchestration graph calls
    `should_stop()` after each assess node.
    """

    @abstractmethod
    async def should_stop(self, state: OrchestrationState) -> StopDecision:
        """Check all stop conditions. Returns first matching condition."""
        ...

    @abstractmethod
    def get_checkers(self) -> list[StopConditionChecker]:
        """Return all registered stop condition checkers."""
        ...


# =============================================================================
# Phase 10: Multi-Agent Contract
# =============================================================================

class AgentRole(str, Enum):
    """Specialized agent roles (V2 §10, V3 §10)."""
    RESEARCHER = "researcher"           # Builds evidence/candidate claims
    SKEPTIC = "skeptic"                 # Attacks strongest conclusion
    ALTERNATIVE_HYPOTHESIS = "alternative_hypothesis"  # Competing explanations
    VERIFIER = "verifier"               # Claim-to-evidence support (reuse Phase 04)
    JUDGE = "judge"                     # Resolves disagreement, constructs final answer


class AgentActivationRule(str, Enum):
    """Rules for when agents activate."""
    ALWAYS = "always"                           # Verifier always active
    HIGH_STAKES = "high_stakes"                 # Skeptic/Alt-hypothesis for high-stakes
    CONFLICTING_EVIDENCE = "conflicting_evidence"  # When evidence conflicts
    HIGH_UNCERTAINTY = "high_uncertainty"       # When uncertainty is high


@dataclass(frozen=True)
class AgentMessage:
    """Message passed between agents."""
    from_agent: AgentRole
    content: str
    to_agent: AgentRole | None = None  # None = broadcast
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class AgentInterface(ABC):
    """Interface for a specialized agent.

    Phase 10 implements concrete agents. The orchestration graph
    coordinates them via this interface.
    """

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The agent's role."""
        ...

    @property
    @abstractmethod
    def activation_rules(self) -> list[AgentActivationRule]:
        """When this agent should activate."""
        ...

    @abstractmethod
    async def process(
        self,
        state: OrchestrationState,
        messages: list[AgentMessage],
    ) -> list[AgentMessage]:
        """Process incoming messages and produce outgoing messages."""
        ...


class AgentCoordinatorInterface(ABC):
    """Interface for coordinating multi-agent debate.

    Phase 10 implements this. The orchestration graph uses this
    to coordinate agents for high-risk questions.
    """

    @abstractmethod
    async def run_debate(
        self,
        state: OrchestrationState,
        agents: list[AgentInterface] | None = None,
        max_rounds: int | None = None,
    ) -> OrchestrationState:
        """Run a multi-agent debate and return updated state."""
        ...

    @abstractmethod
    def should_activate_agents(self, state: OrchestrationState) -> list[AgentRole]:
        """Determine which agents should activate for this state."""
        ...


# =============================================================================
# Extended Orchestration State (for phases that extend it)
# =============================================================================

class ExtendedOrchestrationStateMixin:
    """Mixin for phases that extend OrchestrationState.

    Phases 06, 08, 10 can add fields to the state by composing
    this mixin rather than modifying the core TypedDict.
    """

    # Phase 06: Stopping logic extensions
    stop_conditions_checked: dict[str, bool] = field(default_factory=dict)
    stop_decisions: list[dict[str, Any]] = field(default_factory=list)

    # Phase 08: Memory integration
    memory_context: list[dict[str, Any]] = field(default_factory=list)
    memory_retrieval_queries: list[str] = field(default_factory=list)

    # Phase 10: Multi-agent
    agent_messages: list[dict[str, Any]] = field(default_factory=list)
    agent_round: int = 0
    debate_active: bool = False
    disagreement_detected: bool = False


# =============================================================================
# Factory for creating extended orchestration components
# =============================================================================

class OrchestrationExtensionFactory(ABC):
    """Factory for creating phase-specific orchestration extensions.

    Each phase implements this to provide its extensions.
    """

    @abstractmethod
    def create_stopping_logic(self) -> StoppingLogicInterface | None:
        """Create Phase 06 stopping logic, or None if not implemented."""
        ...

    @abstractmethod
    def create_memory_store(self) -> Any | None:  # MemoryStoreInterface
        """Create Phase 08 memory store, or None if not implemented."""
        ...

    @abstractmethod
    def create_agent_coordinator(self) -> Any | None:  # AgentCoordinatorInterface
        """Create Phase 10 agent coordinator, or None if not implemented."""
        ...


# Default factory that returns None for all extensions
class DefaultOrchestrationExtensionFactory(OrchestrationExtensionFactory):
    def create_stopping_logic(self) -> None:
        return None

    def create_memory_store(self) -> None:
        return None

    def create_agent_coordinator(self) -> None:
        return None


# Global factory instance (phases replace this)
_orchestration_extension_factory: OrchestrationExtensionFactory = DefaultOrchestrationExtensionFactory()


def get_orchestration_extension_factory() -> OrchestrationExtensionFactory:
    return _orchestration_extension_factory


def set_orchestration_extension_factory(factory: OrchestrationExtensionFactory) -> None:
    global _orchestration_extension_factory
    _orchestration_extension_factory = factory


__all__ = [
    "AgentActivationRule",
    "AgentCoordinatorInterface",
    "AgentInterface",
    "AgentMessage",
    # Phase 10
    "AgentRole",
    "DefaultOrchestrationExtensionFactory",
    # Extensions
    "ExtendedOrchestrationStateMixin",
    "OrchestrationExtensionFactory",
    # Phase 06
    "StopCondition",
    "StopConditionChecker",
    "StopDecision",
    "StoppingLogicInterface",
    "get_orchestration_extension_factory",
    "set_orchestration_extension_factory",
]