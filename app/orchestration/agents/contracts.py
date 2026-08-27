"""Agent Role Interfaces (Phase 10).

Defines interfaces for the multi-agent debate system.
Phase 10 implements these. Phase 07 (multi-model fabric) is required
for genuinely independent models per role.

These are CONTRACTS only - Phase 10 implements them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Agent Roles (V2 §10, V3 §10)
# =============================================================================

class AgentRole(str, Enum):
    """Specialized agent roles (V2 §10, V3 §10)."""
    RESEARCHER = "researcher"
    SKEPTIC = "skeptic"
    ALTERNATIVE_HYPOTHESIS = "alternative_hypothesis"
    VERIFIER = "verifier"
    JUDGE = "judge"


class AgentActivationRule(str, Enum):
    """Rules for when agents activate."""
    ALWAYS = "always"
    HIGH_STAKES = "high_stakes"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    EXPLICIT_REQUEST = "explicit_request"


@dataclass(frozen=True)
class AgentCapability:
    """Defines what an agent can do."""
    role: AgentRole
    description: str
    # Which call types this agent uses
    call_types: list[str] = field(default_factory=list)
    # Whether this agent needs a different model from others
    requires_independent_model: bool = True
    # Activation rules
    activation_rules: list[str] = field(default_factory=list)


# Default agent capabilities (V2 §10, V3 §10)
AGENT_CAPABILITIES: dict[str, dict[str, Any]] = {
    "researcher": {
        "role": "researcher",
        "description": "Builds evidence and candidate claims",
        "call_types": ["evidence_extraction", "research_planning"],
        "requires_independent_model": False,
        "activation_rules": ["always"],
    },
    "skeptic": {
        "role": "skeptic",
        "description": "Attacks strongest conclusion, finds weaknesses",
        "call_types": ["verification", "reasoning"],
        "requires_independent_model": True,
        "activation_rules": ["high_stakes", "conflicting_evidence"],
    },
    "alternative_hypothesis": {
        "role": "alternative_hypothesis",
        "description": "Generates competing explanations",
        "call_types": ["research_planning", "reasoning"],
        "requires_independent_model": True,
        "activation_rules": ["high_stakes", "conflicting_evidence", "high_uncertainty"],
    },
    "verifier": {
        "role": "verifier",
        "description": "Claim-to-evidence support verification",
        "call_types": ["verification"],
        "requires_independent_model": True,
        "activation_rules": ["always", "deep_research"],
    },
    "judge": {
        "role": "judge",
        "description": "Resolves disagreement, constructs final answer",
        "call_types": ["synthesis", "reasoning"],
        "requires_independent_model": True,
        "activation_rules": ["always"],
    },
}


# =============================================================================
# Agent Communication
# =============================================================================

class AgentMessageType(str, Enum):
    """Types of messages between agents."""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    CRITIQUE = "critique"
    ALTERNATIVE = "alternative"
    VERIFICATION = "verification"
    JUDGMENT = "judgment"
    QUESTION = "question"


@dataclass(frozen=True)
class AgentMessage:
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
    def capabilities(self) -> dict[str, Any]:
        """The agent's capabilities."""
        ...

    @property
    @abstractmethod
    def activation_rules(self) -> list[str]:
        """When this agent should activate."""
        ...

    @abstractmethod
    async def process(
        self,
        state: Any,  # OrchestrationState
        messages: list[Any],  # list[AgentMessage]
    ) -> list[Any]:  # list[AgentMessage]
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
        state: Any,  # OrchestrationState
        agents: list[Any],  # list[AgentInterface]
        max_rounds: int = 3,
    ) -> Any:  # OrchestrationState
        """Run a multi-agent debate and return updated state."""
        ...

    @abstractmethod
    def should_activate_agents(self, state: Any) -> list[str]:  # list[AgentRole]
        """Determine which agents should activate for this state."""
        ...

    @abstractmethod
    def get_agent(self, role: str) -> Any | None:  # AgentInterface | None
        """Get an agent by role."""
        ...


# =============================================================================
# Disagreement Detection
# =============================================================================

class DisagreementSignal(BaseModel):
    """Signal that agents disagree on a claim."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    agent_roles: list[str]
    positions: dict[str, str]  # role -> position summary
    severity: float = Field(ge=0.0, le=1.0)
    suggested_retrieval: str | None = None


class DisagreementDetectorInterface(ABC):
    """Interface for detecting agent disagreement.

    Phase 10 implements this. The orchestrator uses this to
    detect when agents disagree and trigger targeted retrieval.
    """

    @abstractmethod
    def detect_disagreement(
        self,
        messages: list[Any],  # list[AgentMessage]
        claim_id: str,
    ) -> DisagreementSignal | None:
        """Detect if agents disagree on a claim."""
        ...

    @abstractmethod
    def generate_retrieval_query(self, signal: DisagreementSignal) -> str:
        """Generate a targeted retrieval query from a disagreement signal."""
        ...


# =============================================================================
# Agent Factory
# =============================================================================

class AgentFactoryInterface(ABC):
    """Factory for creating agents.

    Phase 10 implements this. The orchestration graph uses this
    to create agents without depending on concrete implementations.
    """

    @abstractmethod
    def create_agent(self, role: str) -> Any | None:  # AgentInterface
        """Create an agent by role."""
        ...

    @abstractmethod
    def get_all_agents(self) -> list[Any]:  # list[AgentInterface]
        """Get all available agents."""
        ...

    @abstractmethod
    def get_agents_for_activation(self, rules: list[str]) -> list[Any]:
        """Get agents matching activation rules."""
        ...


class DefaultAgentFactory:
    """Default factory returning None (no agents)."""

    def create_agent(self, role: str) -> None:
        return None

    def get_all_agents(self) -> list:
        return []

    def get_agents_for_activation(self, rules: list[str]) -> list:
        return []


# Global factory instance
_agent_factory: Any = None


def get_agent_factory() -> Any:
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = DefaultAgentFactory()
    return _agent_factory


def set_agent_factory(factory: Any) -> None:
    global _agent_factory
    _agent_factory = factory


__all__ = [
    "AGENT_CAPABILITIES",
    "AgentActivationRule",
    "AgentCapability",
    "AgentCoordinatorInterface",
    "AgentFactoryInterface",
    "AgentInterface",
    "AgentMessage",
    "AgentMessageType",
    "AgentRole",
    "DefaultAgentFactory",
    "DisagreementDetectorInterface",
    "DisagreementSignal",
    "get_agent_factory",
    "set_agent_factory",
]