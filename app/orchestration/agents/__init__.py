"""Multi-Agent Challenge (Phase 10).

Specialized agent roles for high-risk/high-uncertainty questions:
- Researcher: builds evidence/candidate claims
- Skeptic: attacks strongest conclusion
- AlternativeHypothesis: competing explanations
- Verifier: claim-to-evidence support (reuse Phase 04)
- Judge: resolves disagreement, constructs final answer

Cross-model disagreement triggers targeted retrieval before final synthesis.
"""

from __future__ import annotations

from app.orchestration.agents.agents import (
    AlternativeHypothesisAgent,
    JudgeAgent,
    ResearcherAgent,
    SkepticAgent,
    VerifierAgent,
)
from app.orchestration.agents.coordinator import AgentCoordinator, create_agent_coordinator

__all__ = [
    "AgentCoordinator",
    "AlternativeHypothesisAgent",
    "JudgeAgent",
    "ResearcherAgent",
    "SkepticAgent",
    "VerifierAgent",
    "create_agent_coordinator",
]