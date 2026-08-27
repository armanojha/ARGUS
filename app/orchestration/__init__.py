"""Agentic RAG orchestration (Phase 02).

Public surface: `run_query()` runs the full query -> plan -> retrieve ->
synthesize loop over the Phase 01 hybrid retriever and the Phase 00.3
LLM gateway. Application code (the API layer) should import from here,
not from `graph`/`nodes` directly.
"""

from __future__ import annotations

from app.orchestration.graph import build_graph, run_query
from app.orchestration.models import (
    ComplexityLevel,
    EvidenceAssessment,
    OrchestrationCitation,
    OrchestrationResult,
    QueryAnalysis,
    ResearchPlan,
    StopReason,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.stopping import (
    AdaptiveStoppingLogic,
    BudgetExhaustedChecker,
    ClaimsSupportedChecker,
    NegligibleEvidenceGainChecker,
    NoUnresolvedContradictionChecker,
    UserEarlyStopChecker,
    build_stopping_logic,
    stop_condition_to_reason,
)

__all__ = [
    "AdaptiveStoppingLogic",
    "BudgetExhaustedChecker",
    "ClaimsSupportedChecker",
    "ComplexityLevel",
    "EvidenceAssessment",
    "NegligibleEvidenceGainChecker",
    "NoUnresolvedContradictionChecker",
    "OrchestrationCitation",
    "OrchestrationResult",
    "OrchestrationState",
    "QueryAnalysis",
    "ResearchPlan",
    "StopReason",
    "UserEarlyStopChecker",
    "build_graph",
    "build_stopping_logic",
    "run_query",
    "stop_condition_to_reason",
]
