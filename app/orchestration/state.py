"""LangGraph state schema for the Agentic RAG loop (Phase 02).

A single `TypedDict` shared across all graph nodes. No checkpointer is
configured (see `graph.py`), so values are held as real Python/Pydantic
objects for the lifetime of one `run_query()` call — nothing here is
serialized, and nothing here is a database of record.

Node functions return partial-update dicts; LangGraph applies
last-write-wins per key by default (no custom reducers), so nodes that
extend a list (evidence, issued subqueries, warnings) must read the
current value and return the full new list rather than a delta.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from app.evidence.models import EvidenceRef
from app.orchestration.models import QueryAnalysis, ResearchPlan


class OrchestrationState(TypedDict):
    # Immutable for the run
    request_id: str | None
    query: str
    max_iterations: int
    token_budget: int

    # Set by analyze/plan nodes
    query_analysis: QueryAnalysis | None
    plan: ResearchPlan | None

    # Retrieval bookkeeping
    pending_subquestions: list[str]
    issued_subqueries: list[str]
    evidence: list[EvidenceRef]  # deduped by chunk_id, most-recent score wins
    consecutive_empty_retrievals: int

    # Loop control
    iteration: int
    tokens_used: int
    sufficient: bool
    stop_reason: str | None

    # Output
    answer: str | None
    warnings: list[str]

    # Phase 06 adaptive policy additions (not required by Phase 02 callers;
    # the graph's `_initial_state` always populates them)
    question_pattern: NotRequired[str | None]
    retrieval_gain_history: NotRequired[list[float]]
    user_early_stop: NotRequired[bool]
    contradiction_signals: NotRequired[list[dict]]
    stop_conditions_checked: NotRequired[list[dict]]
    stop_condition_fired: NotRequired[str | None]
    evidence_tasks: NotRequired[list[dict]]

    # Phase 10 multi-agent additions (not required by Phase 02/06/08 callers;
    # the graph's `_initial_state` always populates them when multi-agent is enabled)
    agent_messages: NotRequired[list[dict[str, Any]]]
    agent_round: NotRequired[int]
    debate_active: NotRequired[bool]
    disagreement_detected: NotRequired[bool]

    # Phase 06.5.3 fast-path marker: set when a simple query skips the
    # analyze/plan/assess nodes and runs a single retrieve -> synthesize pass.
    fast_path: NotRequired[bool]
