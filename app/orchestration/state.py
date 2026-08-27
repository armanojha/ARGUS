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

from typing import TypedDict

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
