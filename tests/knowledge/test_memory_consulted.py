"""Tests for surfacing "memory consulted" in the orchestration result.

Verifies the additive traceability: when the memory-enhance node actually
influenced the plan, the final OrchestrationResult carries ``memory_consulted``
listing the derived-knowledge layers, clearly distinct from document evidence.
"""

from __future__ import annotations

from app.orchestration.graph import _build_result
from app.orchestration.models import (
    ResearchPlan,
    StopReason,
)


def _evidence_ref(store, document, chunk):
    return store._evidence_ref(document, chunk)


def _make_state(memory_consulted=None):
    state = {
        "request_id": "req-test",
        "query": "What did Acme acquire?",
        "plan": ResearchPlan(
            objective="Determine Acme's acquisitions.",
            entities=["Acme"],
            subquestions=["Which companies did Acme acquire?"],
        ),
        "answer": "Acme acquired Beta.",
        "citations": [],
        "evidence": [],
        "iteration": 1,
        "issued_subqueries": [],
        "tokens_used": 100,
        "stop_reason": StopReason.SUFFICIENT_EVIDENCE.value,
        "warnings": [],
    }
    if memory_consulted is not None:
        state["memory_consulted"] = memory_consulted
    return state


def test_memory_consulted_surfaces_layers_when_present():
    result = _build_result(_make_state(memory_consulted=["long_term_knowledge", "user_memory"]))
    assert result.memory_consulted == ["long_term_knowledge", "user_memory"]


def test_memory_consulted_empty_when_not_set():
    result = _build_result(_make_state())
    assert result.memory_consulted == []


def test_memory_consulted_is_distinct_from_citations():
    state = _make_state(memory_consulted=["long_term_knowledge"])
    result = _build_result(state)
    assert result.memory_consulted
    # Derived memory is reported separately; it never masquerades as document evidence.
    assert result.citations == []