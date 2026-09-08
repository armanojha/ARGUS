"""Tests for ResearchStrategy snapshots and reasoning-trace materialization.

No LLM calls. No store access. Pure-function tests with fabricated states.
"""

from types import SimpleNamespace
from uuid import uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.graph.reasoning import NODE_TYPES, build_reasoning_trace
from app.orchestration.graph import _build_strategy_snapshot
from app.orchestration.models import ResearchStrategy


def _ref(source_path: str, text: str = "stub evidence") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(),
        source_path=source_path, source_type=SourceType.TEXT,
        text=text, score=0.9, rank=1,
    )


def _plan(**kw):
    base = {
        "objective": "Find Acme revenue",
        "entities": ["Acme"],
        "time_window": None,
        "subquestions": ["q1", "q2"],
        "evidence_type": "factual",
        "preferred_retrieval_methods": ["hybrid"],
        "required_sources": [],
        "risk_level": "low",
        "token_budget": 6000,
        "iteration_budget": 2,
        "stopping_condition": "stop",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _state(**kw):
    base = {
        "plan": _plan(), "pending_subquestions": ["q1", "q2"],
        "evidence": [], "evidence_tasks": [], "complexity_tier": "BALANCED",
        "iteration": 1, "max_iterations": 3, "token_budget": 6000,
        "tokens_used": 100,
    }
    base.update(kw)
    return base


class TestStrategySnapshot:
    def test_plan_node_marks_initial(self):
        snap = _build_strategy_snapshot("plan", _state(), {"plan": _plan()})
        assert snap is not None
        ResearchStrategy(**snap)  # validates schema
        assert snap["mutated_from"] == "initial"
        assert snap["retrieval_mode"] == "hybrid"
        assert snap["queries"] == ["q1", "q2"]

    def test_gap_detector_tasks_marked(self):
        snap = _build_strategy_snapshot(
            "assess", _state(),
            {"evidence_tasks": [{"suggested_query": "gap q"}], "pending_subquestions": ["q1", "q2"]},
        )
        assert snap["mutated_from"] == "gap_detector"

    def test_tier_change_marked(self):
        snap = _build_strategy_snapshot(
            "assess", _state(complexity_tier="DEEP"),
            {"complexity_tier": "BALANCED", "pending_subquestions": ["q1", "q2"]},
        )
        assert snap["mutated_from"] == "tier_adjustment"
        assert snap["verification_depth"] == 2

    def test_assessor_query_marked(self):
        snap = _build_strategy_snapshot(
            "assess", _state(),
            {"pending_subquestions": ["q1", "q2", "q3"]},
        )
        assert snap["mutated_from"] == "assessor"
        assert snap["queries"] == ["q1", "q2", "q3"]

    def test_terminal_marked(self):
        snap = _build_strategy_snapshot(
            "assess", _state(),
            {"sufficient": True, "stop_reason": "SUFFICIENT_EVIDENCE",
             "pending_subquestions": ["q1", "q2"]},
        )
        assert snap["mutated_from"] == "terminal:SUFFICIENT_EVIDENCE"
        assert snap["stop_reason"] == "SUFFICIENT_EVIDENCE"

    def test_source_priorities_ranked(self):
        refs = [_ref("/a.txt"), _ref("/b.txt"), _ref("/a.txt")]
        snap = _build_strategy_snapshot(
            "assess", _state(evidence=refs), {"pending_subquestions": []})
        assert snap["source_priorities"][0] == "/a.txt"

    def test_no_plan_returns_none(self):
        assert _build_strategy_snapshot("assess", {"plan": None}, {}) is None

    def test_budget_remaining(self):
        snap = _build_strategy_snapshot(
            "assess", _state(token_budget=6000, tokens_used=1500),
            {"pending_subquestions": []})
        assert snap["evidence_budget_tokens"] == 4500


def _result(**kw):
    base = {
        "query": "What was Acme revenue?",
        "plan": {
            "objective": "Find Acme revenue", "entities": ["Acme"],
            "time_window": None, "subquestions": ["sq1"],
            "preferred_retrieval_methods": ["hybrid"],
        },
        "answer": "Acme revenue was $4.7 billion [1]. It grew steadily [2].",
        "citations": [
            {"ref_id": 1, "text": "revenue $4.7 billion", "source_path": "/f.md",
             "score": 0.9, "section_path": None},
            {"ref_id": 2, "text": "steady growth", "source_path": "/e.md",
             "score": 0.8, "section_path": None},
        ],
        "sub_queries_issued": ["sq1"],
        "contradiction_signals": [
            {"conflict_type": "DIFFERENT_TIMEFRAME", "description": "2023 vs 2025",
             "confidence": "MEDIUM", "evidence_indices": [1, 2],
             "semantic_verified": True},
        ],
        "stop_reason": "SUFFICIENT_EVIDENCE", "outcome": "ANSWERED",
        "stop_condition": None, "iterations_used": 2, "warnings": [],
        "verification": {"status": "supported"},
    }
    base.update(kw)
    return base


class TestReasoningTrace:
    def test_all_nine_types_present(self):
        trace = build_reasoning_trace(_result())
        types = {n["node_type"] for n in trace["nodes"]}
        assert set(NODE_TYPES) <= types

    def test_claims_link_evidence(self):
        trace = build_reasoning_trace(_result())
        claims = [n for n in trace["nodes"] if n["node_type"] == "CLAIM"]
        assert len(claims) == 2
        sup = [e for e in trace["edges"] if e["edge_type"] == "supported_by"]
        assert any(e["source"].startswith("claim:") for e in sup)

    def test_counterclaim_rejected(self):
        trace = build_reasoning_trace(_result())
        counters = [n for n in trace["nodes"] if n["node_type"] == "COUNTERCLAIM"]
        assert len(counters) == 1
        rej = [e for e in trace["edges"] if e["edge_type"] == "rejected_alternative"]
        assert len(rej) == 1 and rej[0]["target"] == counters[0]["id"]
        contra = [e for e in trace["edges"] if e["edge_type"] == "contradicted_by"]
        assert len(contra) == 2  # evidence_indices [1, 2]

    def test_answer_traversal_chain(self):
        trace = build_reasoning_trace(_result())
        by_id = {n["id"]: n for n in trace["nodes"]}
        # Answer → Claim → Evidence → Query chain exists
        ans_claim = [e for e in trace["edges"]
                     if e["source"] == "answer:0" and by_id[e["target"]]["node_type"] == "CLAIM"]
        assert ans_claim
        claim_ev = [e for e in trace["edges"]
                    if by_id[e["source"]]["node_type"] == "CLAIM"
                    and by_id[e["target"]]["node_type"] == "EVIDENCE"]
        assert claim_ev
        ev_q = [e for e in trace["edges"]
                if by_id[e["source"]]["node_type"] == "EVIDENCE"
                and e["target"] == "query:0"]
        assert ev_q

    def test_uncited_sentences_never_claims(self):
        trace = build_reasoning_trace(_result(answer="Acme is great. Revenue was high."))
        assert [n for n in trace["nodes"] if n["node_type"] == "CLAIM"] == []

    def test_empty_answer_skeleton(self):
        trace = build_reasoning_trace(_result(answer="", citations=[],
                                              contradiction_signals=[]))
        types = {n["node_type"] for n in trace["nodes"]}
        assert {"QUERY", "HYPOTHESIS", "DECISION", "ANSWER"} <= types

    def test_missing_plan_survives(self):
        r = _result()
        del r["plan"]
        trace = build_reasoning_trace(r)
        assert any(n["node_type"] == "HYPOTHESIS" for n in trace["nodes"])
