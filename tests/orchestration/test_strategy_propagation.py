"""Integration: adaptive strategy mutations propagate into next retrieval.

These tests drive the REAL assess_node → state → retrieve_node path with
fakes only at the boundaries (LLM assessment call, retriever I/O, gap
detector). They prove mutations change what the NEXT retrieval iteration
receives — not merely what mutate_strategy() returns.

No LLM calls. No network. No timing assertions.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.config import Settings
from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.adaptive_research import create_adaptive_research_policy
from app.orchestration.graph import _timed_node
from app.orchestration.models import EvidenceAssessment, ResearchPlan
from app.orchestration.nodes import make_assess_node, make_retrieve_node


def _ref(text="Acme revenue was strong this quarter", score=0.9) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(), document_id=uuid4(), source_id=uuid4(),
        source_path="/corpus/doc.md", source_type=SourceType.TEXT,
        text=text, score=score, rank=1,
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(objective="Find Acme revenue")


def _settings(**kw) -> Settings:
    args = {"adaptive_research_enabled": True}
    args.update(kw)
    return Settings(**args)


def _crit_signal() -> dict:
    return {
        "severity": 1.0, "confidence": "HIGH",
        "conflict_type": "GENUINE_CONTRADICTION",
        "description": "2023 vs 2025 revenue",
        "evidence_indices": [1, 2],
        "resolved": False, "critical": True,
    }


def _base_state(**kw) -> dict:
    state = {
        "request_id": "test-run",
        "query": "What was Acme revenue?",
        "max_iterations": 3,
        "token_budget": 6000,
        "plan": _plan(),
        "pending_subquestions": ["plan q1"],
        "issued_subqueries": [],
        "evidence": [_ref()],
        "consecutive_empty_retrievals": 0,
        "iteration": 1,
        "tokens_used": 100,
        "sufficient": False,
        "stop_reason": None,
        "answer": None,
        "warnings": [],
        "question_pattern": None,
        "retrieval_gain_history": [1.0, 0.5],
        "contradiction_signals": [],
        "evidence_tasks": [],
        "complexity_tier": None,
        "strategy_history": [],
    }
    state.update(kw)
    return state


class _FakeGaps:
    """Gap detector fake: scripted gaps + re-retrieve flag."""

    def __init__(self, gaps=None, re_retrieve=False):
        self._gaps = gaps or []
        self.re_retrieve = re_retrieve

    def detect_gaps(self, state, plan, evidence):
        return list(self._gaps)

    def should_re_retrieve(self, gaps):
        return self.re_retrieve


class _FakeRetriever:
    """Retriever fake: records what retrieval actually received."""

    def __init__(self, refs=None):
        self.calls: list[dict] = []
        self._refs = refs if refs is not None else [_ref("retrieved evidence text")]

    async def search_async(self, query, top_k=None):
        self.calls.append({"query": query, "top_k": top_k})
        return list(self._refs)


class _FakeReranker:
    def rerank(self, query, results, top_k=None):
        return list(results)


def _assess_next_subquery(text="assessor follow-up"):
    return patch(
        "app.orchestration.nodes._safe_structured_call",
        new=AsyncMock(return_value=(
            EvidenceAssessment(sufficient=False, reasoning="need more", next_subquery=text),
            None,
        )),
    )


def _make_assess(settings, gaps=None, re_retrieve=False, policy=None):
    if policy is None:
        policy = create_adaptive_research_policy(settings)
    return make_assess_node(
        router=None,
        settings=settings,
        gap_detector=_FakeGaps(gaps=gaps, re_retrieve=re_retrieve),
        evidence_selector=None,
        adaptive_research_policy=policy,
    )


async def _run_assess(assess_fn, state):
    with _assess_next_subquery():
        return await assess_fn(state)


def _next_state(base, assess_out, iteration=2):
    """Simulate LangGraph last-write-wins merge into the next iteration."""
    nxt = dict(base)
    nxt.update(assess_out)
    nxt["iteration"] = iteration
    return nxt


async def _run_retrieve(state, settings, retriever):
    retrieve_fn = make_retrieve_node(
        retriever=retriever, reranker=_FakeReranker(),
        settings=settings, policy_router=None,
    )
    return await retrieve_fn(state)


RESOLUTION_Q = (
    "What was Acme revenue? - resolution of conflicting evidence, authoritative sources"
)


class TestConflictDrivenPropagation:
    async def test_resolution_query_reaches_next_retrieval(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        state = _base_state(contradiction_signals=[_crit_signal()])
        out = await _run_assess(assess_fn, state)

        assert out["strategy_mutation"] == "conflict_driven"
        assert out["pending_subquestions"][0] == RESOLUTION_Q
        assert out["strategy_top_k"] is None  # conflict path does not escalate depth

        # Iteration 2: retrieval actually receives the mutated query
        retriever = _FakeRetriever()
        await _run_retrieve(_next_state(state, out), settings, retriever)
        assert retriever.calls, "retrieve was never invoked"
        assert retriever.calls[0]["query"] == RESOLUTION_Q
        assert retriever.calls[0]["top_k"] == 8  # configured default, no override

    async def test_existing_resolution_query_not_duplicated(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        state = _base_state(
            contradiction_signals=[_crit_signal()],
            pending_subquestions=[RESOLUTION_Q, "plan q1"],
        )
        out = await _run_assess(assess_fn, state)
        assert out["pending_subquestions"].count(RESOLUTION_Q) == 1
        assert out["pending_subquestions"][0] == RESOLUTION_Q

    async def test_conflict_beats_gain_stall(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        state = _base_state(
            contradiction_signals=[_crit_signal()],
            retrieval_gain_history=[1.0, 0.01],  # stalled AND conflicted
        )
        out = await _run_assess(assess_fn, state)
        assert out["strategy_mutation"] == "conflict_driven"
        assert out["strategy_top_k"] is None
        assert out["pending_subquestions"][0] == RESOLUTION_Q

    async def test_mutation_clears_next_iteration(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        first = await _run_assess(
            assess_fn, _base_state(contradiction_signals=[_crit_signal()]))
        assert first["strategy_mutation"] == "conflict_driven"

        # Next iteration: conflict resolved, healthy gain, no gaps
        second_state = _next_state(
            _base_state(), first, iteration=2)
        second_state["contradiction_signals"] = [
            {**_crit_signal(), "resolved": True, "critical": False}]
        second_state["retrieval_gain_history"] = [1.0, 0.5, 0.4]
        second = await _run_assess(assess_fn, second_state)
        assert second["strategy_mutation"] is None
        assert second["strategy_top_k"] is None


class TestGainStallPropagation:
    async def test_escalated_top_k_reaches_next_retrieval(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        state = _base_state(retrieval_gain_history=[1.0, 0.04])
        out = await _run_assess(assess_fn, state)

        assert out["strategy_mutation"] == "gain_stall_escalation"
        assert out["strategy_top_k"] == 16  # min(8 * 2, 50)

        retriever = _FakeRetriever()
        await _run_retrieve(_next_state(state, out), settings, retriever)
        assert retriever.calls[0]["top_k"] == 16

    async def test_escalation_capped_at_50(self):
        settings = _settings(orchestration_retrieval_top_k=40)
        assess_fn = _make_assess(settings)
        out = await _run_assess(
            assess_fn, _base_state(retrieval_gain_history=[1.0, 0.01]))
        assert out["strategy_top_k"] == 50

        retriever = _FakeRetriever()
        await _run_retrieve(_next_state(_base_state(), out), settings, retriever)
        assert retriever.calls[0]["top_k"] == 50

    async def test_default_top_k_without_override(self):
        settings = _settings()
        assess_fn = _make_assess(settings)
        out = await _run_assess(assess_fn, _base_state())  # healthy gain
        assert out["strategy_top_k"] is None

        retriever = _FakeRetriever()
        await _run_retrieve(_next_state(_base_state(), out), settings, retriever)
        assert retriever.calls[0]["top_k"] == 8

    async def test_none_falls_back_to_configured(self):
        settings = _settings(orchestration_retrieval_top_k=12)
        retriever = _FakeRetriever()
        await _run_retrieve(_base_state(), settings, retriever)
        assert retriever.calls[0]["top_k"] == 12


class TestGapPrioritizationPropagation:
    async def test_gap_query_first_into_retrieval(self):
        settings = _settings()
        gaps = [{"suggested_query": "gap q", "priority": 0.9, "gap_type": "x"}]
        assess_fn = _make_assess(settings, gaps=gaps)
        out = await _run_assess(assess_fn, _base_state())

        assert out["strategy_mutation"] == "gap_prioritized"
        assert out["pending_subquestions"][0] == "gap q"

        retriever = _FakeRetriever()
        await _run_retrieve(_next_state(_base_state(), out), settings, retriever)
        assert retriever.calls[0]["query"] == "gap q"

    async def test_empty_suggestions_ignored(self):
        settings = _settings()
        gaps = [{"suggested_query": "   ", "priority": 0.9}]
        assess_fn = _make_assess(settings, gaps=gaps)
        out = await _run_assess(assess_fn, _base_state())
        assert out["strategy_mutation"] is None
        assert out["pending_subquestions"][0] == "plan q1"


class TestStrategyHistoryRecords:
    async def test_history_records_actual_mutation_reasons(self):
        settings = _settings()
        assess_fn = _timed_node("assess", _make_assess(settings))

        conflicted = await _run_assess(
            assess_fn, _base_state(contradiction_signals=[_crit_signal()]))
        assert conflicted["strategy_history"][-1]["mutated_from"] == "conflict_driven"

        stalled = await _run_assess(
            assess_fn, _base_state(retrieval_gain_history=[1.0, 0.02]))
        assert stalled["strategy_history"][-1]["mutated_from"] == "gain_stall_escalation"
        assert stalled["strategy_history"][-1]["top_k_override"] == 16

        gapped = await _run_assess(
            assess_fn, _base_state(),)
        # No gaps injected, healthy gain, no conflict: assessor appends its
        # follow-up, so pending grows -> derived reason is "assessor".
        assert gapped["strategy_history"][-1]["mutated_from"] == "assessor"

    async def test_history_records_gap_prioritized(self):
        settings = _settings()
        gaps = [{"suggested_query": "gap q", "priority": 0.9}]
        assess_fn = _timed_node("assess", _make_assess(settings, gaps=gaps))
        out = await _run_assess(assess_fn, _base_state())
        assert out["strategy_history"][-1]["mutated_from"] == "gap_prioritized"


class TestDisabledUnchanged:
    async def test_disabled_policy_no_mutation(self):
        settings = _settings(adaptive_research_enabled=False)
        assess_fn = _make_assess(settings)
        out = await _run_assess(
            assess_fn,
            _base_state(contradiction_signals=[_crit_signal()],
                        retrieval_gain_history=[1.0, 0.01]))
        assert out["strategy_mutation"] is None
        assert out["strategy_top_k"] is None
        # pre-existing ordering untouched: plan query first, assessor appends
        assert out["pending_subquestions"][0] == "plan q1"
        assert "assessor follow-up" in out["pending_subquestions"]

    async def test_none_policy_no_mutation(self):
        settings = _settings()
        import app.orchestration.nodes as nodes_mod
        assess_fn = nodes_mod.make_assess_node(
            router=None, settings=settings,
            gap_detector=_FakeGaps(), evidence_selector=None,
            adaptive_research_policy=None,
        )
        out = await _run_assess(assess_fn, _base_state(
            contradiction_signals=[_crit_signal()]))
        assert out["strategy_mutation"] is None
        assert out["strategy_top_k"] is None
        assert out["pending_subquestions"][0] == "plan q1"

    async def test_disabled_retrieval_uses_configured_top_k(self):
        settings = _settings(adaptive_research_enabled=False)
        retriever = _FakeRetriever()
        await _run_retrieve(_base_state(), settings, retriever)
        assert retriever.calls[0]["top_k"] == 8
        assert retriever.calls[0]["query"] == "plan q1"
