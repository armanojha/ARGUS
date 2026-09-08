"""Tests for adaptive strategy mutation (ResearchStrategy behavior).

mutate_strategy is pure (no I/O, no LLM): given this iteration's
discoveries, it decides how the NEXT iteration retrieves. All tests are
deterministic. Default-off behavior (policy disabled → noop) is asserted
to prove the loop is unchanged unless explicitly enabled.
"""

from app.orchestration.adaptive_research import (
    AdaptiveResearchPolicy,
    StrategyMutation,
)


def _policy(**kw):
    args = {"enabled": True}
    args.update(kw)
    return AdaptiveResearchPolicy(**args)


def _base(**kw):
    args = {
        "current_queries": ["q1", "q2"],
        "gain_history": [1.0, 0.5],
        "contradictions_unresolved": False,
        "gaps": [],
        "iteration": 1,
        "max_iterations": 3,
        "base_top_k": 8,
        "query": "What was Acme revenue?",
    }
    args.update(kw)
    return args


class TestDisabledIsNoop:
    def test_disabled_policy_returns_noop(self):
        m = AdaptiveResearchPolicy(enabled=False).mutate_strategy(**_base())
        assert m == StrategyMutation()

    def test_out_of_budget_returns_noop(self):
        m = _policy().mutate_strategy(**_base(iteration=3, max_iterations=3))
        assert m == StrategyMutation()

    def test_steady_state_returns_noop(self):
        m = _policy().mutate_strategy(**_base())
        assert m == StrategyMutation()


class TestConflictDriven:
    def test_contradiction_front_loads_resolution_query(self):
        m = _policy().mutate_strategy(**_base(contradictions_unresolved=True))
        assert m.mutation == "conflict_driven"
        assert m.reordered_queries is not None
        assert m.reordered_queries[0].startswith("What was Acme revenue?")
        assert "conflicting" in m.reordered_queries[0]
        # original queries preserved after the resolution query
        assert m.reordered_queries[1:] == ["q1", "q2"]

    def test_conflict_beats_gain_stall(self):
        m = _policy().mutate_strategy(**_base(
            contradictions_unresolved=True, gain_history=[1.0, 0.01]))
        assert m.mutation == "conflict_driven"
        assert m.top_k_override is None

    def test_no_duplicate_resolution_query(self):
        res_q = "What was Acme revenue? - resolution of conflicting evidence, authoritative sources"
        m = _policy().mutate_strategy(**_base(
            contradictions_unresolved=True, current_queries=[res_q, "q1"]))
        assert m.reordered_queries.count(res_q) == 1


class TestGainStallEscalation:
    def test_stall_widens_top_k(self):
        m = _policy().mutate_strategy(**_base(gain_history=[1.0, 0.05]))
        assert m.mutation == "gain_stall_escalation"
        assert m.top_k_override == 16

    def test_escalation_capped(self):
        m = _policy().mutate_strategy(**_base(gain_history=[1.0, 0.01], base_top_k=40))
        assert m.top_k_override == 50

    def test_short_history_never_escalates(self):
        m = _policy().mutate_strategy(**_base(gain_history=[0.01]))
        assert m == StrategyMutation()

    def test_healthy_gain_no_escalation(self):
        m = _policy().mutate_strategy(**_base(gain_history=[1.0, 0.9]))
        assert m == StrategyMutation()


class TestGapPrioritization:
    def test_gap_queries_front_loaded(self):
        gaps = [{"suggested_query": "gap q", "priority": 0.9, "gap_type": "x"}]
        m = _policy().mutate_strategy(**_base(gaps=gaps))
        assert m.mutation == "gap_prioritized"
        assert m.reordered_queries[0] == "gap q"
        assert m.reordered_queries[1:] == ["q1", "q2"]

    def test_empty_suggestions_ignored(self):
        gaps = [{"suggested_query": "  ", "priority": 0.9}]
        m = _policy().mutate_strategy(**_base(gaps=gaps))
        assert m == StrategyMutation()
