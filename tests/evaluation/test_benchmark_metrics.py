"""Phase 12.3 benchmark metrics tests (deterministic, no LLM/I-O)."""

from __future__ import annotations

import math

import pytest

from benchmarks.metrics import (
    answer_faithfulness,
    adversarial_robustness,
    aggregate_scores,
    by_type_breakdown,
    citation_correctness,
    claim_support_rate,
    compute_item_scores,
    contradiction_recall,
    evidence_precision,
    recall_at_k,
    temporal_accuracy,
    token_f1,
)
from benchmarks.models import BenchmarkItem, BenchmarkRunOutput


def test_token_f1_identical_is_one():
    assert token_f1("Beta Analytics was founded in 2015.", "Beta Analytics was founded in 2015.") == pytest.approx(1.0)


def test_token_f1_disjoint_is_zero():
    assert token_f1("red fox", "blue whale") == 0.0


def test_token_f1_partial():
    score = token_f1("founded in 2015", "founded in 2015 in London")
    assert 0.5 < score < 1.0


def test_recall_at_k_counts_gold_within_top_k():
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, 5) == pytest.approx(1.0)


def test_recall_at_k_not_applicable_when_no_gold():
    assert math.isnan(recall_at_k(["a"], set(), 5))


def test_evidence_precision_fraction_cited_is_gold():
    assert evidence_precision(["g1", "x1", "g2"], {"g1", "g2"}) == pytest.approx(2 / 3)
    assert evidence_precision([], {"g1"}) == 0.0


def test_citation_correctness_resolves_bracket_markers():
    cited = ["g1", "x1", "g2"]
    assert citation_correctness("Answer [1][3]", cited, {"g1", "g2"}) == pytest.approx(1.0)
    assert citation_correctness("Answer [2]", cited, {"g1", "g2"}) == 0.0
    assert citation_correctness("Answer without markers", cited, {"g1"}) == 0.0


def test_claim_support_rate_cited_and_lexically_grounded():
    chunks = {"g1": "Beta Analytics was founded in London in 2015."}
    text = "Beta Analytics was founded in 2015 [1]. London is rainy today."
    assert claim_support_rate(text, ["g1"], {"g1"}, chunks) == pytest.approx(0.5)


def test_contradiction_recall_only_when_expected():
    assert contradiction_recall(True, True) == 1.0
    assert contradiction_recall(False, True) == 0.0
    assert math.isnan(contradiction_recall(False, False))


def test_temporal_accuracy_matches_gold_years():
    assert temporal_accuracy("Approved in May 2023 by the FDA.", ["2023"]) == 1.0
    assert temporal_accuracy("Approved in 2024.", ["2023"]) == 0.0
    assert math.isnan(temporal_accuracy("No date.", []))


def test_answer_faithfulness_lexical_and_judge():
    judge = lambda answer, gold, _: 0.42  # noqa: E731
    assert answer_faithfulness("a b", "a b", judge) == pytest.approx(0.42)
    assert answer_faithfulness("a b", "a b") == pytest.approx(1.0)


def test_adversarial_robustness_distractor_citation():
    assert adversarial_robustness(["g1", "d1"], {"d1"}, adversarial=True) == 0.0
    assert adversarial_robustness(["g1"], {"d1"}, adversarial=True) == 1.0
    assert math.isnan(adversarial_robustness(["g1"], {"d1"}, adversarial=False))


def test_compute_item_scores_full_vector():
    item = BenchmarkItem(
        id="t",
        type="contradiction",
        question="q",
        gold_answer="Figure is $960M.",
        gold_evidence=["Quantix reported 2022 GAAP revenue of $960M."],
        gold_years=["2022"],
        expect_contradiction=True,
    )
    output = BenchmarkRunOutput(
        item_id="t",
        answer="The 2022 figure was $960M under GAAP [1].",
        cited_chunk_ids=["g1"],
        retrieved_chunk_ids=["g1", "x1"],
        loop_count=2,
        tokens_used=120,
        latency_ms=300,
        contradiction_detected=True,
    )
    scores = compute_item_scores(item, output, {"g1"}, {"d1"}, {"g1": item.gold_evidence[0]})
    assert scores["recall_at_10"] == 1.0
    assert scores["evidence_precision"] == 1.0
    assert scores["citation_correctness"] == 1.0
    assert scores["contradiction_recall"] == 1.0
    assert scores["temporal_accuracy"] == 1.0
    assert scores["answer_faithfulness"] > 0.0


def test_aggregate_scores_drops_nan_and_adds_runtime_counters():
    a = BenchmarkRunOutput(item_id="a", answer="", loop_count=1, tokens_used=100, latency_ms=10, failed_calls=0)
    b = BenchmarkRunOutput(item_id="b", answer="", loop_count=3, tokens_used=300, latency_ms=30, failed_calls=1)
    per_item = [(BenchmarkItem(id="a", type="t", question="q", gold_answer="x"), {m: float("nan") for m in [
        "recall_at_5", "recall_at_10", "evidence_precision", "citation_correctness",
        "claim_support_rate", "contradiction_recall", "temporal_accuracy",
        "answer_faithfulness", "adversarial_robustness",
    ]}) for _ in range(2)]
    agg = aggregate_scores(per_item)
    assert math.isnan(agg["contradiction_recall"]["value"])
    assert agg["contradiction_recall"]["applicable"] == 0
    assert agg["avg_loop_count"]["value"] == pytest.approx(2.0)
    assert agg["total_failed_calls"]["value"] == 1.0


def test_by_type_breakdown_groups_headline_metrics():
    per_item = [
        (BenchmarkItem(id="a", type="easy_factual", question="q", gold_answer="x"), {
            "recall_at_10": 1.0, "evidence_precision": 0.8, "answer_faithfulness": 0.6,
        }),
        (BenchmarkItem(id="b", type="multi_hop", question="q", gold_answer="x"), {
            "recall_at_10": 0.5, "evidence_precision": 1.0, "answer_faithfulness": 0.5,
        }),
    ]
    breakdown = by_type_breakdown(per_item)
    assert breakdown["easy_factual"]["recall_at_10"] == pytest.approx(1.0)
    assert breakdown["easy_factual"]["answer_faithfulness"] == pytest.approx(0.6)
    assert breakdown["multi_hop"]["evidence_precision"] == pytest.approx(1.0)