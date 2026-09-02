"""Tests for query complexity classification (HARDEN-06.5.2)."""

from __future__ import annotations

from app.llm_gateway.routing.complexity import (
    ComplexityTier,
    classify_complexity,
)


def test_simple_lookup_is_fast():
    assert classify_complexity("what is the refund period?") == ComplexityTier.FAST


def test_empty_query_defaults_balanced():
    assert classify_complexity("") == ComplexityTier.BALANCED


def test_comparative_is_strong():
    assert classify_complexity("how does policy A differ from policy B?") == ComplexityTier.STRONG


def test_causal_trace_is_strong():
    assert classify_complexity("trace how X affects Y through the document") == ComplexityTier.STRONG


def test_evaluate_abstract_is_strong():
    assert classify_complexity("evaluate the implications of the new policy") == ComplexityTier.STRONG


def test_short_factual_defaults_balanced():
    assert classify_complexity("the capital of France is what") == ComplexityTier.BALANCED


def test_long_query_is_strong_even_without_signal():
    long_q = "q" * 200
    assert classify_complexity(long_q) == ComplexityTier.STRONG


def test_define_prefixed_query_is_fast():
    assert classify_complexity("when was the company founded") == ComplexityTier.FAST