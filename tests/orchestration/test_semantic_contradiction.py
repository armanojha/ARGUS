"""Stage 3: LLM semantic contradiction verification tests.

Tests ``_semantic_contradiction_check`` in app/orchestration/nodes.py.
All LLM calls are mocked — no network access.

Semantics under test:
  - LLM success + CONTRADICTED verdict → kept (semantic_verified=True)
  - LLM success + ENTAILED verdict → dropped
  - LLM success + NEUTRAL verdict → dropped
  - LLM success + INSUFFICIENT_CONTEXT verdict → dropped
  - LLM failure (model None) → retained WITHOUT flag (semantic_verified=False)
  - Empty input → empty output
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.nodes import _semantic_contradiction_check


def _make_ref(text: str) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="test_source",
        source_type=SourceType.TEXT,
        text=text,
        score=0.8,
        rank=1,
    )


def _make_sig(i: int = 1, j: int = 2) -> dict:
    return {
        "severity": 1.0,
        "confidence": "HIGH",
        "conflict_type": "GENUINE_CONTRADICTION",
        "description": "test contradiction",
        "evidence_indices": [i, j],
        "entity_overlap": ["acme"],
        "metric_overlap": ["revenue"],
        "timeframe_i": [2025],
        "timeframe_j": [2025],
        "resolved": False,
        "critical": True,
    }


def _mock_verdict(verdict: str | None):
    """Patch _safe_structured_call to return a fixed verdict.

    verdict=None simulates an LLM failure: (None, "provider error").
    """
    if verdict is None:
        ret = (None, "provider error")
    else:
        ret = (SimpleNamespace(verdict=verdict), None)
    return patch(
        "app.orchestration.nodes._safe_structured_call",
        new=AsyncMock(return_value=ret),
    )


_EVIDENCE = [
    _make_ref("Acme revenue was $3.1 billion in 2025."),
    _make_ref("Acme revenue was $2.4 billion in 2025."),
]


class TestSemanticVerdicts:
    async def test_contradicted_kept_with_flag(self):
        with _mock_verdict("CONTRADICTED"):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert len(out) == 1
        assert out[0]["semantic_verified"] is True

    async def test_entailed_dropped(self):
        with _mock_verdict("ENTAILED"):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert out == []

    async def test_neutral_dropped(self):
        with _mock_verdict("NEUTRAL"):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert out == []

    async def test_insufficient_context_dropped(self):
        with _mock_verdict("INSUFFICIENT_CONTEXT"):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert out == []

    async def test_llm_failure_retains_with_false_flag(self):
        with _mock_verdict(None):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert len(out) == 1
        assert out[0]["semantic_verified"] is False

    async def test_llm_exception_retains_with_false_flag(self):
        with patch(
            "app.orchestration.nodes._safe_structured_call",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert len(out) == 1
        assert out[0]["semantic_verified"] is False

    async def test_empty_input_empty_output(self):
        out = await _semantic_contradiction_check(
            [], _EVIDENCE, router=None, settings=None,
        )
        assert out == []

    async def test_out_of_range_indices_retained_unverified(self):
        bad_sig = _make_sig(i=99, j=100)
        out = await _semantic_contradiction_check(
            [bad_sig], _EVIDENCE, router=None, settings=None,
        )
        assert len(out) == 1
        assert out[0]["semantic_verified"] is False

    async def test_verdict_case_insensitive(self):
        with _mock_verdict("contradicted"):
            out = await _semantic_contradiction_check(
                [_make_sig()], _EVIDENCE, router=None, settings=None,
            )
        assert len(out) == 1
        assert out[0]["semantic_verified"] is True
