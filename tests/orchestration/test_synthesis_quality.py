"""Phase 25: Synthesis quality gate tests.

Tests for:
- Contradiction-aware synthesis prompts
- Evidence quality annotations in prompts
- Deterministic claim grounding check
- Citation fabrication warning
- End-to-end synthesis quality flow
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.models import QueryAnalysis, ResearchPlan
from app.orchestration.nodes import (
    check_claim_grounding,
    extract_cited_indices,
    make_synthesize_node,
)
from app.orchestration.prompts import (
    _format_evidence_block,
    build_synthesis_messages,
)

_FIXED_UUID = UUID("00000000-0000-0000-0000-000000000001")


def _make_evidence(
    text: str = "Test evidence text",
    source_path: str = "doc.txt",
    score: float = 0.85,
    chunk_id: str | UUID = _FIXED_UUID,
) -> EvidenceRef:
    if isinstance(chunk_id, str):
        chunk_id = UUID(chunk_id)
    return EvidenceRef(
        chunk_id=chunk_id,
        document_id=_FIXED_UUID,
        source_id=_FIXED_UUID,
        source_path=source_path,
        source_type=SourceType.TEXT,
        text=text,
        score=score,
        rank=1,
    )


def _make_plan(objective: str = "Test objective") -> ResearchPlan:
    return ResearchPlan(
        objective=objective,
        entities=["test"],
        subquestions=["sub1"],
        stopping_condition="sufficient evidence",
        token_budget=6000,
        iteration_budget=3,
    )


# ─── Claim Grounding Check ───────────────────────────────────────


class TestClaimGrounding:
    def test_all_sentences_cited(self):
        answer = "The sky is blue [1]. Water is wet [2]."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert warnings == []

    def test_uncited_sentence_detected(self):
        answer = "The sky is blue [1]. This has no citation."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert len(warnings) == 1
        assert "unsupported_claim" in warnings[0]

    def test_multiple_uncited_sentences(self):
        answer = "Uncited one. Also uncited [1]. Still uncited."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert len(warnings) == 2

    def test_empty_answer(self):
        warnings = check_claim_grounding("", evidence_count=3)
        assert warnings == []

    def test_no_evidence(self):
        answer = "Something [1]."
        warnings = check_claim_grounding(answer, evidence_count=0)
        assert warnings == []

    def test_fullwidth_citations(self):
        answer = "Claim one 【1】. Claim two 【2】."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert warnings == []

    def test_long_sentence_preview_truncated(self):
        long = "A" * 200
        answer = f"{long}."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert len(warnings) == 1
        assert "..." in warnings[0]

    def test_mixed_cited_and_uncited(self):
        answer = "Cited [1]. Not cited. Also cited [2]. Not cited either."
        warnings = check_claim_grounding(answer, evidence_count=3)
        assert len(warnings) == 2


# ─── Contradiction-Aware Synthesis Prompts ────────────────────────


class TestContradictionAwarePrompts:
    def test_no_contradictions(self):
        plan = _make_plan()
        evidence = [_make_evidence()]
        messages = build_synthesis_messages(plan, evidence)
        assert len(messages) == 2
        assert "CONTRADICTION ALERT" not in messages[1].content

    def test_contradictions_injected(self):
        plan = _make_plan()
        evidence = [_make_evidence()]
        signals = [
            {"severity": "high", "description": "Source A says X, Source B says not-X"},
            {"severity": "medium", "description": "Date mismatch"},
        ]
        messages = build_synthesis_messages(plan, evidence, contradiction_signals=signals)
        user_content = messages[1].content
        assert "CONTRADICTION ALERT" in user_content
        assert "Source A says X" in user_content
        assert "Date mismatch" in user_content

    def test_empty_contradiction_list(self):
        plan = _make_plan()
        evidence = [_make_evidence()]
        messages = build_synthesis_messages(plan, evidence, contradiction_signals=[])
        assert "CONTRADICTION ALERT" not in messages[1].content


# ─── Evidence Quality Annotations ─────────────────────────────────


class TestEvidenceQualityAnnotations:
    def test_scores_included_when_requested(self):
        evidence = [_make_evidence(score=0.92)]
        block = _format_evidence_block(evidence, include_scores=True)
        assert "score: 0.92" in block

    def test_scores_excluded_by_default(self):
        evidence = [_make_evidence(score=0.92)]
        block = _format_evidence_block(evidence)
        assert "score:" not in block

    def test_scores_in_synthesis_prompt(self):
        plan = _make_plan()
        evidence = [_make_evidence(score=0.77)]
        messages = build_synthesis_messages(plan, evidence)
        assert "score: 0.77" in messages[1].content

    def test_empty_evidence_block(self):
        block = _format_evidence_block([])
        assert "no evidence" in block


# ─── Citation Fabrication Warning ─────────────────────────────────


class TestCitationFallbackWarning:
    def test_citation_fallback_produces_warning(self):
        """_build_result should add a warning when fallback citations are used."""
        from app.orchestration.graph import _build_result

        state = {
            "query": "test",
            "plan": _make_plan(),
            "query_analysis": None,
            "iteration": 1,
            "evidence": [_make_evidence(text="Evidence text")],
            "answer": "An answer without any citations.",
            "warnings": [],
            "stop_reason": None,
            "tokens_used": 100,
            "request_id": "req-1",
            "max_iterations": 3,
            "token_budget": 6000,
            "issued_subqueries": [],
            "pending_subquestions": [],
            "contradiction_signals": [],
            "retrieval_gain_history": [],
            "evidence_tasks": [],
            "memory_consulted": [],
            "consecutive_empty_retrievals": 0,
            "sufficient": False,
        }
        result = _build_result(state)
        assert any("citation_fallback" in w for w in result.warnings)
        assert len(result.citations) >= 1

    def test_cited_answer_no_fallback_warning(self):
        """When the answer has citations, no fallback warning is added."""
        from app.orchestration.graph import _build_result

        state = {
            "query": "test",
            "plan": _make_plan(),
            "query_analysis": None,
            "iteration": 1,
            "evidence": [_make_evidence(text="Evidence text")],
            "answer": "The answer is correct [1].",
            "warnings": [],
            "stop_reason": None,
            "tokens_used": 100,
            "request_id": "req-1",
            "max_iterations": 3,
            "token_budget": 6000,
            "issued_subqueries": [],
            "pending_subquestions": [],
            "contradiction_signals": [],
            "retrieval_gain_history": [],
            "evidence_tasks": [],
            "memory_consulted": [],
            "consecutive_empty_retrievals": 0,
            "sufficient": False,
        }
        result = _build_result(state)
        assert not any("citation_fallback" in w for w in result.warnings)


# ─── Synthesize Node Integration ──────────────────────────────────


class TestSynthesizeNodePhase25:
    @pytest.mark.asyncio
    async def test_grounding_warnings_added(self):
        """Synthesize node adds grounding warnings for uncited sentences."""
        router = AsyncMock()
        router.complete = AsyncMock(return_value=MagicMock(content="Uncited claim. Also uncited."))
        settings = MagicMock()
        settings.orchestration_llm_timeout = 30

        node = make_synthesize_node(router, settings)
        state = {
            "plan": _make_plan(),
            "evidence": [_make_evidence()],
            "warnings": [],
            "request_id": "req-1",
            "query": "test",
            "contradiction_signals": [],
        }
        result = await node(state)
        grounding = [w for w in result["warnings"] if w.startswith("unsupported_claim")]
        assert len(grounding) == 2

    @pytest.mark.asyncio
    async def test_contradiction_signals_passed(self):
        """Contradiction signals are passed to the synthesis prompt."""
        router = AsyncMock()
        router.complete = AsyncMock(return_value=MagicMock(content="Answer [1]."))
        settings = MagicMock()
        settings.orchestration_llm_timeout = 30

        node = make_synthesize_node(router, settings)
        signals = [{"severity": "high", "description": "Conflict"}]
        state = {
            "plan": _make_plan(),
            "evidence": [_make_evidence()],
            "warnings": [],
            "request_id": "req-1",
            "query": "test",
            "contradiction_signals": signals,
        }
        result = await node(state)
        router.complete.assert_called_once()
        call_args = router.complete.call_args
        messages = call_args[0][0]
        assert "CONTRADICTION ALERT" in messages[1].content

    @pytest.mark.asyncio
    async def test_empty_evidence_no_synthesis(self):
        """When no evidence, synthesis is skipped with a clear message."""
        router = AsyncMock()
        settings = MagicMock()

        node = make_synthesize_node(router, settings)
        state = {
            "plan": _make_plan(),
            "evidence": [],
            "warnings": [],
            "request_id": "req-1",
            "query": "test",
            "contradiction_signals": [],
        }
        result = await node(state)
        assert "No supporting evidence" in result["answer"]
        router.complete.assert_not_called()
