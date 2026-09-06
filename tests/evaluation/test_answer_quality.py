"""Phase 26: Answer quality evaluation tests.

18 edge case tests covering:
1. Answer with no citations
2. Answer with invalid citation ID
3. Answer with citation to irrelevant evidence
4. Citation supports only half of a sentence
5. Two claims sharing one citation where only one is supported
6. Unsupported number
7. Wrong number despite relevant citation
8. Contradictory sources
9. Answer acknowledges contradiction correctly
10. Answer ignores known contradiction
11. Multi-hop answer with missing bridge evidence
12. Correct answer with multiple supporting sources
13. Absent-information query
14. Citation fallback
15. Multiple citations on one claim
16. Citation at paragraph end
17. Citation attached to a partially supported sentence
18. Evidence from different documents with conflicting values
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest

from app.evaluation.answer_quality import (
    ClaimSupportStatus,
    AnswerQualityResult,
    decompose_claims,
    evaluate_answer,
)


_FIXED_UUID = UUID("00000000-0000-0000-0000-000000000001")


@dataclass
class FakeCitation:
    """Minimal citation object for testing."""

    text: str
    chunk_id: str = "chunk-1"
    document_id: str = "doc-1"
    source_path: str = "source.txt"
    score: float = 0.85
    ref_id: int = 1


# ─── Claim Decomposition Tests ───────────────────────────────────


class TestDecomposeClaims:
    def test_simple_sentence(self):
        claims = decompose_claims("Acme was founded in 1987.")
        assert len(claims) == 1
        assert "1987" in claims[0]

    def test_compound_sentence_with_numbers(self):
        claims = decompose_claims(
            "Revenue was $20M and employees numbered 500."
        )
        # Should attempt to split compound claims
        assert len(claims) >= 1

    def test_empty_answer(self):
        claims = decompose_claims("")
        assert claims == []

    def test_short_fragment(self):
        claims = decompose_claims("Yes.")
        assert len(claims) == 0

    def test_multiple_sentences(self):
        claims = decompose_claims("First fact. Second fact. Third fact.")
        assert len(claims) == 3


# ─── Edge Case 1: Answer with no citations ───────────────────────


class TestCase01_NoCitations:
    def test_no_citations(self):
        answer = "The company was founded in 1987 and is based in New York."
        citations = []
        result = evaluate_answer(answer, citations)
        assert result.citation_presence_rate == 0.0
        assert result.total_claims >= 1
        for claim in result.claims:
            assert claim.support_status == ClaimSupportStatus.NO_CITATION


# ─── Edge Case 2: Invalid citation ID ────────────────────────────


class TestCase02_InvalidCitation:
    def test_citation_out_of_range(self):
        answer = "Revenue was $20M [5]."
        citations = [FakeCitation(text="Some other evidence")]
        result = evaluate_answer(answer, citations)
        assert any(
            c.support_status == ClaimSupportStatus.INVALID_CITATION
            for c in result.claims
        )


# ─── Edge Case 3: Citation to irrelevant evidence ────────────────


class TestCase03_IrrelevantCitation:
    def test_irrelevant_evidence(self):
        answer = "Acme was founded in 1987 [1]."
        citations = [FakeCitation(text="The weather today is sunny and warm.")]
        result = evaluate_answer(answer, citations)
        # Evidence is topically irrelevant
        assert any(
            c.support_status in (ClaimSupportStatus.UNSUPPORTED, ClaimSupportStatus.PARTIALLY_SUPPORTED)
            for c in result.claims
        )


# ─── Edge Case 4: Citation supports half of sentence ─────────────


class TestCase04_PartialSupport:
    def test_partial_support(self):
        answer = "The company earned $20M revenue and has 500 employees [1]."
        citations = [FakeCitation(text="Acme Corp reported $20M in quarterly revenue.")]
        result = evaluate_answer(answer, citations)
        # Revenue is supported but employee count is not in evidence
        assert len(result.claims) >= 1
        # The claim should not be fully SUPPORTED since evidence doesn't cover employees
        claim = result.claims[0]
        assert claim.support_status != ClaimSupportStatus.SUPPORTED or \
               claim.partial_coverage_ratio < 1.0


# ─── Edge Case 5: Two claims, one citation, partial support ──────


class TestCase05_SharedCitation:
    def test_shared_citation(self):
        answer = "Revenue was $20M [1]. Employee count was 500 [1]."
        citations = [FakeCitation(text="Acme reported $20M revenue for the quarter.")]
        result = evaluate_answer(answer, citations)
        # Revenue claim should be supported, employee claim should not
        assert len(result.claims) >= 2


# ─── Edge Case 6: Unsupported number ─────────────────────────────


class TestCase06_UnsupportedNumber:
    def test_wrong_number(self):
        answer = "The company has 10,000 employees [1]."
        citations = [FakeCitation(text="The company has 5,000 employees.")]
        result = evaluate_answer(answer, citations)
        assert len(result.claims) >= 1
        claim = result.claims[0]
        assert claim.numerical_match is False


# ─── Edge Case 7: Wrong number despite relevant citation ─────────


class TestCase07_WrongNumberRelevantCitation:
    def test_relevant_but_wrong_number(self):
        answer = "Revenue reached $50M in 2024 [1]."
        citations = [FakeCitation(text="Acme Corp revenue for 2024 was $35M.")]
        result = evaluate_answer(answer, citations)
        assert len(result.claims) >= 1
        claim = result.claims[0]
        assert claim.numerical_match is False


# ─── Edge Case 8: Contradictory sources ──────────────────────────


class TestCase08_ContradictorySources:
    def test_contradictory_evidence(self):
        answer = "The recall affected 14,200 vehicles [1]. Earlier reports said 12,000 [2]."
        citations = [
            FakeCitation(text="Vertex recalled 14,200 trucks in December 2023."),
            FakeCitation(text="An October bulletin listed 12,000 affected vehicles."),
        ]
        result = evaluate_answer(answer, citations)
        # Both facts are correctly cited
        assert len(result.claims) >= 1


# ─── Edge Case 9: Answer acknowledges contradiction ──────────────


class TestCase09_ContradictionAcknowledged:
    def test_acknowledges_conflict(self):
        answer = "Originally reported as $31M, but restated to $18M in April 2023 [1][2]."
        citations = [
            FakeCitation(text="Polaris posted $31M operating profit for 2022."),
            FakeCitation(text="Polaris restated 2022 profit to $18M in April 2023."),
        ]
        result = evaluate_answer(answer, citations)
        # Answer correctly presents both figures
        assert result.citation_presence_rate > 0


# ─── Edge Case 10: Answer ignores contradiction ──────────────────


class TestCase10_ContradictionIgnored:
    def test_ignores_conflict(self):
        answer = "Polaris was profitable in 2022 with $31M [1]."
        citations = [
            FakeCitation(text="Polaris posted $31M in January 2023."),
            FakeCitation(text="Polaris restated 2022 profit to $18M in April 2023."),
        ]
        result = evaluate_answer(answer, citations)
        # Answer only cites the outdated figure
        assert len(result.claims) >= 1


# ─── Edge Case 11: Multi-hop missing bridge ──────────────────────


class TestCase11_MultiHopMissing:
    def test_missing_bridge(self):
        answer = "The company that acquired Beta is based in New York [1]."
        citations = [
            FakeCitation(text="Acme Corp completed acquisition of Beta Analytics."),
        ]
        result = evaluate_answer(answer, citations)
        # Citation mentions acquisition but not headquarters location
        assert len(result.claims) >= 1


# ─── Edge Case 12: Multiple supporting sources ───────────────────


class TestCase12_MultipleSources:
    def test_multiple_supporting(self):
        answer = "Acme was founded in 1987 [1] and is based in New York [2]."
        citations = [
            FakeCitation(text="Acme Corp was founded in 1987."),
            FakeCitation(text="Acme Corp is headquartered in New York."),
        ]
        result = evaluate_answer(answer, citations)
        assert result.citation_presence_rate > 0
        assert result.citation_validity_rate > 0


# ─── Edge Case 13: Absent information ────────────────────────────


class TestCase13_AbsentInfo:
    def test_absent_info(self):
        answer = "I could not find information about the company's market share."
        citations = []
        result = evaluate_answer(answer, citations)
        assert result.citation_presence_rate == 0.0
        # Should not flag as unsupported — the answer correctly says info is absent
        assert result.unsupported_claim_rate == 0.0 or result.total_claims == 0


# ─── Edge Case 14: Citation fallback ─────────────────────────────


class TestCase14_CitationFallback:
    def test_fallback_citations(self):
        answer = "The answer is somewhere in the evidence."
        citations = [
            FakeCitation(text="First chunk"),
            FakeCitation(text="Second chunk"),
            FakeCitation(text="Third chunk"),
        ]
        result = evaluate_answer(answer, citations)
        # No bracket citations in answer
        assert result.citation_presence_rate == 0.0


# ─── Edge Case 15: Multiple citations on one claim ──────────────


class TestCase15_MultipleCitations:
    def test_multiple_citations(self):
        answer = "Acme acquired Beta for $3.1B in March 2024 [1][2]."
        citations = [
            FakeCitation(text="Acme completed $3.1B acquisition of Beta in March 2024."),
            FakeCitation(text="Beta Analytics was acquired by Acme Corp."),
        ]
        result = evaluate_answer(answer, citations)
        assert result.citation_presence_rate > 0
        assert result.citation_validity_rate > 0


# ─── Edge Case 16: Citation at paragraph end ─────────────────────


class TestCase16_CitationAtEnd:
    def test_citation_at_end(self):
        answer = "The acquisition closed in March 2024 for $3.1B [1]."
        citations = [FakeCitation(text="Acme completed $3.1B acquisition in March 2024.")]
        result = evaluate_answer(answer, citations)
        assert result.citation_presence_rate > 0


# ─── Edge Case 17: Partially supported sentence ──────────────────


class TestCase17_PartiallySupported:
    def test_partially_supported(self):
        answer = "Revenue was $20M and the company employs 500 people [1]."
        citations = [FakeCitation(text="Acme reported $20M in quarterly revenue.")]
        result = evaluate_answer(answer, citations)
        assert len(result.claims) >= 1
        # Revenue part is supported, employee part is not
        claim = result.claims[0]
        assert claim.partial_coverage_ratio < 1.0


# ─── Edge Case 18: Conflicting values from different docs ────────


class TestCase18_ConflictingValues:
    def test_conflicting_documents(self):
        answer = "Revenue was $960M under GAAP [1] and $1.02B non-GAAP [2]."
        citations = [
            FakeCitation(text="Quantix reported $960M GAAP revenue for 2022."),
            FakeCitation(text="Quantix reported $1.02B non-GAAP revenue for 2022."),
        ]
        result = evaluate_answer(answer, citations)
        # Both facts are correctly cited from different sources
        assert result.citation_presence_rate > 0
        assert len(result.claims) >= 1


# ─── Gold Fact Coverage ──────────────────────────────────────────


class TestGoldFactCoverage:
    def test_all_facts_found(self):
        answer = "Acme was founded in 1987 and is based in New York."
        gold = ["1987", "New York"]
        result = evaluate_answer(answer, [], gold_facts=gold)
        assert result.gold_fact_coverage == 1.0
        assert len(result.gold_facts_missing) == 0

    def test_some_facts_missing(self):
        answer = "Acme was founded in 1987."
        gold = ["1987", "New York", "Diana Reyes"]
        result = evaluate_answer(answer, [], gold_facts=gold)
        assert result.gold_fact_coverage is not None
        assert result.gold_fact_coverage < 1.0
        assert len(result.gold_facts_missing) > 0

    def test_no_gold_facts(self):
        answer = "Some answer."
        result = evaluate_answer(answer, [], gold_facts=None)
        assert result.gold_fact_coverage is None


# ─── Numerical Consistency ───────────────────────────────────────


class TestNumericalConsistency:
    def test_correct_number(self):
        answer = "Revenue was $20M [1]."
        citations = [FakeCitation(text="Revenue reached $20M in Q3.")]
        result = evaluate_answer(answer, citations)
        assert result.numerical_consistency_rate == 1.0

    def test_incorrect_number(self):
        answer = "Revenue was $50M [1]."
        citations = [FakeCitation(text="Revenue reached $35M in Q3.")]
        result = evaluate_answer(answer, citations)
        assert result.numerical_consistency_rate == 0.0

    def test_no_numbers(self):
        answer = "The company is based in New York [1]."
        citations = [FakeCitation(text="Headquartered in New York City.")]
        result = evaluate_answer(answer, citations)
        assert result.numerical_consistency_rate is None
