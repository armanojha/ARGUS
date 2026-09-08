"""Phase 43 focused benchmark: correctness regression protection.

Measures:
- Irrelevant evidence rejection rate
- False contradiction rate
- Genuine contradiction recall
- Relevant conflict precision
- Synthesis fallback safety
"""

import json
from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.nodes import (
    _compute_query_relevance,
    _detect_contradictions,
    _filter_evidence_by_relevance,
    _is_topic_coherent,
    filter_contradictions_by_query,
)
from uuid import uuid4


def _make_ref(text: str, score: float = 0.5) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_id=uuid4(),
        source_path="benchmark",
        source_type=SourceType.TEXT,
        text=text,
        score=score,
        rank=1,
    )


# ── Benchmark cases ─────────────────────────────────────────────────────────

BENCHMARK = [
    # 1. Unrelated evidence (real-world failure)
    {
        "name": "unrelated_evidence",
        "query": "What evidence supports the claim that AI adoption is accelerating?",
        "evidence": [
            "Smart India Hackathon urban intelligence platform using public transport fleets for mobile sensing.",
            "Employee attendance system using biometric verification with database records.",
            "LLM tool transparency requirements for AI systems in enterprise environments.",
        ],
        "expected_contradictions": 0,
        "expected_irrelevant": 3,
    },
    # 2. Shared vocabulary (analytics)
    {
        "name": "shared_vocabulary_analytics",
        "query": "How is analytics used in organizations?",
        "evidence": [
            "Analytics can improve city planning by analyzing traffic patterns.",
            "Analytics are used in employee attendance tracking to monitor work hours.",
        ],
        "expected_contradictions": 0,
        "expected_irrelevant": 0,  # Both relevant enough for this broader query
    },
    # 3. Shared vocabulary (database)
    {
        "name": "shared_vocabulary_database",
        "query": "What database systems are used in the company?",
        "evidence": [
            "The database stores employee records including personal information.",
            "The database supports leave management by tracking vacation requests.",
        ],
        "expected_contradictions": 0,
        "expected_irrelevant": 0,
    },
    # 4. Genuine numerical conflict
    {
        "name": "genuine_numerical_conflict",
        "query": "What is the enterprise revenue growth rate in 2024?",
        "evidence": [
            "Enterprise revenue growth was 35% in 2024 according to industry survey.",
            "Enterprise revenue growth reached 52% in 2024 based on annual report.",
        ],
        "expected_contradictions": 1,
        "expected_irrelevant": 0,
    },
    # 5. Different years
    {
        "name": "different_years",
        "query": "What is the trend of AI adoption over time?",
        "evidence": [
            "AI adoption was 35% in 2023 according to industry survey.",
            "AI adoption reached 52% in 2025 based on latest report.",
        ],
        "expected_contradictions": 0,  # DIFFERENT_TIMEFRAME should be filtered
        "expected_irrelevant": 0,
    },
    # 6. Different scopes
    {
        "name": "different_scopes",
        "query": "What is the rate of AI adoption?",
        "evidence": [
            "Enterprise AI adoption reached 52% in 2024 according to Gartner.",
            "Consumer AI adoption was at 31% in 2024 based on consumer survey.",
        ],
        "expected_contradictions": 0,  # Different scopes, not contradictory
        "expected_irrelevant": 0,
    },
    # 7. Genuine opposing claims
    {
        "name": "genuine_opposing_claims",
        "query": "What features does the Acme Cloud Platform support?",
        "evidence": [
            "The Acme Cloud Platform supports feature X for data processing.",
            "The Acme Cloud Platform does not support feature X in current release.",
        ],
        "expected_contradictions": 1,  # POSSIBLE_CONTRADICTION (entity overlap, no metrics)
        "expected_irrelevant": 0,
    },
    # 8. Mixed relevant + irrelevant
    {
        "name": "mixed_relevant_irrelevant",
        "query": "What evidence supports AI adoption acceleration?",
        "evidence": [
            "AI adoption in enterprise sector reached 52% in 2024, showing acceleration.",
            "Employee attendance system using biometric verification.",
            "Gartner reports 40% year-over-year growth in AI adoption.",
            "Smart India Hackathon urban intelligence platform.",
        ],
        "expected_contradictions": 0,
        "expected_irrelevant": 2,
    },
]


def run_benchmark():
    results = []
    for case in BENCHMARK:
        evidence = [_make_ref(t) for t in case["evidence"]]
        query = case["query"]

        # Relevance filtering
        relevant, excluded = _filter_evidence_by_relevance(evidence, query)

        # Contradiction detection (on ALL evidence for raw count, relevant for filtered)
        contradictions_all = _detect_contradictions(evidence, query=query)
        contradictions_relevant = _detect_contradictions(
            relevant if relevant else evidence, query=query
        )

        # Query-aware filtering
        filtered_contradictions = filter_contradictions_by_query(
            contradictions_relevant, evidence, query
        )

        results.append({
            "name": case["name"],
            "total_evidence": len(evidence),
            "relevant": len(relevant),
            "excluded": len(excluded),
            "contradictions_raw": len(contradictions_all),
            "contradictions_filtered": len(filtered_contradictions),
            "expected_contradictions": case["expected_contradictions"],
            "expected_irrelevant": case["expected_irrelevant"],
            "contradiction_correct": (
                len(filtered_contradictions) == case["expected_contradictions"]
            ),
            "relevance_correct": (
                len(excluded) == case["expected_irrelevant"]
            ),
        })

    # Summary
    total = len(results)
    contra_correct = sum(1 for r in results if r["contradiction_correct"])
    relevance_correct = sum(1 for r in results if r["relevance_correct"])

    print("=" * 80)
    print("PHASE 43 CORRECTNESS BENCHMARK")
    print("=" * 80)
    for r in results:
        status = "PASS" if r["contradiction_correct"] and r["relevance_correct"] else "FAIL"
        print(f"\n[{status}] {r['name']}")
        print(f"  Evidence: {r['relevant']} relevant / {r['total_evidence']} total ({r['excluded']} excluded)")
        print(f"  Contradictions: {r['contradictions_filtered']} (raw: {r['contradictions_raw']}, expected: {r['expected_contradictions']})")
        print(f"  Relevance correct: {r['relevance_correct']}, Contradiction correct: {r['contradiction_correct']}")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {total} cases")
    print(f"  Contradiction precision: {contra_correct}/{total} ({contra_correct/total*100:.0f}%)")
    print(f"  Relevance filtering:     {relevance_correct}/{total} ({relevance_correct/total*100:.0f}%)")
    print(f"  Overall:                 {contra_correct + relevance_correct}/{total*2} ({(contra_correct+relevance_correct)/(total*2)*100:.0f}%)")
    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    run_benchmark()
