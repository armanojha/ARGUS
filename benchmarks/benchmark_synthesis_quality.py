"""Phase 25 benchmark: Synthesis quality gate evaluation.

Measures structural synthesis quality improvements:
- Grounding check coverage (sentences with citations)
- Contradiction awareness in prompts
- Evidence quality annotation presence
- Citation fallback warning behavior

Runs deterministically without a live LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import UUID

from app.evidence.models import EvidenceRef, SourceType
from app.orchestration.models import ResearchPlan
from app.orchestration.nodes import check_claim_grounding, extract_cited_indices
from app.orchestration.prompts import _format_evidence_block, build_synthesis_messages

_FIXED_UUID = UUID("00000000-0000-0000-0000-000000000001")
REPORT_DIR = Path("benchmarks")
DATA_DIR = REPORT_DIR / "data" / "benchmark_reports"


def _make_evidence(
    text: str,
    source_path: str = "doc.txt",
    score: float = 0.85,
    rank: int = 1,
) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=_FIXED_UUID,
        document_id=_FIXED_UUID,
        source_id=_FIXED_UUID,
        source_path=source_path,
        source_type=SourceType.TEXT,
        text=text,
        score=score,
        rank=rank,
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


def benchmark_grounding_check():
    """Benchmark the claim grounding check on various answer patterns."""
    test_cases = [
        {
            "name": "all_cited",
            "answer": "The sky is blue [1]. Water is wet [2]. Fire is hot [3].",
            "evidence_count": 3,
            "expected_warnings": 0,
        },
        {
            "name": "half_cited",
            "answer": "Cited [1]. Not cited. Also cited [2]. Not cited either.",
            "evidence_count": 3,
            "expected_warnings": 2,
        },
        {
            "name": "none_cited",
            "answer": "This has no citations at all. Neither does this sentence.",
            "evidence_count": 3,
            "expected_warnings": 2,
        },
        {
            "name": "fullwidth_citations",
            "answer": "Claim one 【1】. Claim two 【2】.",
            "evidence_count": 3,
            "expected_warnings": 0,
        },
        {
            "name": "empty_answer",
            "answer": "",
            "evidence_count": 3,
            "expected_warnings": 0,
        },
        {
            "name": "long_answer_mixed",
            "answer": (
                "The first claim is well supported [1]. "
                "This claim lacks any citation whatsoever. "
                "The second claim also has evidence [2]. "
                "Another unsupported assertion goes here. "
                "Final supported statement [3]."
            ),
            "evidence_count": 3,
            "expected_warnings": 2,
        },
    ]

    results = []
    for tc in test_cases:
        start = time.perf_counter()
        for _ in range(1000):
            warnings = check_claim_grounding(tc["answer"], tc["evidence_count"])
        elapsed_ms = (time.perf_counter() - start) * 1000 / 1000
        results.append({
            "name": tc["name"],
            "warnings_found": len(warnings),
            "expected_warnings": tc["expected_warnings"],
            "pass": len(warnings) == tc["expected_warnings"],
            "avg_latency_ms": round(elapsed_ms, 4),
        })
    return results


def benchmark_contradiction_prompt():
    """Benchmark contradiction-aware prompt generation."""
    plan = _make_plan()
    evidence = [_make_evidence("Test evidence")]

    # Baseline: no contradictions
    start = time.perf_counter()
    for _ in range(1000):
        msgs = build_synthesis_messages(plan, evidence)
    baseline_ms = (time.perf_counter() - start) * 1000 / 1000

    # With contradictions
    signals = [
        {"severity": "high", "description": "Source A says X"},
        {"severity": "medium", "description": "Date mismatch"},
    ]
    start = time.perf_counter()
    for _ in range(1000):
        msgs = build_synthesis_messages(plan, evidence, contradiction_signals=signals)
    with_contradictions_ms = (time.perf_counter() - start) * 1000 / 1000

    # Verify contradiction section present
    user_content = msgs[1].content
    has_alert = "CONTRADICTION ALERT" in user_content
    has_instructions = "Acknowledge the conflict" in user_content
    has_severity = "Severity high" in user_content

    return {
        "baseline_prompt_ms": round(baseline_ms, 4),
        "with_contradictions_prompt_ms": round(with_contradictions_ms, 4),
        "overhead_ms": round(with_contradictions_ms - baseline_ms, 4),
        "contradiction_alert_present": has_alert,
        "synthesis_instructions_present": has_instructions,
        "severity_info_present": has_severity,
    }


def benchmark_evidence_annotations():
    """Benchmark evidence quality annotation in prompts."""
    evidence = [_make_evidence(f"Evidence {i}", score=0.5 + i * 0.1) for i in range(5)]

    # Baseline: no scores
    start = time.perf_counter()
    for _ in range(1000):
        block = _format_evidence_block(evidence)
    baseline_ms = (time.perf_counter() - start) * 1000 / 1000
    has_scores_baseline = "score:" in block

    # With scores
    start = time.perf_counter()
    for _ in range(1000):
        block_scores = _format_evidence_block(evidence, include_scores=True)
    scores_ms = (time.perf_counter() - start) * 1000 / 1000
    has_scores = "score:" in block_scores

    return {
        "baseline_no_scores_ms": round(baseline_ms, 4),
        "with_scores_ms": round(scores_ms, 4),
        "overhead_ms": round(scores_ms - baseline_ms, 4),
        "scores_present_in_baseline": has_scores_baseline,
        "scores_present_with_flag": has_scores,
    }


def benchmark_citation_fallback():
    """Benchmark citation fallback detection."""
    # Answer with citations
    cited_answer = "The sky is blue [1]. Water is wet [2]."
    indices_cited = extract_cited_indices(cited_answer, 3)

    # Answer without citations
    uncited_answer = "This answer has no citations."
    indices_uncited = extract_cited_indices(uncited_answer, 3)

    return {
        "cited_answer_indices": indices_cited,
        "uncited_answer_indices": indices_uncited,
        "fallback_triggered_correctly": len(indices_uncited) == 0 and len(indices_cited) > 0,
    }


def run_all_benchmarks():
    """Run all Phase 25 benchmarks."""
    grounding = benchmark_grounding_check()
    contradiction = benchmark_contradiction_prompt()
    evidence = benchmark_evidence_annotations()
    citation = benchmark_citation_fallback()

    # All grounding checks pass?
    all_grounding_pass = all(r["pass"] for r in grounding)
    # Average grounding latency
    avg_grounding_latency = sum(r["avg_latency_ms"] for r in grounding) / len(grounding)

    summary = {
        "phase": 25,
        "title": "Synthesis Quality Gate Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {
            "grounding_check": {
                "test_cases": grounding,
                "all_pass": all_grounding_pass,
                "avg_latency_ms": round(avg_grounding_latency, 4),
            },
            "contradiction_prompt": contradiction,
            "evidence_annotations": evidence,
            "citation_fallback": citation,
        },
        "overall_pass": all_grounding_pass and contradiction["contradiction_alert_present"],
    }

    # Write report
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DATA_DIR / "phase25_synthesis_quality.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_all_benchmarks()
    print(json.dumps(summary, indent=2))
