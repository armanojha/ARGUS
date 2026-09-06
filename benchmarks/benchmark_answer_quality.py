"""Phase 26 benchmark: Answer quality evaluation baseline.

Establishes baseline metrics by evaluating answer quality against
the existing benchmark datasets. Uses gold_facts as synthetic answers
to establish upper-bound metrics.

Reports per-category breakdowns and overall metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.evaluation.answer_quality import (
    AnswerQualityResult,
    evaluate_answer,
)

BENCHMARK_DIR = Path("benchmarks")
DATA_DIR = BENCHMARK_DIR / "data"
EVAL_PLAN_PATH = DATA_DIR / "eval_plan_v1.json" if (DATA_DIR / "eval_plan_v1.json").exists() else BENCHMARK_DIR / "eval_data" / "eval_plan_v1.json"
QUESTIONS_PATH = DATA_DIR / "questions_v1.json"
REPORT_DIR = DATA_DIR / "benchmark_reports"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_fake_citations(gold_facts: list[str]) -> list[Any]:
    """Create fake citation objects for synthetic answers."""
    from dataclasses import dataclass

    @dataclass
    class FakeCitation:
        text: str
        ref_id: int = 1

    return [FakeCitation(text=fact, ref_id=i + 1) for i, fact in enumerate(gold_facts)]


def _build_cited_answer(gold_facts: list[str]) -> str:
    """Build a synthetic answer with bracket citations [N]."""
    if not gold_facts:
        return ""
    parts = []
    for i, fact in enumerate(gold_facts, 1):
        fact_text = fact.strip().rstrip(".")
        parts.append(f"{fact_text} [{i}]")
    return ". ".join(parts) + "."


def evaluate_eval_plan() -> dict[str, Any]:
    """Evaluate the 38-query eval plan with gold_facts as answers."""
    plan = _load_json(EVAL_PLAN_PATH)
    queries = plan["queries"]
    query_classes = plan.get("query_classes", {})

    results_by_class: dict[str, list[dict]] = {}
    all_results = []

    for q in queries:
        qid = q["id"]
        qclass = q.get("class", "unknown")
        gold_facts = q.get("gold_facts", [])
        absent = q.get("absent", False)
        conflict = q.get("conflict", False)

        # Build synthetic answer from gold facts with citations
        if absent:
            answer = "Information not available in the provided evidence."
            citations = []
        elif gold_facts:
            answer = _build_cited_answer(gold_facts)
            citations = _make_fake_citations(gold_facts)
        else:
            answer = "No specific facts available."
            citations = []

        # Evaluate
        start = time.perf_counter()
        result = evaluate_answer(answer, citations, gold_facts=gold_facts)
        latency_ms = (time.perf_counter() - start) * 1000

        item_result = {
            "id": qid,
            "class": qclass,
            "query": q["query"],
            "absent": absent,
            "conflict": conflict,
            "gold_facts_count": len(gold_facts),
            "claims_count": result.total_claims,
            "claim_support_rate": result.claim_support_rate,
            "unsupported_claim_rate": result.unsupported_claim_rate,
            "citation_presence_rate": result.citation_presence_rate,
            "citation_validity_rate": result.citation_validity_rate,
            "gold_fact_coverage": result.gold_fact_coverage,
            "gold_facts_found": len(result.gold_facts_found),
            "gold_facts_missing": len(result.gold_facts_missing),
            "numerical_consistency_rate": result.numerical_consistency_rate,
            "latency_ms": round(latency_ms, 4),
            "warnings": result.warnings,
        }

        all_results.append(item_result)
        results_by_class.setdefault(qclass, []).append(item_result)

    # Aggregate per class
    class_summary = {}
    for cls, items in results_by_class.items():
        n = len(items)
        class_summary[cls] = {
            "count": n,
            "avg_claim_support": round(sum(i["claim_support_rate"] for i in items) / n, 4),
            "avg_citation_presence": round(sum(i["citation_presence_rate"] for i in items) / n, 4),
            "avg_gold_coverage": round(
                sum(i["gold_fact_coverage"] for i in items if i["gold_fact_coverage"] is not None) /
                max(1, sum(1 for i in items if i["gold_fact_coverage"] is not None)),
                4,
            ),
            "avg_latency_ms": round(sum(i["latency_ms"] for i in items) / n, 4),
        }

    # Overall aggregate
    n = len(all_results)
    overall = {
        "total_queries": n,
        "avg_claim_support_rate": round(sum(i["claim_support_rate"] for i in all_results) / n, 4),
        "avg_citation_presence_rate": round(sum(i["citation_presence_rate"] for i in all_results) / n, 4),
        "avg_citation_validity_rate": round(sum(i["citation_validity_rate"] for i in all_results) / n, 4),
        "avg_unsupported_claim_rate": round(sum(i["unsupported_claim_rate"] for i in all_results) / n, 4),
        "avg_latency_ms": round(sum(i["latency_ms"] for i in all_results) / n, 4),
    }

    return {
        "dataset": "eval_plan_v1",
        "dataset_size": n,
        "overall": overall,
        "by_class": class_summary,
        "items": all_results,
    }


def evaluate_questions_v1() -> dict[str, Any]:
    """Evaluate the 110-question dataset with gold_answer as answers."""
    data = _load_json(QUESTIONS_PATH)
    items = data.get("items", [])
    adversarial = data.get("adversarial_cases", [])

    results_by_type: dict[str, list[dict]] = {}
    all_results = []

    for q in items + adversarial:
        qid = q.get("id", "unknown")
        qtype = q.get("type", q.get("adversarial_type", "unknown"))
        gold_answer = q.get("gold_answer", "")
        gold_evidence = q.get("gold_evidence", [])
        gold_facts = q.get("gold_facts", gold_evidence)

        # Build answer with citations from gold evidence
        if gold_evidence:
            answer = _build_cited_answer(gold_evidence)
        elif gold_answer:
            answer = gold_answer
        else:
            answer = ". ".join(gold_facts) + "." if gold_facts else "No facts."
        citations = _make_fake_citations(gold_evidence) if gold_evidence else []

        # Evaluate
        start = time.perf_counter()
        result = evaluate_answer(gold_answer, citations, gold_facts=gold_facts)
        latency_ms = (time.perf_counter() - start) * 1000

        item_result = {
            "id": qid,
            "type": qtype,
            "gold_answer_length": len(gold_answer),
            "gold_evidence_count": len(gold_evidence),
            "claims_count": result.total_claims,
            "claim_support_rate": result.claim_support_rate,
            "unsupported_claim_rate": result.unsupported_claim_rate,
            "citation_presence_rate": result.citation_presence_rate,
            "gold_fact_coverage": result.gold_fact_coverage,
            "numerical_consistency_rate": result.numerical_consistency_rate,
            "latency_ms": round(latency_ms, 4),
        }

        all_results.append(item_result)
        results_by_type.setdefault(qtype, []).append(item_result)

    # Aggregate per type
    type_summary = {}
    for t, type_items in results_by_type.items():
        n = len(type_items)
        type_summary[t] = {
            "count": n,
            "avg_claim_support": round(sum(i["claim_support_rate"] for i in type_items) / n, 4),
            "avg_citation_presence": round(sum(i["citation_presence_rate"] for i in type_items) / n, 4),
            "avg_latency_ms": round(sum(i["latency_ms"] for i in type_items) / n, 4),
        }

    n = len(all_results)
    overall = {
        "total_queries": n,
        "avg_claim_support_rate": round(sum(i["claim_support_rate"] for i in all_results) / n, 4),
        "avg_citation_presence_rate": round(sum(i["citation_presence_rate"] for i in all_results) / n, 4),
        "avg_latency_ms": round(sum(i["latency_ms"] for i in all_results) / n, 4),
    }

    return {
        "dataset": "questions_v1",
        "dataset_size": n,
        "overall": overall,
        "by_type": type_summary,
        "items": all_results,
    }


def run_benchmark():
    """Run both benchmarks and produce a combined report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    eval_plan_results = evaluate_eval_plan()
    questions_results = evaluate_questions_v1()

    report = {
        "phase": 26,
        "title": "Answer Quality Evaluation Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": {
            "eval_plan_v1": eval_plan_results["overall"],
            "questions_v1": questions_results["overall"],
        },
        "by_category": {
            "eval_plan_by_class": eval_plan_results["by_class"],
            "questions_by_type": questions_results["by_type"],
        },
    }

    # Write report
    report_path = REPORT_DIR / "phase26_answer_quality_baseline.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write detailed results
    detail_path = REPORT_DIR / "phase26_eval_plan_details.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(eval_plan_results, f, indent=2)

    detail_path2 = REPORT_DIR / "phase26_questions_details.json"
    with open(detail_path2, "w", encoding="utf-8") as f:
        json.dump(questions_results, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report, indent=2))
