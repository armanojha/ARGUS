#!/usr/bin/env python3
"""Phase 27: Real ARGUS Answer Quality Benchmark.

Runs the ACTUAL ARGUS pipeline end-to-end on curated queries and
evaluates the synthesized answers using the improved evaluator.

Produces per-query results with quality layer decomposition.

Usage:
    python benchmarks/benchmark_phase27.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.answer_quality import (
    AnswerQualityResult,
    ClaimSupportStatus,
    evaluate_answer,
)

BENCHMARK_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Curated Core Set ────────────────────────────────────────────
# 40 queries covering all difficult categories.
# Gold facts are human-verified from the corpus documents.

CORE_SET: list[dict[str, Any]] = [
    # ── Simple Lookup (baseline sanity) ──────────────────────────
    {
        "id": "P27-A01", "class": "simple_lookup", "difficulty": "easy",
        "query": "Where is Acme Corporation headquartered?",
        "gold_facts": ["New York City"],
        "corpus_docs": ["doc-a"],
    },
    {
        "id": "P27-A02", "class": "simple_lookup", "difficulty": "easy",
        "query": "What year was Acme Corporation founded?",
        "gold_facts": ["1987"],
        "corpus_docs": ["doc-a"],
    },
    # ── Multi-doc Synthesis ──────────────────────────────────────
    {
        "id": "P27-D01", "class": "multi_doc_synthesis", "difficulty": "medium",
        "query": "How many employees did Acme report for 2025 and what was the revenue trend?",
        "gold_facts": ["12,400", "revenue", "2025"],
        "corpus_docs": ["doc-a", "doc-f"],
    },
    {
        "id": "P27-D02", "class": "multi_doc_synthesis", "difficulty": "medium",
        "query": "What is the total manufacturing capacity across Acme's facilities?",
        "gold_facts": ["manufacturing", "capacity", "facilities"],
        "corpus_docs": ["doc-a", "doc-k"],
    },
    {
        "id": "P27-D03", "class": "multi_doc_synthesis", "difficulty": "hard",
        "query": "Compare Acme's 2023 and 2025 revenue figures and explain the trend.",
        "gold_facts": ["revenue", "2023", "2025", "trend"],
        "corpus_docs": ["doc-e", "doc-f"],
    },
    # ── Multi-hop ────────────────────────────────────────────────
    {
        "id": "P27-E01", "class": "multi_hop", "difficulty": "medium",
        "query": "What database engine does the supply chain system use and what is its default storage format?",
        "gold_facts": ["Atlas", "columnar", "supply chain"],
        "corpus_docs": ["doc-b", "doc-c"],
    },
    {
        "id": "P27-E02", "class": "multi_hop", "difficulty": "hard",
        "query": "What is the Polaris probe depth and what data does it collect for the supply chain analysis?",
        "gold_facts": ["1,400 meters", "Polaris probe", "supply chain"],
        "corpus_docs": ["doc-j", "doc-c"],
    },
    {
        "id": "P27-E03", "class": "multi_hop", "difficulty": "hard",
        "query": "How does the Atlas database handle time-series data from the Polaris probe?",
        "gold_facts": ["Atlas", "time-series", "Polaris", "probe"],
        "corpus_docs": ["doc-b", "doc-j"],
    },
    # ── Conflict ─────────────────────────────────────────────────
    {
        "id": "P27-F01", "class": "conflict", "difficulty": "hard",
        "query": "What were Acme's revenue figures for 2023 according to different sources?",
        "gold_facts": ["revenue", "2023", "conflict"],
        "corpus_docs": ["doc-e", "doc-f"],
        "conflict": True,
    },
    {
        "id": "P27-F02", "class": "conflict", "difficulty": "hard",
        "query": "Are there conflicting reports about Acme's employee count?",
        "gold_facts": ["employees", "conflict"],
        "corpus_docs": ["doc-a", "doc-f"],
        "conflict": True,
    },
    # ── Numerical ────────────────────────────────────────────────
    {
        "id": "P27-H01", "class": "numerical", "difficulty": "medium",
        "query": "What was Acme's revenue per employee in 2025?",
        "gold_facts": ["revenue", "employees", "per employee", "2025"],
        "corpus_docs": ["doc-a", "doc-f"],
    },
    {
        "id": "P27-H02", "class": "numerical", "difficulty": "hard",
        "query": "What is the Atlas database query latency at P99?",
        "gold_facts": ["latency", "P99", "Atlas"],
        "corpus_docs": ["doc-b"],
    },
    {
        "id": "P27-H03", "class": "numerical", "difficulty": "medium",
        "query": "What was the Polaris probe's data collection rate?",
        "gold_facts": ["data collection", "rate", "Polaris"],
        "corpus_docs": ["doc-j"],
    },
    # ── Technical Explanation ────────────────────────────────────
    {
        "id": "P27-C01", "class": "technical_explanation", "difficulty": "medium",
        "query": "Explain how the Atlas columnar storage engine works for analytical queries.",
        "gold_facts": ["Atlas", "columnar", "analytical", "storage"],
        "corpus_docs": ["doc-b"],
    },
    {
        "id": "P27-C02", "class": "technical_explanation", "difficulty": "hard",
        "query": "What is the data center architecture used for the Frontier Fusion system?",
        "gold_facts": ["data center", "Frontier Fusion", "architecture"],
        "corpus_docs": ["doc-g", "doc-l"],
    },
    # ── Complex Research ─────────────────────────────────────────
    {
        "id": "P27-I01", "class": "complex_research", "difficulty": "hard",
        "query": "What is Acme's overall technology strategy based on their product roadmap and recent investments?",
        "gold_facts": ["strategy", "roadmap", "investments", "technology"],
        "corpus_docs": ["doc-i", "doc-a"],
    },
    {
        "id": "P27-I02", "class": "complex_research", "difficulty": "hard",
        "query": "How does the Q3 metrics report compare to the product roadmap targets?",
        "gold_facts": ["Q3", "metrics", "roadmap", "targets"],
        "corpus_docs": ["doc-h", "doc-i"],
    },
    # ── Absent Info ──────────────────────────────────────────────
    {
        "id": "P27-G01", "class": "absent_info", "difficulty": "medium",
        "query": "What is Acme's market share in the European robotics market?",
        "gold_facts": [],
        "corpus_docs": ["doc-a"],
        "absent": True,
    },
    {
        "id": "P27-G02", "class": "absent_info", "difficulty": "medium",
        "query": "How many patents does Atlas database hold?",
        "gold_facts": [],
        "corpus_docs": ["doc-b"],
        "absent": True,
    },
    # ── Adversarial ──────────────────────────────────────────────
    {
        "id": "P27-J01", "class": "adversarial", "difficulty": "hard",
        "query": "Was the Polaris probe deployed at 1,400 meters or 1,600 meters?",
        "gold_facts": ["1,400 meters"],
        "corpus_docs": ["doc-j"],
    },
    {
        "id": "P27-J02", "class": "adversarial", "difficulty": "hard",
        "query": "What is the difference between Atlas's columnar and row-based storage?",
        "gold_facts": ["columnar", "Atlas", "storage"],
        "corpus_docs": ["doc-b"],
    },
]


# ─── Benchmark Runner ────────────────────────────────────────────


async def run_single_query(
    query_item: dict[str, Any],
    retriever: Any,
) -> dict[str, Any]:
    """Run ARGUS on a single query using the full pipeline."""
    query = query_item["query"]
    query_id = query_item["id"]

    start = time.time()
    try:
        from app.orchestration.graph import run_query
        result = await run_query(query, retriever=retriever)
        elapsed = time.time() - start

        return {
            "query_id": query_id,
            "query": query,
            "class": query_item["class"],
            "difficulty": query_item.get("difficulty", "unknown"),
            "answer": result.answer,
            "citations": [
                {"text": c.text, "document_id": str(c.document_id), "score": c.score}
                for c in result.citations
            ],
            "iterations_used": result.iterations_used,
            "stop_reason": result.stop_reason.value if hasattr(result.stop_reason, "value") else str(result.stop_reason),
            "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
            "warnings": result.warnings,
            "elapsed_seconds": elapsed,
            "success": True,
            "gold_facts": query_item.get("gold_facts", []),
            "absent": query_item.get("absent", False),
            "conflict": query_item.get("conflict", False),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "query_id": query_id,
            "query": query,
            "class": query_item["class"],
            "difficulty": query_item.get("difficulty", "unknown"),
            "answer": "",
            "citations": [],
            "iterations_used": 0,
            "stop_reason": "error",
            "outcome": "error",
            "warnings": [],
            "elapsed_seconds": elapsed,
            "success": False,
            "error": str(e),
            "gold_facts": query_item.get("gold_facts", []),
            "absent": query_item.get("absent", False),
            "conflict": query_item.get("conflict", False),
        }


def evaluate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single benchmark result using the answer quality evaluator."""
    if not result["success"] or not result["answer"]:
        return {**result, "evaluation": None}

    class FakeCitation:
        def __init__(self, text: str):
            self.text = text

    citations = [FakeCitation(text=c["text"]) for c in result["citations"]]

    eval_result = evaluate_answer(
        result["answer"],
        citations,
        gold_facts=result.get("gold_facts") or None,
        query=result["query"],
    )

    return {
        **result,
        "evaluation": {
            "claim_support_rate": eval_result.claim_support_rate,
            "unsupported_claim_rate": eval_result.unsupported_claim_rate,
            "partially_supported_rate": eval_result.partially_supported_rate,
            "contradicted_claim_rate": eval_result.contradicted_claim_rate,
            "citation_presence_rate": eval_result.citation_presence_rate,
            "citation_validity_rate": eval_result.citation_validity_rate,
            "citation_precision": eval_result.citation_precision,
            "gold_fact_coverage": eval_result.gold_fact_coverage,
            "gold_facts_found": eval_result.gold_facts_found,
            "gold_facts_missing": eval_result.gold_facts_missing,
            "gold_facts_negated": eval_result.gold_facts_negated,
            "numerical_consistency_rate": eval_result.numerical_consistency_rate,
            "query_relevance": eval_result.query_relevance,
            "total_claims": eval_result.total_claims,
            "cited_claims": eval_result.cited_claims,
            "warnings": eval_result.warnings,
            "claim_details": [
                {
                    "text": c.claim_text[:120],
                    "status": c.support_status.value,
                    "citations": c.citation_ids,
                    "key_terms_in_evidence": c.key_terms_in_evidence,
                    "numerical_match": c.numerical_match,
                }
                for c in eval_result.claims
            ],
        },
    }


def compute_aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across all results."""
    evaluated = [r for r in results if r.get("evaluation")]
    if not evaluated:
        return {"error": "no_evaluated_results"}

    n = len(evaluated)

    by_class: dict[str, list] = {}
    for r in evaluated:
        cls = r["class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(r)

    class_metrics = {}
    for cls, items in by_class.items():
        class_metrics[cls] = {
            "count": len(items),
            "avg_claim_support_rate": sum(
                i["evaluation"]["claim_support_rate"] for i in items
            ) / len(items),
            "avg_citation_precision": sum(
                i["evaluation"]["citation_precision"] for i in items
            ) / len(items),
            "avg_query_relevance": sum(
                i["evaluation"].get("query_relevance") or 0.0 for i in items
            ) / len(items),
            "total_gold_found": sum(
                len(i["evaluation"]["gold_facts_found"]) for i in items
            ),
            "total_gold_missing": sum(
                len(i["evaluation"]["gold_facts_missing"]) for i in items
            ),
            "total_gold_negated": sum(
                len(i["evaluation"]["gold_facts_negated"]) for i in items
            ),
        }

    total_claims = sum(r["evaluation"]["total_claims"] for r in evaluated)
    total_gold_found = sum(len(r["evaluation"]["gold_facts_found"]) for r in evaluated)
    total_gold_missing = sum(len(r["evaluation"]["gold_facts_missing"]) for r in evaluated)
    total_gold_negated = sum(len(r["evaluation"]["gold_facts_negated"]) for r in evaluated)
    total_gold = total_gold_found + total_gold_missing + total_gold_negated

    return {
        "total_queries": n,
        "successful_queries": sum(1 for r in results if r["success"]),
        "failed_queries": sum(1 for r in results if not r["success"]),
        "avg_claim_support_rate": sum(
            r["evaluation"]["claim_support_rate"] for r in evaluated
        ) / n,
        "avg_citation_presence": sum(
            r["evaluation"]["citation_presence_rate"] for r in evaluated
        ) / n,
        "avg_citation_precision": sum(
            r["evaluation"]["citation_precision"] for r in evaluated
        ) / n,
        "avg_query_relevance": sum(
            r["evaluation"].get("query_relevance") or 0.0 for r in evaluated
        ) / n,
        "overall_gold_coverage": total_gold_found / total_gold if total_gold > 0 else None,
        "total_gold_negated": total_gold_negated,
        "class_breakdown": class_metrics,
    }


def classify_failures(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify failures by root cause using quality layer decomposition."""
    failures = {
        "retrieval_failure": [],
        "evidence_failure": [],
        "synthesis_failure": [],
        "citation_failure": [],
        "off_topic": [],
        "absent_info_handled": [],
        "absent_info_not_handled": [],
        "numerical_error": [],
        "contradiction_missed": [],
    }

    for r in results:
        if not r["success"]:
            failures["retrieval_failure"].append(r["query_id"])
            continue

        eval_ = r.get("evaluation")
        if not eval_:
            continue

        qid = r["query_id"]

        relevance = eval_.get("query_relevance")
        if relevance is not None and relevance < 0.2:
            failures["off_topic"].append(qid)

        if r.get("absent"):
            if eval_["unsupported_claim_rate"] > 0.5:
                failures["absent_info_handled"].append(qid)
            else:
                failures["absent_info_not_handled"].append(qid)

        if eval_["citation_presence_rate"] < 0.5:
            failures["citation_failure"].append(qid)

        if eval_["numerical_consistency_rate"] is not None:
            if eval_["numerical_consistency_rate"] < 0.8:
                failures["numerical_error"].append(qid)

        if r.get("conflict") and eval_["contradicted_claim_rate"] == 0:
            failures["contradiction_missed"].append(qid)

        if eval_["claim_support_rate"] < 0.5:
            failures["synthesis_failure"].append(qid)

        if eval_["gold_facts_missing"]:
            failures["evidence_failure"].append(qid)

    return failures


async def main():
    """Run the full Phase 27 benchmark."""
    print("=" * 70)
    print("Phase 27: Real ARGUS Answer Quality Benchmark")
    print("=" * 70)

    # 1. Build benchmark store
    print("\n[1/5] Building benchmark store...")
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs")

    # 2. Initialize retriever
    print("[2/5] Initializing retriever...")
    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    # 3. Run queries
    print(f"[3/5] Running {len(CORE_SET)} queries through ARGUS...")
    results = []
    for i, query_item in enumerate(CORE_SET):
        print(f"  [{i+1}/{len(CORE_SET)}] {query_item['id']}: {query_item['query'][:60]}...")
        result = await run_single_query(query_item, retriever)
        results.append(result)

    # 4. Evaluate
    print("[4/5] Evaluating answers...")
    evaluated = [evaluate_result(r) for r in results]

    # 5. Compute aggregates
    print("[5/5] Computing aggregates...")
    aggregates = compute_aggregate_metrics(evaluated)
    failures = classify_failures(evaluated)

    # Save results
    output_file = OUTPUT_DIR / "phase27_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "core_set": CORE_SET,
                "results": evaluated,
                "aggregates": aggregates,
                "failures": failures,
            },
            f,
            indent=2,
            default=str,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total queries: {aggregates['total_queries']}")
    print(f"Successful: {aggregates['successful_queries']}")
    print(f"Failed: {aggregates['failed_queries']}")
    print(f"\nAvg Claim Support Rate: {aggregates['avg_claim_support_rate']:.1%}")
    print(f"Avg Citation Presence: {aggregates['avg_citation_presence']:.1%}")
    print(f"Avg Citation Precision: {aggregates['avg_citation_precision']:.1%}")
    print(f"Avg Query Relevance: {aggregates['avg_query_relevance']:.1%}")
    if aggregates["overall_gold_coverage"] is not None:
        print(f"Overall Gold Coverage: {aggregates['overall_gold_coverage']:.1%}")
    print(f"Gold Facts Negated: {aggregates['total_gold_negated']}")

    print("\n--- Per-Class Breakdown ---")
    for cls, metrics in aggregates["class_breakdown"].items():
        print(f"\n  {cls} (n={metrics['count']}):")
        print(f"    Claim Support: {metrics['avg_claim_support_rate']:.1%}")
        print(f"    Citation Precision: {metrics['avg_citation_precision']:.1%}")
        print(f"    Query Relevance: {metrics['avg_query_relevance']:.1%}")

    print("\n--- Failure Classification ---")
    for category, query_ids in failures.items():
        if query_ids:
            print(f"  {category}: {len(query_ids)} queries ({query_ids})")

    print(f"\nDetailed results saved to: {output_file}")
    return evaluated, aggregates, failures


if __name__ == "__main__":
    results = asyncio.run(main())
