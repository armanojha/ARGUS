"""Phase 18: Failure Forensics - Comprehensive diagnostic for hard queries.

For each evaluation query, produces:
- query, canonical class, EvidenceNeeds, generated subqueries
- retrieved candidates with scores and ranks
- verifier result (if applicable)
- missing EvidenceNeed
- recovery activation and result
- graph traversal result if applicable
- failure classification

DO NOT implement anything until this failure classification is complete.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.evidence.models import EvidenceRef
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import FAISSVectorStore
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.planner import EvidenceNeedPlanner, QueryPlan, EvidenceNeed
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.recovery import TargetedRecovery, CoverageAnalysis
from app.retrieval.router import RetrievalPolicyRouter


class FailureClass(Enum):
    """Classification of retrieval failures."""
    MISSING_TOPIC = "A_missing_topic"
    WRONG_ENTITY = "B_wrong_entity"
    WEAK_SEMANTIC = "C_weak_semantic"
    LEXICAL_MISMATCH = "D_lexical_mismatch"
    MISSING_RELATIONSHIP = "E_missing_relationship"
    INSUFFICIENT_CONTEXT = "F_insufficient_context"
    CHUNK_BOUNDARY = "G_chunk_boundary"
    INSUFFICIENT_SOURCE_COVERAGE = "H_insufficient_source_coverage"
    GRAPH_TRAVERSAL_FAILURE = "I_graph_traversal_failure"
    RERANKING_FAILURE = "J_reranking_failure"
    ABSENT_EVIDENCE = "K_genuinely_absent"
    OTHER = "L_other"


@dataclass
class QueryForensics:
    """Complete forensic analysis of a single query."""
    # Query metadata
    query_id: str
    query_text: str
    canonical_class: str
    eval_class: str
    supporting_docs: list[str] = field(default_factory=list)
    gold_facts: list[str] = field(default_factory=list)

    # Evidence needs (from planner)
    evidence_needs: list[dict[str, Any]] = field(default_factory=list)
    planner_activated: bool = False
    planner_latency_ms: float = 0.0

    # Subqueries generated
    subqueries: list[str] = field(default_factory=list)

    # Retrieved candidates
    candidates: list[dict[str, Any]] = field(default_factory=list)
    total_candidates: int = 0

    # Gold chunk info
    gold_chunk_ids: set[str] = field(default_factory=set)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Verifier result (if applicable)
    verifier_activated: bool = False
    verifier_coverage: dict[str, float] = field(default_factory=dict)
    verifier_result: str = ""

    # Recovery result
    recovery_activated: bool = False
    recovery_type: str = ""
    recovery_attempts: int = 0
    recovery_candidates_added: int = 0
    coverage_before_recovery: float = 0.0
    coverage_after_recovery: float = 0.0

    # Graph traversal (if applicable)
    graph_traversal_activated: bool = False
    graph_traversal_result: str = ""

    # Latency breakdown
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0

    # Failure classification
    failure_class: FailureClass = FailureClass.OTHER
    failure_reason: str = ""
    failure_details: dict[str, Any] = field(default_factory=dict)


def load_eval_plan() -> dict:
    """Load the evaluation plan."""
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        return json.load(f)


def build_store():
    """Build the evaluation store."""
    from benchmarks.benchmark_fusion import build_benchmark_store
    return build_benchmark_store()


def trace_query(
    query: dict,
    retriever: HybridRetriever,
    router: RetrievalPolicyRouter,
    planner: EvidenceNeedPlanner,
    multi_query: MultiQueryRetriever,
    chunk_id_map: dict[str, list[str]],
    top_k: int = 10,
) -> QueryForensics:
    """Trace a single query with full forensic detail."""
    forensics = QueryForensics(
        query_id=query["id"],
        query_text=query["query"],
        canonical_class=query.get("canonical_class", ""),
        eval_class=query.get("eval_class", ""),
        supporting_docs=query.get("supporting_docs", []),
        gold_facts=query.get("gold_facts", []),
    )

    # Compute gold chunk IDs
    for doc_id in forensics.supporting_docs:
        if doc_id in chunk_id_map:
            forensics.gold_chunk_ids.update(chunk_id_map[doc_id])

    # Classify query
    from app.retrieval.policy import QuestionPattern
    pattern = router.classify_question(query["query"])
    pattern_value = pattern.value if hasattr(pattern, 'value') else pattern

    # Check if planner should activate
    plannable = pattern_value in EvidenceNeedPlanner.PLANNABLE_PATTERNS

    # Run planned retrieval if applicable
    t0 = time.perf_counter()
    if plannable:
        # Create query plan
        plan_start = time.perf_counter()
        plan = planner.plan(query["query"], pattern_value)
        forensics.planner_latency_ms = (time.perf_counter() - plan_start) * 1000
        forensics.planner_activated = plan.is_planned

        if plan.is_planned:
            # Extract evidence needs
            for need in plan.evidence_needs:
                forensics.evidence_needs.append({
                    "id": need.id,
                    "topic": need.topic,
                    "entities": need.entities,
                    "claim_type": need.claim_type.value,
                    "search_query": need.search_query,
                    "priority": need.priority.value,
                    "original_need": need.original_need,
                })

            # Extract subqueries
            forensics.subqueries = [n.search_query for n in plan.evidence_needs if n.search_query]

            # Run multi-query retrieval
            result = asyncio.run(multi_query.retrieve(plan))

            # Extract candidates
            for ref in result.selected:
                forensics.candidates.append({
                    "chunk_id": str(ref.chunk_id),
                    "document_id": str(ref.document_id),
                    "score": ref.score,
                    "rank": ref.rank,
                    "source_path": ref.source_path,
                    "text_preview": ref.text[:100],
                    "evidence_need_id": ref.metadata.get("evidence_need_id", ""),
                })

            forensics.total_candidates = result.total_candidates
            forensics.retrieved_chunk_ids = [str(r.chunk_id) for r in result.selected]

            # Recovery info
            forensics.recovery_activated = result.recovery_activated
            forensics.recovery_type = result.recovery_type.value if result.recovery_activated else ""
            forensics.recovery_attempts = result.recovery_attempts
            forensics.recovery_candidates_added = result.recovery_candidates_added
            forensics.coverage_before_recovery = result.coverage_before_recovery
            forensics.coverage_after_recovery = result.coverage_after_recovery

            # Verifier coverage
            forensics.verifier_coverage = result.need_coverage
            forensics.verifier_activated = bool(result.need_coverage)
            uncovered_needs = [k for k, v in result.need_coverage.items() if v < 1.0]
            if uncovered_needs:
                forensics.verifier_result = f"{len(uncovered_needs)}/{len(result.need_coverage)} needs uncovered"
            elif result.need_coverage:
                forensics.verifier_result = "all needs covered"
    else:
        # Simple retrieval (no planner)
        refs = retriever.search(query["query"], top_k=top_k)
        forensics.total_candidates = len(refs)
        forensics.retrieved_chunk_ids = [str(r.chunk_id) for r in refs]
        for ref in refs:
            forensics.candidates.append({
                "chunk_id": str(ref.chunk_id),
                "document_id": str(ref.document_id),
                "score": ref.score,
                "rank": ref.rank,
                "source_path": ref.source_path,
                "text_preview": ref.text[:100],
                "evidence_need_id": "",
            })

    forensics.retrieval_latency_ms = (time.perf_counter() - t0) * 1000
    forensics.total_latency_ms = forensics.retrieval_latency_ms

    # Compute hits and misses
    retrieved_set = set(forensics.retrieved_chunk_ids)
    forensics.hits = list(retrieved_set & forensics.gold_chunk_ids)
    forensics.misses = list(forensics.gold_chunk_ids - retrieved_set)

    # Compute recall
    if forensics.gold_chunk_ids:
        forensics.recall_at_5 = len(set(forensics.retrieved_chunk_ids[:5]) & forensics.gold_chunk_ids) / len(forensics.gold_chunk_ids)
        forensics.recall_at_10 = len(set(forensics.retrieved_chunk_ids[:10]) & forensics.gold_chunk_ids) / len(forensics.gold_chunk_ids)

    # Classify failure
    if not forensics.misses:
        forensics.failure_class = FailureClass.OTHER
        forensics.failure_reason = "No misses - query passed"
    elif forensics.recall_at_10 >= 0.5:
        forensics.failure_class = FailureClass.OTHER
        forensics.failure_reason = f"Partial success (R@10={forensics.recall_at_10:.3f})"
    else:
        # Analyze miss patterns
        forensics.failure_class, forensics.failure_reason = _classify_failure(forensics, chunk_id_map)

    return forensics


def _classify_failure(f: QueryForensics, chunk_id_map: dict[str, list[str]]) -> tuple[FailureClass, str]:
    """Classify the failure based on forensic evidence."""
    # Check if planner generated wrong evidence needs
    if f.planner_activated and not f.evidence_needs:
        return FailureClass.MISSING_TOPIC, "Planner activated but generated no evidence needs"

    # Check if recovery was needed but failed
    if f.recovery_activated and f.recovery_candidates_added == 0:
        return FailureClass.INSUFFICIENT_SOURCE_COVERAGE, "Recovery activated but added no candidates"

    # Check if coverage was low after retrieval
    if f.verifier_coverage:
        uncovered = [k for k, v in f.verifier_coverage.items() if v < 0.5]
        if uncovered:
            return FailureClass.MISSING_TOPIC, f"{len(uncovered)}/{len(f.verifier_coverage)} needs uncovered"

    # Check for chunk boundary issues (gold chunks from same doc but different chunks)
    if f.misses and f.hits:
        hit_docs = set()
        miss_docs = set()
        for chunk_id in f.hits:
            for doc_id, chunks in chunk_id_map.items():
                if chunk_id in chunks:
                    hit_docs.add(doc_id)
        for chunk_id in f.misses:
            for doc_id, chunks in chunk_id_map.items():
                if chunk_id in chunks:
                    miss_docs.add(doc_id)
        if hit_docs & miss_docs:
            return FailureClass.CHUNK_BOUNDARY, f"Hit some chunks from docs {hit_docs & miss_docs} but missed others"

    # Check for insufficient source coverage
    retrieved_docs = set()
    for candidate in f.candidates:
        retrieved_docs.add(candidate.get("document_id", ""))
    if len(f.supporting_docs) > 1:
        covered_docs = retrieved_docs & set(f.supporting_docs)
        if len(covered_docs) < len(f.supporting_docs):
            return FailureClass.INSUFFICIENT_SOURCE_COVERAGE, f"Only {len(covered_docs)}/{len(f.supporting_docs)} supporting docs covered"

    # Default classification
    return FailureClass.WEAK_SEMANTIC, "Gold chunks not retrieved in top-10"


def run_forensics():
    """Run comprehensive failure forensics on all evaluation queries."""
    print("=" * 90)
    print("PHASE 18: FAILURE FORENSICS")
    print("=" * 90)

    # Load eval plan
    plan = load_eval_plan()
    print(f"Loaded eval plan: {len(plan.get('queries', []))} queries")

    # Build store
    print("\n[1/4] Building evaluation corpus...")
    t0 = time.monotonic()
    store, chunk_id_map = build_store()
    total_chunks = sum(len(v) for v in chunk_id_map.values())
    print(f"  Corpus: {total_chunks} chunks from {len(chunk_id_map)} docs ({time.monotonic()-t0:.1f}s)")

    # Build index
    print("\n[2/4] Building retrieval index...")
    t0 = time.monotonic()
    retriever = HybridRetriever(store)
    retriever.ensure_indexes()
    print(f"  Index built in {time.monotonic()-t0:.1f}s")

    # Build components
    router = RetrievalPolicyRouter()
    planner = EvidenceNeedPlanner()
    multi_query = MultiQueryRetriever(router=router, retriever=retriever, top_k=10)

    # Warm up
    print("\n[3/4] Warming up...")
    for q in plan.get("queries", [])[:2]:
        retriever.search(q["query"], top_k=10)

    # Run forensics on all queries
    print(f"\n[4/4] Running forensics on {len(plan.get('queries', []))} queries...")
    all_forensics = []

    for q in plan.get("queries", []):
        # Convert eval class to canonical
        from app.retrieval.policy import QuestionPattern
        eval_class = q.get("class", "unknown")
        canonical_pattern = QuestionPattern.from_eval_class(eval_class)

        query_data = {
            "id": q.get("id", "unknown"),
            "query": q.get("query", ""),
            "eval_class": eval_class,
            "canonical_class": canonical_pattern.value,
            "supporting_docs": q.get("supporting_docs", []),
            "gold_facts": q.get("gold_facts", []),
        }

        f = trace_query(query_data, retriever, router, planner, multi_query, chunk_id_map)
        all_forensics.append(f)

    # Print summary
    print("\n" + "=" * 90)
    print("FORENSICS SUMMARY")
    print("=" * 90)

    # Group by pattern
    by_pattern = defaultdict(list)
    for f in all_forensics:
        by_pattern[f.canonical_class].append(f)

    # Print failures for hard patterns
    hard_patterns = ["complex_research", "multi_hop", "adversarial"]
    for pattern in hard_patterns:
        pf = by_pattern.get(pattern, [])
        if not pf:
            continue

        print(f"\n{'=' * 90}")
        print(f"PATTERN: {pattern} ({len(pf)} queries)")
        print(f"{'=' * 90}")

        for f in pf:
            status = "PASS" if f.recall_at_10 >= 0.5 else "FAIL"
            print(f"\n--- [{status}] {f.query_id}: R@5={f.recall_at_5:.3f} R@10={f.recall_at_10:.3f} ---")
            print(f"  Query: {f.query_text[:100]}")
            print(f"  Canonical: {f.canonical_class}, Eval: {f.eval_class}")
            print(f"  Gold docs: {f.supporting_docs}")
            print(f"  Gold facts: {f.gold_facts[:3]}")
            print(f"  Planner activated: {f.planner_activated}")
            print(f"  Evidence needs: {len(f.evidence_needs)}")
            print(f"  Subqueries: {len(f.subqueries)}")
            print(f"  Candidates: {f.total_candidates}")
            print(f"  Hits: {len(f.hits)}, Misses: {len(f.misses)}")
            print(f"  Failure class: {f.failure_class.value}")
            print(f"  Failure reason: {f.failure_reason}")

            if f.evidence_needs:
                print(f"  Evidence needs:")
                for i, need in enumerate(f.evidence_needs[:3]):
                    print(f"    [{i}] topic={need['topic']}, claim={need['claim_type']}, query={need['search_query'][:60]}")

            if f.verifier_coverage:
                print(f"  Verifier coverage: {f.verifier_coverage}")

            if f.recovery_activated:
                print(f"  Recovery: type={f.recovery_type}, attempts={f.recovery_attempts}, added={f.recovery_candidates_added}")

    # Summary by pattern
    print(f"\n{'=' * 90}")
    print("PATTERN SUMMARY")
    print(f"{'=' * 90}")
    for pattern in sorted(by_pattern.keys()):
        pf = by_pattern[pattern]
        avg_r5 = np.mean([f.recall_at_5 for f in pf])
        avg_r10 = np.mean([f.recall_at_10 for f in pf])
        fail_count = sum(1 for f in pf if f.recall_at_10 < 0.5)
        print(f"  {pattern:<25} R@5={avg_r5:.3f} R@10={avg_r10:.3f}  queries={len(pf)}  failed={fail_count}")

    # Failure class distribution
    print(f"\n{'=' * 90}")
    print("FAILURE CLASS DISTRIBUTION")
    print(f"{'=' * 90}")
    failure_counts = defaultdict(int)
    for f in all_forensics:
        if f.recall_at_10 < 0.5:
            failure_counts[f.failure_class.value] += 1
    for fc, count in sorted(failure_counts.items()):
        print(f"  {fc}: {count}")

    # Save full forensics
    report_path = Path("data/benchmark_reports/phase18_forensics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = []
    for f in all_forensics:
        serializable.append({
            "query_id": f.query_id,
            "query_text": f.query_text,
            "canonical_class": f.canonical_class,
            "eval_class": f.eval_class,
            "supporting_docs": f.supporting_docs,
            "gold_facts": f.gold_facts,
            "evidence_needs": f.evidence_needs,
            "planner_activated": f.planner_activated,
            "planner_latency_ms": f.planner_latency_ms,
            "subqueries": f.subqueries,
            "candidates": f.candidates[:10],  # Limit to top 10
            "total_candidates": f.total_candidates,
            "gold_chunk_ids": list(f.gold_chunk_ids),
            "retrieved_chunk_ids": f.retrieved_chunk_ids,
            "hits": f.hits,
            "misses": f.misses,
            "recall_at_5": f.recall_at_5,
            "recall_at_10": f.recall_at_10,
            "verifier_activated": f.verifier_activated,
            "verifier_coverage": f.verifier_coverage,
            "verifier_result": f.verifier_result,
            "recovery_activated": f.recovery_activated,
            "recovery_type": f.recovery_type,
            "recovery_attempts": f.recovery_attempts,
            "recovery_candidates_added": f.recovery_candidates_added,
            "coverage_before_recovery": f.coverage_before_recovery,
            "coverage_after_recovery": f.coverage_after_recovery,
            "total_latency_ms": f.total_latency_ms,
            "retrieval_latency_ms": f.retrieval_latency_ms,
            "failure_class": f.failure_class.value,
            "failure_reason": f.failure_reason,
        })

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nFull forensics saved to {report_path}")

    return all_forensics


if __name__ == "__main__":
    run_forensics()
