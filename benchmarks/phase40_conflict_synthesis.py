#!/usr/bin/env python3
"""Phase 40: Conflict-Aware Synthesis Benchmark.

Builds a focused conflict benchmark from the existing eval_plan_v1 corpus,
runs controlled synthesis strategies, and measures conflict-specific quality.

Usage:
    python -m benchmarks.phase40_conflict_synthesis
"""
from __future__ import annotations

import io
import json
import sys
import time
import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Conflict Benchmark Queries ───────────────────────────────────────────────

CONFLICT_QUERIES = [
    # Direct numerical contradictions (2023 vs 2025)
    {
        "id": "C1",
        "query": "What was Acme's annual revenue?",
        "class": "direct_numerical",
        "source_a_claim": "Acme reported $3.1 billion in 2023",
        "source_b_claim": "Acme reported $4.7 billion in 2025",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["4.7", "billion", "3.1"],
        "forbidden_resolution": "The correct revenue is $4.7 billion.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
    {
        "id": "C2",
        "query": "How many employees does Acme have?",
        "class": "direct_numerical",
        "source_a_claim": "Acme had 10,800 employees in 2023",
        "source_b_claim": "Acme had 12,400 employees in 2025",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["12,400", "10,800"],
        "forbidden_resolution": "Acme has 12,400 employees.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
    {
        "id": "C3",
        "query": "What was the Ohio plant utilization?",
        "class": "direct_numerical",
        "source_a_claim": "Ohio plant utilization was 61% in 2023",
        "source_b_claim": "Ohio plant utilization was 91% in 2025",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["91%", "61%"],
        "forbidden_resolution": "The Ohio plant utilization was 91%.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
    # Different factual claims
    {
        "id": "C4",
        "query": "What was the industrial sealant growth rate?",
        "class": "different_factual",
        "source_a_claim": "Industrial sealant grew 9% in 2023",
        "source_b_claim": "Industrial sealant grew 23% in 2025",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["23%", "9%"],
        "forbidden_resolution": "The growth rate was 23%.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
    # Source disagreement (different sources, same topic)
    {
        "id": "C5",
        "query": "What is the Ohio plant utilization rate?",
        "class": "source_disagreement",
        "source_a_claim": "Revenue report says 91% (2025)",
        "source_b_claim": "Q3 metrics say 90%",
        "conflict_type": "minor_numerical_disagreement",
        "gold_facts": ["91%", "90%"],
        "forbidden_resolution": "The utilization was 91%.",
        "required_elements": ["both", "conflict"],
    },
    # Temporal disagreement (different time periods)
    {
        "id": "C6",
        "query": "What is Acme's total revenue?",
        "class": "temporal_disagreement",
        "source_a_claim": "2023 report: $3.1 billion",
        "source_b_claim": "2025 report: $4.7 billion",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["4.7", "3.1"],
        "forbidden_resolution": "Acme's revenue is $4.7 billion.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
    # Qualified disagreement
    {
        "id": "C7",
        "query": "How many units did the Ohio plant produce?",
        "class": "qualified_disagreement",
        "source_a_claim": "Regional report: 1.8 million units in 2025",
        "source_b_claim": "Q3 metrics: 470,000 units in Q3 alone",
        "conflict_type": "partial_complement",
        "gold_facts": ["1.8", "470,000"],
        "forbidden_resolution": "Ohio produced 1.8 million units.",
        "required_elements": ["both", "context"],
    },
    # Partial contradiction
    {
        "id": "C8",
        "query": "What was the total output across all plants in 2025?",
        "class": "partial_contradiction",
        "source_a_claim": "Regional report: 3.9 million units combined",
        "source_b_claim": "Q3 alone: 1,020,000 units",
        "conflict_type": "partial_complement",
        "gold_facts": ["3.9", "1,020,000"],
        "forbidden_resolution": "Total output was 3.9 million units.",
        "required_elements": ["both", "context"],
    },
    # One source contradicting another
    {
        "id": "C9",
        "query": "What is the flagship product line at the Ohio plant?",
        "class": "source_contradiction",
        "source_a_claim": "Regional report: industrial sealants",
        "source_b_claim": "Q3 metrics: does not specify flagship",
        "conflict_type": "information_asymmetry",
        "gold_facts": ["industrial sealants"],
        "forbidden_resolution": "The flagship is industrial sealants.",
        "required_elements": ["source_a", "context"],
    },
    # Multiple conflicting sources
    {
        "id": "C10",
        "query": "What was the industrial sealant division's growth?",
        "class": "multiple_conflicts",
        "source_a_claim": "2023 report: 9% growth",
        "source_b_claim": "2025 report: 23% growth",
        "conflict_type": "temporal_disagreement",
        "gold_facts": ["23%", "9%"],
        "forbidden_resolution": "The growth was 23%.",
        "required_elements": ["both", "conflict", "2023", "2025"],
    },
]


# ── Conflict Evaluation Metrics ──────────────────────────────────────────────

@dataclass
class ConflictMetrics:
    """Metrics specific to conflict query evaluation."""
    conflict_detected: bool = False
    conflict_acknowledged: bool = False
    source_attribution_correct: bool = False
    false_resolution: bool = False
    conflict_grounding_rate: float = 0.0
    conflict_hallucination_rate: float = 0.0
    citation_precision: float = 0.0
    gold_fact_coverage: float = 0.0
    overall_claim_support: float = 0.0
    required_elements_present: float = 0.0


def evaluate_conflict_answer(
    answer: str,
    query_data: dict,
    evidence_texts: list[str],
    contradiction_signals: list[dict],
) -> ConflictMetrics:
    """Evaluate a conflict answer against the gold-standard conflict specification."""
    metrics = ConflictMetrics()
    answer_lower = answer.lower()

    # A. Conflict detection
    metrics.conflict_detected = len(contradiction_signals) > 0

    # B. Conflict acknowledgement
    conflict_phrases = [
        "conflict", "contradict", "discrepan", "differ",
        "inconsistent", "disagree", "varies", "while",
        "on the other hand", "however", "whereas",
        "different report", "legacy", "superseded",
    ]
    metrics.conflict_acknowledged = any(p in answer_lower for p in conflict_phrases)

    # C. Source attribution
    source_phrases = [
        "2023", "2025", "legacy", "current", "authoritative",
        "report", "according", "stated",
    ]
    metrics.source_attribution_correct = any(p in answer_lower for p in source_phrases)

    # D. False resolution detection
    forbidden = query_data.get("forbidden_resolution", "").lower()
    if forbidden:
        # Check if the answer contains the forbidden resolution
        forbidden_words = set(re.findall(r"\b\w{4,}\b", forbidden))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer_lower))
        overlap = forbidden_words & answer_words
        # If >70% of forbidden words appear, it's a false resolution
        metrics.false_resolution = len(overlap) / len(forbidden_words) > 0.7 if forbidden_words else False
    else:
        metrics.false_resolution = False

    # E. Required elements
    required = query_data.get("required_elements", [])
    if required:
        present = sum(1 for elem in required if elem.lower() in answer_lower)
        metrics.required_elements_present = present / len(required)
    else:
        metrics.required_elements_present = 1.0

    # F. Gold fact coverage
    gold_facts = query_data.get("gold_facts", [])
    if gold_facts:
        found = sum(1 for fact in gold_facts if fact.lower() in answer_lower)
        metrics.gold_fact_coverage = found / len(gold_facts)
    else:
        metrics.gold_fact_coverage = 1.0

    # G. Citation precision
    citation_pattern = re.compile(r"\[(\d+)\]")
    citations = [int(m.group(1)) for m in citation_pattern.finditer(answer)]
    if citations and evidence_texts:
        valid_citations = [c for c in citations if 1 <= c <= len(evidence_texts)]
        metrics.citation_precision = len(valid_citations) / len(citations) if citations else 1.0
    else:
        metrics.citation_precision = 1.0

    return metrics


# ── Strategy Implementations ─────────────────────────────────────────────────

def build_strategy_a_prompt(plan, evidence, contradiction_signals):
    """Strategy A: Baseline - use existing synthesis prompt."""
    from app.orchestration.prompts import build_synthesis_messages
    return build_synthesis_messages(plan, evidence, contradiction_signals=contradiction_signals)


def build_strategy_b_prompt(plan, evidence, contradiction_signals):
    """Strategy B: Explicit conflict block with evidence indices."""
    from app.orchestration.prompts import _format_evidence_block, _UNTRUSTED_NOTICE
    from app.llm_gateway.providers.models import Message, MessageRole

    system = (
        "You are a research assistant synthesizing evidence into a cited answer.\n\n"
        "RULES:\n"
        "1. If evidence contains conflicts, you MUST explicitly acknowledge them.\n"
        "2. Present both sides with their respective sources.\n"
        "3. Do NOT present contradictory claims as settled fact.\n"
        "4. If one source is more authoritative or recent, note that.\n"
        "5. Use bracket citations [N] for every claim.\n"
        "6. Do NOT use external knowledge.\n"
        f"{_UNTRUSTED_NOTICE}"
    )

    conflict_section = ""
    if contradiction_signals:
        conflict_section = "\n=== DETECTED CONFLICTS ===\n"
        for i, sig in enumerate(contradiction_signals, 1):
            indices = sig.get("evidence_indices", [])
            desc = sig.get("description", "")
            conflict_section += f"\nConflict {i}:\n"
            if indices:
                for idx in indices:
                    if 1 <= idx <= len(evidence):
                        e = evidence[idx - 1]
                        conflict_section += f"  [{idx}] states: {e.text[:200]}\n"
            conflict_section += f"  Relationship: CONTRADICTORY\n"
            conflict_section += f"  {desc}\n"
        conflict_section += "\nYou MUST acknowledge these conflicts in your answer.\n"
        conflict_section += "=== END DETECTED CONFLICTS ===\n"

    user = (
        f"Objective: {plan.objective}\n"
        f"{conflict_section}\n"
        f"--- NUMBERED EVIDENCE ---\n"
        f"{_format_evidence_block(evidence, include_scores=True)}\n"
        f"--- END EVIDENCE ---\n\n"
        "Write the answer now, with bracket citations."
    )
    return [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]


def build_strategy_c_prompt(plan, evidence, contradiction_signals):
    """Strategy C: Conflict-first synthesis."""
    from app.orchestration.prompts import _format_evidence_block, _UNTRUSTED_NOTICE
    from app.llm_gateway.providers.models import Message, MessageRole

    system = (
        "You are a research assistant. Before synthesizing any answer, you must "
        "first analyze whether the evidence contains conflicts.\n\n"
        "STEP 1 - CONFLICT ANALYSIS:\n"
        "For each source, identify what it claims.\n"
        "Determine if the claims conflict.\n"
        "Determine what can be safely concluded.\n"
        "Determine what remains uncertain.\n\n"
        "STEP 2 - FINAL ANSWER:\n"
        "Using ONLY the conflict analysis above, write the answer.\n"
        "If conflicts exist, explicitly represent the disagreement.\n"
        "Do NOT invent certainty where evidence conflicts.\n"
        "Use bracket citations [N] for every claim.\n"
        f"{_UNTRUSTED_NOTICE}"
    )

    conflict_section = ""
    if contradiction_signals:
        conflict_section = "\n=== DETECTED CONFLICTS ===\n"
        for i, sig in enumerate(contradiction_signals, 1):
            indices = sig.get("evidence_indices", [])
            desc = sig.get("description", "")
            conflict_section += f"\nConflict {i}: {desc}\n"
            if indices:
                for idx in indices:
                    if 1 <= idx <= len(evidence):
                        e = evidence[idx - 1]
                        conflict_section += f"  [{idx}]: {e.text[:200]}\n"
        conflict_section += "=== END DETECTED CONFLICTS ===\n"

    user = (
        f"Objective: {plan.objective}\n"
        f"{conflict_section}\n"
        f"--- NUMBERED EVIDENCE ---\n"
        f"{_format_evidence_block(evidence, include_scores=True)}\n"
        f"--- END EVIDENCE ---\n\n"
        "STEP 1: Analyze conflicts in the evidence.\n"
        "STEP 2: Write the final answer."
    )
    return [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]


def build_strategy_d_prompt(plan, evidence, contradiction_signals):
    """Strategy D: Structured conflict output."""
    from app.orchestration.prompts import _format_evidence_block, _UNTRUSTED_NOTICE
    from app.llm_gateway.providers.models import Message, MessageRole

    system = (
        "You are a research assistant. You MUST use this exact output structure:\n\n"
        "For EACH conflict in the evidence:\n"
        "1. Claim A (with citation)\n"
        "2. Claim B (with citation)\n"
        "3. Conflict description\n"
        "4. Supported conclusion (if any)\n\n"
        "If no conflicts exist, answer normally.\n"
        "If conflicts exist, you MUST present both sides.\n"
        "Do NOT invent certainty where evidence conflicts.\n"
        f"{_UNTRUSTED_NOTICE}"
    )

    conflict_section = ""
    if contradiction_signals:
        conflict_section = "\n=== DETECTED CONFLICTS ===\n"
        for i, sig in enumerate(contradiction_signals, 1):
            indices = sig.get("evidence_indices", [])
            conflict_section += f"\nConflict {i}:\n"
            if indices and len(indices) >= 2:
                idx_a, idx_b = indices[0], indices[1]
                if 1 <= idx_a <= len(evidence) and 1 <= idx_b <= len(evidence):
                    conflict_section += f"  Claim A [{idx_a}]: {evidence[idx_a-1].text[:150]}\n"
                    conflict_section += f"  Claim B [{idx_b}]: {evidence[idx_b-1].text[:150]}\n"
            conflict_section += f"  Relationship: CONTRADICTORY\n"
        conflict_section += "=== END DETECTED CONFLICTS ===\n"

    user = (
        f"Objective: {plan.objective}\n"
        f"{conflict_section}\n"
        f"--- NUMBERED EVIDENCE ---\n"
        f"{_format_evidence_block(evidence, include_scores=True)}\n"
        f"--- END EVIDENCE ---\n\n"
        "Use the structured format to present conflicts."
    )
    return [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]


# ── Main Benchmark Runner ────────────────────────────────────────────────────

STRATEGIES = {
    "A": ("Baseline", build_strategy_a_prompt),
    "B": ("Explicit Conflict Block", build_strategy_b_prompt),
    "C": ("Conflict-First", build_strategy_c_prompt),
    "D": ("Structured Conflict", build_strategy_d_prompt),
}


async def run_conflict_benchmark():
    """Run the conflict benchmark across all strategies."""
    from app.config import Settings
    from app.orchestration.graph import build_graph, run_query
    from app.orchestration.models import ResearchPlan
    from app.retrieval.hybrid import HybridRetriever
    from app.reranking.reranker import NoOpReranker

    settings = Settings()

    # Build benchmark store
    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()
    retriever = HybridRetriever(store=store)
    reranker = NoOpReranker()

    results = {}

    for strategy_id, (strategy_name, prompt_builder) in STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"STRATEGY {strategy_id}: {strategy_name}")
        print(f"{'='*70}")

        strategy_results = []

        for qd in CONFLICT_QUERIES:
            query = qd["query"]
            print(f"\n  [{qd['id']}] {query[:60]}...")

            try:
                # Run the query
                t0 = time.time()
                result = await run_query(
                    query=query,
                    settings=settings,
                    router=router,
                    retriever=retriever,
                    reranker=reranker,
                )
                total_ms = (time.time() - t0) * 1000

                # Get contradiction signals
                contradiction_signals = getattr(result, 'contradiction_signals', []) or []

                # Evaluate
                evidence_texts = [c.text for c in (result.citations or [])]
                metrics = evaluate_conflict_answer(
                    answer=result.answer or "",
                    query_data=qd,
                    evidence_texts=evidence_texts,
                    contradiction_signals=contradiction_signals,
                )

                record = {
                    "query_id": qd["id"],
                    "query": query,
                    "strategy": strategy_id,
                    "strategy_name": strategy_name,
                    "answer": result.answer,
                    "answer_length": len(result.answer or ""),
                    "evidence_count": len(result.citations or []),
                    "contradiction_signals_count": len(contradiction_signals),
                    "conflict_detected": metrics.conflict_detected,
                    "conflict_acknowledged": metrics.conflict_acknowledged,
                    "source_attribution_correct": metrics.source_attribution_correct,
                    "false_resolution": metrics.false_resolution,
                    "required_elements_present": metrics.required_elements_present,
                    "gold_fact_coverage": metrics.gold_fact_coverage,
                    "citation_precision": metrics.citation_precision,
                    "total_e2e_ms": round(total_ms, 2),
                    "outcome": str(getattr(result, "outcome", "unknown")),
                    "warnings": getattr(result, "warnings", []) or [],
                    "providers": getattr(result, "providers_used", []) or [],
                }
                strategy_results.append(record)

                ack = "ACK" if metrics.conflict_acknowledged else "NO-ACK"
                fr = "FALSE-RES" if metrics.false_resolution else "OK"
                print(f"    {ack} | {fr} | GFC={metrics.gold_fact_coverage:.0%} | CP={metrics.citation_precision:.0%}")

            except Exception as e:
                print(f"    ERROR: {e}")
                strategy_results.append({
                    "query_id": qd["id"],
                    "query": query,
                    "strategy": strategy_id,
                    "strategy_name": strategy_name,
                    "error": str(e),
                })

        results[strategy_id] = strategy_results

    return results


def print_comparison_table(results: dict):
    """Print the strategy comparison table."""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 100)

    header = f"{'Strategy':<25} {'Ack':>5} {'FalseRes':>9} {'GFC':>6} {'CitePrec':>9} {'Detected':>9} {'Queries':>8}"
    print(header)
    print("-" * 100)

    for sid, sresults in results.items():
        valid = [r for r in sresults if "error" not in r]
        if not valid:
            print(f"{sid}: ALL ERRORS")
            continue

        n = len(valid)
        ack = sum(1 for r in valid if r.get("conflict_acknowledged", False))
        fr = sum(1 for r in valid if r.get("false_resolution", False))
        gfc = sum(r.get("gold_fact_coverage", 0) for r in valid) / n
        cp = sum(r.get("citation_precision", 0) for r in valid) / n
        det = sum(1 for r in valid if r.get("conflict_detected", False))
        name = valid[0].get("strategy_name", sid)

        print(f"{sid}. {name:<22} {ack}/{n} {fr}/{n} {gfc:.0%} {cp:.0%} {det}/{n} {n}")


def main():
    """Run the conflict benchmark."""
    import asyncio

    print("=" * 70)
    print("PHASE 40: CONFLICT-AWARE SYNTHESIS VALIDATION")
    print("=" * 70)
    print(f"Conflict queries: {len(CONFLICT_QUERIES)}")
    print(f"Strategies: {len(STRATEGIES)}")
    print()

    results = asyncio.run(run_conflict_benchmark())

    # Save results
    output_path = Path("benchmarks/results/phase40_conflict_synthesis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    main()
