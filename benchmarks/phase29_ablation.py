#!/usr/bin/env python3
"""Phase 29: Two-pass verified synthesis benchmark.

Compares:
A: Current baseline (single-pass free-form synthesis)
D: Two-pass verified synthesis (structured claims + deterministic verify + final synthesis)

Runs on the curated 21-query benchmark with rate limit detection.
"""
import asyncio
import io
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.evidence.models import EvidenceRef
from app.evaluation.answer_quality import evaluate_answer
from app.llm_gateway.providers.models import Message, MessageRole


# --- Benchmark queries ---
BENCHMARK_QUERIES = [
    {"id": "P27-A01", "class": "simple_lookup", "query": "Where is Acme Corporation headquartered?", "gold_facts": ["New York City"]},
    {"id": "P27-A02", "class": "simple_lookup", "query": "What is the name of Acme's CEO?", "gold_facts": ["CEO"]},
    {"id": "P27-D01", "class": "multi_doc_synthesis", "query": "What are the key differences between Atlas v1 and Atlas v2?", "gold_facts": ["Atlas", "v1", "v2"]},
    {"id": "P27-D02", "class": "multi_doc_synthesis", "query": "How does Acme's employee count compare across different reports?", "gold_facts": ["employees", "count"]},
    {"id": "P27-D03", "class": "multi_doc_synthesis", "query": "Compare Acme's 2023 and 2025 revenue figures and explain the trend.", "gold_facts": ["revenue", "2023", "2025"]},
    {"id": "P27-E01", "class": "multi_hop", "query": "Who manages the team that developed Atlas?", "gold_facts": ["Atlas", "team", "manager"]},
    {"id": "P27-E02", "class": "multi_hop", "query": "What technology stack does the Acme analytics platform use?", "gold_facts": ["analytics", "technology", "stack"]},
    {"id": "P27-E03", "class": "multi_hop", "query": "Which division's product uses the highest-performing database?", "gold_facts": ["database", "performance", "division"]},
    {"id": "P27-F01", "class": "conflict", "query": "What were Acme's revenue figures for 2023 according to different sources?", "gold_facts": ["revenue", "2023"], "conflict": True},
    {"id": "P27-F02", "class": "conflict", "query": "What are the different employee count figures reported for Acme?", "gold_facts": ["employees", "count"], "conflict": True},
    {"id": "P27-H01", "class": "numerical", "query": "What was Acme's revenue growth rate from 2022 to 2023?", "gold_facts": ["revenue", "growth", "rate"]},
    {"id": "P27-H02", "class": "numerical", "query": "What is the Atlas database query latency at P99?", "gold_facts": ["latency", "P99", "Atlas"]},
    {"id": "P27-H03", "class": "numerical", "query": "How many customers does Acme serve across all products?", "gold_facts": ["customers", "count"]},
    {"id": "P27-C01", "class": "technical_explanation", "query": "How does the Atlas database achieve low latency?", "gold_facts": ["Atlas", "latency", "architecture"]},
    {"id": "P27-C02", "class": "technical_explanation", "query": "Explain Acme's approach to data security.", "gold_facts": ["security", "approach"]},
    {"id": "P27-I01", "class": "complex_research", "query": "What is Acme's competitive advantage in the analytics market?", "gold_facts": ["competitive", "advantage", "analytics"]},
    {"id": "P27-I02", "class": "complex_research", "query": "How does Acme plan to expand internationally?", "gold_facts": ["international", "expansion", "plan"]},
    {"id": "P27-G01", "class": "absent_info", "query": "What is Acme's market share in the European robotics market?", "gold_facts": [], "absent": True},
    {"id": "P27-G02", "class": "absent_info", "query": "What is the salary range for Acme's software engineers?", "gold_facts": [], "absent": True},
    {"id": "P27-J01", "class": "adversarial", "query": "What undisclosed legal issues has Acme faced?", "gold_facts": [], "absent": True},
    {"id": "P27-J02", "class": "adversarial", "query": "What internal documents reveal Acme's planned layoffs?", "gold_facts": [], "absent": True},
]


@dataclass
class ProviderInfo:
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    is_fallback: bool = False
    failure_reason: str = ""


@dataclass
class QueryResult:
    query_id: str
    query_class: str
    answer: str
    provider_info: ProviderInfo
    evaluation: dict = field(default_factory=dict)
    gold_facts: list = field(default_factory=list)
    absent: bool = False
    conflict: bool = False


async def retrieve_evidence(retriever, query: str, top_k: int = 20) -> list[EvidenceRef]:
    """Retrieve evidence for a query."""
    refs = retriever.search(query, top_k=top_k)
    return refs


def format_evidence(evidence: list[EvidenceRef], include_scores: bool = True) -> str:
    lines = []
    for i, ref in enumerate(evidence, 1):
        snippet = ref.text.strip().replace("\n", " ")[:500]
        if include_scores and ref.score is not None:
            lines.append(f"[{i}] (score: {ref.score:.2f}, source: {ref.source_path}) {snippet}")
        else:
            lines.append(f"[{i}] (source: {ref.source_path}) {snippet}")
    return "\n".join(lines)


async def run_baseline_strategy(
    router, query: str, evidence: list[EvidenceRef], settings
) -> tuple[str, ProviderInfo]:
    """Run current baseline single-pass synthesis."""
    from app.orchestration.prompts import build_synthesis_messages
    from app.orchestration.models import ResearchPlan

    plan = ResearchPlan(objective=query, subquestions=[query])
    messages = build_synthesis_messages(plan, evidence)

    info = ProviderInfo()
    t0 = time.time()
    try:
        response = await router.complete(
            messages,
            temperature=0.2,
            timeout=settings.orchestration_llm_timeout,
            call_type="synthesis",
            request_id=None,
            query=query,
            tier="strong",
        )
        info.provider = getattr(response, "provider", "unknown")
        info.model = getattr(response, "model", "unknown")
        info.latency_ms = (time.time() - t0) * 1000
        return response.content or "", info
    except Exception as e:
        info.failure_reason = str(e)
        info.latency_ms = (time.time() - t0) * 1000
        return "", info


async def run_two_pass_strategy(
    router, query: str, evidence: list[EvidenceRef], settings
) -> tuple[str, ProviderInfo]:
    """Run two-pass verified synthesis."""
    from app.orchestration.two_pass_synthesis import two_pass_synthesize
    from app.orchestration.models import ResearchPlan

    plan = ResearchPlan(objective=query, subquestions=[query])

    info = ProviderInfo()
    t0 = time.time()
    try:
        answer, warnings, claim_set = await two_pass_synthesize(
            plan, evidence, router=router, settings=settings, request_id=None
        )
        info.provider = "two_pass"
        info.model = "groq+verify"
        info.latency_ms = (time.time() - t0) * 1000
        return answer, info
    except Exception as e:
        info.failure_reason = str(e)
        info.latency_ms = (time.time() - t0) * 1000
        return "", info


async def run_experiment():
    print("=" * 70)
    print("Phase 29: Two-Pass Verified Synthesis Benchmark")
    print("=" * 70)

    from benchmarks.benchmark_fusion import build_benchmark_store
    store, chunk_id_map = build_benchmark_store()

    from app.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever(store=store)
    retriever.ensure_indexes()

    from app.llm_gateway import get_router
    from app.config import get_settings
    router = get_router()
    settings = get_settings()

    strategies = {
        "A_baseline": run_baseline_strategy,
        "D_two_pass": run_two_pass_strategy,
    }

    all_results = {}

    for strat_name, strat_fn in strategies.items():
        print(f"\n--- Strategy: {strat_name} ---")
        results = []

        for qi in BENCHMARK_QUERIES:
            query = qi["query"]
            print(f"  {qi['id']}: {query[:60]}...", end=" ", flush=True)

            # Retrieve evidence
            refs = await retrieve_evidence(retriever, query)
            evidence = refs[:8]

            # Rate limit delay
            await asyncio.sleep(1.5)

            # Run strategy
            answer, provider_info = await strat_fn(router, query, evidence, settings)

            # Check for provider failure
            if provider_info.failure_reason:
                print(f"FAIL ({provider_info.failure_reason[:50]})")
            else:
                print(f"OK ({provider_info.latency_ms:.0f}ms)")

            # Evaluate
            if answer:
                ev = evaluate_answer(
                    answer,
                    evidence,
                    gold_facts=qi.get("gold_facts") or None,
                    query=query,
                )
                eval_dict = {
                    "claim_support_rate": ev.claim_support_rate,
                    "citation_presence_rate": ev.citation_presence_rate,
                    "citation_precision": ev.citation_precision,
                    "query_relevance": ev.query_relevance,
                    "gold_fact_coverage": ev.gold_fact_coverage,
                }
            else:
                eval_dict = {
                    "claim_support_rate": 0.0,
                    "citation_presence_rate": 0.0,
                    "citation_precision": 0.0,
                    "query_relevance": 0.0,
                    "gold_fact_coverage": None,
                }

            results.append(QueryResult(
                query_id=qi["id"],
                query_class=qi["class"],
                answer=answer[:500] if answer else "",
                provider_info=provider_info,
                evaluation=eval_dict,
                gold_facts=qi.get("gold_facts", []),
                absent=qi.get("absent", False),
                conflict=qi.get("conflict", False),
            ))

        all_results[strat_name] = results

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for strat_name, results in all_results.items():
        valid = [r for r in results if r.provider_info.failure_reason == ""]
        if not valid:
            print(f"{strat_name}: ALL FAILED")
            continue

        avg_support = sum(r.evaluation.get("claim_support_rate", 0) for r in valid) / len(valid)
        avg_cite_pres = sum(r.evaluation.get("citation_presence_rate", 0) for r in valid) / len(valid)
        avg_cite_prec = sum(r.evaluation.get("citation_precision", 0) for r in valid) / len(valid)
        avg_relevance = sum(r.evaluation.get("query_relevance", 0) for r in valid) / len(valid)

        gold_vals = [r.evaluation.get("gold_fact_coverage") for r in valid if r.evaluation.get("gold_fact_coverage") is not None]
        avg_gold = sum(gold_vals) / len(gold_vals) if gold_vals else None

        absent_correct = sum(1 for r in valid if r.absent and "does not" in r.answer.lower())
        absent_total = sum(1 for r in valid if r.absent)

        avg_latency = sum(r.provider_info.latency_ms for r in valid) / len(valid)
        failed = len(results) - len(valid)

        print(f"\n{strat_name}:")
        print(f"  Claim Support:    {avg_support:.1%}")
        print(f"  Citation Presence: {avg_cite_pres:.1%}")
        print(f"  Citation Precision: {avg_cite_prec:.1%}")
        print(f"  Query Relevance:  {avg_relevance:.1%}")
        print(f"  Gold Coverage:    {avg_gold:.1%}" if avg_gold is not None else "  Gold Coverage:    N/A")
        print(f"  Absent Info:      {absent_correct}/{absent_total} correct abstentions")
        print(f"  Avg Latency:      {avg_latency:.0f}ms")
        print(f"  Failed Queries:   {failed}/{len(results)}")

    # Per-category breakdown
    print("\n" + "=" * 70)
    print("PER-QUERY CLAIM SUPPORT")
    print("=" * 70)

    header = f"{'Query':<15} {'Class':<22}"
    for s in all_results:
        header += f" {s[:10]:>10}"
    print(header)
    print("-" * len(header))

    for i in range(len(BENCHMARK_QUERIES)):
        qi = BENCHMARK_QUERIES[i]
        row = f"{qi['id']:<15} {qi['class']:<22}"
        for s in all_results:
            r = all_results[s][i]
            v = r.evaluation.get("claim_support_rate", 0)
            row += f" {v:>9.0%}"
        print(row)

    # Save results
    output = Path("benchmarks/results")
    output.mkdir(exist_ok=True)

    save_data = {}
    for strat_name, results in all_results.items():
        save_data[strat_name] = [
            {
                "query_id": r.query_id,
                "class": r.query_class,
                "answer": r.answer,
                "provider": r.provider_info.provider,
                "model": r.provider_info.model,
                "latency_ms": r.provider_info.latency_ms,
                "failure": r.provider_info.failure_reason,
                **r.evaluation,
            }
            for r in results
        ]

    with open(output / "phase29_ablation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to benchmarks/results/phase29_ablation.json")


if __name__ == "__main__":
    asyncio.run(run_experiment())
