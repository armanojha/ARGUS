#!/usr/bin/env python3
"""Phase 30: Optimized verified synthesis benchmark.

Compares 5 strategies on the curated 21-query benchmark:
A: Current baseline single-pass synthesis
B: Phase 29 two-pass (with raw evidence in Pass 2)
C: Restricted Pass 2 (no raw evidence, strict renderer prompt)
D: Deterministic renderer (no LLM Pass 2)
E: Early-exit hybrid (simple→deterministic, complex→restricted LLM)

All with instrumentation for latency, claims, citations.
"""
import asyncio
import dataclasses
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.evidence.models import EvidenceRef
from app.evaluation.answer_quality import evaluate_answer
from app.orchestration.models import ResearchPlan


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


@dataclasses.dataclass
class StrategyResult:
    query_id: str
    query_class: str
    answer: str
    provider: str
    latency_ms: float
    evaluation: dict
    pass1_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0
    pass2_method: str = ""


async def retrieve_evidence(retriever, query: str) -> list[EvidenceRef]:
    return retriever.search(query, top_k=20)


async def strategy_a_baseline(router, query, evidence, settings):
    """Current single-pass synthesis."""
    from app.orchestration.prompts import build_synthesis_messages
    plan = ResearchPlan(objective=query, subquestions=[query])
    messages = build_synthesis_messages(plan, evidence)
    t0 = time.time()
    try:
        response = await router.complete(messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        return response.content or "", "groq", (time.time() - t0) * 1000, 0, 0, 0, "single_pass"
    except Exception as e:
        return "", "error", (time.time() - t0) * 1000, 0, 0, 0, f"error: {e}"


async def strategy_b_two_pass_old(router, query, evidence, settings):
    """Phase 29 two-pass with raw evidence in Pass 2."""
    from app.orchestration.two_pass_synthesis import two_pass_synthesize, verify_claims
    from app.orchestration.two_pass_prompts import build_claim_generation_messages, build_verified_synthesis_messages
    from app.orchestration.two_pass_synthesis import ClaimGenerationOutput
    from pydantic import ValidationError
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t0 = time.time()

    # Pass 1
    claim_messages = build_claim_generation_messages(plan, evidence)
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return "", "error", (time.time() - t0) * 1000, 0, 0, 0, "pass1_error"

    # Verify
    claim_set = verify_claims(claims, evidence)
    supported = claim_set.supported_claims
    rejected = claim_set.rejected_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return v.claim.claim, "two_pass", (time.time() - t0) * 1000, len(claims), len(supported), len(rejected), "absent"
        return "The available evidence does not contain sufficient information to answer this question.", "two_pass", (time.time() - t0) * 1000, len(claims), len(supported), len(rejected), "no_support"

    # Pass 2 WITH raw evidence (Phase 29 behavior)
    verified_for_pass2 = [{"claim": v.claim.claim, "evidence_ids": v.claim.evidence_ids, "status": v.status.value}
                          for v in claim_set.verified if v.status.value in ("supported", "partially_supported")]
    synth_messages = build_verified_synthesis_messages(plan, evidence, verified_for_pass2)
    try:
        response = await router.complete(synth_messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        return response.content or "", "two_pass", (time.time() - t0) * 1000, len(claims), len(supported), len(rejected), "llm_with_evidence"
    except Exception:
        return "", "error", (time.time() - t0) * 1000, len(claims), len(supported), len(rejected), "pass2_error"


async def strategy_c_restricted(router, query, evidence, settings):
    """Restricted Pass 2 — no raw evidence, strict renderer prompt."""
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages
    from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
    from pydantic import ValidationError
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t0 = time.time()

    # Pass 1
    claim_messages = build_claim_generation_messages(plan, evidence)
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return "", "error", (time.time() - t0) * 1000, 0, 0, 0, "pass1_error"

    # Verify
    claim_set = verify_claims(claims, evidence)
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return v.claim.claim, "restricted", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), "absent"
        return "The available evidence does not contain sufficient information to answer this question.", "restricted", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), "no_support"

    # Pass 2 WITHOUT raw evidence (Phase 30 fix)
    verified_for_pass2 = [{"claim": v.claim.claim, "evidence_ids": v.claim.evidence_ids, "status": v.status.value}
                          for v in claim_set.verified if v.status.value in ("supported", "partially_supported")]
    synth_messages = build_verified_synthesis_messages(plan, evidence, verified_for_pass2)
    try:
        response = await router.complete(synth_messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        return response.content or "", "restricted", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), "llm_restricted"
    except Exception:
        return "", "error", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error"


async def strategy_d_deterministic(router, query, evidence, settings):
    """Deterministic renderer — no LLM Pass 2."""
    from app.orchestration.two_pass_synthesis import verify_claims, _render_verified_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t0 = time.time()

    # Pass 1 (still needs LLM for claim extraction)
    claim_messages = build_claim_generation_messages(plan, evidence)
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return "", "error", (time.time() - t0) * 1000, 0, 0, 0, "pass1_error"

    # Verify
    claim_set = verify_claims(claims, evidence)

    # Deterministic rendering (NO Pass 2 LLM call)
    answer = _render_verified_claims(claim_set.verified, query)
    return answer, "deterministic", (time.time() - t0) * 1000, len(claims), len(claim_set.supported_claims), len(claim_set.rejected_claims), "deterministic_render"


async def strategy_e_hybrid(router, query, evidence, settings):
    """Early-exit hybrid: simple→deterministic, complex→restricted LLM."""
    from app.orchestration.two_pass_synthesis import (
        verify_claims, _render_verified_claims, _detect_question_pattern, _should_early_exit,
        ClaimGenerationOutput,
    )
    from app.orchestration.two_pass_prompts import build_claim_generation_messages, build_verified_synthesis_messages
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t0 = time.time()
    pattern = _detect_question_pattern(query)

    # Pass 1
    claim_messages = build_claim_generation_messages(plan, evidence)
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return "", "error", (time.time() - t0) * 1000, 0, 0, 0, "pass1_error"

    # Verify
    claim_set = verify_claims(claims, evidence)
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return v.claim.claim, "hybrid", (time.time() - t0) * 1000, len(claims), 0, len(claim_set.rejected_claims), "absent"
        return "The available evidence does not contain sufficient information to answer this question.", "hybrid", (time.time() - t0) * 1000, len(claims), 0, len(claim_set.rejected_claims), "no_support"

    # Early exit: deterministic rendering for simple/numerical/absent
    if _should_early_exit(claim_set, pattern):
        answer = _render_verified_claims(claim_set.verified, query)
        return answer, "hybrid", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), f"deterministic_{pattern}"

    # Complex: restricted LLM Pass 2
    verified_for_pass2 = [{"claim": v.claim.claim, "evidence_ids": v.claim.evidence_ids, "status": v.status.value}
                          for v in claim_set.verified if v.status.value in ("supported", "partially_supported")]
    synth_messages = build_verified_synthesis_messages(plan, evidence, verified_for_pass2)
    try:
        response = await router.complete(synth_messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        return response.content or "", "hybrid", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), f"llm_{pattern}"
    except Exception:
        return "", "error", (time.time() - t0) * 1000, len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error"


async def run_experiment():
    print("=" * 70)
    print("Phase 30: Optimized Verified Synthesis Benchmark")
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
        "A_baseline": strategy_a_baseline,
        "B_two_pass_old": strategy_b_two_pass_old,
        "C_restricted": strategy_c_restricted,
        "D_deterministic": strategy_d_deterministic,
        "E_hybrid": strategy_e_hybrid,
    }

    all_results = {}

    for strat_name, strat_fn in strategies.items():
        print(f"\n--- Strategy: {strat_name} ---")
        results = []

        for qi in BENCHMARK_QUERIES:
            query = qi["query"]
            print(f"  {qi['id']}: {query[:50]}...", end=" ", flush=True)

            refs = await retrieve_evidence(retriever, query)
            evidence = refs[:8]
            await asyncio.sleep(1.5)

            answer, provider, latency, p1_claims, v_claims, r_claims, method = await strat_fn(router, query, evidence, settings)
            print(f"OK ({latency:.0f}ms, {method})")

            if answer:
                ev = evaluate_answer(answer, evidence, gold_facts=qi.get("gold_facts") or None, query=query)
                eval_dict = {
                    "claim_support_rate": ev.claim_support_rate,
                    "citation_presence_rate": ev.citation_presence_rate,
                    "citation_precision": ev.citation_precision,
                    "query_relevance": ev.query_relevance,
                    "gold_fact_coverage": ev.gold_fact_coverage,
                }
            else:
                eval_dict = {"claim_support_rate": 0, "citation_presence_rate": 0, "citation_precision": 0, "query_relevance": 0, "gold_fact_coverage": None}

            results.append(StrategyResult(
                query_id=qi["id"], query_class=qi["class"],
                answer=answer[:500] if answer else "", provider=provider,
                latency_ms=latency, evaluation=eval_dict,
                pass1_claims=p1_claims, verified_claims=v_claims,
                rejected_claims=r_claims, pass2_method=method,
            ))

        all_results[strat_name] = results

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for strat_name, results in all_results.items():
        valid = [r for r in results if r.provider != "error"]
        if not valid:
            print(f"{strat_name}: ALL FAILED"); continue

        avg = lambda key: sum(r.evaluation.get(key, 0) for r in valid) / len(valid)
        gold = [r.evaluation.get("gold_fact_coverage") for r in valid if r.evaluation.get("gold_fact_coverage") is not None]
        absent_correct = sum(1 for r in valid if r.query_class in ("absent_info", "adversarial") and "does not" in r.answer.lower())
        absent_total = sum(1 for r in valid if r.query_class in ("absent_info", "adversarial"))
        avg_lat = sum(r.latency_ms for r in valid) / len(valid)
        failed = len(results) - len(valid)

        print(f"\n{strat_name}:")
        print(f"  Claim Support:     {avg('claim_support_rate'):.1%}")
        print(f"  Citation Presence:  {avg('citation_presence_rate'):.1%}")
        print(f"  Citation Precision: {avg('citation_precision'):.1%}")
        print(f"  Query Relevance:    {avg('query_relevance'):.1%}")
        gc = f"{sum(gold)/len(gold):.1%}" if gold else "N/A"
        print(f"  Gold Coverage:      {gc}")
        print(f"  Absent Info:        {absent_correct}/{absent_total}")
        print(f"  Avg Latency:        {avg_lat:.0f}ms")
        print(f"  Failed:             {failed}/{len(results)}")

    # Per-query
    print("\n" + "=" * 70)
    print("PER-QUERY CLAIM SUPPORT")
    print("=" * 70)
    header = f"{'Query':<15} {'Class':<22}"
    for s in all_results: header += f" {s[:10]:>10}"
    print(header)
    print("-" * len(header))
    for i in range(21):
        qi = BENCHMARK_QUERIES[i]
        row = f"{qi['id']:<15} {qi['class']:<22}"
        for s in all_results:
            r = all_results[s][i]
            v = r.evaluation.get("claim_support_rate", 0)
            row += f" {v:>9.0%}"
        print(row)

    # Save
    output = Path("benchmarks/results")
    output.mkdir(exist_ok=True)
    save_data = {}
    for strat_name, results in all_results.items():
        save_data[strat_name] = [
            {"query_id": r.query_id, "class": r.query_class, "answer": r.answer,
             "provider": r.provider, "latency_ms": r.latency_ms,
             "pass1_claims": r.pass1_claims, "verified_claims": r.verified_claims,
             "rejected_claims": r.rejected_claims, "pass2_method": r.pass2_method,
             **r.evaluation}
            for r in results
        ]
    with open(output / "phase30_ablation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to benchmarks/results/phase30_ablation.json")


if __name__ == "__main__":
    asyncio.run(run_experiment())
