#!/usr/bin/env python3
"""Phase 31: Pattern-specific verified synthesis & latency optimization.

Compares strategies on the curated 21-query benchmark with detailed
instrumentation for latency, claims, citations, and safety invariants.

Strategies:
A: Baseline single-pass synthesis
B: Phase 30 Strategy C (restricted Pass 2, no raw evidence)
C: Deterministic renderer only (no LLM Pass 2)
D: Pattern-specific hybrid (simple→deterministic, complex→restricted)
E: Optimized restricted Pass 2 (shorter prompt, tighter contract)
"""
import asyncio
import dataclasses
import io
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.evidence.models import EvidenceRef
from app.evaluation.answer_quality import evaluate_answer
from app.orchestration.models import ResearchPlan


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
class DetailedResult:
    query_id: str
    query_class: str
    answer: str
    strategy: str
    latency_ms: float
    pass1_latency_ms: float = 0.0
    pass2_latency_ms: float = 0.0
    verify_latency_ms: float = 0.0
    pass1_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0
    pass2_method: str = ""
    pass1_tokens_in: int = 0
    pass1_tokens_out: int = 0
    pass2_tokens_in: int = 0
    pass2_tokens_out: int = 0
    citations_in_answer: int = 0
    evaluation: dict = dataclasses.field(default_factory=dict)
    pattern: str = ""


CITATION_RE = re.compile(r"\[(\d+)\]")

def count_citations(text: str) -> list[int]:
    return [int(m.group(1)) for m in CITATION_RE.finditer(text)]


def detect_pattern(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["where is", "what is the name", "who is", "when was", "how many employees"]):
        return "simple_lookup"
    if any(w in q for w in ["latency", "p99", "revenue growth rate", "percentage", "ratio", "how many customers"]):
        return "numerical"
    if any(w in q for w in ["conflict", "different sources", "disagreement", "discrepancies", "different employee count"]):
        return "conflict"
    if any(w in q for w in ["not contain", "not available", "no information", "undisclosed", "planned layoffs", "salary range", "market share"]):
        return "absent_info"
    if any(w in q for w in ["how does", "explain", "approach", "why does", "architecture"]):
        return "technical_explanation"
    if any(w in q for w in ["compare", "trend", "difference", "growth"]):
        return "multi_doc_synthesis"
    return "general"


async def retrieve_evidence(retriever, query: str) -> list[EvidenceRef]:
    return retriever.search(query, top_k=20)


# ---- Strategy A: Baseline single-pass ----
async def strategy_a_baseline(router, query, evidence, settings):
    from app.orchestration.prompts import build_synthesis_messages
    plan = ResearchPlan(objective=query, subquestions=[query])
    messages = build_synthesis_messages(plan, evidence)
    t0 = time.time()
    try:
        response = await router.complete(messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        answer = response.content or ""
        latency = (time.time() - t0) * 1000
        tokens_in = getattr(response, "usage", None)
        tokens_in = getattr(tokens_in, "prompt_tokens", 0) if tokens_in else 0
        tokens_out = getattr(response, "usage", None)
        tokens_out = getattr(tokens_out, "completion_tokens", 0) if tokens_out else 0
        return answer, latency, 0, 0, 0, 0, 0, "single_pass", tokens_in, tokens_out, 0, 0
    except Exception as e:
        return "", (time.time() - t0) * 1000, 0, 0, 0, 0, 0, f"error: {e}", 0, 0, 0, 0


# ---- Strategy B: Restricted Pass 2 (Phase 30 Strategy C) ----
async def strategy_b_restricted(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages, build_verified_synthesis_messages
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    # Pass 1
    claim_messages = build_claim_generation_messages(plan, evidence)
    t_p1 = time.time()
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0)

    # Verify
    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                        len(claims), len(supported), len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0)
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0)

    # Pass 2 WITHOUT raw evidence
    verified_for_pass2 = [{"claim": v.claim.claim, "evidence_ids": v.claim.evidence_ids, "status": v.status.value}
                          for v in claim_set.verified if v.status.value in ("supported", "partially_supported")]
    synth_messages = build_verified_synthesis_messages(plan, evidence, verified_for_pass2)
    t_p2 = time.time()
    try:
        response = await router.complete(synth_messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p2_latency = (time.time() - t_p2) * 1000
        p2_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p2_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        answer = response.content or ""
    except Exception:
        p2_latency = (time.time() - t_p2) * 1000
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency, v_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0)

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency, v_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "llm_restricted",
            p1_in, p1_out, p2_in, p2_out)


# ---- Strategy C: Deterministic renderer only ----
async def strategy_c_deterministic(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, _render_verified_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    claim_messages = build_claim_generation_messages(plan, evidence)
    t_p1 = time.time()
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0)

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000

    t_p2 = time.time()
    answer = _render_verified_claims(claim_set.verified, query)
    p2_latency = (time.time() - t_p2) * 1000

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency, v_latency,
            len(claims), len(claim_set.supported_claims), len(claim_set.rejected_claims),
            "deterministic_render", p1_in, p1_out, 0, 0)


# ---- Strategy D: Pattern-specific hybrid ----
async def strategy_d_hybrid(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import (
        verify_claims, _render_verified_claims, _detect_question_pattern, _should_early_exit,
        ClaimGenerationOutput,
    )
    from app.orchestration.two_pass_prompts import build_claim_generation_messages, build_verified_synthesis_messages
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()
    pattern = _detect_question_pattern(query)

    claim_messages = build_claim_generation_messages(plan, evidence)
    t_p1 = time.time()
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0)

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0)
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0)

    if _should_early_exit(claim_set, pattern):
        t_p2 = time.time()
        answer = _render_verified_claims(claim_set.verified, query)
        p2_latency = (time.time() - t_p2) * 1000
        total = (time.time() - t_total) * 1000
        return (answer, total, p1_latency, p2_latency, v_latency,
                len(claims), len(supported), len(claim_set.rejected_claims),
                f"deterministic_{pattern}", p1_in, p1_out, 0, 0)

    verified_for_pass2 = [{"claim": v.claim.claim, "evidence_ids": v.claim.evidence_ids, "status": v.status.value}
                          for v in claim_set.verified if v.status.value in ("supported", "partially_supported")]
    synth_messages = build_verified_synthesis_messages(plan, evidence, verified_for_pass2)
    t_p2 = time.time()
    try:
        response = await router.complete(synth_messages, temperature=0.2, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p2_latency = (time.time() - t_p2) * 1000
        p2_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p2_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        answer = response.content or ""
    except Exception:
        p2_latency = (time.time() - t_p2) * 1000
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency, v_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0)

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency, v_latency,
            len(claims), len(supported), len(claim_set.rejected_claims),
            f"llm_{pattern}", p1_in, p1_out, p2_in, p2_out)


# ---- Strategy E: Optimized restricted (shorter prompt) ----
async def strategy_e_optimized(router, query, evidence, settings):
    """Optimized restricted Pass 2 with shorter prompt and tighter contract."""
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages
    from app.llm_gateway.providers.models import Message, MessageRole
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    # Pass 1
    claim_messages = build_claim_generation_messages(plan, evidence)
    t_p1 = time.time()
    try:
        response = await router.complete(claim_messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0)

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0)
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0, v_latency,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0)

    # Optimized Pass 2: minimal prompt, no filler allowed
    verified_for_pass2 = []
    for v in claim_set.verified:
        if v.status.value in ("supported", "partially_supported"):
            verified_for_pass2.append({
                "claim": v.claim.claim,
                "evidence_ids": v.claim.evidence_ids,
                "status": v.status.value,
            })

    claims_text = ""
    for i, vc in enumerate(verified_for_pass2, 1):
        citations = "".join(f"[{eid}]" for eid in vc["evidence_ids"])
        claims_text += f"\n{i}. {vc['claim']} {citations}"

    # Minimal prompt — no system preamble, just the contract
    messages = [
        Message(role=MessageRole.SYSTEM, content=(
            "Renderer. Not researcher.\n"
            "Rules: verified claims only. No new facts. No new numbers. "
            "No introductions. No conclusions. No filler.\n"
            "Answer the question directly with citations."
        )),
        Message(role=MessageRole.USER, content=(
            f"Q: {plan.objective}\n"
            f"Verified claims:\n{claims_text}\n"
            "Answer:"
        )),
    ]

    t_p2 = time.time()
    try:
        response = await router.complete(messages, temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p2_latency = (time.time() - t_p2) * 1000
        p2_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p2_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        answer = response.content or ""
    except Exception:
        p2_latency = (time.time() - t_p2) * 1000
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency, v_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0)

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency, v_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "optimized_restricted",
            p1_in, p1_out, p2_in, p2_out)


STRATEGIES = {
    "A_baseline": strategy_a_baseline,
    "B_restricted": strategy_b_restricted,
    "C_deterministic": strategy_c_deterministic,
    "D_hybrid": strategy_d_hybrid,
    "E_optimized": strategy_e_optimized,
}


async def run_experiment():
    print("=" * 70)
    print("Phase 31: Pattern-Specific Verified Synthesis & Latency Optimization")
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

    all_results = {}

    for strat_name, strat_fn in STRATEGIES.items():
        print(f"\n--- Strategy: {strat_name} ---")
        results = []

        for qi in BENCHMARK_QUERIES:
            query = qi["query"]
            print(f"  {qi['id']}: {query[:50]}...", end=" ", flush=True)

            refs = await retrieve_evidence(retriever, query)
            evidence = refs[:8]
            await asyncio.sleep(1.5)

            try:
                out = await strat_fn(router, query, evidence, settings)
                answer, total_lat, p1_lat, p2_lat, v_lat, p1_cl, v_cl, r_cl, method, p1_in, p1_out, p2_in, p2_out = out
            except Exception as e:
                print(f"ERROR: {e}")
                answer, total_lat, p1_lat, p2_lat, v_lat = "", 0, 0, 0, 0
                p1_cl, v_cl, r_cl, method = 0, 0, 0, f"error: {e}"
                p1_in, p1_out, p2_in, p2_out = 0, 0, 0, 0

            print(f"OK ({total_lat:.0f}ms, {method})")

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

            pattern = detect_pattern(query)
            results.append(DetailedResult(
                query_id=qi["id"], query_class=qi["class"],
                answer=answer[:500] if answer else "", strategy=strat_name,
                latency_ms=total_lat, pass1_latency_ms=p1_lat, pass2_latency_ms=p2_lat,
                verify_latency_ms=v_lat, pass1_claims=p1_cl, verified_claims=v_cl,
                rejected_claims=r_cl, pass2_method=method,
                pass1_tokens_in=p1_in, pass1_tokens_out=p1_out,
                pass2_tokens_in=p2_in, pass2_tokens_out=p2_out,
                citations_in_answer=len(count_citations(answer)) if answer else 0,
                evaluation=eval_dict, pattern=pattern,
            ))

        all_results[strat_name] = results

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for strat_name, results in all_results.items():
        valid = [r for r in results if "error" not in r.pass2_method]
        if not valid:
            print(f"{strat_name}: ALL FAILED"); continue

        avg = lambda key: sum(r.evaluation.get(key, 0) for r in valid) / len(valid)
        gold = [r.evaluation.get("gold_fact_coverage") for r in valid if r.evaluation.get("gold_fact_coverage") is not None]
        absent_correct = sum(1 for r in valid if r.query_class in ("absent_info", "adversarial") and "does not" in r.answer.lower())
        absent_total = sum(1 for r in valid if r.query_class in ("absent_info", "adversarial"))
        latencies = [r.latency_ms for r in valid]
        p1_lats = [r.pass1_latency_ms for r in valid if r.pass1_latency_ms > 0]
        p2_lats = [r.pass2_latency_ms for r in valid if r.pass2_latency_ms > 0]

        print(f"\n{strat_name}:")
        print(f"  Claim Support:     {avg('claim_support_rate'):.1%}")
        print(f"  Citation Presence:  {avg('citation_presence_rate'):.1%}")
        print(f"  Citation Precision: {avg('citation_precision'):.1%}")
        print(f"  Query Relevance:    {avg('query_relevance'):.1%}")
        gc = f"{sum(gold)/len(gold):.1%}" if gold else "N/A"
        print(f"  Gold Coverage:      {gc}")
        print(f"  Absent Info:        {absent_correct}/{absent_total}")
        print(f"  Avg Latency:        {statistics.mean(latencies):.0f}ms")
        print(f"  P50 Latency:        {statistics.median(latencies):.0f}ms")
        if len(latencies) >= 4:
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            print(f"  P95 Latency:        {sorted_lat[min(p95_idx, len(sorted_lat)-1)]:.0f}ms")
        print(f"  Avg Pass1 Latency:  {statistics.mean(p1_lats):.0f}ms" if p1_lats else "  Avg Pass1 Latency:  N/A")
        print(f"  Avg Pass2 Latency:  {statistics.mean(p2_lats):.0f}ms" if p2_lats else "  Avg Pass2 Latency:  N/A (deterministic)")
        total_tokens_in = sum(r.pass1_tokens_in + r.pass2_tokens_in for r in valid)
        total_tokens_out = sum(r.pass1_tokens_out + r.pass2_tokens_out for r in valid)
        print(f"  Total Tokens:       {total_tokens_in} in / {total_tokens_out} out")
        failed = len(results) - len(valid)
        print(f"  Failed:             {failed}/{len(results)}")

    # ---- Per-query pattern matrix ----
    print("\n" + "=" * 70)
    print("PER-QUERY PATTERN / COST MATRIX")
    print("=" * 70)
    header = f"{'ID':<12} {'Pattern':<22}"
    for s in all_results:
        header += f" {s[:12]:>12}"
    print(header)
    print("-" * len(header))
    for i, qi in enumerate(BENCHMARK_QUERIES):
        row = f"{qi['id']:<12} {detect_pattern(qi['query']):<22}"
        for s in all_results:
            r = all_results[s][i]
            lat = r.latency_ms
            support = r.evaluation.get("claim_support_rate", 0)
            row += f" {lat:>6.0f}ms {support:>4.0%}"
        print(row)

    # ---- Latency breakdown ----
    print("\n" + "=" * 70)
    print("LATENCY BREAKDOWN (per strategy)")
    print("=" * 70)
    for strat_name, results in all_results.items():
        valid = [r for r in results if "error" not in r.pass2_method]
        if not valid: continue
        p1 = [r.pass1_latency_ms for r in valid if r.pass1_latency_ms > 0]
        v = [r.verify_latency_ms for r in valid if r.verify_latency_ms > 0]
        p2 = [r.pass2_latency_ms for r in valid if r.pass2_latency_ms > 0]
        print(f"\n{strat_name}:")
        if p1: print(f"  Pass1 (claim gen):  {statistics.mean(p1):.0f}ms avg ({min(p1):.0f}-{max(p1):.0f})")
        if v: print(f"  Verify (deterministic): {statistics.mean(v):.0f}ms avg ({min(v):.0f}-{max(v):.0f})")
        if p2: print(f"  Pass2 (LLM):        {statistics.mean(p2):.0f}ms avg ({min(p2):.0f}-{max(p2):.0f})")
        if not p2: print(f"  Pass2:              0ms (deterministic render)")
        total = [r.latency_ms for r in valid]
        print(f"  Total:              {statistics.mean(total):.0f}ms avg")

    # ---- Safety invariants ----
    print("\n" + "=" * 70)
    print("SAFETY INVARIANT CHECK")
    print("=" * 70)
    for strat_name, results in all_results.items():
        absent_q = [r for r in results if r.query_class in ("absent_info", "adversarial")]
        absent_correct = sum(1 for r in absent_q if "does not" in r.answer.lower() or "no information" in r.answer.lower())
        print(f"  {strat_name}: Absent info correct: {absent_correct}/{len(absent_q)}")

    # ---- Save ----
    output = Path("benchmarks/results")
    output.mkdir(exist_ok=True)
    save_data = {}
    for strat_name, results in all_results.items():
        save_data[strat_name] = [
            {
                "query_id": r.query_id, "class": r.query_class, "pattern": r.pattern,
                "answer": r.answer, "strategy": r.strategy,
                "latency_ms": r.latency_ms, "pass1_latency_ms": r.pass1_latency_ms,
                "pass2_latency_ms": r.pass2_latency_ms, "verify_latency_ms": r.verify_latency_ms,
                "pass1_claims": r.pass1_claims, "verified_claims": r.verified_claims,
                "rejected_claims": r.rejected_claims, "pass2_method": r.pass2_method,
                "pass1_tokens_in": r.pass1_tokens_in, "pass1_tokens_out": r.pass1_tokens_out,
                "pass2_tokens_in": r.pass2_tokens_in, "pass2_tokens_out": r.pass2_tokens_out,
                "citations_in_answer": r.citations_in_answer,
                **r.evaluation,
            }
            for r in results
        ]
    with open(output / "phase31_ablation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to benchmarks/results/phase31_ablation.json")


if __name__ == "__main__":
    asyncio.run(run_experiment())
