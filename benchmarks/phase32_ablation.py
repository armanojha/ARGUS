#!/usr/bin/env python3
"""Phase 32: Claim Generation Optimization & Inference Cost Reduction.

Tests whether Pass 1 claim generation can be made faster/cheaper while
preserving grounding quality.

Experiments:
A: Phase 30 baseline (8 chunks, full prompt)
B: Reduced evidence (4 chunks)
C: Reduced evidence (6 chunks)
D: Minimal prompt (shorter system message)
E: Claim count cap (max 8 claims)
F: Best safe combination (4 chunks + minimal prompt + claim cap)
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

CITATION_RE = re.compile(r"\[(\d+)\]")

def count_citations(text: str) -> list[int]:
    return [int(m.group(1)) for m in CITATION_RE.finditer(text)]


@dataclasses.dataclass
class ExpResult:
    query_id: str
    query_class: str
    answer: str
    strategy: str
    latency_ms: float
    pass1_latency_ms: float = 0.0
    pass2_latency_ms: float = 0.0
    pass1_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0
    pass1_tokens_in: int = 0
    pass1_tokens_out: int = 0
    pass2_tokens_in: int = 0
    pass2_tokens_out: int = 0
    evidence_chunks: int = 0
    evaluation: dict = dataclasses.field(default_factory=dict)


async def retrieve_evidence(retriever, query: str) -> list[EvidenceRef]:
    return retriever.search(query, top_k=20)


# ---- Strategy A: Phase 30 baseline (8 chunks, full prompt) ----
async def strategy_a_baseline(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_claim_generation_messages, build_verified_synthesis_messages
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
                "pass1_error", 0, 0, 0, 0, len(evidence))

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0, len(evidence))
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0, len(evidence))

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
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0, len(evidence))

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "llm_restricted",
            p1_in, p1_out, p2_in, p2_out, len(evidence))


# ---- Strategy B: Reduced evidence (4 chunks) ----
async def strategy_b_evidence_4(router, query, evidence, settings):
    evidence = evidence[:4]
    return await strategy_a_baseline(router, query, evidence, settings)


# ---- Strategy C: Reduced evidence (6 chunks) ----
async def strategy_c_evidence_6(router, query, evidence, settings):
    evidence = evidence[:6]
    return await strategy_a_baseline(router, query, evidence, settings)


# ---- Strategy D: Minimal prompt ----
async def strategy_d_minimal_prompt(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
    from app.llm_gateway.providers.models import Message, MessageRole
    from app.orchestration.prompts import _format_evidence_block
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    # Minimal Pass 1 prompt — stripped of verbose instructions
    evidence_block = _format_evidence_block(evidence, include_scores=True)
    system = (
        "Extract factual claims from evidence. Output JSON: "
        '{"claims":[{"claim":"...","evidence_ids":[1],"claim_type":"factual","numerical_values":[]}]}'
        "\nRules: every claim must cite [N]. No external knowledge. One fact per claim."
    )
    user = f"Q: {query}\nEvidence:\n{evidence_block}\nExtract claims as JSON."
    messages = [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]

    t_p1 = time.time()
    try:
        response = await router.complete(messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0, len(evidence))

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0, len(evidence))
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0, len(evidence))

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
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0, len(evidence))

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "minimal_prompt",
            p1_in, p1_out, p2_in, p2_out, len(evidence))


# ---- Strategy E: Claim count cap (max 8) ----
async def strategy_e_claim_cap_8(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
    from app.llm_gateway.providers.models import Message, MessageRole
    from app.orchestration.prompts import _format_evidence_block
    import json as _json

    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    evidence_block = _format_evidence_block(evidence, include_scores=True)
    system = (
        "You are the claim-generation stage of a research assistant. "
        "Given the research objective and numbered evidence passages, "
        "extract every factual claim the evidence supports.\n\n"
        "RULES:\n"
        "1. EVERY claim MUST cite at least one evidence passage using its [N] number.\n"
        "2. Claims with NO evidence ID are INVALID and must not be generated.\n"
        "3. For numerical claims, include the exact numbers from evidence.\n"
        "4. For comparison claims, cite evidence for EACH side.\n"
        "5. For contradictions, create SEPARATE claims for each conflicting fact.\n"
        "6. If the evidence does NOT contain information to answer the question, "
        "generate a single claim of type 'absent' explaining this.\n"
        "7. Do NOT use external knowledge. Only extract what is IN the evidence.\n"
        "8. Each claim should be ONE factual statement, not a compound paragraph.\n"
        "9. MAXIMUM 8 claims. Prioritize the most important facts.\n\n"
        "OUTPUT FORMAT: JSON with a 'claims' array. Each claim has:\n"
        '- "claim": the factual statement\n'
        '- "evidence_ids": list of 1-based evidence indices supporting it\n'
        '- "confidence": 0.0-1.0\n'
        '- "claim_type": "factual" | "numerical" | "comparison" | "relationship" | "absent"\n'
        '- "numerical_values": list of specific numbers referenced (if any)\n'
    )
    user = (
        f"Objective: {query}\n"
        f"--- NUMBERED EVIDENCE ---\n{evidence_block}\n--- END EVIDENCE ---\n\n"
        "Extract all factual claims from the evidence. Output JSON with 'claims' array. MAX 8 claims."
    )
    messages = [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]

    t_p1 = time.time()
    try:
        response = await router.complete(messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims[:8]  # Hard cap
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0, len(evidence))

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0, len(evidence))
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0, len(evidence))

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
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0, len(evidence))

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "claim_cap_8",
            p1_in, p1_out, p2_in, p2_out, len(evidence))


# ---- Strategy F: Best safe combination (6 chunks + minimal prompt + claim cap 8) ----
async def strategy_f_combined(router, query, evidence, settings):
    from app.orchestration.two_pass_synthesis import verify_claims, ClaimGenerationOutput
    from app.orchestration.two_pass_prompts import build_verified_synthesis_messages
    from app.llm_gateway.providers.models import Message, MessageRole
    from app.orchestration.prompts import _format_evidence_block
    import json as _json

    evidence = evidence[:6]
    plan = ResearchPlan(objective=query, subquestions=[query])
    t_total = time.time()

    evidence_block = _format_evidence_block(evidence, include_scores=True)
    system = (
        "Extract factual claims from evidence. Output JSON: "
        '{"claims":[{"claim":"...","evidence_ids":[1],"claim_type":"factual","numerical_values":[]}]}'
        "\nRules: every claim must cite [N]. No external knowledge. One fact per claim. MAX 8 claims."
    )
    user = f"Q: {query}\nEvidence:\n{evidence_block}\nExtract claims as JSON. MAX 8."
    messages = [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]

    t_p1 = time.time()
    try:
        response = await router.complete(messages, response_format=ClaimGenerationOutput,
                                         temperature=0.1, timeout=settings.orchestration_llm_timeout,
                                         call_type="synthesis", request_id=None, query=query, tier="strong")
        p1_latency = (time.time() - t_p1) * 1000
        p1_in = getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        p1_out = getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        parsed = ClaimGenerationOutput.model_validate(_json.loads(response.content))
        claims = parsed.claims[:8]
    except Exception:
        return ("", (time.time() - t_total) * 1000, (time.time() - t_p1) * 1000, 0, 0, 0, 0, 0,
                "pass1_error", 0, 0, 0, 0, len(evidence))

    t_v = time.time()
    claim_set = verify_claims(claims, evidence)
    v_latency = (time.time() - t_v) * 1000
    supported = claim_set.supported_claims

    if not supported:
        for v in claim_set.verified:
            if v.claim.claim_type.value == "absent":
                return (v.claim.claim, (time.time() - t_total) * 1000, p1_latency, 0,
                        len(claims), 0, len(claim_set.rejected_claims), "absent",
                        p1_in, p1_out, 0, 0, len(evidence))
        return ("The available evidence does not contain sufficient information to answer this question.",
                (time.time() - t_total) * 1000, p1_latency, 0,
                len(claims), 0, len(claim_set.rejected_claims), "no_support",
                p1_in, p1_out, 0, 0, len(evidence))

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
        return ("", (time.time() - t_total) * 1000, p1_latency, p2_latency,
                len(claims), len(supported), len(claim_set.rejected_claims), "pass2_error",
                p1_in, p1_out, 0, 0, len(evidence))

    total = (time.time() - t_total) * 1000
    return (answer, total, p1_latency, p2_latency,
            len(claims), len(supported), len(claim_set.rejected_claims), "combined",
            p1_in, p1_out, p2_in, p2_out, len(evidence))


STRATEGIES = {
    "A_baseline_8chunks": strategy_a_baseline,
    "B_evidence_4": strategy_b_evidence_4,
    "C_evidence_6": strategy_c_evidence_6,
    "D_minimal_prompt": strategy_d_minimal_prompt,
    "E_claim_cap_8": strategy_e_claim_cap_8,
    "F_combined": strategy_f_combined,
}


async def run_experiment():
    print("=" * 70)
    print("Phase 32: Claim Generation Optimization & Inference Cost Reduction")
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
                answer, total_lat, p1_lat, p2_lat, p1_cl, v_cl, r_cl, method, p1_in, p1_out, p2_in, p2_out, ev_count = out
            except Exception as e:
                print(f"ERROR: {e}")
                answer, total_lat, p1_lat, p2_lat = "", 0, 0, 0
                p1_cl, v_cl, r_cl, method = 0, 0, 0, f"error: {e}"
                p1_in, p1_out, p2_in, p2_out, ev_count = 0, 0, 0, 0, len(evidence)

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

            results.append(ExpResult(
                query_id=qi["id"], query_class=qi["class"],
                answer=answer[:500] if answer else "", strategy=strat_name,
                latency_ms=total_lat, pass1_latency_ms=p1_lat, pass2_latency_ms=p2_lat,
                pass1_claims=p1_cl, verified_claims=v_cl, rejected_claims=r_cl,
                pass1_tokens_in=p1_in, pass1_tokens_out=p1_out,
                pass2_tokens_in=p2_in, pass2_tokens_out=p2_out,
                evidence_chunks=ev_count, evaluation=eval_dict,
            ))

        all_results[strat_name] = results

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for strat_name, results in all_results.items():
        valid = [r for r in results if "error" not in r.pass1_claims.__class__.__name__ and r.latency_ms > 0]
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
        print(f"  Avg Pass2 Latency:  {statistics.mean(p2_lats):.0f}ms" if p2_lats else "  Avg Pass2 Latency:  N/A")
        total_p1_in = sum(r.pass1_tokens_in for r in valid)
        total_p1_out = sum(r.pass1_tokens_out for r in valid)
        total_p2_in = sum(r.pass2_tokens_in for r in valid)
        total_p2_out = sum(r.pass2_tokens_out for r in valid)
        print(f"  Pass1 Tokens:       {total_p1_in} in / {total_p1_out} out")
        print(f"  Pass2 Tokens:       {total_p2_in} in / {total_p2_out} out")
        avg_ev = statistics.mean([r.evidence_chunks for r in valid])
        print(f"  Avg Evidence Chunks: {avg_ev:.1f}")
        failed = len(results) - len(valid)
        print(f"  Failed:             {failed}/{len(results)}")

    # ---- Latency comparison (baseline vs best) ----
    print("\n" + "=" * 70)
    print("LATENCY COMPARISON vs BASELINE")
    print("=" * 70)
    baseline = all_results.get("A_baseline_8chunks", [])
    baseline_valid = [r for r in baseline if r.latency_ms > 0]
    if baseline_valid:
        baseline_p1 = statistics.mean([r.pass1_latency_ms for r in baseline_valid if r.pass1_latency_ms > 0])
        baseline_total = statistics.mean([r.latency_ms for r in baseline_valid])
        for strat_name, results in all_results.items():
            if strat_name == "A_baseline_8chunks":
                continue
            valid = [r for r in results if r.latency_ms > 0]
            if not valid: continue
            p1 = statistics.mean([r.pass1_latency_ms for r in valid if r.pass1_latency_ms > 0])
            total = statistics.mean([r.latency_ms for r in valid])
            p1_saved = (1 - p1 / baseline_p1) * 100
            total_saved = (1 - total / baseline_total) * 100
            print(f"  {strat_name}: Pass1 {p1_saved:+.1f}% ({p1:.0f}ms), Total {total_saved:+.1f}% ({total:.0f}ms)")

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
                "query_id": r.query_id, "class": r.query_class, "answer": r.answer,
                "strategy": r.strategy, "latency_ms": r.latency_ms,
                "pass1_latency_ms": r.pass1_latency_ms, "pass2_latency_ms": r.pass2_latency_ms,
                "pass1_claims": r.pass1_claims, "verified_claims": r.verified_claims,
                "rejected_claims": r.rejected_claims, "pass1_tokens_in": r.pass1_tokens_in,
                "pass1_tokens_out": r.pass1_tokens_out, "pass2_tokens_in": r.pass2_tokens_in,
                "pass2_tokens_out": r.pass2_tokens_out, "evidence_chunks": r.evidence_chunks,
                **r.evaluation,
            }
            for r in results
        ]
    with open(output / "phase32_ablation.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to benchmarks/results/phase32_ablation.json")


if __name__ == "__main__":
    asyncio.run(run_experiment())
