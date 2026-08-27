"""Graph node implementations for the Agentic RAG loop (Phase 02).

Each `make_*_node` factory closes over the dependencies it needs (the
LLM router, the Phase 01 hybrid retriever, the reranker, and settings)
and returns an async callable matching LangGraph's `(state) -> dict`
node signature. Dependencies are injected rather than imported as
module-level singletons so tests can substitute a `MockProvider` /
in-memory store without monkeypatching.

Failure handling: every LLM call is wrapped so a provider error or a
malformed structured response degrades to a deterministic fallback
instead of raising out of the graph. Evidence already accumulated is
never discarded on an LLM failure — synthesis always runs over
whatever evidence exists, per the vault's evidence-first rule.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.evidence.models import EvidenceRef
from app.llm_gateway.providers.exceptions import LLMProviderError
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.models import (
    ComplexityLevel,
    EvidenceAssessment,
    QueryAnalysis,
    ResearchPlan,
    StopReason,
)
from app.orchestration.prompts import (
    build_analysis_messages,
    build_assessment_messages,
    build_planning_messages,
    build_synthesis_messages,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.stopping import stop_condition_to_reason
from app.reranking.reranker import NoOpReranker, Reranker
from app.retrieval.hybrid import HybridRetriever

logger = get_logger("argus.orchestration.nodes")

NodeFn = Callable[[OrchestrationState], Coroutine[Any, Any, dict]]

_MAX_CONSECUTIVE_EMPTY_RETRIEVALS = 2


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budget accounting.

    Not a substitute for a real tokenizer — good enough to bound loop
    growth without pulling in a model-specific tokenizer dependency.
    """
    return max(1, len(text) // 4)


async def _safe_structured_call(
    router: LLMRouter,
    *,
    messages: list,
    response_model: type[BaseModel],
    call_type: str,
    settings: Settings,
    request_id: str | None,
) -> tuple[Any | None, str | None]:
    """Run a structured LLM call, returning (parsed_model_or_None, error_message_or_None).

    Never raises: provider errors, malformed JSON, and schema validation
    failures are all normalized into an error string so callers can
    apply a deterministic fallback and keep the loop moving.
    """
    try:
        response = await router.complete(
            messages,
            response_format=response_model,
            timeout=settings.orchestration_llm_timeout,
            call_type=call_type,
            request_id=request_id,
        )
    except LLMProviderError as exc:
        logger.warning("orchestration_llm_call_failed", call_type=call_type, error=str(exc))
        return None, f"{call_type} call failed: {exc}"

    if not response.content:
        logger.warning("orchestration_llm_empty_response", call_type=call_type)
        return None, f"{call_type} call returned no content"

    try:
        parsed = response_model.model_validate(json.loads(response.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("orchestration_llm_malformed_response", call_type=call_type, error=str(exc))
        return None, f"{call_type} response did not match schema: {exc}"

    return parsed, None


def make_analyze_node(router: LLMRouter, settings: Settings) -> NodeFn:
    async def analyze_node(state: OrchestrationState) -> dict:
        messages = build_analysis_messages(state["query"])
        analysis, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=QueryAnalysis,
            call_type="query_analysis",
            settings=settings,
            request_id=state["request_id"],
        )
        warnings = list(state["warnings"])
        if analysis is None:
            warnings.append(f"query_analysis_fallback: {error}")
            analysis = QueryAnalysis(
                complexity=ComplexityLevel.MODERATE,
                reasoning="Fallback: query analysis LLM call failed.",
                suggested_subquestion_count=settings.orchestration_default_subquestions,
            )
        return {"query_analysis": analysis, "warnings": warnings}

    return analyze_node


def make_plan_node(router: LLMRouter, settings: Settings) -> NodeFn:
    async def plan_node(state: OrchestrationState) -> dict:
        analysis = state["query_analysis"] or QueryAnalysis(
            complexity=ComplexityLevel.MODERATE,
            reasoning="No analysis available.",
            suggested_subquestion_count=settings.orchestration_default_subquestions,
        )
        messages = build_planning_messages(state["query"], analysis)
        plan, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=ResearchPlan,
            call_type="research_planning",
            settings=settings,
            request_id=state["request_id"],
        )
        warnings = list(state["warnings"])
        if plan is None:
            warnings.append(f"research_plan_fallback: {error}")
            plan = ResearchPlan(
                objective=state["query"],
                subquestions=[state["query"]],
                token_budget=settings.orchestration_token_budget,
                iteration_budget=settings.orchestration_max_iterations,
                stopping_condition="Fallback single-pass plan: planner LLM call failed.",
            )

        # The planner LLM may propose budgets; the orchestrator has the
        # final word. Clamp to configured hard ceilings (never allow the
        # plan to request more than ARGUS is configured to spend).
        clamped_iteration_budget = min(plan.iteration_budget, settings.orchestration_max_iterations)
        clamped_token_budget = min(plan.token_budget, settings.orchestration_token_budget)
        plan = plan.model_copy(
            update={
                "iteration_budget": max(1, clamped_iteration_budget),
                "token_budget": max(1, clamped_token_budget),
            }
        )

        pending = list(plan.subquestions) if plan.subquestions else [plan.objective or state["query"]]

        return {
            "plan": plan,
            "pending_subquestions": pending,
            "max_iterations": plan.iteration_budget,
            "token_budget": plan.token_budget,
            "warnings": warnings,
        }

    return plan_node


def _merge_evidence(existing: list[EvidenceRef], new: list[EvidenceRef]) -> tuple[list[EvidenceRef], int]:
    """Dedup by chunk_id, keeping the highest score seen. Returns (merged, new_count)."""
    by_id: dict[UUID, EvidenceRef] = {ref.chunk_id: ref for ref in existing}
    new_count = 0
    for ref in new:
        prior = by_id.get(ref.chunk_id)
        if prior is None:
            new_count += 1
            by_id[ref.chunk_id] = ref
        elif ref.score > prior.score:
            by_id[ref.chunk_id] = ref
    merged = sorted(by_id.values(), key=lambda r: r.score, reverse=True)
    return merged, new_count


def make_retrieve_node(
    retriever: HybridRetriever,
    reranker: Reranker | NoOpReranker,
    settings: Settings,
    policy_router: Any | None = None,
) -> NodeFn:
    async def retrieve_node(state: OrchestrationState) -> dict:
        pending = list(state["pending_subquestions"])
        subquery = pending.pop(0) if pending else state["query"]

        question_pattern = state.get("question_pattern")

        try:
            if policy_router is not None and settings.retrieval_policy_enabled:
                pattern = policy_router.classify_question(subquery)
                if pattern is not None:
                    question_pattern = pattern.value
                results = await policy_router.execute_retrieval(
                    subquery,
                    pattern,
                    retriever,
                    top_k=settings.orchestration_retrieval_top_k,
                    reranker=reranker,
                )
                if not results:
                    # Deterministic fallback: plain hybrid pass. Never fabricate.
                    results = retriever.search(subquery, top_k=settings.orchestration_retrieval_top_k)
                    if results:
                        results = reranker.rerank(subquery, results, top_k=settings.orchestration_retrieval_top_k)
            else:
                results = retriever.search(subquery, top_k=settings.orchestration_retrieval_top_k)
                if results:
                    results = reranker.rerank(subquery, results, top_k=settings.orchestration_retrieval_top_k)
        except Exception as exc:
            logger.exception("orchestration_retrieval_critical", subquery=subquery, error=str(exc))
            results = []

        merged_evidence, new_count = _merge_evidence(state["evidence"], results)
        issued = list(state["issued_subqueries"]) + [subquery]

        tokens_used = state["tokens_used"] + sum(_estimate_tokens(r.text) for r in results)

        consecutive_empty = state["consecutive_empty_retrievals"] + 1 if new_count == 0 else 0

        prev_total = len(state["evidence"])
        gain = (new_count / prev_total) if prev_total else (1.0 if new_count else 0.0)
        gain_history = list(state.get("retrieval_gain_history") or []) + [round(gain, 4)]

        logger.info(
            "orchestration_retrieve_iteration",
            request_id=state["request_id"],
            subquery=subquery[:80],
            new_evidence=new_count,
            total_evidence=len(merged_evidence),
            iteration=state["iteration"] + 1,
            pattern=question_pattern,
        )

        return {
            "pending_subquestions": pending,
            "issued_subqueries": issued,
            "evidence": merged_evidence,
            "tokens_used": tokens_used,
            "iteration": state["iteration"] + 1,
            "consecutive_empty_retrievals": consecutive_empty,
            "question_pattern": question_pattern,
            "retrieval_gain_history": gain_history,
        }

    return retrieve_node


def make_assess_node(
    router: LLMRouter,
    settings: Settings,
    gap_detector: Any | None = None,
) -> NodeFn:
    async def assess_node(state: OrchestrationState) -> dict:
        plan = state["plan"]
        if plan is None:
            return {}

        # Deterministic short-circuits: never spend an LLM call once a
        # hard bound is already exceeded.
        if state["consecutive_empty_retrievals"] >= _MAX_CONSECUTIVE_EMPTY_RETRIEVALS:
            return {"sufficient": True, "stop_reason": StopReason.NO_NEW_EVIDENCE.value}

        if state["iteration"] >= state["max_iterations"] or state["tokens_used"] >= state["token_budget"]:
            return {"sufficient": True, "stop_reason": StopReason.BUDGET_EXHAUSTED.value}

        messages = build_assessment_messages(
            plan, state["evidence"], state["issued_subqueries"], state["pending_subquestions"]
        )
        assessment, error = await _safe_structured_call(
            router,
            messages=messages,
            response_model=EvidenceAssessment,
            call_type="evidence_extraction",
            settings=settings,
            request_id=state["request_id"],
        )

        warnings = list(state["warnings"])
        if assessment is None:
            warnings.append(f"assessment_fallback: {error}")
            # Fail safe: stop rather than loop forever on repeated LLM errors.
            return {
                "sufficient": True,
                "stop_reason": StopReason.ASSESSMENT_ERROR.value,
                "warnings": warnings,
            }

        # Active evidence seeking (Phase 06.2): formulate targeted retrieval
        # actions whenever the assessment concludes the evidence does not
        # answer the question. Deterministic — never an LLM call.
        gaps: list[dict[str, Any]] = []
        if gap_detector is not None and not assessment.sufficient:
            gaps = gap_detector.detect_gaps(state, plan, state["evidence"])
            evidence_tasks = list(state.get("evidence_tasks") or []) + list(gaps)
        else:
            evidence_tasks = list(state.get("evidence_tasks") or [])

        if assessment.sufficient or not assessment.next_subquery:
            if not assessment.sufficient and gap_detector is not None and gap_detector.should_re_retrieve(gaps):
                # The assessor ran out of ideas but the policy sees a real
                # evidence gap: queue its highest-priority targeted action.
                top_gap = max(gaps, key=lambda g: g.get("priority", 0.0))
                next_q = (top_gap.get("suggested_query") or "").strip()
                already_issued = {q.strip().lower() for q in state["issued_subqueries"]}
                pending = list(state["pending_subquestions"])
                if next_q and next_q.lower() not in already_issued and next_q not in pending:
                    pending.append(next_q)
                    return {
                        "sufficient": False,
                        "pending_subquestions": pending,
                        "evidence_tasks": evidence_tasks,
                        "warnings": warnings,
                    }

            stop_reason = (
                StopReason.SUFFICIENT_EVIDENCE.value
                if assessment.sufficient
                else StopReason.NO_SUBQUESTIONS.value
            )
            return {
                "sufficient": True,
                "stop_reason": stop_reason,
                "warnings": warnings,
                "evidence_tasks": evidence_tasks,
            }

        # Not sufficient, and a next query was proposed: queue it unless
        # it's a near-duplicate of one we've already issued.
        already_issued = {q.strip().lower() for q in state["issued_subqueries"]}
        pending = list(state["pending_subquestions"])
        next_q = assessment.next_subquery.strip()
        if next_q and next_q.lower() not in already_issued and next_q not in pending:
            pending.append(next_q)
        elif not pending:
            # Nothing new to try and nothing queued: stop rather than loop.
            return {
                "sufficient": True,
                "stop_reason": StopReason.NO_SUBQUESTIONS.value,
                "warnings": warnings,
                "evidence_tasks": evidence_tasks,
            }

        return {
            "sufficient": False,
            "pending_subquestions": pending,
            "warnings": warnings,
            "evidence_tasks": evidence_tasks,
        }

    return assess_node


def make_stop_check_node(stopping_logic: Any | None) -> NodeFn:
    """Evaluate the Phase 06 stopping logic after each assessment (V2 §5.4).

    When no stopping logic is wired, the node is a no-op and the router
    keeps the Phase 02 assess-based behavior unchanged.
    """

    async def stop_check_node(state: OrchestrationState) -> dict:
        if stopping_logic is None:
            return {}
        decision = await stopping_logic.should_stop(state)
        checked_all = list((decision.metadata or {}).get("checked", []))
        fired = decision.condition.value if decision.condition is not None else None
        if decision.should_stop:
            reason = stop_condition_to_reason(decision.condition)
            return {
                "sufficient": True,
                "stop_reason": reason.value if reason else state.get("stop_reason"),
                "stop_conditions_checked": checked_all,
                "stop_condition_fired": fired,
            }
        return {
            "stop_conditions_checked": checked_all,
            "stop_condition_fired": fired,
        }

    return stop_check_node


_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def make_synthesize_node(router: LLMRouter, settings: Settings) -> NodeFn:
    async def synthesize_node(state: OrchestrationState) -> dict:
        plan = state["plan"]
        if plan is None:
            return {}
        evidence = state["evidence"]
        warnings = list(state["warnings"])

        if not evidence:
            answer = (
                "No supporting evidence was retrieved for this question. "
                "I can't produce a cited answer without evidence to draw on."
            )
            return {"answer": answer, "warnings": warnings}

        messages = build_synthesis_messages(plan, evidence)
        try:
            response = await router.complete(
                messages,
                temperature=0.2,
                timeout=settings.orchestration_llm_timeout,
                call_type="synthesis",
                request_id=state["request_id"],
            )
            answer = response.content or ""
        except LLMProviderError as exc:
            logger.warning("orchestration_synthesis_failed", error=str(exc))
            warnings.append(f"synthesis_fallback: {exc}")
            answer = ""

        if not answer.strip():
            # Degrade to a deterministic, still evidence-grounded summary
            # rather than returning nothing.
            top = evidence[: min(3, len(evidence))]
            bullets = "\n".join(f"- {r.text.strip()[:300]} [{i}]" for i, r in enumerate(top, 1))
            answer = (
                "Synthesis was unavailable; returning the top retrieved evidence instead:\n"
                f"{bullets}"
            )
            warnings.append("synthesis_degraded_to_raw_evidence")

        return {"answer": answer, "warnings": warnings}

    return synthesize_node


def extract_cited_indices(answer: str, evidence_count: int) -> list[int]:
    """Parse bracket citation markers like `[2]` out of the synthesized answer.

    Returns 1-based indices that are within range, in first-seen order,
    deduplicated. Used to build the final citation list from only the
    evidence the model actually referenced.
    """
    seen: list[int] = []
    for match in _CITATION_MARKER_RE.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= evidence_count and idx not in seen:
            seen.append(idx)
    return seen
