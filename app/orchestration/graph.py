"""Agentic RAG orchestration graph (Phase 02 / Phase 06 policy / Phase 08 memory hook / Phase 10 multi-agent).

Builds and runs the LangGraph state machine:

    analyze -> plan -> [memory_enhance] -> retrieve -> assess -> stop_check -+-> retrieve (loop)
                                                                              +-> [debate] -> synthesize -> END

No checkpointer is configured; each `run_query()` call is a single
in-memory execution with no cross-request persistence. That's
deliberate: Phase 02 is bounded, single-shot orchestration, not
long-running agent memory (memory is a later phase per the vault's
phase boundary).

This module is the only place that wires the LLM gateway (Phase 00.3),
hybrid retriever (Phase 01), the adaptive policy (Phase 06), the
optional memory store (Phase 08), and the multi-agent coordinator (Phase 10)
together for the agentic loop; individual nodes stay dependency-injected
and gateway-agnostic.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.llm_gateway import get_router
from app.llm_gateway.routing.complexity import ComplexityTier, classify_complexity
from app.llm_gateway.telemetry import check_call_ceiling
from app.logging_config import get_logger
from app.orchestration.agents.coordinator import (
    AgentCoordinator,
    create_agent_coordinator,
    should_skip_verification,
)
from app.orchestration.models import (
    ComplexityLevel,
    OrchestrationCitation,
    OrchestrationResult,
    OrchestrationVerification,
    Outcome,
    QueryAnalysis,
    ResearchPlan,
    StopReason,
)
from app.orchestration.nodes import (
    extract_cited_indices,
    make_analyze_node,
    make_assess_node,
    make_plan_node,
    make_retrieve_node,
    make_stop_check_node,
    make_synthesize_node,
)
from app.orchestration.state import OrchestrationState
from app.orchestration.stopping import build_stopping_logic
from app.reranking import get_reranker
from app.reranking.reranker import NoOpReranker, Reranker
from app.retrieval.hybrid import HybridRetriever, get_hybrid_retriever
from app.retrieval.router import get_retrieval_policy_router
from app.retrieval.seeking import get_adaptive_gap_detector

logger = get_logger("argus.orchestration.graph")


def _is_simple_query(query: str) -> bool:
    """Whether a query is simple enough to skip the analyze/plan/assess LLM calls.

    Uses the zero-LLM complexity classifier (HARDEN-06.5.2); a FAST tier
    means a single retrieve -> synthesize pass is sufficient — no research
    planning or iterative sufficiency assessment needed (HARDEN-06.5.3).
    """
    return classify_complexity(query) == ComplexityTier.FAST


def _fast_path_plan(query: str) -> ResearchPlan:
    """Deterministic, no-LLM ResearchPlan for a simple query.

    A simple lookup needs no decomposition: the single research objective is
    the query itself, and we do a single-pass grounding retrieval.
    """
    return ResearchPlan(
        objective=query.strip(),
        entities=[],
        time_window=None,
        subquestions=[query.strip()],
        evidence_type="factual",
        preferred_retrieval_methods=["hybrid"],
        required_sources=[],
        risk_level="low",
    )


def _route_entry(state: OrchestrationState) -> str:
    """Graph entry router: simple queries skip analyze/plan via the fast path."""
    return "retrieve" if state.get("fast_path") else "analyze"


def _route_after_retrieve(state: OrchestrationState) -> str:
    """After retrieval: fast-path queries synthesize immediately (no assess loop)."""
    if state.get("fast_path"):
        return "synthesize"
    return "assess"


def _route_after_assess(state: OrchestrationState) -> str:
    if state["sufficient"]:
        return "synthesize"
    return "retrieve"


async def _memory_enhance_node(state: OrchestrationState, memory_store: Any) -> dict:
    """Enhance the research plan using persistent memory."""
    plan = state.get("plan")
    query = state.get("query")
    if not plan or not query:
        return {}

    if memory_store is None:
        return {}

    try:
        from app.memory.planner_integration import create_memory_aware_planner
        planner = await create_memory_aware_planner(memory_store)
        enhanced_plan = await planner.enhance_plan_with_memory(plan, query, memory_store)
        if enhanced_plan is not plan:
            logger.info("plan_enhanced_with_memory", request_id=state["request_id"])
            return {"plan": enhanced_plan}
    except Exception as exc:  # noqa: BLE001 - memory enhancement is non-critical
        logger.warning("memory_enhance_failed", error=str(exc), request_id=state["request_id"])

    return {}


async def _debate_node(state: OrchestrationState, agent_coordinator: AgentCoordinator) -> dict:
    """Run multi-agent debate for high-risk/uncertainty questions (Phase 10)."""
    if agent_coordinator is None:
        return {}

    try:
        logger.info("multi_agent_debate_node_started", request_id=state["request_id"])
        updated_state = await agent_coordinator.run_debate(state)
        logger.info("multi_agent_debate_node_finished", request_id=state["request_id"])

        # Extract judge's final answer if available
        agent_messages = updated_state.get("agent_messages", [])
        judge_answer = ""
        for msg_dict in reversed(agent_messages):
            if msg_dict.get("from_agent") == "judge":
                judge_answer = msg_dict.get("payload", {}).get("final_answer", "")
                if judge_answer:
                    break

        return {
            "agent_messages": updated_state.get("agent_messages", []),
            "agent_round": updated_state.get("agent_round", 0),
            "debate_active": updated_state.get("debate_active", False),
            "disagreement_detected": updated_state.get("disagreement_detected", False),
            "answer": judge_answer if judge_answer else state.get("answer"),
        }
    except Exception as exc:  # noqa: BLE001 - debate is non-critical, fall back to normal synthesis
        logger.warning("multi_agent_debate_failed", error=str(exc), request_id=state["request_id"])
        return {"debate_active": False}


def build_graph(
    router: Any,  # LLMRouter | MultiModelRouter (gateway-agnostic wiring)
    retriever: HybridRetriever,
    reranker: Reranker | NoOpReranker,
    settings: Settings,
    policy_router: Any | None = None,
    gap_detector: Any | None = None,
    stopping_logic: Any | None = None,
    memory_store: Any | None = None,
    agent_coordinator: AgentCoordinator | None = None,
):
    """Compile the orchestration StateGraph. Cheap; safe to call per-request.

    Phase 06 wiring (all deterministic, all optional so Phase 02 callers
    keep working unchanged):
      * `policy_router`   — adaptive retrieval dispatch on each retrieve node.
      * `gap_detector`    — active evidence seeking on each assess node.
      * `stopping_logic`  — full V2 §5.4 stop-condition evaluation after each
                            assessment (the stop_check node).
    Phase 08 wiring (optional):
      * `memory_store`    — persistent memory for plan enhancement.
    Phase 10 wiring (optional):
      * `agent_coordinator` — multi-agent debate for high-risk questions.
    """
    workflow = StateGraph(OrchestrationState)

    workflow.add_node("analyze", make_analyze_node(router, settings))  # type: ignore
    workflow.add_node("plan", make_plan_node(router, settings))  # type: ignore
    if memory_store is not None:
        workflow.add_node("memory_enhance", partial(_memory_enhance_node, memory_store=memory_store))  # type: ignore
    workflow.add_node("retrieve", make_retrieve_node(retriever, reranker, settings, policy_router=policy_router))  # type: ignore
    workflow.add_node("assess", make_assess_node(router, settings, gap_detector=gap_detector))  # type: ignore
    workflow.add_node("stop_check", make_stop_check_node(stopping_logic))  # type: ignore
    if agent_coordinator is not None:
        workflow.add_node("debate", partial(_debate_node, agent_coordinator=agent_coordinator))  # type: ignore
    workflow.add_node("synthesize", make_synthesize_node(router, settings))  # type: ignore

    workflow.add_conditional_edges(
        START,
        _route_entry,
        {"analyze": "analyze", "retrieve": "retrieve"},
    )
    workflow.add_edge("analyze", "plan")
    if memory_store is not None:
        workflow.add_edge("plan", "memory_enhance")
        workflow.add_edge("memory_enhance", "retrieve")
    else:
        workflow.add_edge("plan", "retrieve")
    # Fast-path queries skip the assess/stop_check loop entirely.
    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"assess": "assess", "synthesize": "synthesize"},
    )
    workflow.add_edge("assess", "stop_check")

    # Route after stop_check: if debate is enabled and agents should activate, go to debate
    # otherwise go directly to synthesize
    def _route_after_stop_check(state: OrchestrationState) -> str:
        if state["sufficient"]:
            if agent_coordinator is not None and settings.multiagent_enabled:
                # Check if agents should activate for this state
                active_roles = agent_coordinator.should_activate_agents(state)
                if len(active_roles) > 2:  # More than just Researcher + Verifier + Judge
                    return "debate"
            return "synthesize"
        return "retrieve"

    if agent_coordinator is not None:
        workflow.add_conditional_edges(
            "stop_check", _route_after_stop_check, {"retrieve": "retrieve", "synthesize": "synthesize", "debate": "debate"}
        )
        workflow.add_edge("debate", "synthesize")
    else:
        workflow.add_conditional_edges(
            "stop_check", _route_after_assess, {"retrieve": "retrieve", "synthesize": "synthesize"}
        )

    workflow.add_edge("synthesize", END)

    return workflow.compile()


def _initial_state(query: str, request_id: str | None, settings: Settings) -> OrchestrationState:
    fast_path = _is_simple_query(query)
    pre_analysis: QueryAnalysis | None = None
    pre_plan: ResearchPlan | None = None
    if fast_path:
        logger.info("orchestration_fast_path", request_id=request_id, query=query[:100])
        pre_analysis = QueryAnalysis(
            complexity=ComplexityLevel.SIMPLE,
            reasoning="Query classified as simple (fast tier); skipping research planning.",
            suggested_subquestion_count=1,
        )
        pre_plan = _fast_path_plan(query)

    return OrchestrationState(
        request_id=request_id,
        query=query,
        max_iterations=settings.orchestration_max_iterations,
        token_budget=settings.orchestration_token_budget,
        query_analysis=pre_analysis,
        plan=pre_plan,
        pending_subquestions=[],
        issued_subqueries=[],
        evidence=[],
        consecutive_empty_retrievals=0,
        iteration=0,
        tokens_used=0,
        sufficient=False,
        stop_reason=None,
        answer=None,
        warnings=[],
        # Phase 06 adaptive policy fields (populated as the run proceeds)
        question_pattern=None,
        retrieval_gain_history=[],
        user_early_stop=False,
        contradiction_signals=[],
        stop_conditions_checked=[],
        stop_condition_fired=None,
        evidence_tasks=[],
        # Phase 10 multi-agent fields
        agent_messages=[],
        agent_round=0,
        debate_active=False,
        disagreement_detected=False,
        # Phase 06.5.3 fast-path marker
        fast_path=fast_path,
    )


def _derive_outcome(final_state: OrchestrationState) -> Outcome:
    """Derive the truthful run outcome from what was actually delivered.

    Independent of ``stop_reason`` (which only tells us why the loop stopped):
    the outcome reflects whether a usable, grounded answer was produced and
    whether the run degraded or relied on a fallback. Classification:
      * empty evidence + "no evidence" answer  -> NOT_FOUND (truthful no-answer)
      * empty / failed answer (evidence present) -> NO_ANSWER
      * any fallback/degradation warning        -> ANSWERED_FALLBACK / DEGRADED
      * otherwise                                -> ANSWERED
    """
    answer = (final_state.get("answer") or "").strip()
    evidence = final_state.get("evidence") or []
    warnings = final_state.get("warnings") or []

    # Truthful "no supporting evidence" statement (synthesize node, evidence empty).
    if not evidence:
        if not answer or answer.startswith("No supporting evidence was retrieved"):
            return Outcome.NOT_FOUND
        # Evidence gate normally blocks synthesis without evidence, but be safe.
        return Outcome.NOT_FOUND

    if not answer:
        return Outcome.NO_ANSWER

    if any(w.startswith("synthesis_fallback") or w == "synthesis_degraded_to_raw_evidence" for w in warnings):
        return Outcome.ANSWERED_DEGRADED

    if any(
        w.startswith(("research_plan_fallback", "query_analysis_fallback", "assessment_fallback"))
        for w in warnings
    ):
        return Outcome.ANSWERED_FALLBACK

    return Outcome.ANSWERED


def _build_result(final_state: OrchestrationState) -> OrchestrationResult:
    plan = final_state["plan"]
    assert plan is not None
    evidence = final_state["evidence"]
    answer = final_state["answer"] or ""

    cited_indices = extract_cited_indices(answer, len(evidence))
    if not cited_indices and evidence:
        # Model produced no bracket citations (or synthesis degraded) —
        # never silently drop provenance: fall back to surfacing the
        # top evidence actually used to ground the answer.
        cited_indices = list(range(1, min(3, len(evidence)) + 1))

    citations = [
        OrchestrationCitation(
            ref_id=idx,
            chunk_id=str(evidence[idx - 1].chunk_id),
            document_id=str(evidence[idx - 1].document_id),
            source_id=str(evidence[idx - 1].source_id),
            source_path=evidence[idx - 1].source_path,
            source_type=evidence[idx - 1].source_type.value,
            text=evidence[idx - 1].text,
            page_start=evidence[idx - 1].page_start,
            page_end=evidence[idx - 1].page_end,
            section_path=evidence[idx - 1].section_path,
            score=evidence[idx - 1].score,
            metadata=evidence[idx - 1].metadata,
        )
        for idx in cited_indices
    ]

    stop_reason_raw = final_state["stop_reason"] or StopReason.BUDGET_EXHAUSTED.value

    return OrchestrationResult(
        query=final_state["query"],
        plan=plan,
        answer=answer,
        citations=citations,
        iterations_used=final_state["iteration"],
        sub_queries_issued=final_state["issued_subqueries"],
        stop_reason=StopReason(stop_reason_raw),
        outcome=_derive_outcome(final_state),
        token_usage_estimate=final_state["tokens_used"],
        request_id=final_state["request_id"],
        warnings=final_state["warnings"],
        question_pattern=final_state.get("question_pattern"),
        stop_condition=final_state.get("stop_condition_fired"),
        stop_decisions=final_state.get("stop_conditions_checked") or [],
        evidence_tasks=final_state.get("evidence_tasks") or [],
        # Phase 10 multi-agent traceability
        agent_round=final_state.get("agent_round"),
        agent_messages=final_state.get("agent_messages") or [],
        disagreement_detected=final_state.get("disagreement_detected"),
    )


async def _run_selective_verification(
    final_state: OrchestrationState,
    *,
    router: Any,
    settings: Settings,
    request_id: str | None,
) -> OrchestrationVerification | None:
    """Phase 07b: run the deterministic, selective verification post-step.

    Disabled entirely unless ``settings.verification_enabled`` is True.

    The stage is *fail-safe by design*: it runs *after* the synthesize node
    has already produced a grounded, cited answer, and it can only annotate
    that answer — never replace or collapse it. Verification is invoked at
    most once and adds at most a single ``verification`` LLM call, gated by:

      * ``check_call_ceiling()``  — never spend the last of the run's call
        budget (which the synthesizer already used) on verification.
      * ``should_skip_verification`` — the shared 06.5.4 gate: simple/low-
        risk, high-confidence, non-conflicting evidence is NOT verified.

    Any failure (LLM error, malformed output, timeout) is captured into
    :class:`OrchestrationVerification` with ``status="error"`` and the
    grounded answer is returned unchanged.
    """
    if not getattr(settings, "verification_enabled", False):
        return OrchestrationVerification(triggered=False, skipped_reason="disabled")

    plan = final_state.get("plan")
    evidence = final_state.get("evidence") or []

    if check_call_ceiling():
        logger.info("verification_skipped_call_budget", request_id=request_id)
        return OrchestrationVerification(triggered=False, skipped_reason="call_budget")

    if should_skip_verification(plan, evidence, settings):
        logger.info("verification_skipped_low_risk", request_id=request_id)
        return OrchestrationVerification(triggered=False, skipped_reason="low_risk")

    answer = final_state.get("answer") or ""
    if not answer:
        logger.info("verification_skipped_no_answer", request_id=request_id)
        return OrchestrationVerification(triggered=False, skipped_reason="no_answer")

    # Map the cited chunk ids used to ground the answer into the verifier.
    built_result = _build_result(final_state)
    from uuid import UUID, uuid4

    from app.verification.engine import verify_claim
    from app.verification.models import VerificationRequest

    entity_names = list(plan.entities) if plan is not None else []
    temporal_context = plan.time_window if plan is not None else None
    supporting_chunk_ids = []
    for c in built_result.citations:
        try:
            supporting_chunk_ids.append(UUID(c.chunk_id))
        except (ValueError, AttributeError):
            logger.debug("verification_skip_invalid_chunk_id", chunk_id=c.chunk_id)
    verification_request = VerificationRequest(
        claim_id=uuid4(),
        claim_text=answer,
        supporting_chunk_ids=supporting_chunk_ids,
        entity_names=entity_names,
        temporal_context=temporal_context,
    )
    try:
        from app.evidence.store import get_evidence_store
        from app.graph.store import get_graph_store

        verification_result = await asyncio.wait_for(
            verify_claim(
                verification_request,
                router=router,
                evidence_store=get_evidence_store(),
                graph_store=get_graph_store(),
                settings=settings,
                request_id=request_id,
            ),
            timeout=getattr(settings, "orchestration_llm_timeout", 30.0) + 5.0,
        )
        status_value = getattr(verification_result.status, "value", str(verification_result.status))
        return OrchestrationVerification(
            triggered=True,
            status=status_value,
            confidence=verification_result.confidence,
            contradiction_detected=bool(verification_result.contradictions),
            reasoning=verification_result.reasoning or None,
            error=str(verification_result.metadata.get("error", "")) or None
            if status_value == "error"
            else None,
        )
    except Exception as exc:  # noqa: BLE001 - fail-safe: annotate, don't crash
        logger.warning("verification_stage_failed", request_id=request_id, error=str(exc))
        return OrchestrationVerification(
            triggered=True,
            status="error",
            error=str(exc),
            reasoning="verification failed; returning the grounded, cited answer unchanged",
        )


async def run_query(
    query: str,
    *,
    request_id: str | None = None,
    user_early_stop: bool = False,
    router: Any = None,  # LLMRouter | MultiModelRouter — accepts any LLM-gateway router
    retriever: HybridRetriever | None = None,
    reranker: Reranker | NoOpReranker | None = None,
    settings: Settings | None = None,
    policy_router: Any | None = None,
    gap_detector: Any | None = None,
    stopping_logic: Any | None = None,
) -> OrchestrationResult:
    """Run one query through the Agentic RAG loop end to end.

    Resolves gateway/retriever/reranker singletons from Phase 00.3 /
    Phase 01 if not provided, ensures retrieval indexes are current,
    builds and invokes the graph, and maps the final state onto
    `OrchestrationResult` (including provenance-preserving citations).

    Phase 06 components are built from settings toggles when not injected:
      * `policy_router`  — `settings.retrieval_policy_enabled`
      * `gap_detector`   — `settings.active_evidence_seeking_enabled`
      * `stopping_logic` — `settings.stopping_logic_enabled`

    Phase 08 component (optional, non-fatal):
      * `memory_store`   — `settings.memory_enabled`. Imported lazily so a
                           still-in-progress memory module can never break
                           the Phase 02/06 loop.

    Phase 10 component (optional, non-fatal):
      * `agent_coordinator` — `settings.multiagent_enabled`. Multi-agent
                              debate for high-risk/uncertainty questions.
    """
    settings = settings or get_settings()
    router = router or get_router()
    retriever = retriever or get_hybrid_retriever()
    reranker = reranker or get_reranker()

    # Build indexes once up front rather than per retrieval iteration.
    retriever.ensure_indexes()

    if policy_router is None and settings.retrieval_policy_enabled:
        policy_router = get_retrieval_policy_router()
    if gap_detector is None and settings.active_evidence_seeking_enabled:
        gap_detector = get_adaptive_gap_detector()
    if stopping_logic is None and settings.stopping_logic_enabled:
        stopping_logic = build_stopping_logic(settings)

    # Phase 08: initialize memory system if enabled (non-fatal when the
    # memory module is unavailable or in progress — the loop still runs).
    memory_store: Any = None
    if getattr(settings, "memory_enabled", False):
        try:
            from app.memory import get_memory_factory, initialize_memory_system

            await initialize_memory_system()
            factory = get_memory_factory()
            if hasattr(factory, "create_memory_store"):
                memory_store = factory.create_memory_store()
        except Exception as exc:  # noqa: BLE001 - memory is an optional enhancement
            logger.warning("memory_system_unavailable", error=str(exc))

    # Phase 10: initialize agent coordinator if enabled (non-fatal)
    agent_coordinator: AgentCoordinator | None = None
    if getattr(settings, "multiagent_enabled", False):
        try:
            agent_coordinator = create_agent_coordinator(router, settings, retriever, reranker)
        except Exception as exc:  # noqa: BLE001 - multi-agent is an optional enhancement
            logger.warning("agent_coordinator_unavailable", error=str(exc))

    graph = build_graph(
        router,
        retriever,
        reranker,
        settings,
        policy_router=policy_router,
        gap_detector=gap_detector,
        stopping_logic=stopping_logic,
        memory_store=memory_store,
        agent_coordinator=agent_coordinator,
    )
    initial_state = _initial_state(query, request_id, settings)
    if user_early_stop:
        initial_state["user_early_stop"] = True

    logger.info("orchestration_run_started", request_id=request_id, query=query[:100])
    final_state: OrchestrationState = await asyncio.wait_for(
        graph.ainvoke(initial_state),
        timeout=getattr(settings, "orchestration_timeout", None) or 120,
    )
    logger.info(
        "orchestration_run_finished",
        request_id=request_id,
        iterations=final_state["iteration"],
        stop_reason=final_state["stop_reason"],
        stop_condition=final_state.get("stop_condition_fired"),
        evidence_count=len(final_state["evidence"]),
    )

    result = _build_result(final_state)

    # Phase 07b: selective claim verification, post-synthesis and fail-safe.
    # Runs after the grounded answer exists and can only annotate it (never
    # replace it). Bounded to a single `verification` LLM call, opt-in via
    # `verification_enabled`, and skipped by the 06.5.4 gate or the call
    # ceiling. Import here so a verifier-sharing/LLM hiccup can never break
    # the Phase 02/06 loop.
    try:
        verification = await _run_selective_verification(
            final_state,
            router=router,
            settings=settings,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001 - fail-safe by construction
        logger.warning("verification_stage_unavailable", request_id=request_id, error=str(exc))
        verification = OrchestrationVerification(triggered=False, skipped_reason="disabled")
    if verification is not None:
        result = result.model_copy(update={"verification": verification})

    return result
