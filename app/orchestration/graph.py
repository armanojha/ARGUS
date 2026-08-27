"""Agentic RAG orchestration graph (Phase 02).

Builds and runs the LangGraph state machine:

    analyze -> plan -> retrieve -> assess -+-> retrieve (loop)
                                            +-> synthesize -> END

No checkpointer is configured — each `run_query()` call is a single
in-memory execution with no cross-request persistence. That's
deliberate: Phase 02 is bounded, single-shot orchestration, not
long-running agent memory (memory is a later phase per the vault's
phase boundary).

This module is the only place that wires the LLM gateway (Phase 00.3)
and hybrid retriever (Phase 01) together for the agentic loop —
individual nodes stay dependency-injected and gateway-agnostic.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.config import Settings, get_settings
from app.llm_gateway import get_router
from app.llm_gateway.routing.router import LLMRouter
from app.logging_config import get_logger
from app.orchestration.models import OrchestrationCitation, OrchestrationResult, StopReason
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


def _route_after_assess(state: OrchestrationState) -> str:
    if state["sufficient"]:
        return "synthesize"
    return "retrieve"


def build_graph(
    router: LLMRouter,
    retriever: HybridRetriever,
    reranker: Reranker | NoOpReranker,
    settings: Settings,
    policy_router: Any | None = None,
    gap_detector: Any | None = None,
    stopping_logic: Any | None = None,
):
    """Compile the orchestration StateGraph. Cheap; safe to call per-request.

    Phase 06 wiring (all deterministic, all optional so Phase 02 callers
    keep working unchanged):
      * `policy_router`   — adaptive retrieval dispatch on each retrieve node.
      * `gap_detector`    — active evidence seeking on each assess node.
      * `stopping_logic`  — full V2 §5.4 stop-condition evaluation after each
                            assessment (the stop_check node).
    """
    workflow = StateGraph(OrchestrationState)

    workflow.add_node("analyze", make_analyze_node(router, settings))  # type: ignore
    workflow.add_node("plan", make_plan_node(router, settings))  # type: ignore
    workflow.add_node(
        "retrieve", make_retrieve_node(retriever, reranker, settings, policy_router=policy_router)  # type: ignore
    )
    workflow.add_node(  # type: ignore
        "assess", make_assess_node(router, settings, gap_detector=gap_detector)
    )
    workflow.add_node(  # type: ignore
        "stop_check", make_stop_check_node(stopping_logic)
    )
    workflow.add_node("synthesize", make_synthesize_node(router, settings))  # type: ignore

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "assess")
    workflow.add_edge("assess", "stop_check")
    workflow.add_conditional_edges(
        "stop_check", _route_after_assess, {"retrieve": "retrieve", "synthesize": "synthesize"}
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def _initial_state(query: str, request_id: str | None, settings: Settings) -> OrchestrationState:
    return OrchestrationState(
        request_id=request_id,
        query=query,
        max_iterations=settings.orchestration_max_iterations,
        token_budget=settings.orchestration_token_budget,
        query_analysis=None,
        plan=None,
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
    )


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
        token_usage_estimate=final_state["tokens_used"],
        request_id=final_state["request_id"],
        warnings=final_state["warnings"],
        question_pattern=final_state.get("question_pattern"),
        stop_condition=final_state.get("stop_condition_fired"),
        stop_decisions=final_state.get("stop_conditions_checked") or [],
        evidence_tasks=final_state.get("evidence_tasks") or [],
    )


async def run_query(
    query: str,
    *,
    request_id: str | None = None,
    user_early_stop: bool = False,
    router: LLMRouter | None = None,
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

    graph = build_graph(
        router,
        retriever,
        reranker,
        settings,
        policy_router=policy_router,
        gap_detector=gap_detector,
        stopping_logic=stopping_logic,
    )
    initial_state = _initial_state(query, request_id, settings)
    if user_early_stop:
        initial_state["user_early_stop"] = True

    logger.info("orchestration_run_started", request_id=request_id, query=query[:100])
    final_state: OrchestrationState = await graph.ainvoke(initial_state)
    logger.info(
        "orchestration_run_finished",
        request_id=request_id,
        iterations=final_state["iteration"],
        stop_reason=final_state["stop_reason"],
        stop_condition=final_state.get("stop_condition_fired"),
        evidence_count=len(final_state["evidence"]),
    )

    return _build_result(final_state)
