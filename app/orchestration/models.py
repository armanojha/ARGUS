"""Agentic RAG data models (Phase 02).

Typed schemas for the orchestration loop: query complexity analysis,
the ResearchPlan produced by the planner, the sufficiency assessment
used to decide whether to retrieve again, and the final result returned
to callers. All LLM-produced structures flow through these Pydantic
models (via the LLM gateway's `response_format`) so the graph never
trusts free-form model output for control flow.

None of these models are a database of record — they describe a single
in-flight query's control state. Durable evidence/provenance remains in
the Phase 01 `EvidenceStore`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ComplexityLevel(str, Enum):
    """Coarse complexity bucket used for gating planner depth."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class QueryAnalysis(BaseModel):
    """Fast structured pass over the raw query (sub-phase 02.2).

    Cheap, single LLM call. Used only to size the plan (how many
    subquestions to ask for) — it does not itself decide retrieval or
    models.
    """

    model_config = ConfigDict(extra="forbid")

    complexity: ComplexityLevel
    reasoning: str = Field(description="One or two sentences on why this complexity was assigned.")
    suggested_subquestion_count: int = Field(ge=1, le=6)


class ResearchPlan(BaseModel):
    """Structured research plan produced by the planner node (V2 §5.1).

    Fields mirror the vault's Phase 02 spec. `token_budget` and
    `iteration_budget` are always clamped by the orchestrator to the
    configured hard ceilings (`Settings.orchestration_token_budget` /
    `orchestration_max_iterations`) after the plan is produced — the
    planner LLM proposes them, it does not have the final word.
    """

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(description="The single overall research objective, restated from the query.")
    entities: list[str] = Field(default_factory=list, description="Key named entities/topics involved.")
    time_window: str | None = Field(default=None, description="Relevant time window, if any (free text).")
    subquestions: list[str] = Field(default_factory=list, description="Decomposed subquestions to retrieve for.")
    evidence_type: str = Field(default="general", description="Kind of evidence sought, e.g. 'factual', 'comparative'.")
    preferred_retrieval_methods: list[str] = Field(
        default_factory=lambda: ["hybrid"],
        description="Retrieval modes to prefer, e.g. 'hybrid', 'bm25', 'vector'.",
    )
    required_sources: list[str] = Field(default_factory=list, description="Specific sources the plan calls out, if any.")
    risk_level: str = Field(default="low", description="Coarse risk/uncertainty label for the question, e.g. 'low'/'medium'/'high'.")
    token_budget: int = Field(default=6000, ge=1, description="Proposed token budget; clamped by orchestrator config.")
    iteration_budget: int = Field(default=2, ge=1, description="Proposed retrieval-iteration budget; clamped by orchestrator config.")
    stopping_condition: str = Field(
        default="Stop once every subquestion has supporting evidence or the budget is exhausted.",
        description="Free-text description of when the loop should stop.",
    )


class ResearchStrategy(BaseModel):
    """First-class per-iteration research strategy (unified control state).

    The orchestration loop previously scattered strategy across six state
    keys (pending_subquestions, max_iterations, token_budget,
    complexity_tier, question_pattern, evidence_tasks). ResearchStrategy
    unifies them into one object snapshotted every assess iteration, so the
    Brain UI can show HOW the strategy evolved — not just the final plan.

    Mutation drivers (existing mechanisms, now recorded — no new control
    flow): the sufficiency assessor (queries), the gap detector
    (targeted queries), the complexity-tier adjustment (verification
    depth / model routing), and budget consumption (iterations/tokens).
    """

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(description="Loop iteration this snapshot was taken at.")
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval mode driving this iteration (from question pattern / policy).",
    )
    queries: list[str] = Field(
        default_factory=list,
        description="Active retrieval queries: pending subquestions plus any gap-suggested queries.",
    )
    source_priorities: list[str] = Field(
        default_factory=list,
        description="Source paths ranked by evidence yield so far (most hits first).",
    )
    verification_depth: int = Field(
        default=2, ge=0,
        description="Verification depth for this iteration (from complexity tier).",
    )
    max_iterations: int = Field(description="Iteration ceiling for the run.")
    evidence_budget_tokens: int = Field(description="Token budget remaining for evidence work.")
    stop_reason: str | None = Field(
        default=None, description="Stop reason if the loop ended on this iteration.")
    mutated_from: str | None = Field(
        default=None,
        description="What drove this iteration's strategy change: "
        "'assessor', 'gap_detector', 'tier_adjustment', 'budget', or 'initial'.",
    )


class EvidenceAssessment(BaseModel):
    """Sufficiency check produced after each retrieval iteration (sub-phase 02.3/02.5).

    Drives the retrieve/synthesize branch. `next_subquery` is only
    consulted when `sufficient` is false and the iteration budget has
    not been exhausted.
    """

    model_config = ConfigDict(extra="forbid")

    sufficient: bool = Field(description="Whether accumulated evidence adequately covers the plan's objective.")
    reasoning: str = Field(description="One or two sentences explaining the sufficiency judgement.")
    next_subquery: str | None = Field(
        default=None,
        description="If not sufficient, the next retrieval query to run. Null if no further query is useful.",
    )


class Outcome(str, Enum):
    """The truthful *outcome* of a run, distinct from the loop's ``stop_reason``.

    ``stop_reason`` (``StopReason``) answers "why did the loop stop?" (control
    flow). ``Outcome`` answers "what actually got delivered?" (result quality).
    They can legitimately diverge — e.g. a hard-degraded run that never made a
    single usable provider call can still stop with
    ``NO_UNRESOLVED_CONTRADICTION``, and a healthy run can stop with
    ``BUDGET_EXHAUSTED``. Callers branching on "did we succeed?" should use
    ``outcome``, not ``stop_reason`` (HARDEN-07d.2).

    Ranks: ANSWERED* > NOT_FOUND > NO_ANSWER.
    """

    ANSWERED = "answered"  # grounded, cited answer delivered
    ANSWERED_FALLBACK = "answered_fallback"  # delivered via a fallback provider/plan
    ANSWERED_DEGRADED = "answered_degraded"  # delivered but degraded (raw-evidence synthesis, etc.)
    NOT_FOUND = "not_found"  # delivered a truthful "no evidence found" statement
    NO_ANSWER = "no_answer"  # nothing usable delivered (e.g. total provider failure)


class StopReason(str, Enum):
    """Why the orchestration loop stopped, for observability/acceptance checks."""

    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_NEW_EVIDENCE = "no_new_evidence"
    ASSESSMENT_ERROR = "assessment_error"
    NO_SUBQUESTIONS = "no_subquestions"
    # Phase 06 adaptive policy stop conditions (V2 §5.4)
    CLAIMS_SUPPORTED = "claims_supported"
    NO_UNRESOLVED_CONTRADICTION = "no_unresolved_contradiction"
    NEGLIGIBLE_EVIDENCE_GAIN = "negligible_evidence_gain"
    USER_EARLY_STOP = "user_early_stop"


class OrchestrationCitation(BaseModel):
    """A citation surfaced in the final answer, tracing back to Phase 01 evidence."""

    model_config = ConfigDict(extra="forbid")

    ref_id: int = Field(description="1-based citation marker used in the synthesized answer, e.g. '[2]'.")
    chunk_id: str
    document_id: str
    source_id: str
    source_path: str
    source_type: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrationVerification(BaseModel):
    """Selective claim verification metadata attached to a query result (Phase 07b).

    Additive and defaulted-``None`` so existing consumers of
    ``OrchestrationResult`` are unaffected. ``triggered`` distinguishes
    "verification was considered and skipped" from "not applicable"; the
    fail-safe principle is that verification can *annotate* but never
    *replace* a grounded, cited answer.
    """

    model_config = ConfigDict(extra="forbid")

    triggered: bool = Field(
        description="Whether the selective verification stage fired for this query "
        "(False when disabled, over call budget, or skipped by the 06.5.4 gate)."
    )
    skipped_reason: str | None = Field(
        default=None, description="Why verification was skipped (disabled / call_budget / low_risk)."
    )
    status: str | None = Field(
        default=None, description="VerificationStatus value when triggered: supported/partial/contradicted/unsupported/error."
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall verification confidence (0-1) when triggered."
    )
    contradiction_detected: bool | None = Field(
        default=None, description="Whether a material contradiction was detected, when triggered."
    )
    reasoning: str | None = Field(default=None, description="Verifier's explanation, when triggered.")
    error: str | None = Field(
        default=None, description="Verification failure detail, if any (never discards the cited answer)."
    )


class OrchestrationResult(BaseModel):
    """Final result of a query → plan → retrieve → synthesize run."""

    model_config = ConfigDict(extra="forbid")

    query: str
    plan: ResearchPlan
    answer: str
    citations: list[OrchestrationCitation]
    iterations_used: int
    sub_queries_issued: list[str]
    stop_reason: StopReason
    outcome: Outcome = Field(
        default=Outcome.ANSWERED,
        description="Truthful result outcome, derived from what was actually "
        "delivered (answer + grounding + fallback/degradation). Independent of "
        "the control-flow ``stop_reason`` (HARDEN-07d.2). The default is "
        "``answered`` only for direct/backwards-compatible construction; the "
        "graph always derives the real value.",
    )
    token_usage_estimate: int
    request_id: str | None = None
    # Phase 12.2 run-trace observability (additive; set by the API layer)
    telemetry: dict[str, Any] | None = Field(
        default=None,
        description="Run summary from the Phase 07 telemetry fabric (latency, tokens, "
        "provider/model, call counts), when a trace was active for this request.",
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal degradations, e.g. a fallback plan was used.")
    # Phase 06 adaptive policy traceability (additive; empty when policy off)
    question_pattern: str | None = Field(
        default=None, description="Question pattern selected by the Phase 06 policy router, if enabled."
    )
    stop_condition: str | None = Field(
        default=None, description="Which Phase 06 stop condition fired, if any (V2 §5.4)."
    )
    stop_decisions: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-condition stop evaluations from the last policy stop check."
    )
    evidence_tasks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Targeted retrieval actions formulated by active evidence seeking.",
    )
    # Phase 10 multi-agent traceability (additive; empty when multi-agent off)
    agent_round: int | None = Field(
        default=None, description="Number of debate rounds executed, if multi-agent was active."
    )
    agent_messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Messages exchanged between agents during debate."
    )
    disagreement_detected: bool | None = Field(
        default=None, description="Whether material disagreement was detected during debate."
    )
    # Phase 08 / knowledge-system traceability (additive; empty when memory off/not consulted)
    memory_consulted: list[str] = Field(
        default_factory=list,
        description="Memory layers (derived knowledge) consulted for this query, "
        "distinct from user-document evidence. Empty when no persistent memory "
        "influenced the plan.",
    )
    # Phase 39 / Phase 40 contradiction signals (additive; empty when none detected)
    contradiction_signals: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deterministic contradiction signals detected between evidence chunks.",
    )
    # Phase 07b selective verification traceability (additive; None when skipped/off)
    verification: OrchestrationVerification | None = Field(
        default=None,
        description="Selective claim verification metadata for this query, when the 07b "
        "verification stage fired. Verification annotates but never replaces the answer.",
    )
    # Per-node runtime traces for Brain UI (additive; empty when not recorded)
    node_traces: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-node orchestration trace objects recorded during this run. "
        "Each trace includes: node name, status, why (node purpose), latency_ms, "
        "evidence_count, iteration, tokens_used, plus node-specific fields copied "
        "from the node result (sufficient, stop_reason, question_pattern, "
        "stop_condition_fired, contradiction_signals). This is backend tracing "
        "for the Brain UI — not a claim about per-node input/confidence/"
        "alternatives, which are not currently captured.",
    )
    # Per-iteration research strategy history (additive; empty when not recorded)
    strategy_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ResearchStrategy snapshots taken at plan time and every "
        "assess iteration: retrieval mode, active queries, source priorities, "
        "verification depth, remaining budgets, and what mutated the strategy.",
    )


def sanitize_result_for_user(result: OrchestrationResult) -> OrchestrationResult:
    """Strip internal filesystem paths and sensitive metadata from a result.

    Replaces full source_path values with just the filename to prevent
    internal directory structure disclosure to API consumers.
    """
    sanitized_citations = []
    for c in result.citations:
        # Mask source_path to just the filename
        parts = c.source_path.replace("\\", "/").split("/")
        masked = parts[-1] if parts else c.source_path
        sanitized_citations.append(c.model_copy(update={"source_path": masked}))

    # Strip agent debate traces and internal policy details
    return result.model_copy(update={
        "citations": sanitized_citations,
        "agent_messages": [],  # Never expose inter-agent reasoning
        "stop_decisions": [],  # Never expose internal policy thresholds
        "evidence_tasks": [],  # Never expose gap-detection internals
    })
