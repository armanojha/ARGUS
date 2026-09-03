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
    # Phase 07b selective verification traceability (additive; None when skipped/off)
    verification: OrchestrationVerification | None = Field(
        default=None,
        description="Selective claim verification metadata for this query, when the 07b "
        "verification stage fired. Verification annotates but never replaces the answer.",
    )
