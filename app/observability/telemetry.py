"""Telemetry/Observability Schema (Phase 07, 12).

Defines the telemetry data model for LLM calls, orchestration runs,
and system performance. Phase 07 implements the collection;
Phase 12 surfaces it in the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Telemetry Event Types
# =============================================================================

class TelemetryEventType(str, Enum):
    """Types of telemetry events."""
    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    ORCHESTRATION_STEP = "orchestration_step"
    VERIFICATION = "verification"
    GRAPH_OPERATION = "graph_operation"
    MEMORY_OPERATION = "memory_operation"
    INGESTION = "ingestion"
    ERROR = "error"


class LLMCallPhase(str, Enum):
    """Phases of an LLM call for detailed timing."""
    QUEUE = "queue"
    CONNECT = "connect"
    REQUEST = "request"
    RESPONSE = "response"
    PARSE = "parse"
    COMPLETE = "complete"


# =============================================================================
# LLM Call Telemetry (Phase 07)
# =============================================================================

class LLMCallTelemetry(BaseModel):
    """Telemetry for a single LLM call (Phase 07 §4.1, V3 §14)."""
    model_config = ConfigDict(extra="forbid")

    # Identity
    call_id: UUID = Field(default_factory=UUID)
    request_id: str | None = None
    # Call details
    provider: str
    model: str
    call_type: str
    # Timing (all in milliseconds)
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    # Phase timings
    phase_timings: dict[str, int] = field(default_factory=dict)
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Cost
    estimated_cost_usd: float | None = None
    # Outcome
    success: bool = True
    error: str | None = None
    finish_reason: str | None = None
    # Context
    temperature: float = 0.0
    max_tokens: int | None = None
    used_structured_output: bool = False
    used_tools: bool = False
    # Quota
    quota_remaining: int | None = None
    quota_reset_seconds: int | None = None


class LLMCallTelemetryBatch(BaseModel):
    """Batch of LLM call telemetry for aggregation."""
    model_config = ConfigDict(extra="forbid")

    calls: list[Any] = field(default_factory=list)  # list[LLMCallTelemetry]
    period_start: datetime
    period_end: datetime
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0


# =============================================================================
# Orchestration Run Telemetry (Phase 06, 07, 12)
# =============================================================================

class OrchestrationStepTelemetry(BaseModel):
    """Telemetry for a single orchestration step."""
    model_config = ConfigDict(extra="forbid")

    step_name: str  # analyze, plan, retrieve, assess, synthesize
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: int | None = None
    # LLM calls made during this step
    llm_calls: list[Any] = field(default_factory=list)  # list[LLMCallTelemetry]
    # Retrieval
    retrieval_count: int = 0
    retrieval_latency_ms: int = 0
    evidence_count: int = 0
    # Outcome
    success: bool = True
    error: str | None = None


class OrchestrationRunTelemetry(BaseModel):
    """Complete telemetry for an orchestration run."""
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    request_id: str | None = None
    query: str
    started_at: datetime
    ended_at: datetime | None = None
    total_duration_ms: int | None = None
    # Steps
    steps: list[Any] = field(default_factory=list)  # list[OrchestrationStepTelemetry]
    # Aggregated LLM calls
    all_llm_calls: list[Any] = field(default_factory=list)  # list[LLMCallTelemetry]
    # Aggregated metrics
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_retrievals: int = 0
    total_evidence: int = 0
    iterations: int = 0
    stop_reason: str | None = None
    # Outcome
    success: bool = True
    error: str | None = None
    final_answer_length: int = 0
    citation_count: int = 0


# =============================================================================
# Retrieval Telemetry
# =============================================================================

class RetrievalTelemetry(BaseModel):
    """Telemetry for a retrieval operation."""
    model_config = ConfigDict(extra="forbid")

    retrieval_id: UUID
    query: str
    pattern: str | None = None  # QuestionPattern
    methods: list[str] = field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    candidates_found: int = 0
    results_returned: int = 0
    reranked: bool = False
    rerank_latency_ms: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Verification Telemetry
# =============================================================================

class VerificationTelemetry(BaseModel):
    """Telemetry for a verification operation."""
    model_config = ConfigDict(extra="forbid")

    verification_id: UUID
    claim_id: str
    status: str
    confidence: float
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    evidence_count: int = 0
    contradictions_found: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Graph Operation Telemetry
# =============================================================================

class GraphOperationTelemetry(BaseModel):
    """Telemetry for graph operations."""
    model_config = ConfigDict(extra="forbid")

    operation_id: UUID
    operation_type: str  # upsert_entity, upsert_claim, add_edge, query
    entity_type: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    nodes_affected: int = 0
    edges_affected: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Memory Operation Telemetry
# =============================================================================

class MemoryOperationTelemetry(BaseModel):
    """Telemetry for memory operations."""
    model_config = ConfigDict(extra="forbid")

    operation_id: UUID
    operation_type: str  # store, retrieve, update, delete
    layer: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    records_affected: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Ingestion Telemetry
# =============================================================================

class IngestionTelemetry(BaseModel):
    """Telemetry for ingestion operations."""
    model_config = ConfigDict(extra="forbid")

    ingestion_id: UUID
    source_path: str
    content_type: str
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: int | None = None
    chunks_created: int = 0
    chunks_updated: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Error Telemetry
# =============================================================================

class ErrorTelemetry(BaseModel):
    """Telemetry for errors."""
    model_config = ConfigDict(extra="forbid")

    error_id: UUID
    error_type: str
    message: str
    component: str
    severity: str = "error"  # error, warning, info
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None


# =============================================================================
# Telemetry Collector Interface
# =============================================================================

class TelemetryCollectorInterface(ABC):
    """Interface for collecting telemetry.

    Phase 07 implements this. Phase 12 UI reads from it.
    """

    @abstractmethod
    async def record_llm_call(self, telemetry: Any) -> None:  # LLMCallTelemetry
        ...

    @abstractmethod
    async def record_orchestration_step(self, telemetry: Any) -> None:  # OrchestrationStepTelemetry
        ...

    @abstractmethod
    async def record_orchestration_run(self, telemetry: Any) -> None:  # OrchestrationRunTelemetry
        ...

    @abstractmethod
    async def record_retrieval(self, telemetry: Any) -> None:  # RetrievalTelemetry
        ...

    @abstractmethod
    async def record_verification(self, telemetry: Any) -> None:  # VerificationTelemetry
        ...

    @abstractmethod
    async def record_graph_operation(self, telemetry: Any) -> None:  # GraphOperationTelemetry
        ...

    @abstractmethod
    async def record_memory_operation(self, telemetry: Any) -> None:  # MemoryOperationTelemetry
        ...

    @abstractmethod
    async def record_ingestion(self, telemetry: Any) -> None:  # IngestionTelemetry
        ...

    @abstractmethod
    async def record_error(self, telemetry: Any) -> None:  # ErrorTelemetry
        ...

    @abstractmethod
    async def get_run_telemetry(self, run_id: UUID) -> Any | None:  # OrchestrationRunTelemetry
        ...

    @abstractmethod
    async def get_llm_call_stats(
        self,
        provider: str | None = None,
        model: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def get_orchestration_stats(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        ...


# =============================================================================
# Telemetry Collector Factory
# =============================================================================

class TelemetryCollectorFactoryInterface(ABC):
    @abstractmethod
    def create_collector(self) -> Any:  # TelemetryCollectorInterface
        ...


class DefaultTelemetryCollectorFactory:
    def create_collector(self) -> None:
        return None


_telemetry_factory: Any = None


def get_telemetry_factory() -> Any:
    global _telemetry_factory
    if _telemetry_factory is None:
        _telemetry_factory = DefaultTelemetryCollectorFactory()
    return _telemetry_factory


def set_telemetry_factory(factory: Any) -> None:
    global _telemetry_factory
    _telemetry_factory = factory


__all__ = [
    "DefaultTelemetryCollectorFactory",
    "ErrorTelemetry",
    "GraphOperationTelemetry",
    "IngestionTelemetry",
    "LLMCallPhase",
    "LLMCallTelemetry",
    "LLMCallTelemetryBatch",
    "MemoryOperationTelemetry",
    "OrchestrationRunTelemetry",
    "OrchestrationStepTelemetry",
    "RetrievalTelemetry",
    "TelemetryCollectorFactoryInterface",
    "TelemetryCollectorInterface",
    "TelemetryEventType",
    "VerificationTelemetry",
    "get_telemetry_factory",
    "set_telemetry_factory",
]