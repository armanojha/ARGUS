"""Retrieval Policy Interface (Phase 06).

Defines the interface for the adaptive retrieval policy that Phase 06
implements. The policy maps question patterns to retrieval strategies
and controls active evidence seeking.

Phase 01/02 retrieval components depend on this interface.
Phase 06 implements it. Phase 07 may use it for routing decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar
from typing import Any

from pydantic import BaseModel, ConfigDict


class RetrievalMethod(str, Enum):
    """Retrieval methods available in the policy."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"
    GRAPH_BM25 = "graph_bm25"
    GRAPH_VECTOR = "graph_vector"
    WEB = "web"
    METADATA_FILTER = "metadata_filter"
    TEMPORAL = "temporal"
    # BGE-M3 experimental methods (A/B testing)
    BGE_M3_DENSE = "bge_m3_dense"
    BGE_M3_SPARSE = "bge_m3_sparse"
    BGE_M3_HYBRID = "bge_m3_hybrid"


class QuestionPattern(str, Enum):
    """Question patterns that map to retrieval strategies (V2 §5.2 / V3 §5).

    This is the canonical classification used by:
    - Production retrieval (router.py)
    - EvidenceNeedPlanner
    - Benchmark evaluation
    - Diagnostics
    - Telemetry

    Patterns marked [PLANNABLE] trigger the EvidenceNeedPlanner for
    multi-query retrieval with evidence need decomposition.
    """
    # Original patterns
    EXACT_TERM = "exact_term"                    # Exact term lookup
    CONCEPTUAL = "conceptual"                    # Conceptual/semantic
    ENTITY_RELATIONSHIP = "entity_relationship"  # Entity relationships
    HISTORICAL = "historical"                    # Historical/temporal
    LONG_REPORT = "long_report"                  # Long report/synthesis
    FRESH_MISSING = "fresh_missing"              # Fresh/missing evidence
    MULTIMODAL = "multimodal"                    # Multimodal (text+image/table)
    COMPARATIVE = "comparative"                  # Comparative
    CAUSAL = "causal"                            # Causal reasoning
    PROCEDURAL = "procedural"                    # How-to/procedural
    # Evaluation/complex patterns [PLANNABLE where noted]
    SIMPLE_LOOKUP = "simple_lookup"              # Simple fact lookup
    NORMAL_QA = "normal_qa"                      # Normal Q&A
    NUMERICAL = "numerical"                      # Numerical/quantitative
    TECHNICAL_EXPLANATION = "technical_explanation"  # Technical explanation
    MULTI_DOC_SYNTHESIS = "multi_doc_synthesis"  # Multi-document synthesis
    CONFLICT = "conflict"                        # [PLANNABLE] Conflicting evidence
    COMPLEX_RESEARCH = "complex_research"        # [PLANNABLE] Complex research
    MULTI_HOP = "multi_hop"                      # [PLANNABLE] Multi-hop reasoning
    ABSENT_INFO = "absent_info"                  # Information genuinely absent
    ADVERSARIAL = "adversarial"                  # Adversarial/misleading queries

    # Mapping from evaluation plan classes to canonical patterns
    # This ensures consistency between eval plan and production classification
    @classmethod
    def from_eval_class(cls, eval_class: str) -> "QuestionPattern":
        """Convert an evaluation plan class to the canonical pattern.

        This is the SINGLE ENTRY POINT for converting eval plan classes
        to canonical patterns. All benchmarks and diagnostics MUST use this.
        """
        mapping = {
            "simple_lookup": cls.SIMPLE_LOOKUP,
            "normal_qa": cls.NORMAL_QA,
            "numerical": cls.NUMERICAL,
            "technical_explanation": cls.TECHNICAL_EXPLANATION,
            "multi_doc_synthesis": cls.MULTI_DOC_SYNTHESIS,
            "conflict": cls.CONFLICT,
            "complex_research": cls.COMPLEX_RESEARCH,
            "multi_hop": cls.MULTI_HOP,
            "absent_info": cls.ABSENT_INFO,
            "adversarial": cls.ADVERSARIAL,
            # Legacy mappings
            "exact_term": cls.EXACT_TERM,
            "conceptual": cls.CONCEPTUAL,
            "entity_relationship": cls.ENTITY_RELATIONSHIP,
            "historical": cls.HISTORICAL,
            "long_report": cls.LONG_REPORT,
            "fresh_missing": cls.FRESH_MISSING,
            "multimodal": cls.MULTIMODAL,
            "comparative": cls.COMPARATIVE,
            "causal": cls.CAUSAL,
            "procedural": cls.PROCEDURAL,
        }
        return mapping.get(eval_class, cls.CONCEPTUAL)


@dataclass(frozen=True)
class RetrievalMix:
    """A retrieval strategy mix for a question pattern.

    Defines which retrieval methods to use and their weights.
    """
    methods: list[RetrievalMethod] = field(default_factory=lambda: [RetrievalMethod.HYBRID])
    weights: dict[RetrievalMethod, float] = field(default_factory=dict)
    # Method-specific parameters
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    graph_max_hops: int = 2
    temporal_window_days: int | None = None
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    # Budget
    max_results_per_method: int = 20
    rerank_top_k: int = 10


class RetrievalPolicyEntry(BaseModel):
    """A single entry in the retrieval policy table.

    Maps a question pattern to a retrieval mix.
    """
    model_config = ConfigDict(extra="forbid")

    pattern: QuestionPattern
    retrieval_mix: RetrievalMix
    # Optional: conditions when this entry applies
    required_entities: list[str] = field(default_factory=list)
    required_time_window: str | None = None
    min_confidence: float = 0.0
    # Priority for conflict resolution (higher = more specific)
    priority: int = 0


class RetrievalPolicy(BaseModel):
    """Complete retrieval policy configuration.

    Phase 06 implements the full policy table from V2 §5.2 / V3 §5.
    """
    model_config = ConfigDict(extra="forbid")

    entries: list[RetrievalPolicyEntry] = field(default_factory=list)
    default_mix: RetrievalMix = field(default_factory=lambda: RetrievalMix(
        methods=[RetrievalMethod.HYBRID],
        weights={RetrievalMethod.HYBRID: 1.0},
    ))

    def get_mix_for_pattern(self, pattern: QuestionPattern) -> RetrievalMix:
        """Get the retrieval mix for a question pattern."""
        matching = [e for e in self.entries if e.pattern == pattern]
        if not matching:
            return self.default_mix
        # Return highest priority match
        return max(matching, key=lambda e: e.priority).retrieval_mix


class RetrievalPolicyInterface(ABC):
    """Interface for the retrieval policy router.

    Phase 06 implements this. The orchestration loop and retrieval
    components use this interface to get retrieval strategies.
    """

    @abstractmethod
    def get_policy(self) -> RetrievalPolicy:
        """Get the current retrieval policy."""
        ...

    @abstractmethod
    def classify_question(self, query: str, context: dict[str, Any] | None = None) -> QuestionPattern:
        """Classify a question into a pattern.

        Uses LLM or heuristic classification.
        """
        ...

    @abstractmethod
    def get_retrieval_mix(self, pattern: QuestionPattern) -> RetrievalMix:
        """Get the retrieval mix for a classified pattern."""
        ...

    @abstractmethod
    async def execute_retrieval(
        self,
        query: str,
        pattern: QuestionPattern,
        retriever: Any,  # HybridRetriever
        top_k: int | None = None,
    ) -> list[Any]:  # list[EvidenceRef]
        """Execute retrieval using the policy for a pattern."""
        ...


class EvidenceGapDetectorInterface(ABC):
    """Interface for detecting evidence gaps and triggering re-retrieval.

    Phase 06 implements this. The orchestration loop uses this
    after the assess node to detect gaps and trigger re-retrieval.
    """

    @abstractmethod
    def detect_gaps(
        self,
        state: Any,  # OrchestrationState
        plan: Any,  # ResearchPlan
        evidence: list[Any],  # list[EvidenceRef]
    ) -> list[dict[str, Any]]:  # list of gap descriptions with suggested queries
        """Detect evidence gaps in the current state.

        Returns list of gaps with suggested follow-up queries.
        """
        ...

    @abstractmethod
    def should_re_retrieve(self, gaps: list[dict[str, Any]]) -> bool:
        """Determine if re-retrieval should be triggered."""
        ...


class RetrievalPolicyFactoryInterface(ABC):
    """Factory for creating retrieval policy components.

    Phase 06 implements this. The orchestration graph uses this
    to create policy components.
    """

    @abstractmethod
    def create_policy_router(self) -> Any:  # RetrievalPolicyInterface
        """Create the policy router."""
        ...

    @abstractmethod
    def create_gap_detector(self) -> Any:  # EvidenceGapDetectorInterface
        """Create the evidence gap detector."""
        ...


# Default policy matching V2 §5.2 / V3 §5 table
DEFAULT_RETRIEVAL_POLICY = RetrievalPolicy(
    entries=[
        RetrievalPolicyEntry(
            pattern=QuestionPattern.EXACT_TERM,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.BM25, RetrievalMethod.HYBRID],
                weights={RetrievalMethod.BM25: 0.7, RetrievalMethod.HYBRID: 0.3},
                bm25_weight=0.7,
                vector_weight=0.3,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.CONCEPTUAL,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.VECTOR, RetrievalMethod.HYBRID],
                weights={RetrievalMethod.VECTOR: 0.7, RetrievalMethod.HYBRID: 0.3},
                bm25_weight=0.3,
                vector_weight=0.7,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.ENTITY_RELATIONSHIP,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.GRAPH, RetrievalMethod.BM25],
                weights={RetrievalMethod.GRAPH: 0.6, RetrievalMethod.BM25: 0.4},
                graph_max_hops=2,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.HISTORICAL,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.TEMPORAL, RetrievalMethod.VECTOR, RetrievalMethod.HYBRID],
                weights={RetrievalMethod.TEMPORAL: 0.5, RetrievalMethod.VECTOR: 0.3, RetrievalMethod.HYBRID: 0.2},
                temporal_window_days=365,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.LONG_REPORT,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.VECTOR, RetrievalMethod.HYBRID, RetrievalMethod.GRAPH],
                weights={RetrievalMethod.VECTOR: 0.5, RetrievalMethod.HYBRID: 0.3, RetrievalMethod.GRAPH: 0.2},
                bm25_weight=0.3,
                vector_weight=0.7,
                graph_max_hops=3,
            ),
            priority=8,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.FRESH_MISSING,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.WEB, RetrievalMethod.HYBRID],
                weights={RetrievalMethod.WEB: 0.5, RetrievalMethod.HYBRID: 0.5},
            ),
            priority=8,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.MULTIMODAL,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.HYBRID, RetrievalMethod.VECTOR],
                weights={RetrievalMethod.HYBRID: 0.5, RetrievalMethod.VECTOR: 0.5},
            ),
            priority=5,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.COMPARATIVE,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.BM25, RetrievalMethod.VECTOR, RetrievalMethod.GRAPH],
                weights={RetrievalMethod.BM25: 0.4, RetrievalMethod.VECTOR: 0.4, RetrievalMethod.GRAPH: 0.2},
                bm25_weight=0.5,
                vector_weight=0.5,
                graph_max_hops=2,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.CAUSAL,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.VECTOR, RetrievalMethod.BM25],
                weights={RetrievalMethod.VECTOR: 0.6, RetrievalMethod.BM25: 0.4},
                bm25_weight=0.4,
                vector_weight=0.6,
            ),
            priority=10,
        ),
        RetrievalPolicyEntry(
            pattern=QuestionPattern.PROCEDURAL,
            retrieval_mix=RetrievalMix(
                methods=[RetrievalMethod.BM25, RetrievalMethod.VECTOR],
                weights={RetrievalMethod.BM25: 0.6, RetrievalMethod.VECTOR: 0.4},
                bm25_weight=0.6,
                vector_weight=0.4,
            ),
            priority=10,
        ),
    ],
)


def get_default_retrieval_policy() -> RetrievalPolicy:
    """Get the default retrieval policy (V2 §5.2 / V3 §5 table)."""
    return DEFAULT_RETRIEVAL_POLICY


__all__ = [
    "DEFAULT_RETRIEVAL_POLICY",
    "EvidenceGapDetectorInterface",
    "QuestionPattern",
    "RetrievalMethod",
    "RetrievalMix",
    "RetrievalPolicy",
    "RetrievalPolicyEntry",
    "RetrievalPolicyFactoryInterface",
    "RetrievalPolicyInterface",
    "get_default_retrieval_policy",
]