"""Obsidian Classification and Writer Contracts (Phase 09).

Defines interfaces for the full Obsidian integration:
- 7-class knowledge taxonomy classifier
- Hypothesis-to-research-objective conversion
- Vault-graph alignment
- Safe write-back/proposal workflow

Phase 09 implements these. Phase 05 provides the base ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Phase 09.1: 7-Class Knowledge Taxonomy (V3 §4.2)
# =============================================================================

class KnowledgeClass(str, Enum):
    """7-class knowledge taxonomy (V3 §4.2).

    Each class has a specific ARGUS treatment rule.
    """
    SOURCE_NOTE = "source_note"
    KNOWLEDGE_NOTE = "knowledge_note"
    HYPOTHESIS = "hypothesis"
    PROJECT_NOTE = "project_note"
    TASK_QUESTION = "task_question"
    RESEARCH_CAPTURE = "research_capture"
    REFERENCE_INDEX = "reference_index"


class NoteTreatmentRule(str, Enum):
    """How ARGUS treats each knowledge class."""
    SOURCE_NOTE = "require_provenance"
    KNOWLEDGE_NOTE = "personalization_only"
    HYPOTHESIS = "drive_research"
    PROJECT_NOTE = "track_progress"
    TASK_QUESTION = "answer_question"
    RESEARCH_CAPTURE = "archive_output"
    REFERENCE_INDEX = "navigation_only"


@dataclass(frozen=True)
class ClassificationRule:
    """Rule for how a knowledge class is treated."""
    knowledge_class: KnowledgeClass
    treatment: NoteTreatmentRule
    requires_provenance: bool = False
    drives_research: bool = False
    is_personal: bool = True
    is_argus_generated: bool = False


# Default classification rules (V3 §4.2)
CLASSIFICATION_RULES: dict[KnowledgeClass, ClassificationRule] = {
    "source_note": ClassificationRule(
        knowledge_class="source_note",
        treatment="require_provenance",
        requires_provenance=True,
        drives_research=False,
        is_personal=False,
        is_argus_generated=False,
    ),
    "knowledge_note": ClassificationRule(
        knowledge_class="knowledge_note",
        treatment="personalization_only",
        requires_provenance=False,
        drives_research=False,
        is_personal=True,
        is_argus_generated=False,
    ),
    "hypothesis": ClassificationRule(
        knowledge_class="hypothesis",
        treatment="drive_research",
        requires_provenance=False,
        drives_research=True,
        is_personal=True,
        is_argus_generated=False,
    ),
    "project_note": ClassificationRule(
        knowledge_class="project_note",
        treatment="track_progress",
        requires_provenance=False,
        drives_research=False,
        is_personal=True,
        is_argus_generated=False,
    ),
    "task_question": ClassificationRule(
        knowledge_class="task_question",
        treatment="drive_research",
        requires_provenance=False,
        drives_research=True,
        is_personal=True,
        is_argus_generated=False,
    ),
    "research_capture": ClassificationRule(
        knowledge_class="research_capture",
        treatment="archive_output",
        requires_provenance=True,
        drives_research=False,
        is_personal=False,
        is_argus_generated=True,
    ),
    "reference_index": ClassificationRule(
        knowledge_class="reference_index",
        treatment="navigation_only",
        requires_provenance=False,
        drives_research=False,
        is_personal=True,
        is_argus_generated=False,
    ),
}


class ClassificationResult(BaseModel):
    """Result of classifying a note."""
    model_config = ConfigDict(extra="forbid")

    knowledge_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    treatment_rule: str
    features: dict[str, Any] = field(default_factory=dict)


class ObsidianClassifierInterface(ABC):
    """Interface for the 7-class Obsidian note classifier.

    Phase 09 implements this. The ingestion pipeline uses this
    to classify notes during ingestion.
    """

    @abstractmethod
    async def classify_note(
        self,
        note_path: str,
        content: str,
        frontmatter: dict[str, Any],
        sections: list[Any],
    ) -> ClassificationResult:
        """Classify a single note into one of 7 knowledge classes."""
        ...

    @abstractmethod
    def get_treatment_rule(self, knowledge_class: str) -> str:
        """Get the treatment rule for a knowledge class."""
        ...

    @abstractmethod
    def get_all_rules(self) -> dict[str, Any]:
        """Get all classification rules."""
        ...


# =============================================================================
# Phase 09.2: Hypothesis → Research Objective
# =============================================================================

class HypothesisResearchObjective(BaseModel):
    """A hypothesis converted to a research objective."""
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    hypothesis_text: str
    research_objective: str
    subquestions: list[str] = field(default_factory=list)
    suggested_patterns: list[str] = field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_note_path: str


class HypothesisConverterInterface(ABC):
    """Interface for converting hypotheses to research objectives.

    Phase 09 implements this. The orchestration loop uses this
    when a hypothesis-class note is detected.
    """

    @abstractmethod
    async def convert_hypothesis(
        self,
        hypothesis_text: str,
        note_path: str,
        context: dict[str, Any] | None = None,
    ) -> HypothesisResearchObjective:
        """Convert a hypothesis note to a research objective."""
        ...

    @abstractmethod
    def should_convert(self, note_class: str, frontmatter: dict[str, Any]) -> bool:
        """Determine if a note should be converted to a research objective."""
        ...


# =============================================================================
# Phase 09.4: Safe Write-Back Workflow
# =============================================================================

class WriteBackProposal(BaseModel):
    """A proposed change to a user's note (V3 §12).

    Default workflow: propose, don't mutate.
    """
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    target_note_path: str
    change_type: str
    proposed_content: str
    section_heading: str | None = None
    research_id: str | None = None
    evidence_citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WriteBackProposalInterface(ABC):
    """Interface for creating and managing write-back proposals.

    Phase 09 implements this. The writer creates proposals;
    the user reviews them.
    """

    @abstractmethod
    async def create_proposal(
        self,
        target_note_path: str,
        change_type: str,
        proposed_content: str,
        section_heading: str | None = None,
        research_id: str | None = None,
        evidence_citations: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
    ) -> WriteBackProposal:
        """Create a write-back proposal."""
        ...

    @abstractmethod
    async def get_proposal(self, proposal_id: str) -> WriteBackProposal | None:
        """Get a proposal by ID."""
        ...

    @abstractmethod
    async def list_proposals(
        self,
        status: str | None = None,
        note_path: str | None = None,
    ) -> list[WriteBackProposal]:
        """List proposals with optional filters."""
        ...

    @abstractmethod
    async def accept_proposal(self, proposal_id: str, reviewer: str = "user") -> bool:
        """Accept a proposal (apply the change)."""
        ...

    @abstractmethod
    async def reject_proposal(self, proposal_id: str, reviewer: str = "user") -> bool:
        """Reject a proposal."""
        ...


class ResearchCaptureWriterInterface(ABC):
    """Interface for writing research capture notes to 90_ARGUS/.

    Phase 05 provides basic writer; Phase 09 extends with full
    frontmatter and proposal workflow.
    """

    @abstractmethod
    def write_research_capture(
        self,
        research_id: str,
        title: str,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        confidence: float | None = None,
        caveats: list[str] | None = None,
        sources: list[str] | None = None,
        claims: list[str] | None = None,
    ) -> Path:
        """Write a research capture note to 90_ARGUS/Research_Output/."""
        ...

    @abstractmethod
    def write_evidence_report(
        self,
        research_id: str,
        title: str,
        evidence_summary: str,
        citations: list[dict[str, Any]],
    ) -> Path:
        """Write an evidence report to 90_ARGUS/Evidence_Reports/."""
        ...

    @abstractmethod
    def write_research_trace(
        self,
        research_id: str,
        trace_data: dict[str, Any],
    ) -> Path:
        """Write a research trace to 90_ARGUS/Research_Traces/."""
        ...

    @abstractmethod
    def write_sync_log(self, log_data: dict[str, Any]) -> Path:
        """Write a sync log to 90_ARGUS/Sync_Logs/."""
        ...


# =============================================================================
# Obsidian Extension Factory
# =============================================================================

class ObsidianExtensionFactoryInterface(ABC):
    """Factory for creating Obsidian extension components.

    Phase 09 implements this.
    """

    @abstractmethod
    def create_classifier(self) -> Any | None:
        """Create Phase 09 classifier, or None."""
        ...

    @abstractmethod
    def create_hypothesis_converter(self) -> Any | None:
        """Create Phase 09 hypothesis converter, or None."""
        ...

    @abstractmethod
    def create_writeback_proposal(self) -> Any | None:
        """Create Phase 09 write-back proposal manager, or None."""
        ...

    @abstractmethod
    def create_research_writer(self) -> Any | None:
        """Create Phase 09 extended writer, or None."""
        ...


class DefaultObsidianExtensionFactory:
    """Default factory returning None (no Phase 09 extensions)."""

    def create_classifier(self) -> None:
        return None

    def create_hypothesis_converter(self) -> None:
        return None

    def create_writeback_proposal(self) -> None:
        return None

    def create_research_writer(self) -> None:
        return None


# Global factory instance
_obsidian_extension_factory: Any = None


def get_obsidian_extension_factory() -> Any:
    global _obsidian_extension_factory
    if _obsidian_extension_factory is None:
        _obsidian_extension_factory = DefaultObsidianExtensionFactory()
    return _obsidian_extension_factory


def set_obsidian_extension_factory(factory: Any) -> None:
    global _obsidian_extension_factory
    _obsidian_extension_factory = factory


__all__ = [
    "CLASSIFICATION_RULES",
    "ClassificationResult",
    "ClassificationRule",
    "DefaultObsidianExtensionFactory",
    "HypothesisConverterInterface",
    "HypothesisResearchObjective",
    "KnowledgeClass",
    "NoteTreatmentRule",
    "ObsidianClassifierInterface",
    "ObsidianExtensionFactoryInterface",
    "ResearchCaptureWriterInterface",
    "WriteBackProposal",
    "WriteBackProposalInterface",
    "get_obsidian_extension_factory",
    "set_obsidian_extension_factory",
]