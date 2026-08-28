"""Obsidian Integration exports (Phase 05 + Phase 09 full integration)."""

from app.integrations.obsidian.alignment import (
    GraphAlignmentResult,
    VaultGraphAligner,
    VaultMemoryCoordinator,
)
from app.integrations.obsidian.classifier import (
    RuleBasedHypothesisConverter,
    RuleBasedObsidianClassifier,
    extract_hypothesis_text,
)
from app.integrations.obsidian.contracts import (
    CLASSIFICATION_RULES,
    ClassificationResult,
    ClassificationRule,
    KnowledgeClass,
    NoteTreatmentRule,
)
from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline, ingest_obsidian_vault
from app.integrations.obsidian.models import (
    NoteType,
    ObsidianCallout,
    ObsidianCodeBlock,
    ObsidianFrontmatter,
    ObsidianIngestionResult,
    ObsidianNoteRecord,
    ObsidianSection,
    ObsidianTag,
    ObsidianWikilink,
    ParsedObsidianNote,
    ResearchCaptureNote,
    SyncManifest,
)
from app.integrations.obsidian.parser import parse_obsidian_note
from app.integrations.obsidian.proposals import WriteBackProposalManager
from app.integrations.obsidian.research import (
    HypothesisResearchOutcome,
    HypothesisResearchRunner,
    ObsidianResearchCoordinator,
)
from app.integrations.obsidian.scanner import VaultScanner
from app.integrations.obsidian.sync import SyncManager
from app.integrations.obsidian.writer import ObsidianWriter

__all__ = [
    "CLASSIFICATION_RULES",
    "ClassificationResult",
    "ClassificationRule",
    "GraphAlignmentResult",
    "HypothesisResearchOutcome",
    "HypothesisResearchRunner",
    "KnowledgeClass",
    "NoteTreatmentRule",
    "NoteType",
    "ObsidianCallout",
    "ObsidianCodeBlock",
    "ObsidianFrontmatter",
    "ObsidianIngestionPipeline",
    "ObsidianIngestionResult",
    "ObsidianNoteRecord",
    "ObsidianResearchCoordinator",
    "ObsidianSection",
    "ObsidianTag",
    "ObsidianWikilink",
    "ObsidianWriter",
    "ParsedObsidianNote",
    "ResearchCaptureNote",
    "RuleBasedHypothesisConverter",
    "RuleBasedObsidianClassifier",
    "SyncManager",
    "SyncManifest",
    "VaultGraphAligner",
    "VaultMemoryCoordinator",
    "VaultScanner",
    "WriteBackProposalManager",
    "extract_hypothesis_text",
    "ingest_obsidian_vault",
    "parse_obsidian_note",
]
