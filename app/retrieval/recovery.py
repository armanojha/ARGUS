"""Phase 16: Targeted Second-Stage Evidence Recovery.

Recovery activates ONLY when initial evidence set does not adequately cover
planned evidence requirements. Does NOT globally increase top-k.

Architecture:
    Initial planned retrieval
    → Evidence coverage analysis
    → Identify missing evidence need
    → Targeted recovery
    → Merge recovered candidates
    → Deduplicate
    → Final evidence selection
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.planner import EvidenceNeed, QueryPlan
    from app.evidence.models import EvidenceRef

logger = logging.getLogger(__name__)


class RecoveryType(Enum):
    """Types of evidence recovery."""
    DISCLAIMER = "disclaimer"         # Short note/chunk about conflicts
    CAVEAT = "caveat"                 # Exception/limitation chunk
    BRIDGE = "bridge"                 # Multi-hop bridge document
    PARENT_CONTEXT = "parent_context" # Expand from parent section
    RELATED_CLAIM = "related_claim"   # Related claim for conflict
    SECTION_EXPANSION = "section_expansion"  # Neighboring chunks
    NONE = "none"                     # No recovery needed


@dataclass(frozen=True)
class RecoveryBudget:
    """Budget constraints for recovery operations."""
    max_recovery_attempts: int = 3
    max_additional_candidates: int = 6
    max_graph_hops: int = 1
    max_additional_retrieval_calls: int = 2
    short_chunk_threshold: int = 30  # tokens


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    recovery_type: RecoveryType
    recovered_refs: list  # list[EvidenceRef]
    attempts: int = 0
    success: bool = False
    reason: str = ""


@dataclass
class CoverageAnalysis:
    """Analysis of evidence coverage for a query plan."""
    total_needs: int
    satisfied_needs: int
    unsatisfied_need_indices: list[int]
    unsatisfied_need_topics: list[str]
    coverage_ratio: float
    recovery_type: RecoveryType = RecoveryType.NONE
    missing_entity: str = ""


class TargetedRecovery:
    """Targeted second-stage evidence recovery.

    Activates only when initial evidence set does not adequately cover
    planned evidence requirements.
    """

    # Short disclaimer/note patterns
    _DISCLAIMER_PATTERNS = re.compile(
        r"\b(important\s+note|note:|disclaimer|warning|caution|"
        r"this\s+(document|report)\s+(does\s+not|contains|is)|"
        r"where\s+figures\s+conflict|the\s+\d+\s+report\s+is\s+authoritative|"
        r"legacy\s+report|superseded|not\s+audited|illustrative|"
        r"must\s+use\s+the\s+current|any\s+question\s+about)\b",
        re.IGNORECASE,
    )

    def __init__(self, budget: RecoveryBudget | None = None):
        self.budget = budget or RecoveryBudget()

    def analyze_coverage(
        self,
        plan: QueryPlan,
        retrieved_refs: list[EvidenceRef],
    ) -> CoverageAnalysis:
        """Analyze whether initial retrieval covered all evidence needs.

        Args:
            plan: The query plan with evidence needs
            retrieved_refs: Retrieved evidence references

        Returns:
            CoverageAnalysis with unsatisfied need indices
        """
        if not plan.is_planned or not plan.evidence_needs:
            return CoverageAnalysis(
                total_needs=0,
                satisfied_needs=0,
                unsatisfied_need_indices=[],
                unsatisfied_need_topics=[],
                coverage_ratio=1.0,
            )

        retrieved_texts = [r.text.lower() for r in retrieved_refs]
        retrieved_chunks = " ".join(retrieved_texts)

        satisfied = 0
        unsatisfied_indices = []
        unsatisfied_topics = []

        for i, need in enumerate(plan.evidence_needs):
            # Check if any retrieved chunk satisfies this need
            need_satisfied = self._check_need_satisfaction(need, retrieved_chunks, retrieved_refs)
            if need_satisfied:
                satisfied += 1
            else:
                unsatisfied_indices.append(i)
                unsatisfied_topics.append(need.topic)

        total = len(plan.evidence_needs)
        coverage_ratio = satisfied / total if total > 0 else 1.0

        # Determine recovery type
        recovery_type = self._determine_recovery_type(plan, unsatisfied_indices)

        return CoverageAnalysis(
            total_needs=total,
            satisfied_needs=satisfied,
            unsatisfied_need_indices=unsatisfied_indices,
            unsatisfied_need_topics=unsatisfied_topics,
            coverage_ratio=coverage_ratio,
            recovery_type=recovery_type,
        )

    def _check_need_satisfaction(
        self,
        need: EvidenceNeed,
        retrieved_chunks: str,
        retrieved_refs: list[EvidenceRef],
    ) -> bool:
        """Check if a specific evidence need is satisfied.

        For CAVEAT needs, we require that a chunk containing disclaimer/note
        patterns is actually retrieved (not just that the topic appears).
        For PRIMARY/CONTRADICTORY needs, we require that a chunk mentions
        both the topic AND a specific data point or entity.
        """
        # For CAVEAT needs, check if any retrieved chunk has disclaimer patterns
        if need.claim_type.value == "caveat":
            for ref in retrieved_refs:
                if self._DISCLAIMER_PATTERNS.search(ref.text):
                    return True
            return False

        # For PRIMARY needs, check if a chunk mentions the topic with specifics
        if need.claim_type.value == "primary":
            # Need a chunk that mentions the topic AND has specific data
            for ref in retrieved_refs:
                ref_text = ref.text.lower()
                topic_words = need.topic.lower().split()
                topic_match = any(w in ref_text for w in topic_words if len(w) > 3)
                entity_match = any(e.lower() in ref_text for e in need.entities)
                # Check for specific data indicators (numbers, years, percentages)
                has_data = bool(re.search(r'\d{4}|\d+%|\$[\d,]+|\d+\.\d+', ref.text))
                if topic_match and entity_match and has_data:
                    return True
            return False

        # For CONTRADICTORY needs, check for conflicting claims
        if need.claim_type.value == "contradictory":
            conflict_indicators = re.compile(
                r"\b(different|conflict|contradict|however|although|"
                r"but|legacy|superseded|alternative|disagree)\b",
                re.IGNORECASE,
            )
            for ref in retrieved_refs:
                if conflict_indicators.search(ref.text):
                    ref_text = ref.text.lower()
                    topic_words = need.topic.lower().split()
                    if any(w in ref_text for w in topic_words if len(w) > 3):
                        return True
            return False

        # For SUPPORTING needs, topic mention is sufficient
        topic_words = need.topic.lower().split()
        topic_satisfied = any(w in retrieved_chunks for w in topic_words if len(w) > 3)

        # Check by entity
        entity_satisfied = False
        for entity in need.entities:
            if entity.lower() in retrieved_chunks:
                entity_satisfied = True
                break

        return topic_satisfied or entity_satisfied

    def _determine_recovery_type(
        self,
        plan: QueryPlan,
        unsatisfied_indices: list[int],
    ) -> RecoveryType:
        """Determine recovery type based on unsatisfied needs."""
        if not unsatisfied_indices:
            return RecoveryType.NONE

        # Check what types of needs are unsatisfied
        need_types = [plan.evidence_needs[i].claim_type.value for i in unsatisfied_indices]

        if plan.pattern == "conflict":
            if "caveat" in need_types:
                return RecoveryType.DISCLAIMER
            if "contradictory" in need_types:
                return RecoveryType.RELATED_CLAIM
            return RecoveryType.PARENT_CONTEXT

        if plan.pattern == "multi_hop":
            return RecoveryType.BRIDGE

        if plan.pattern == "complex_research":
            return RecoveryType.SECTION_EXPANSION

        return RecoveryType.PARENT_CONTEXT

    async def recover(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> list[EvidenceRef]:
        """Execute targeted recovery for unsatisfied evidence needs.

        Args:
            plan: The query plan
            analysis: Coverage analysis results
            initial_refs: Initial retrieval results
            retriever: Hybrid retriever instance
            store: Evidence store

        Returns:
            List of recovered evidence references
        """
        if analysis.coverage_ratio >= 1.0:
            return []

        recovery_type = analysis.recovery_type
        recovered = []
        attempts = 0

        if recovery_type == RecoveryType.DISCLAIMER:
            recovered, attempts = await self._recover_disclaimers(
                plan, analysis, initial_refs, retriever, store,
            )
        elif recovery_type == RecoveryType.RELATED_CLAIM:
            recovered, attempts = await self._recover_related_claims(
                plan, analysis, initial_refs, retriever, store,
            )
        elif recovery_type == RecoveryType.BRIDGE:
            recovered, attempts = await self._recover_bridge_documents(
                plan, analysis, initial_refs, retriever, store,
            )
        elif recovery_type == RecoveryType.SECTION_EXPANSION:
            recovered, attempts = await self._recover_section_expansion(
                plan, analysis, initial_refs, retriever, store,
            )
        elif recovery_type == RecoveryType.PARENT_CONTEXT:
            recovered, attempts = await self._recover_parent_context(
                plan, analysis, initial_refs, retriever, store,
            )

        # Apply budget limits
        if len(recovered) > self.budget.max_additional_candidates:
            recovered = recovered[:self.budget.max_additional_candidates]

        logger.info(
            "recovery_completed",
            recovery_type=recovery_type.value,
            attempts=attempts,
            recovered_count=len(recovered),
            coverage_after=min(1.0, analysis.coverage_ratio + len(recovered) * 0.1),
        )

        return recovered

    async def _recover_disclaimers(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> tuple[list, int]:
        """Recover short disclaimer/note chunks for conflict queries."""
        recovered = []
        attempts = 0

        # Strategy 1: Search for disclaimer-like terms
        for idx in analysis.unsatisfied_need_indices:
            if attempts >= self.budget.max_additional_retrieval_calls:
                break

            need = plan.evidence_needs[idx]
            # Generate disclaimer-specific search queries
            disclaimer_queries = [
                f"{need.topic} disclaimer note",
                f"{need.topic} authoritative report",
                f"{need.topic} legacy superseded",
                f"{need.topic} figures conflict",
            ]

            for dq in disclaimer_queries[:2]:  # Limit attempts
                if attempts >= self.budget.max_additional_retrieval_calls:
                    break

                refs = retriever.search(dq, top_k=5)
                attempts += 1

                for ref in refs:
                    if self._DISCLAIMER_PATTERNS.search(ref.text):
                        # Check if not already in initial results
                        initial_ids = {str(r.chunk_id) for r in initial_refs}
                        if str(ref.chunk_id) not in initial_ids:
                            recovered.append(ref)
                            break

        # Strategy 2: Search for parent sections of retrieved chunks
        if not recovered and initial_refs:
            recovered, attempts2 = await self._expand_to_parent_sections(
                initial_refs, store,
            )
            attempts += attempts2

        return recovered, attempts

    async def _recover_related_claims(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> tuple[list, int]:
        """Recover related contradictory evidence for conflict queries."""
        recovered = []
        attempts = 0

        for idx in analysis.unsatisfied_need_indices:
            if attempts >= self.budget.max_additional_retrieval_calls:
                break

            need = plan.evidence_needs[idx]
            # Search for contradictory evidence with different framing
            conflict_queries = [
                f"{need.topic} disagree contradict",
                f"{need.topic} however although",
                f"{need.topic} different source alternative",
            ]

            for cq in conflict_queries[:2]:
                if attempts >= self.budget.max_additional_retrieval_calls:
                    break

                refs = retriever.search(cq, top_k=5)
                attempts += 1

                initial_ids = {str(r.chunk_id) for r in initial_refs}
                for ref in refs:
                    if str(ref.chunk_id) not in initial_ids:
                        recovered.append(ref)
                        if len(recovered) >= self.budget.max_additional_candidates:
                            break

        return recovered, attempts

    async def _recover_bridge_documents(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> tuple[list, int]:
        """Recover bridge documents for multi-hop queries.

        Multi-hop: A → B → C where A/B was found but B/C was missed.
        Strategy: Identify intermediate entity B, use B as second-hop anchor.
        """
        recovered = []
        attempts = 0

        for idx in analysis.unsatisfied_need_indices:
            if attempts >= self.budget.max_additional_retrieval_calls:
                break

            need = plan.evidence_needs[idx]

            # Extract intermediate entities from retrieved text
            retrieved_text = " ".join(r.text for r in initial_refs)
            intermediate_entities = self._extract_intermediate_entities(
                need, retrieved_text,
            )

            for entity in intermediate_entities:
                if attempts >= self.budget.max_additional_retrieval_calls:
                    break

                # Use intermediate entity as second-hop anchor
                hop_query = f"{entity} {need.topic}"
                refs = retriever.search(hop_query, top_k=5)
                attempts += 1

                initial_ids = {str(r.chunk_id) for r in initial_refs}
                for ref in refs:
                    if str(ref.chunk_id) not in initial_ids:
                        recovered.append(ref)
                        if len(recovered) >= self.budget.max_additional_candidates:
                            break

                if recovered:
                    break

        return recovered, attempts

    async def _recover_section_expansion(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> tuple[list, int]:
        """Recover missing evidence by expanding to related sections."""
        recovered = []
        attempts = 0

        # For complex research, expand to neighboring chunks of hits
        hit_refs = [r for r in initial_refs if r.score > 0.5]

        for ref in hit_refs[:3]:  # Limit expansion
            if attempts >= self.budget.max_additional_retrieval_calls:
                break
            if len(recovered) >= self.budget.max_additional_candidates:
                break

            # Search for chunks in same document
            doc_query = f"document:{ref.document_id}"
            refs = retriever.search(ref.text[:100], top_k=5)
            attempts += 1

            initial_ids = {str(r.chunk_id) for r in initial_refs}
            for r in refs:
                if str(r.chunk_id) not in initial_ids and r.document_id == ref.document_id:
                    recovered.append(r)

        return recovered, attempts

    async def _recover_parent_context(
        self,
        plan: QueryPlan,
        analysis: CoverageAnalysis,
        initial_refs: list[EvidenceRef],
        retriever: HybridRetriever,
        store: Any,
    ) -> tuple[list, int]:
        """Recover by expanding to parent sections."""
        recovered = []
        attempts = 0

        # Find low-score candidates that might have relevant parents
        low_score_refs = [r for r in initial_refs if r.score < 0.3]

        for ref in low_score_refs[:2]:
            if attempts >= self.budget.max_additional_retrieval_calls:
                break

            # Get chunks from same document with higher scores
            doc_chunks = store.get_chunks_by_document(ref.document_id)
            if doc_chunks:
                # Find the "parent" chunk (ordinal before this one)
                parent_chunks = [
                    c for c in doc_chunks
                    if c.ordinal == ref.ordinal - 1 or c.ordinal == ref.ordinal + 1
                ]
                for pc in parent_chunks:
                    # Create ref from parent
                    parent_refs = store.get_evidence_refs([pc.id], [0.3])
                    if parent_refs:
                        initial_ids = {str(r.chunk_id) for r in initial_refs}
                        if str(pc.id) not in initial_ids:
                            recovered.append(parent_refs[0])
                            attempts += 1
                            break

        return recovered, attempts

    async def _expand_to_parent_sections(
        self,
        refs: list[EvidenceRef],
        store: Any,
    ) -> tuple[list, int]:
        """Expand retrieval to parent sections of retrieved chunks."""
        recovered = []
        attempts = 0

        for ref in refs[:3]:
            if len(recovered) >= self.budget.max_additional_candidates:
                break

            # Get all chunks from same document
            doc_chunks = store.get_chunks_by_document(ref.document_id)
            if not doc_chunks:
                continue

            # Find chunks that might be parents (section headers, etc.)
            for chunk in doc_chunks:
                if chunk.id == ref.chunk_id:
                    continue
                # Check if this chunk is a section header or parent
                if (chunk.text.startswith("#") or
                    chunk.text.startswith("##") or
                    "note" in chunk.text.lower() or
                    "disclaimer" in chunk.text.lower()):
                    parent_refs = store.get_evidence_refs([chunk.id], [0.2])
                    if parent_refs:
                        recovered.append(parent_refs[0])
                        attempts += 1
                        break

        return recovered, attempts

    def _extract_intermediate_entities(
        self,
        need: EvidenceNeed,
        retrieved_text: str,
    ) -> list[str]:
        """Extract intermediate entities from retrieved text for multi-hop.

        For query "A depends on C via B", identify B as the intermediate.
        """
        entities = []

        # Look for entity patterns in retrieved text
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        found_entities = entity_pattern.findall(retrieved_text)

        # Filter to entities that are in the need but not the original query entities
        need_entities_lower = {e.lower() for e in need.entities}
        for entity in found_entities:
            if entity.lower() not in need_entities_lower:
                entities.append(entity)

        # Return unique entities, limited to 3
        return list(dict.fromkeys(entities))[:3]
