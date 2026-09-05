"""Evidence Need Planner (Phase 15).

Decomposes complex queries into structured evidence requirements.
For COMPLEX_RESEARCH, MULTI_HOP, and CONFLICT patterns, generates
multiple targeted search queries instead of a single monolithic search.

Simple queries bypass this layer entirely.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)


class ClaimType(Enum):
    """Types of evidence claims."""
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    CONTRADICTORY = "contradictory"
    CONTEXTUAL = "contextual"
    CAVEAT = "caveat"


class NeedPriority(Enum):
    """Evidence need priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EvidenceNeed:
    """A structured evidence requirement extracted from a complex query."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    entities: list[str] = field(default_factory=list)
    claim_type: ClaimType = ClaimType.PRIMARY
    search_query: str = ""
    priority: NeedPriority = NeedPriority.HIGH
    requires_opposing_evidence: bool = False
    source_constraints: list[str] = field(default_factory=list)
    original_need: str = ""  # Human-readable description of what's needed

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvidenceNeed):
            return NotImplemented
        return self.id == other.id


@dataclass
class QueryPlan:
    """A plan consisting of one or more evidence needs for a query."""
    original_query: str
    pattern: str
    evidence_needs: list[EvidenceNeed] = field(default_factory=list)
    search_variants: list[str] = field(default_factory=list)
    is_planned: bool = False  # True if planner generated needs; False = pass-through

    @property
    def need_count(self) -> int:
        return len(self.evidence_needs)

    @property
    def query_count(self) -> int:
        """Total number of searches to run (needs + variants)."""
        return max(self.need_count, len(self.search_variants), 1)


class EvidenceNeedPlanner:
    """Decomposes complex queries into structured evidence needs.

    For simple patterns (SIMPLE_LOOKUP, NORMAL_QA, NUMERICAL, etc.),
    returns an unplanned pass-through QueryPlan.

    For complex patterns (CONFLICT, COMPLEX_RESEARCH, MULTI_HOP),
    generates targeted EvidenceNeed objects with specific search queries.
    """

    # Patterns that benefit from evidence need planning
    # Uses QuestionPattern enum values for canonical classification
    PLANNABLE_PATTERNS = {
        "conflict", "complex_research", "multi_hop",  # String values
        # Note: Also accepts QuestionPattern.CONFLICT.value, etc.
    }

    # Conflict indicator keywords
    _CONFLICT_INDICATORS = re.compile(
        r"\b(compare|vs|versus|difference|disagree|conflict|contradict|"
        r"contrary|however|but|although|despite|on the other hand|"
        r"legacy.*superseded|superseded.*current|authoritative)\b",
        re.IGNORECASE,
    )

    # Entity extraction: capitalized words (including camelCase like PetroKem)
    _ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b")

    # Topic extraction: key nouns from query
    _TOPIC_STOP_WORDS = frozenset({
        "what", "how", "why", "when", "where", "who", "which", "is", "are",
        "was", "were", "does", "do", "did", "has", "have", "had", "can",
        "could", "should", "would", "will", "the", "a", "an", "and", "or",
        "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "this", "that", "these", "those", "it", "its", "their", "they",
        "we", "our", "you", "your", "he", "she", "his", "her", "my",
        "about", "into", "through", "during", "before", "after", "between",
        "under", "above", "over", "more", "most", "than", "then", "also",
        "not", "no", "any", "each", "every", "all", "both", "few", "many",
        "much", "some", "such", "only", "other", "very", "just", "if",
        "so", "too", "here", "there", "now", "still", "even", "well",
        "back", "being", "get", "got", "make", "made", "take", "took",
        "come", "came", "go", "went", "see", "saw", "know", "knew",
        "think", "thought", "say", "said", "tell", "told", "give", "gave",
        "use", "used", "find", "found", "want", "want", "need", "needed",
        "the", "it", "and", "for", "that", "with", "you", "this", "but",
        "from", "they", "have", "been", "one", "were", "which", "when",
        "their", "will", "way", "about", "many", "then", "them", "would",
        "like", "than", "each", "those", "its", "how", "just", "his",
        "her", "are", "our", "out", "what", "some", "could", "other",
        "into", "more", "time", "very", "when", "come", "made", "after",
        "also", "did", "any", "only", "new", "year", "old", "great",
    })

    def plan(self, query: str, pattern: str) -> QueryPlan:
        """Create a query plan based on the classified pattern.

        Args:
            query: The original user query.
            pattern: The classified QuestionPattern value.

        Returns:
            QueryPlan with evidence needs (if complex) or pass-through (if simple).
        """
        if pattern not in self.PLANNABLE_PATTERNS:
            return QueryPlan(
                original_query=query,
                pattern=pattern,
                is_planned=False,
            )

        if pattern == "conflict":
            return self._plan_conflict(query)
        elif pattern == "complex_research":
            return self._plan_complex_research(query)
        elif pattern == "multi_hop":
            return self._plan_multi_hop(query)

        return QueryPlan(original_query=query, pattern=pattern, is_planned=False)

    def _plan_conflict(self, query: str) -> QueryPlan:
        """Plan conflict queries: need primary claim + contradictory evidence.

        Conflict queries require balanced evidence from multiple sources.
        We generate search variants targeting different aspects of the conflict.
        """
        entities = self._extract_entities(query)
        topic = self._extract_topic(query)

        needs = []

        # Primary claim retrieval
        needs.append(EvidenceNeed(
            topic=topic,
            entities=entities,
            claim_type=ClaimType.PRIMARY,
            search_query=query,
            priority=NeedPriority.HIGH,
            requires_opposing_evidence=False,
            original_need=f"Primary evidence about {topic}",
        ))

        # Search for the specific data points mentioned
        # E.g., "What was Acme's revenue?" -> also search for "Acme revenue 2025"
        data_query = self._generate_data_query(query, entities)
        if data_query and data_query != query:
            needs.append(EvidenceNeed(
                topic=topic,
                entities=entities,
                claim_type=ClaimType.SUPPORTING,
                search_query=data_query,
                priority=NeedPriority.HIGH,
                original_need=f"Supporting data for {topic}",
            ))

        # Search for conflicting/contradictory evidence
        conflict_query = self._generate_conflict_query(query, entities, topic)
        if conflict_query:
            needs.append(EvidenceNeed(
                topic=topic,
                entities=entities,
                claim_type=ClaimType.CONTRADICTORY,
                search_query=conflict_query,
                priority=NeedPriority.HIGH,
                requires_opposing_evidence=True,
                original_need=f"Conflicting evidence about {topic}",
            ))

        # Search for caveats/exceptions/limitations
        caveat_query = self._generate_caveat_query(query, entities, topic)
        if caveat_query:
            needs.append(EvidenceNeed(
                topic=topic,
                entities=entities,
                claim_type=ClaimType.CAVEAT,
                search_query=caveat_query,
                priority=NeedPriority.MEDIUM,
                original_need=f"Caveats or exceptions about {topic}",
            ))

        return QueryPlan(
            original_query=query,
            pattern="conflict",
            evidence_needs=needs,
            is_planned=True,
        )

    def _plan_complex_research(self, query: str) -> QueryPlan:
        """Plan complex research: decompose multi-topic queries.

        Complex research queries often contain multiple unrelated information
        needs joined by "and", "evaluate", "assess", etc.
        """
        topics = self._decompose_topics(query)
        entities = self._extract_entities(query)
        needs = []

        for i, subtopic in enumerate(topics):
            # Generate a focused search query for each subtopic
            sub_query = self._focus_query(query, subtopic, entities)
            priority = NeedPriority.HIGH if i == 0 else NeedPriority.MEDIUM

            needs.append(EvidenceNeed(
                topic=subtopic,
                entities=entities,
                claim_type=ClaimType.PRIMARY,
                search_query=sub_query,
                priority=priority,
                original_need=f"Evidence about {subtopic}",
            ))

        # If we couldn't decompose, fall back to the original query
        if not needs:
            needs.append(EvidenceNeed(
                topic=self._extract_topic(query),
                entities=entities,
                claim_type=ClaimType.PRIMARY,
                search_query=query,
                priority=NeedPriority.HIGH,
                original_need="Full query evidence",
            ))

        # Generate the original query as an additional search variant
        search_variants = [query]
        for need in needs:
            if need.search_query != query:
                search_variants.append(need.search_query)

        return QueryPlan(
            original_query=query,
            pattern="complex_research",
            evidence_needs=needs,
            search_variants=search_variants,
            is_planned=True,
        )

    def _plan_multi_hop(self, query: str) -> QueryPlan:
        """Plan multi-hop queries: decompose into intermediate steps.

        Multi-hop queries require connecting information across documents.
        We extract the key entities and relationships to search for.

        Phase 18: improved subquery generation to trace relationship chains.
        """
        entities = self._extract_entities(query)
        topic = self._extract_topic(query)
        needs = []

        # Primary query
        needs.append(EvidenceNeed(
            topic=topic,
            entities=entities,
            claim_type=ClaimType.PRIMARY,
            search_query=query,
            priority=NeedPriority.HIGH,
            original_need=f"Direct evidence for {topic}",
        ))

        # Phase 18: Generate relationship-tracing subqueries
        # For "X depends on Y via Z" patterns, search for each hop
        if len(entities) >= 2:
            # Generate entity-pair relationship queries
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1:]:
                    # Direct relationship
                    rel_query = f"{entity_a} {entity_b}"
                    needs.append(EvidenceNeed(
                        topic=f"{entity_a} - {entity_b}",
                        entities=[entity_a, entity_b],
                        claim_type=ClaimType.CONTEXTUAL,
                        search_query=rel_query,
                        priority=NeedPriority.MEDIUM,
                        original_need=f"Relationship between {entity_a} and {entity_b}",
                    ))

                    # Phase 18: Also search for intermediate hops
                    # "X supplies Y" or "X feeds Y" or "X depends on Y"
                    for verb in ["supplies", "feeds", "depends on", "connects to", "produces for"]:
                        hop_query = f"{entity_a} {verb} {entity_b}"
                        needs.append(EvidenceNeed(
                            topic=f"{entity_a} {verb} {entity_b}",
                            entities=[entity_a, entity_b],
                            claim_type=ClaimType.CONTEXTUAL,
                            search_query=hop_query,
                            priority=NeedPriority.MEDIUM,
                            original_need=f"{entity_a} {verb} {entity_b}",
                        ))

        # Also search for each entity individually with the topic
        for entity in entities:
            entity_query = f"{entity} {topic}"
            needs.append(EvidenceNeed(
                topic=entity,
                entities=[entity],
                claim_type=ClaimType.SUPPORTING,
                search_query=entity_query,
                priority=NeedPriority.MEDIUM,
                original_need=f"Information about {entity}",
            ))

        search_variants = [query] + [n.search_query for n in needs if n.search_query != query]

        return QueryPlan(
            original_query=query,
            pattern="multi_hop",
            evidence_needs=needs,
            search_variants=search_variants,
            is_planned=True,
        )

    # --- Query generation helpers ---

    def _generate_data_query(self, query: str, entities: list[str]) -> str:
        """Generate a query targeting specific data points."""
        # If the query asks for a specific metric, add the year/context
        if entities:
            entity_str = " ".join(entities)
            # Check if there's already a year in the query
            if not re.search(r"\b(20\d{2})\b", query):
                return f"{entity_str} latest data figures"
        return ""

    def _generate_conflict_query(self, query: str, entities: list[str], topic: str) -> str:
        """Generate a query targeting conflicting evidence."""
        # Try to find what the query is comparing
        if entities:
            entity_str = " ".join(entities)
            # Search for alternative/disagreeing sources
            return f"{entity_str} {topic} discrepancy alternative source"
        return ""

    def _generate_caveat_query(self, query: str, entities: list[str], topic: str) -> str:
        """Generate a query targeting caveats and exceptions."""
        if entities:
            entity_str = " ".join(entities)
            return f"{entity_str} {topic} caveat exception limitation note"
        return ""

    def _decompose_topics(self, query: str) -> list[str]:
        """Decompose a complex query into subtopics.

        Uses structural cues like "and evaluate", "assess...and",
        semicolons, and clause boundaries.
        """
        topics = []

        # Split on common multi-topic connectors
        # Phase 18: relaxed terminators - allow end-of-string without punctuation
        connectors = [
            r"\band\s+(?:evaluate|assess|explain|analyze|consider|discuss|review|describe|compare)\b",  # "and evaluate..."
            r"\band\s+(?:what|how|why|which|where|who|when)\b",  # "and what/ how..."
            r"\band\s+(?:it|its|this|that|the)\b",  # "and it depends on..."
            r"(?<=\.)\s+(?:Also|Additionally|Furthermore|Moreover)\b",
            r";\s*",
            r"(?<=\w)\s+while\s+",
            r"(?<=\w)\s+whereas\s+",
            r"(?<=\w)\s+but\s+(?:also|additionally)\b",
        ]

        parts = [query]
        for connector in connectors:
            new_parts = []
            for part in parts:
                new_parts.extend(re.split(connector, part, flags=re.IGNORECASE))
            parts = new_parts

        for part in parts:
            part = part.strip().strip(".")
            if len(part) > 10:  # Skip very short fragments
                topic = self._extract_topic(part)
                if topic and len(topic) > 3:
                    topics.append(topic)

        # If decomposition produced only one topic, try phrase-level extraction
        if len(topics) <= 1:
            # Look for "X and Y" patterns within the query
            compound = re.findall(
                r"(\w+(?:\s+\w+){0,3})\s+and\s+(\w+(?:\s+\w+){0,3})",
                query, re.IGNORECASE
            )
            if compound:
                topics = []
                for a, b in compound:
                    a_clean = a.strip()
                    b_clean = b.strip()
                    if len(a_clean) > 3:
                        topics.append(a_clean)
                    if len(b_clean) > 3:
                        topics.append(b_clean)

        return topics if topics else [self._extract_topic(query)]

    def _focus_query(self, original: str, subtopic: str, entities: list[str]) -> str:
        """Create a focused search query for a specific subtopic."""
        # Combine entities with the subtopic for a focused query
        parts = []
        if entities:
            parts.append(" ".join(entities[:2]))  # Limit to top 2 entities
        parts.append(subtopic)
        return " ".join(parts)

    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities from the query."""
        # Find capitalized words (potential entities)
        raw = self._ENTITY_PATTERN.findall(query)

        # Filter out common false positives
        sentence_starts = set()
        for match in re.finditer(r"(?:^|[.!?]\s+)(\w+)", query):
            sentence_starts.add(match.group(1))

        entities = []
        seen = set()
        for entity in raw:
            if entity in seen or entity in sentence_starts:
                continue
            if entity.lower() in self._TOPIC_STOP_WORDS:
                continue
            seen.add(entity)
            entities.append(entity)

        return entities[:5]  # Limit entity count

    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        words = query.split()
        topic_words = []
        for word in words:
            clean = re.sub(r"[^a-zA-Z0-9]", "", word).lower()
            if clean and clean not in self._TOPIC_STOP_WORDS and len(clean) > 2:
                topic_words.append(clean)

        return " ".join(topic_words[:6])  # Limit topic length
