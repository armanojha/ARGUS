"""Lightweight Corpus Entity & Concept Linking (Phase 20).

Extracts entities/concepts from corpus documents at ingestion time,
builds a persistent relationship index, and expands planner sub-queries
with relevant linked entities at query time.

Design principles:
- Lightweight: no LLM calls, no spaCy, no heavy NLP
- Deterministic: regex + RapidFuzz for entity extraction and matching
- Persistent: JSON index file alongside EvidenceStore
- Additive: entity expansion is additive to existing sub-queries
- Bounded: max 3 expansions per sub-query (configurable)
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

EntityType = str  # "PERSON" | "ORGANIZATION" | "LOCATION" | "TECHNOLOGY" | etc.


@dataclass
class CorpusEntity:
    """A canonical entity extracted from the corpus."""
    id: str                          # deterministic: normalized_name lowercase
    canonical_name: str              # e.g. "Atlas"
    entity_type: EntityType          # e.g. "TECHNOLOGY"
    aliases: list[str] = field(default_factory=list)  # ["Atlas database", "Atlas DB"]
    source_documents: list[str] = field(default_factory=list)  # document IDs
    source_chunks: list[str] = field(default_factory=list)     # chunk IDs
    mention_count: int = 0

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class EntityRelationship:
    """A relationship between two entities, grounded in corpus evidence."""
    source_entity: str               # canonical ID
    target_entity: str               # canonical ID
    relation_type: str               # "depends_on" | "related_to" | "part_of" | etc.
    evidence_chunks: list[str] = field(default_factory=list)  # chunk IDs
    confidence: float = 1.0

    def __hash__(self) -> int:
        return hash((self.source_entity, self.target_entity, self.relation_type))


@dataclass
class EntityExpansion:
    """A recommended entity expansion for a planner sub-query."""
    entity: CorpusEntity
    score: float                     # 0-1, how relevant this expansion is
    reason: str                      # "direct_mention" | "linked_to_query_entity" | etc.


@dataclass
class CorpusEntityIndex:
    """The full entity index for a corpus."""
    entities: dict[str, CorpusEntity] = field(default_factory=dict)
    relationships: list[EntityRelationship] = field(default_factory=list)
    # Inverted index: entity_name_lower -> entity_id
    _alias_index: dict[str, str] = field(default_factory=dict, repr=False)
    # Inverted index: chunk_id -> list of entity_ids
    _chunk_entities: dict[str, list[str]] = field(default_factory=dict, repr=False)
    version: int = 1
    created_at: str = ""
    source_checksum: str = ""

    def build_indices(self) -> None:
        """Build inverted indices for fast lookup."""
        self._alias_index.clear()
        self._chunk_entities.clear()
        for eid, entity in self.entities.items():
            # Index canonical name
            self._alias_index[entity.canonical_name.lower()] = eid
            # Index all aliases
            for alias in entity.aliases:
                self._alias_index[alias.lower()] = eid
            # Index chunk -> entity mapping
            for chunk_id in entity.source_chunks:
                self._chunk_entities.setdefault(chunk_id, []).append(eid)


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Match capitalized phrases (proper nouns): "Acme", "Atlas database engine"
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

# Match technical terms with mixed case: "Delta Sync", "BM25", "FAISS"
_MIXED_CASE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Match ALL-CAPS terms (acronyms): "SQL", "OCC", "CEO"
_ALL_CAPS_RE = re.compile(
    r"\b([A-Z]{2,})\b"
)

# Relationship indicator phrases
_RELATION_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"depends?\s+on", re.IGNORECASE), "depends_on"),
    (re.compile(r"integrates?\s+with", re.IGNORECASE), "integrates_with"),
    (re.compile(r"part\s+of", re.IGNORECASE), "part_of"),
    (re.compile(r"uses?\b", re.IGNORECASE), "uses"),
    (re.compile(r"supplied?\s+by", re.IGNORECASE), "supplied_by"),
    (re.compile(r"located?\s+in", re.IGNORECASE), "located_in"),
    (re.compile(r"replaces?\b", re.IGNORECASE), "replaces"),
    (re.compile(r"supports?\b", re.IGNORECASE), "supports"),
    (re.compile(r"contains?\b", re.IGNORECASE), "contains"),
    (re.compile(r"downstream\s+of", re.IGNORECASE), "downstream_of"),
    (re.compile(r"upstream\s+of", re.IGNORECASE), "upstream_of"),
]

# Stop words to skip as standalone entities
_ENTITY_STOP_WORDS = frozenset({
    "the", "this", "that", "these", "those", "what", "which", "when",
    "where", "how", "why", "who", "does", "will", "should", "could",
    "would", "can", "may", "might", "must", "shall", "not", "also",
    "than", "then", "into", "over", "into", "each", "some", "more",
    "most", "than", "other", "such", "only", "very", "just", "even",
    "still", "already", "here", "there", "now", "new", "old", "first",
    "last", "next", "same", "different", "both", "many", "much", "few",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million", "billion", "about", "after",
    "before", "during", "under", "above", "below", "between", "through",
    "across", "along", "around", "without", "within", "among", "against",
    "level", "type", "model", "version", "grade", "class", "kind",
    "section", "chapter", "part", "area", "region", "zone", "group",
})

# Entity type heuristics based on context
_LOCATION_WORDS = {
    "ohio", "monterrey", "mexico", "memphis", "new york", "california",
    "texas", "china", "japan", "germany", "france", "brazil", "india",
    "canada", "london", "paris", "tokyo", "beijing", "berlin", "madrid",
}

_TECHNOLOGY_WORDS = {
    "atlas", "delta sync", "incremental sync", "faiss", "bm25", "sql",
    "nosql", "database", "engine", "columnar", "ocr", "bge-m3",
    "llm", "gpt", "bert", "transformer", "embedding", "index",
    "pipeline", "ingestion", "chunking", "retrieval", "vector",
}

_ORGANIZATION_WORDS = {
    "acme", "petrokem", "helion", "darpa", "openai", "google",
    "microsoft", "amazon", "meta", "nvidia", "tesla", "apple",
}

_PRODUCT_WORDS = {
    "inspector", "robot", "robotics", "sealant", "adhesive",
    "fastener", "torque", "polaris", "probe",
}


def _classify_entity_type(name: str, context: str = "") -> EntityType:
    """Classify entity type using simple heuristics."""
    lower = name.lower()
    ctx_lower = context.lower()

    if lower in _LOCATION_WORDS or any(loc in lower for loc in _LOCATION_WORDS):
        return "LOCATION"
    if lower in _TECHNOLOGY_WORDS or any(tech in lower for tech in _TECHNOLOGY_WORDS):
        return "TECHNOLOGY"
    if lower in _ORGANIZATION_WORDS or any(org in lower for org in _ORGANIZATION_WORDS):
        return "ORGANIZATION"
    if lower in _PRODUCT_WORDS or any(prod in lower for prod in _PRODUCT_WORDS):
        return "PRODUCT"
    # Check context for type hints
    for tech in _TECHNOLOGY_WORDS:
        if tech in ctx_lower:
            return "TECHNOLOGY"
    for org in _ORGANIZATION_WORDS:
        if org in ctx_lower:
            return "ORGANIZATION"
    # Check if it looks like a person name (2 capitalized words, no org/tech context)
    words = name.split()
    if len(words) == 2 and all(w[0].isupper() and w[1:].islower() for w in words):
        if not any(w.lower() in _TECHNOLOGY_WORDS | _ORGANIZATION_WORDS for w in words):
            return "PERSON"
    return "OTHER"


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities_from_text(
    text: str,
    chunk_id: str = "",
    document_id: str = "",
) -> list[CorpusEntity]:
    """Extract entities from a text chunk using regex patterns.

    Returns a list of CorpusEntity objects with canonical names, types,
    and source tracking.
    """
    entities: dict[str, CorpusEntity] = {}

    # Extract proper nouns
    for match in _PROPER_NOUN_RE.finditer(text):
        name = match.group(1).strip()
        if name.lower() in _ENTITY_STOP_WORDS:
            continue
        if len(name) < 2:
            continue
        canonical = _normalize_entity_name(name)
        eid = canonical.lower()
        if eid in entities:
            entities[eid].mention_count += 1
            if chunk_id and chunk_id not in entities[eid].source_chunks:
                entities[eid].source_chunks.append(chunk_id)
            if document_id and document_id not in entities[eid].source_documents:
                entities[eid].source_documents.append(document_id)
        else:
            etype = _classify_entity_type(name, text)
            entities[eid] = CorpusEntity(
                id=eid,
                canonical_name=canonical,
                entity_type=etype,
                aliases=[name],
                source_documents=[document_id] if document_id else [],
                source_chunks=[chunk_id] if chunk_id else [],
                mention_count=1,
            )

    # Extract ALL-CAPS terms (acronyms)
    for match in _ALL_CAPS_RE.finditer(text):
        name = match.group(1).strip()
        if len(name) < 2 or name in ("THE", "AND", "FOR", "BUT", "NOT", "ARE", "WAS"):
            continue
        eid = name.lower()
        if eid not in entities:
            entities[eid] = CorpusEntity(
                id=eid,
                canonical_name=name,
                entity_type="OTHER",
                aliases=[],
                source_documents=[document_id] if document_id else [],
                source_chunks=[chunk_id] if chunk_id else [],
                mention_count=1,
            )

    return list(entities.values())


def extract_entities_from_chunks(
    chunks: list[dict[str, Any]],
) -> list[CorpusEntity]:
    """Extract entities from a list of chunk dicts.

    Each chunk dict should have: text, chunk_id, document_id.
    """
    all_entities: dict[str, CorpusEntity] = {}

    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        document_id = chunk.get("document_id", "")

        chunk_entities = extract_entities_from_text(text, chunk_id, document_id)
        for entity in chunk_entities:
            if entity.id in all_entities:
                existing = all_entities[entity.id]
                existing.mention_count += entity.mention_count
                for cid in entity.source_chunks:
                    if cid not in existing.source_chunks:
                        existing.source_chunks.append(cid)
                for did in entity.source_documents:
                    if did not in existing.source_documents:
                        existing.source_documents.append(did)
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
            else:
                all_entities[entity.id] = entity

    return list(all_entities.values())


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Common suffixes to strip for normalization
_STRIP_SUFFIXES = [
    " database", " db", " engine", " system", " platform",
    " line", " project", " feature", " tool", " api",
    " inc", " corp", " co", " ltd", " llc",
]

# Common prefix/suffix noise
_NOISE_RE = re.compile(r"^(the|a|an)\s+|\s+(inc|corp|co|ltd|llc)\.?$", re.IGNORECASE)


def _normalize_entity_name(name: str) -> str:
    """Normalize an entity name to its canonical form.

    Examples:
        "Atlas database engine" -> "Atlas"
        "Atlas DB" -> "Atlas"
        "Delta synchronization" -> "Delta Sync"
        "Acme Corp" -> "Acme"
    """
    result = name.strip()

    # Remove leading articles
    result = _NOISE_RE.sub("", result).strip()

    # Strip common suffixes
    for suffix in _STRIP_SUFFIXES:
        if result.lower().endswith(suffix):
            candidate = result[: -len(suffix)].strip()
            if len(candidate) >= 2:
                result = candidate
                break

    # Capitalize first letter of each word (preserve proper casing)
    result = " ".join(w.capitalize() if w.islower() else w for w in result.split())

    return result


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------

def extract_relationships_from_text(
    text: str,
    entities: list[CorpusEntity],
    chunk_id: str = "",
) -> list[EntityRelationship]:
    """Extract entity relationships from text based on dependency phrases.

    Looks for patterns like:
    - "X depends on Y" -> depends_on
    - "X integrates with Y" -> integrates_with
    """
    relationships: list[EntityRelationship] = []

    # Build a lookup of entity names present in this text
    text_lower = text.lower()
    present_entities = [
        e for e in entities
        if any(alias.lower() in text_lower for alias in [e.canonical_name] + e.aliases)
    ]

    if len(present_entities) < 2:
        return relationships

    # Look for relationship phrases between entity pairs
    for pattern, relation_type in _RELATION_PHRASES:
        for match in pattern.finditer(text):
            # Get the text around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()

            # Find entities in the before/after context
            before_text = text[start:match.start()].lower()
            after_text = text[match.end():end].lower()

            source_candidates = [
                e for e in present_entities
                if e.canonical_name.lower() in before_text
                or any(a.lower() in before_text for a in e.aliases)
            ]
            target_candidates = [
                e for e in present_entities
                if e.canonical_name.lower() in after_text
                or any(a.lower() in after_text for a in e.aliases)
            ]

            # If we found source but not target, check full context for remaining entities
            if source_candidates and not target_candidates:
                for e in present_entities:
                    if e not in source_candidates and (
                        e.canonical_name.lower() in context
                        or any(a.lower() in context for a in e.aliases)
                    ):
                        target_candidates.append(e)

            for source in source_candidates:
                for target in target_candidates:
                    if source.id != target.id:
                        rel = EntityRelationship(
                            source_entity=source.id,
                            target_entity=target.id,
                            relation_type=relation_type,
                            evidence_chunks=[chunk_id] if chunk_id else [],
                            confidence=0.9,
                        )
                        relationships.append(rel)

    return relationships


# ---------------------------------------------------------------------------
# Query-time entity detection
# ---------------------------------------------------------------------------

def detect_query_entities(
    query: str,
    index: CorpusEntityIndex,
    threshold: float = 0.6,
) -> list[tuple[CorpusEntity, float]]:
    """Detect known corpus entities in a user query.

    Uses fuzzy matching via RapidFuzz to handle variations.

    Returns list of (entity, match_score) tuples sorted by score descending.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback to simple substring matching
        return _detect_query_entities_simple(query, index)

    query_lower = query.lower()
    matches: list[tuple[CorpusEntity, float]] = []

    for eid, entity in index.entities.items():
        # Check canonical name
        score = fuzz.partial_ratio(entity.canonical_name.lower(), query_lower) / 100.0
        if score >= threshold:
            matches.append((entity, score))
            continue

        # Check aliases
        for alias in entity.aliases:
            alias_score = fuzz.partial_ratio(alias.lower(), query_lower) / 100.0
            if alias_score >= threshold:
                matches.append((entity, max(score, alias_score)))
                break

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def _detect_query_entities_simple(
    query: str,
    index: CorpusEntityIndex,
    threshold: float = 0.6,
) -> list[tuple[CorpusEntity, float]]:
    """Fallback entity detection without RapidFuzz."""
    query_lower = query.lower()
    matches: list[tuple[CorpusEntity, float]] = []

    for eid, entity in index.entities.items():
        # Simple substring check
        if entity.canonical_name.lower() in query_lower:
            matches.append((entity, 1.0))
            continue
        for alias in entity.aliases:
            if alias.lower() in query_lower:
                matches.append((entity, 0.9))
                break

    return matches


# ---------------------------------------------------------------------------
# Expansion generation
# ---------------------------------------------------------------------------

def generate_expansions(
    query: str,
    detected_entities: list[tuple[CorpusEntity, float]],
    index: CorpusEntityIndex,
    max_expansions: int = 3,
    min_score: float = 0.3,
) -> list[EntityExpansion]:
    """Generate entity expansions for a planner sub-query.

    Given detected entities in the query, find linked entities that
    should be added to improve retrieval.

    Args:
        query: The planner sub-query.
        detected_entities: Entities already detected in the query.
        index: The corpus entity index.
        max_expansions: Maximum number of expansions to return.
        min_score: Minimum expansion score to include.

    Returns:
        List of EntityExpansion sorted by score descending, bounded by max_expansions.
    """
    if not detected_entities:
        return []

    expansions: dict[str, EntityExpansion] = {}

    for entity, match_score in detected_entities:
        # Find entities linked to this entity via relationships
        for rel in index.relationships:
            linked_id = None
            direction = ""
            if rel.source_entity == entity.id:
                linked_id = rel.target_entity
                direction = "outgoing"
            elif rel.target_entity == entity.id:
                linked_id = rel.source_entity
                direction = "incoming"

            if linked_id and linked_id in index.entities:
                linked = index.entities[linked_id]
                # Don't expand with entities already in the query
                if linked.canonical_name.lower() in query.lower():
                    continue

                # Score based on: relationship confidence, entity frequency, match quality
                score = (
                    rel.confidence * 0.4
                    + min(1.0, linked.mention_count / 3.0) * 0.3
                    + match_score * 0.3
                )

                if linked_id not in expansions or score > expansions[linked_id].score:
                    expansions[linked_id] = EntityExpansion(
                        entity=linked,
                        score=score,
                        reason=f"linked_to_{entity.canonical_name}_via_{rel.relation_type}",
                    )

    # Sort by score and return top N
    sorted_expansions = sorted(expansions.values(), key=lambda e: e.score, reverse=True)
    return [e for e in sorted_expansions[:max_expansions] if e.score >= min_score]


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def save_entity_index(index: CorpusEntityIndex, path: Path) -> None:
    """Save the entity index to a JSON file."""
    import datetime
    index.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data = {
        "version": index.version,
        "created_at": index.created_at,
        "source_checksum": index.source_checksum,
        "entities": {eid: asdict(e) for eid, e in index.entities.items()},
        "relationships": [asdict(r) for r in index.relationships],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(
        "entity_index_saved",
        path=str(path),
        entities=len(index.entities),
        relationships=len(index.relationships),
    )


def load_entity_index(path: Path) -> CorpusEntityIndex | None:
    """Load the entity index from a JSON file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        index = CorpusEntityIndex(
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            source_checksum=data.get("source_checksum", ""),
        )
        for eid, edata in data.get("entities", {}).items():
            index.entities[eid] = CorpusEntity(**edata)
        for rdata in data.get("relationships", []):
            index.relationships.append(EntityRelationship(**rdata))
        index.build_indices()
        logger.info(
            "entity_index_loaded",
            path=str(path),
            entities=len(index.entities),
            relationships=len(index.relationships),
        )
        return index
    except Exception as exc:
        logger.warning("entity_index_load_failed", path=str(path), error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Full pipeline: build index from chunks
# ---------------------------------------------------------------------------

def build_entity_index(
    chunks: list[dict[str, Any]],
    source_checksum: str = "",
) -> CorpusEntityIndex:
    """Build a complete entity index from corpus chunks.

    Args:
        chunks: List of dicts with keys: text, chunk_id, document_id.
        source_checksum: Checksum of the source corpus for cache invalidation.

    Returns:
        CorpusEntityIndex with entities, relationships, and inverted indices.
    """
    start = time.perf_counter()

    # Step 1: Extract entities from all chunks
    entities = extract_entities_from_chunks(chunks)
    entity_map = {e.id: e for e in entities}

    # Step 2: Extract relationships
    all_relationships: list[EntityRelationship] = []
    entity_list = list(entity_map.values())
    seen_rels: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        rels = extract_relationships_from_text(text, entity_list, chunk_id)
        for rel in rels:
            key = (rel.source_entity, rel.target_entity, rel.relation_type)
            if key not in seen_rels:
                seen_rels.add(key)
                all_relationships.append(rel)
            else:
                # Merge evidence chunks into existing relationship
                for existing in all_relationships:
                    if (existing.source_entity == rel.source_entity
                            and existing.target_entity == rel.target_entity
                            and existing.relation_type == rel.relation_type):
                        for cid in rel.evidence_chunks:
                            if cid not in existing.evidence_chunks:
                                existing.evidence_chunks.append(cid)
                        break

    # Step 3: Build index
    index = CorpusEntityIndex(
        entities=entity_map,
        relationships=all_relationships,
        source_checksum=source_checksum,
    )
    index.build_indices()

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "entity_index_built",
        entities=len(entities),
        relationships=len(all_relationships),
        elapsed_ms=round(elapsed_ms, 1),
    )

    return index


# ---------------------------------------------------------------------------
# Main integration class
# ---------------------------------------------------------------------------

class EntityLinker:
    """Lightweight entity/concept linking layer for ARGUS retrieval.

    Provides:
    - Ingestion-time entity extraction and index building
    - Query-time entity detection and sub-query expansion
    - Persistent index storage
    """

    def __init__(
        self,
        index_path: Path | None = None,
        max_expansions: int = 3,
        min_expansion_score: float = 0.3,
        entity_detection_threshold: float = 0.8,
    ) -> None:
        self.index_path = index_path
        self.max_expansions = max_expansions
        self.min_expansion_score = min_expansion_score
        self.entity_detection_threshold = entity_detection_threshold
        self._index: CorpusEntityIndex | None = None

    @property
    def index(self) -> CorpusEntityIndex:
        if self._index is None:
            if self.index_path:
                self._index = load_entity_index(self.index_path)
            if self._index is None:
                self._index = CorpusEntityIndex()
        return self._index

    def build_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_checksum: str = "",
    ) -> CorpusEntityIndex:
        """Build the entity index from corpus chunks."""
        self._index = build_entity_index(chunks, source_checksum)
        if self.index_path:
            save_entity_index(self._index, self.index_path)
        return self._index

    def load(self) -> bool:
        """Load the entity index from disk."""
        if self.index_path:
            self._index = load_entity_index(self.index_path)
            return self._index is not None
        return False

    def save(self) -> None:
        """Save the entity index to disk."""
        if self.index_path and self._index:
            save_entity_index(self._index, self.index_path)

    def expand_subquery(
        self,
        subquery: str,
        pattern: str = "",
    ) -> str:
        """Expand a planner sub-query with relevant entity links.

        This is the main query-time entry point. Takes a sub-query
        from the planner and returns an expanded version with linked
        entities appended.

        Args:
            subquery: The original planner sub-query.
            pattern: The query pattern (for protection logic).

        Returns:
            Expanded sub-query string. May be identical to input if no
            expansions are found.
        """
        if not self._index or not self._index.entities:
            return subquery

        start = time.perf_counter()

        # Detect entities in the sub-query (not the original query)
        detected = detect_query_entities(
            subquery, self._index, self.entity_detection_threshold
        )

        if not detected:
            return subquery

        # Generate expansions only for entities detected in THIS sub-query
        expansions = generate_expansions(
            subquery, detected, self._index,
            max_expansions=self.max_expansions,
            min_score=self.min_expansion_score,
        )

        if not expansions:
            return subquery

        # Only add expansions that are topically relevant
        # Filter out expansions that don't share context with the sub-query
        subquery_lower = subquery.lower()
        filtered_expansions = []
        for exp in expansions:
            # Check if the expansion entity is topically related to the sub-query
            # by verifying it shares at least one co-occurring entity with the sub-query
            entity_id = exp.entity.id
            # Check if any entity in the sub-query has a relationship with this expansion
            has_relationship = False
            for rel in self._index.relationships:
                if (rel.source_entity in {e.id for e, _ in detected}
                        and rel.target_entity == entity_id):
                    has_relationship = True
                    break
                if (rel.target_entity in {e.id for e, _ in detected}
                        and rel.source_entity == entity_id):
                    has_relationship = True
                    break
            if has_relationship:
                filtered_expansions.append(exp)

        if not filtered_expansions:
            return subquery

        # Build expanded query: original terms + expansion entity names
        expansion_terms = [e.entity.canonical_name for e in filtered_expansions[:self.max_expansions]]

        expanded = subquery + " " + " ".join(expansion_terms)

        elapsed_us = (time.perf_counter() - start) * 1_000_000
        logger.debug(
            "entity_expansion",
            original=subquery,
            expanded=expanded,
            detected=len(detected),
            expansions=len(filtered_expansions),
            elapsed_us=round(elapsed_us, 1),
        )

        return expanded

    def expand_evidence_needs(
        self,
        evidence_needs: list[Any],  # EvidenceNeed list
        pattern: str = "",
    ) -> list[Any]:
        """Generate expansion search variants for evidence needs.

        Instead of modifying existing sub-queries (which can degrade retrieval),
        this method generates NEW search variants with entity expansions and
        adds them to the evidence needs as additional search targets.

        The original sub-queries are preserved unchanged.
        Returns the same list for chaining.
        """
        if not self._index or not self._index.entities:
            return evidence_needs

        for need in evidence_needs:
            if not hasattr(need, 'search_query') or not need.search_query:
                continue

            # Detect entities in this specific sub-query
            detected = detect_query_entities(
                need.search_query, self._index, self.entity_detection_threshold
            )

            if not detected:
                continue

            # Generate expansions
            expansions = generate_expansions(
                need.search_query, detected, self._index,
                max_expansions=self.max_expansions,
                min_score=self.min_expansion_score,
            )

            if not expansions:
                continue

            # Only add expansions that have direct relationships with detected entities
            subquery_lower = need.search_query.lower()
            direct_expansions = []
            for exp in expansions:
                for entity, _ in detected:
                    for rel in self._index.relationships:
                        if ((rel.source_entity == entity.id and rel.target_entity == exp.entity.id)
                                or (rel.target_entity == entity.id and rel.source_entity == exp.entity.id)):
                            direct_expansions.append(exp)
                            break
                    if len(direct_expansions) >= self.max_expansions:
                        break
                if len(direct_expansions) >= self.max_expansions:
                    break

            if not direct_expansions:
                continue

            # Create expansion terms string
            expansion_terms = " ".join(e.entity.canonical_name for e in direct_expansions[:self.max_expansions])

            # Store the expansion as a separate attribute on the need
            # The planner integration will use this to run an additional search
            if not hasattr(need, '_entity_expansion'):
                need._entity_expansion = expansion_terms

        return evidence_needs
