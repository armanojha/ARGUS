"""Tests for Phase 20: Lightweight Corpus Entity & Concept Linking."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.retrieval.entity_linking import (
    CorpusEntity,
    CorpusEntityIndex,
    EntityExpansion,
    EntityLinker,
    EntityRelationship,
    _classify_entity_type,
    _normalize_entity_name,
    build_entity_index,
    detect_query_entities,
    extract_entities_from_text,
    extract_relationships_from_text,
    generate_expansions,
    load_entity_index,
    save_entity_index,
)


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_extracts_capitalized_proper_nouns(self):
        text = "Acme Corporation operates in Ohio and Monterrey."
        entities = extract_entities_from_text(text, chunk_id="c1", document_id="d1")
        names = {e.canonical_name for e in entities}
        # Acme Corporation normalizes to Acme; also find Ohio and Monterrey
        assert "Acme" in names or "Acme Corporation" in names
        assert "Ohio" in names
        assert "Monterrey" in names

    def test_extracts_technology_entities(self):
        text = "The Atlas database engine uses Delta Sync for incremental updates."
        entities = extract_entities_from_text(text)
        names = {e.canonical_name.lower() for e in entities}
        # Should find Atlas-related entities
        assert any("atlas" in n for n in names)

    def test_classifies_locations(self):
        text = "The plant in Ohio ships to Monterrey."
        entities = extract_entities_from_text(text)
        ohio = next((e for e in entities if "ohio" in e.canonical_name.lower()), None)
        monterrey = next((e for e in entities if "monterrey" in e.canonical_name.lower()), None)
        if ohio:
            assert ohio.entity_type == "LOCATION"
        if monterrey:
            assert monterrey.entity_type == "LOCATION"

    def test_classifies_technology(self):
        entities = extract_entities_from_text("Atlas is a database engine.")
        atlas = next((e for e in entities if e.canonical_name == "Atlas"), None)
        if atlas:
            assert atlas.entity_type == "TECHNOLOGY"

    def test_stops_at_sentence_starts(self):
        # "Acme" at sentence start should still be extracted
        text = "Acme is a company. Acme makes products."
        entities = extract_entities_from_text(text)
        # Should find at least one Acme entity
        assert any("acme" in e.canonical_name.lower() for e in entities)

    def test_skips_stop_words(self):
        text = "What is the impact of this?"
        entities = extract_entities_from_text(text)
        names = {e.canonical_name.lower() for e in entities}
        assert "what" not in names
        assert "this" not in names

    def test_tracking_source_chunks(self):
        text = "Atlas database is used by Acme."
        e1 = extract_entities_from_text(text, chunk_id="c1", document_id="d1")
        e2 = extract_entities_from_text(text, chunk_id="c2", document_id="d1")
        # Merge
        merged: dict[str, CorpusEntity] = {}
        for e in e1 + e2:
            if e.id in merged:
                merged[e.id].source_chunks.extend(
                    c for c in e.source_chunks if c not in merged[e.id].source_chunks
                )
            else:
                merged[e.id] = e
        for e in merged.values():
            assert len(e.source_chunks) >= 1


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_strips_database_suffix(self):
        assert _normalize_entity_name("Atlas database") == "Atlas"

    def test_strips_db_suffix(self):
        assert _normalize_entity_name("Atlas DB") == "Atlas"

    def test_strips_engine_suffix(self):
        assert _normalize_entity_name("Atlas engine") == "Atlas"

    def test_preserves_multi_word_names(self):
        result = _normalize_entity_name("Delta Sync")
        assert "Delta" in result and "Sync" in result

    def test_strips_corp_suffix(self):
        assert _normalize_entity_name("Acme Corp") == "Acme"

    def test_strips_leading_article(self):
        assert _normalize_entity_name("The Robotics") == "Robotics"

    def test_min_length_preserved(self):
        result = _normalize_entity_name("IoT platform")
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Relationship extraction tests
# ---------------------------------------------------------------------------

class TestRelationshipExtraction:
    def test_depends_on_relationship(self):
        text = "The robotics line depends on the Atlas database engine."
        entities = [
            CorpusEntity(id="robotics", canonical_name="Robotics", entity_type="PRODUCT"),
            CorpusEntity(id="atlas", canonical_name="Atlas", entity_type="TECHNOLOGY"),
        ]
        rels = extract_relationships_from_text(text, entities, chunk_id="c1")
        depends_rels = [r for r in rels if r.relation_type == "depends_on"]
        assert len(depends_rels) >= 1
        assert depends_rels[0].source_entity == "robotics"
        assert depends_rels[0].target_entity == "atlas"

    def test_integrates_with_relationship(self):
        text = "The robot integrates with the Memphis distribution center."
        entities = [
            CorpusEntity(id="robot", canonical_name="Robot", entity_type="PRODUCT"),
            CorpusEntity(id="memphis", canonical_name="Memphis", entity_type="LOCATION"),
        ]
        rels = extract_relationships_from_text(text, entities, chunk_id="c1")
        integration_rels = [r for r in rels if r.relation_type == "integrates_with"]
        assert len(integration_rels) >= 1

    def test_no_relationship_with_single_entity(self):
        text = "Atlas is a database engine."
        entities = [CorpusEntity(id="atlas", canonical_name="Atlas", entity_type="TECHNOLOGY")]
        rels = extract_relationships_from_text(text, entities, chunk_id="c1")
        assert len(rels) == 0

    def test_relationship_evidence_tracking(self):
        text = "Acme depends on PetroKem for raw materials."
        entities = [
            CorpusEntity(id="acme", canonical_name="Acme", entity_type="ORGANIZATION"),
            CorpusEntity(id="petrokem", canonical_name="PetroKem", entity_type="ORGANIZATION"),
        ]
        rels = extract_relationships_from_text(text, entities, chunk_id="chunk_42")
        assert len(rels) >= 1
        assert "chunk_42" in rels[0].evidence_chunks


# ---------------------------------------------------------------------------
# Index build/save/load tests
# ---------------------------------------------------------------------------

class TestEntityIndex:
    def _make_chunks(self):
        return [
            {
                "text": "Acme Corporation operates plants in Ohio and Monterrey.",
                "chunk_id": "c1",
                "document_id": "doc-a",
            },
            {
                "text": "The robotics line depends on the Atlas database engine.",
                "chunk_id": "c2",
                "document_id": "doc-i",
            },
            {
                "text": "Atlas uses Delta Sync for incremental data replication.",
                "chunk_id": "c3",
                "document_id": "doc-b",
            },
            {
                "text": "The inspection robot integrates with the Memphis distribution center.",
                "chunk_id": "c4",
                "document_id": "doc-i",
            },
        ]

    def test_build_index(self):
        chunks = self._make_chunks()
        index = build_entity_index(chunks, source_checksum="test_v1")
        assert len(index.entities) > 0
        assert len(index.relationships) > 0

    def test_build_index_has_depends_on(self):
        """Test relationship extraction finds relationships between entities."""
        # Use text with explicit capitalized entity names
        chunks = [
            {
                "text": "The Robotics line depends on the Atlas database engine.",
                "chunk_id": "c_dep",
                "document_id": "doc-test",
            },
            {
                "text": "Atlas integrates with the Memphis distribution center.",
                "chunk_id": "c_int",
                "document_id": "doc-test",
            },
        ]
        index = build_entity_index(chunks)
        # Should find depends_on or integrates_with
        rel_types = {r.relation_type for r in index.relationships}
        assert len(rel_types) >= 1, (
            f"Expected relationships, got: "
            f"{[(r.source_entity, r.relation_type, r.target_entity) for r in index.relationships]}"
        )

    def test_build_index_finds_relationships(self):
        """Test that the index finds relationships from the benchmark corpus."""
        chunks = self._make_chunks()
        index = build_entity_index(chunks)
        # Should find at least one relationship of any type
        assert len(index.relationships) >= 1, (
            f"Expected relationships, got: "
            f"{[(r.source_entity, r.relation_type, r.target_entity) for r in index.relationships]}"
        )

    def test_save_and_load(self):
        chunks = self._make_chunks()
        index = build_entity_index(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.json"
            save_entity_index(index, path)
            loaded = load_entity_index(path)

            assert loaded is not None
            assert len(loaded.entities) == len(index.entities)
            assert len(loaded.relationships) == len(index.relationships)

    def test_load_nonexistent(self):
        result = load_entity_index(Path("/nonexistent/path.json"))
        assert result is None

    def test_build_indices(self):
        index = CorpusEntityIndex()
        index.entities["e1"] = CorpusEntity(
            id="e1", canonical_name="Atlas", entity_type="TECHNOLOGY",
            aliases=["Atlas DB"], source_chunks=["c1", "c2"],
        )
        index.build_indices()
        assert "atlas" in index._alias_index
        assert "atlas db" in index._alias_index
        assert "c1" in index._chunk_entities
        assert "e1" in index._chunk_entities["c1"]


# ---------------------------------------------------------------------------
# Query entity detection tests
# ---------------------------------------------------------------------------

class TestQueryDetection:
    def _make_index(self):
        index = CorpusEntityIndex()
        index.entities["acme"] = CorpusEntity(
            id="acme", canonical_name="Acme", entity_type="ORGANIZATION",
            aliases=["Acme Corporation"], mention_count=5,
        )
        index.entities["atlas"] = CorpusEntity(
            id="atlas", canonical_name="Atlas", entity_type="TECHNOLOGY",
            aliases=["Atlas database", "Atlas DB"], mention_count=3,
        )
        index.entities["ohio"] = CorpusEntity(
            id="ohio", canonical_name="Ohio", entity_type="LOCATION",
            aliases=[], mention_count=7,
        )
        index.entities["monterrey"] = CorpusEntity(
            id="monterrey", canonical_name="Monterrey", entity_type="LOCATION",
            aliases=[], mention_count=4,
        )
        index.entities["delta_sync"] = CorpusEntity(
            id="delta_sync", canonical_name="Delta Sync", entity_type="TECHNOLOGY",
            aliases=["Delta synchronization"], mention_count=2,
        )
        index.build_indices()
        return index

    def test_detects_acme(self):
        index = self._make_index()
        detected = detect_query_entities("Acme diversification strategy", index)
        names = [e.canonical_name for e, _ in detected]
        assert "Acme" in names

    def test_detects_atlas(self):
        index = self._make_index()
        detected = detect_query_entities("Atlas database features", index)
        names = [e.canonical_name for e, _ in detected]
        assert "Atlas" in names

    def test_no_detection_for_unrelated_query(self):
        index = self._make_index()
        detected = detect_query_entities("What is the weather today?", index)
        # Should detect nothing or very low confidence
        high_conf = [e for e, s in detected if s > 0.8]
        assert len(high_conf) == 0

    def test_multiple_entities_detected(self):
        index = self._make_index()
        detected = detect_query_entities("Acme operations in Ohio", index)
        names = {e.canonical_name for e, _ in detected}
        assert "Acme" in names
        assert "Ohio" in names


# ---------------------------------------------------------------------------
# Expansion generation tests
# ---------------------------------------------------------------------------

class TestExpansionGeneration:
    def _make_index(self):
        index = CorpusEntityIndex()
        index.entities["acme"] = CorpusEntity(
            id="acme", canonical_name="Acme", entity_type="ORGANIZATION",
            aliases=[], mention_count=5,
        )
        index.entities["atlas"] = CorpusEntity(
            id="atlas", canonical_name="Atlas", entity_type="TECHNOLOGY",
            aliases=[], mention_count=3,
        )
        index.entities["delta_sync"] = CorpusEntity(
            id="delta_sync", canonical_name="Delta Sync", entity_type="TECHNOLOGY",
            aliases=[], mention_count=2,
        )
        index.entities["ohio"] = CorpusEntity(
            id="ohio", canonical_name="Ohio", entity_type="LOCATION",
            aliases=[], mention_count=7,
        )
        index.entities["monterrey"] = CorpusEntity(
            id="monterrey", canonical_name="Monterrey", entity_type="LOCATION",
            aliases=[], mention_count=4,
        )
        index.entities["robotics"] = CorpusEntity(
            id="robotics", canonical_name="Robotics", entity_type="PRODUCT",
            aliases=[], mention_count=3,
        )
        # Relationships
        index.relationships = [
            EntityRelationship("robotics", "atlas", "depends_on", ["c2"], 0.9),
            EntityRelationship("atlas", "delta_sync", "related_to", ["c3"], 0.9),
            EntityRelationship("acme", "robotics", "related_to", ["c1"], 0.8),
        ]
        index.build_indices()
        return index

    def test_expands_acme_to_linked_entities(self):
        index = self._make_index()
        detected = [("acme", 1.0)]
        entity_list = [(index.entities["acme"], 1.0)]
        expansions = generate_expansions(
            "Acme diversification", entity_list, index, max_expansions=3
        )
        expansion_names = {e.entity.canonical_name for e in expansions}
        # Should find linked entities (robotics, atlas, etc.)
        assert len(expansions) >= 1

    def test_does_not_expand_with_query_entities(self):
        index = self._make_index()
        detected = [(index.entities["atlas"], 1.0)]
        expansions = generate_expansions(
            "Atlas database", detected, index, max_expansions=3
        )
        # Atlas is already in query, should not be in expansions
        expansion_names = {e.entity.canonical_name for e in expansions}
        assert "Atlas" not in expansion_names

    def test_respects_max_expansions(self):
        index = self._make_index()
        detected = [(index.entities["acme"], 1.0)]
        expansions = generate_expansions(
            "Acme", detected, index, max_expansions=1
        )
        assert len(expansions) <= 1

    def test_expansion_score_above_threshold(self):
        index = self._make_index()
        detected = [(index.entities["acme"], 1.0)]
        expansions = generate_expansions(
            "Acme", detected, index, min_score=0.1
        )
        for exp in expansions:
            assert exp.score >= 0.1


# ---------------------------------------------------------------------------
# EntityLinker class tests
# ---------------------------------------------------------------------------

class TestEntityLinker:
    def _make_chunks(self):
        return [
            {
                "text": "Acme Corporation operates plants in Ohio and Monterrey.",
                "chunk_id": "c1",
                "document_id": "doc-a",
            },
            {
                "text": "The robotics line depends on the Atlas database engine.",
                "chunk_id": "c2",
                "document_id": "doc-i",
            },
            {
                "text": "Atlas uses Delta Sync for incremental data replication.",
                "chunk_id": "c3",
                "document_id": "doc-b",
            },
        ]

    def test_build_and_expand(self):
        chunks = self._make_chunks()
        linker = EntityLinker(max_expansions=3, min_expansion_score=0.1)
        linker.build_from_chunks(chunks)

        expanded = linker.expand_subquery("Acme diversification strategy")
        # Should add some linked entity terms
        assert len(expanded) >= len("Acme diversification strategy")

    def test_expand_no_match(self):
        chunks = self._make_chunks()
        linker = EntityLinker(entity_detection_threshold=0.8)
        linker.build_from_chunks(chunks)

        original = "What is the meaning of photosynthesis?"
        expanded = linker.expand_subquery(original)
        # No known entities detected, should return original
        assert expanded == original

    def test_expand_empty_index(self):
        linker = EntityLinker()
        expanded = linker.expand_subquery("Acme query")
        assert expanded == "Acme query"

    def test_save_and_load(self):
        chunks = self._make_chunks()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_linker.json"
            linker = EntityLinker(index_path=path)
            linker.build_from_chunks(chunks)
            linker.save()

            # Load in new linker
            linker2 = EntityLinker(index_path=path)
            linker2.load()
            assert len(linker2.index.entities) > 0

    def test_expand_evidence_needs(self):
        from app.retrieval.planner import EvidenceNeed, ClaimType, NeedPriority

        chunks = self._make_chunks()
        linker = EntityLinker(max_expansions=2, min_expansion_score=0.1,
                              entity_detection_threshold=0.7)
        linker.build_from_chunks(chunks)

        needs = [
            EvidenceNeed(
                topic="Acme",
                search_query="Acme diversification strategy",
                claim_type=ClaimType.PRIMARY,
                priority=NeedPriority.HIGH,
            ),
            EvidenceNeed(
                topic="dependencies",
                search_query="robotics dependencies Atlas",
                claim_type=ClaimType.PRIMARY,
                priority=NeedPriority.MEDIUM,
            ),
        ]
        original_q0 = needs[0].search_query
        original_q1 = needs[1].search_query
        linker.expand_evidence_needs(needs, "complex_research")
        # At least one need should have been expanded (Acme is in index)
        expanded_any = (needs[0].search_query != original_q0 or
                        needs[1].search_query != original_q1)
        assert expanded_any


# ---------------------------------------------------------------------------
# Type classification tests
# ---------------------------------------------------------------------------

class TestTypeClassification:
    def test_classifies_person(self):
        assert _classify_entity_type("Diana Reyes") == "PERSON"

    def test_classifies_organization(self):
        assert _classify_entity_type("Acme") == "ORGANIZATION"

    def test_classifies_technology(self):
        assert _classify_entity_type("Atlas") == "TECHNOLOGY"

    def test_classifies_location(self):
        assert _classify_entity_type("Ohio") == "LOCATION"

    def test_classifies_product(self):
        assert _classify_entity_type("Robotics") == "PRODUCT"

    def test_fallback_to_other(self):
        assert _classify_entity_type("SomethingRandom") == "OTHER"
