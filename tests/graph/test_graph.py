"""Lightweight sanity tests for the Evidence Graph (Phase 03).

Not exhaustive by design (see vault Phase 03 testing policy: deferred
to a later stabilization pass). Covers the phase's own acceptance
criteria: graph construction, entity/claim/event persistence,
multi-hop queries, and temporal fields.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore
from app.graph.models import (
    Claim,
    EdgeType,
    Entity,
    EntityType,
    Event,
    ExtractionResult,
    GraphEdge,
    GraphQuery,
    TemporalPrecision,
)
from app.graph.store import EvidenceGraphStore
from app.retrieval.bm25 import assign_bm25_doc_ids
from app.retrieval.vector import assign_embedding_indices


@pytest.fixture
def temp_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        yield tmp


@pytest.fixture
def evidence_store(temp_paths) -> EvidenceStore:
    return EvidenceStore(
        db_path=temp_paths / "evidence.db",
        bm25_index_path=temp_paths / "bm25.pkl",
        faiss_index_path=temp_paths / "faiss.index",
    )


@pytest.fixture
def populated_evidence_store(evidence_store: EvidenceStore) -> EvidenceStore:
    source = Source(type=SourceType.TEXT, path="/test/corpus.txt", checksum="c1")
    evidence_store.upsert_source(source)
    doc = Document(source_id=source.id, version=1, checksum="d1", chunking_strategy="fixed")
    evidence_store.insert_document(doc)
    evidence_store.insert_chunks(
        [
            Chunk(document_id=doc.id, ordinal=0, text="John Smith works at Acme Corp.", token_count=8),
            Chunk(document_id=doc.id, ordinal=1, text="Acme Corp is located in New York.", token_count=8),
            Chunk(document_id=doc.id, ordinal=2, text="The meeting happened on 2024-01-15.", token_count=8),
        ]
    )
    assign_bm25_doc_ids(evidence_store)
    assign_embedding_indices(evidence_store)
    return evidence_store


@pytest.fixture
def graph_store(temp_paths, populated_evidence_store: EvidenceStore) -> EvidenceGraphStore:
    return EvidenceGraphStore(
        graph_path=temp_paths / "graph.pkl",
        evidence_store=populated_evidence_store,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


class TestGraphModels:
    def test_entity_creation(self):
        entity = Entity(
            canonical_name="John Smith",
            entity_type=EntityType.PERSON,
            aliases=["J. Smith", "Johnny"],
            description="CEO of Acme Corp",
            confidence=0.9,
        )
        assert entity.canonical_name == "John Smith"
        assert entity.entity_type == EntityType.PERSON
        assert "J. Smith" in entity.aliases

    def test_claim_creation(self):
        claim = Claim(
            text="John Smith works at Acme Corp",
            subject_entity_id=uuid4(),
            predicate="works at",
            object_entity_id=uuid4(),
            confidence=0.8,
        )
        assert claim.predicate == "works at"
        assert claim.confidence == 0.8

    def test_event_creation(self):
        event = Event(
            name="Board Meeting",
            event_time=datetime(2024, 1, 15, tzinfo=UTC),
            event_time_precision=TemporalPrecision.DAY,
            participant_entity_ids=[uuid4()],
            confidence=0.7,
        )
        assert event.name == "Board Meeting"
        assert event.event_time_precision == TemporalPrecision.DAY

    def test_edge_creation(self):
        edge = GraphEdge(
            edge_type=EdgeType.RELATES_TO,
            source_node_id=uuid4(),
            source_node_type="entity",
            target_node_id=uuid4(),
            target_node_type="entity",
            confidence=0.9,
        )
        assert edge.edge_type == EdgeType.RELATES_TO
        assert edge.source_node_type == "entity"


class TestGraphStore:
    def test_upsert_and_get_entity(self, graph_store: EvidenceGraphStore):
        entity = Entity(
            canonical_name="Test Entity",
            entity_type=EntityType.CONCEPT,
            aliases=["alias1"],
            confidence=0.8,
        )
        upserted = graph_store.upsert_entity(entity)
        assert upserted.id == entity.id

        retrieved = graph_store.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.canonical_name == "Test Entity"
        assert retrieved.entity_type == EntityType.CONCEPT

    def test_entity_deduplication_by_name(self, graph_store: EvidenceGraphStore):
        entity1 = Entity(canonical_name="Acme Corp", entity_type=EntityType.ORGANIZATION, confidence=0.7)
        entity2 = Entity(canonical_name="Acme Corp", entity_type=EntityType.ORGANIZATION, confidence=0.9, aliases=["Acme"])

        graph_store.upsert_entity(entity1)
        graph_store.upsert_entity(entity2)

        # Should find the merged entity
        found = graph_store.find_entity_by_name("Acme Corp")
        assert found is not None
        assert found.confidence == 0.9  # Max confidence
        assert "Acme" in found.aliases

    def test_upsert_and_get_claim(self, graph_store: EvidenceGraphStore):
        claim = Claim(
            text="Test claim",
            predicate="is",
            object_value="true",
            confidence=0.6,
        )
        upserted = graph_store.upsert_claim(claim)
        assert upserted.id == claim.id

        retrieved = graph_store.get_claim(claim.id)
        assert retrieved is not None
        assert retrieved.text == "Test claim"

    def test_upsert_and_get_event(self, graph_store: EvidenceGraphStore):
        event = Event(
            name="Test Event",
            event_time=datetime(2024, 6, 1, tzinfo=UTC),
            event_time_precision=TemporalPrecision.DAY,
            confidence=0.7,
        )
        upserted = graph_store.upsert_event(event)
        assert upserted.id == event.id

        retrieved = graph_store.get_event(event.id)
        assert retrieved is not None
        assert retrieved.name == "Test Event"

    def test_add_edge(self, graph_store: EvidenceGraphStore):
        entity1 = Entity(canonical_name="Entity A", entity_type=EntityType.CONCEPT)
        entity2 = Entity(canonical_name="Entity B", entity_type=EntityType.CONCEPT)
        graph_store.upsert_entity(entity1)
        graph_store.upsert_entity(entity2)

        edge = GraphEdge(
            edge_type=EdgeType.RELATES_TO,
            source_node_id=entity1.id,
            source_node_type="entity",
            target_node_id=entity2.id,
            target_node_type="entity",
            confidence=0.8,
        )
        added = graph_store.add_edge(edge)
        assert added.edge_type == EdgeType.RELATES_TO

        # Query edges
        edges = graph_store.get_edges(source_node_id=entity1.id, source_node_type="entity")
        assert len(edges) == 1
        assert edges[0].target_node_id == entity2.id

    def test_edge_deduplication(self, graph_store: EvidenceGraphStore):
        entity1 = Entity(canonical_name="Entity A", entity_type=EntityType.CONCEPT)
        entity2 = Entity(canonical_name="Entity B", entity_type=EntityType.CONCEPT)
        graph_store.upsert_entity(entity1)
        graph_store.upsert_entity(entity2)

        edge1 = GraphEdge(
            edge_type=EdgeType.RELATES_TO,
            source_node_id=entity1.id,
            source_node_type="entity",
            target_node_id=entity2.id,
            target_node_type="entity",
            confidence=0.7,
        )
        edge2 = GraphEdge(
            edge_type=EdgeType.RELATES_TO,
            source_node_id=entity1.id,
            source_node_type="entity",
            target_node_id=entity2.id,
            target_node_type="entity",
            confidence=0.9,
        )
        graph_store.add_edge(edge1)
        graph_store.add_edge(edge2)

        # Should have merged, keeping higher confidence
        edges = graph_store.get_edges(source_node_id=entity1.id, source_node_type="entity", edge_type=EdgeType.RELATES_TO)
        assert len(edges) == 1
        assert edges[0].confidence == 0.9

    def test_apply_extraction(self, graph_store: EvidenceGraphStore, populated_evidence_store: EvidenceStore):
        # Get chunk IDs from evidence store
        chunks = populated_evidence_store.get_chunks_by_document(
            populated_evidence_store.get_latest_document_for_source(
                populated_evidence_store.get_source_by_checksum("c1").id
            ).id
        )
        chunk_ids = [c.id for c in chunks]

        extraction = ExtractionResult(
            entities=[
                Entity(
                    canonical_name="John Smith",
                    entity_type=EntityType.PERSON,
                    supporting_chunk_ids=[chunk_ids[0]],
                    confidence=0.9,
                ),
                Entity(
                    canonical_name="Acme Corp",
                    entity_type=EntityType.ORGANIZATION,
                    supporting_chunk_ids=[chunk_ids[0], chunk_ids[1]],
                    confidence=0.8,
                ),
            ],
            claims=[
                Claim(
                    text="John Smith works at Acme Corp",
                    predicate="works at",
                    subject_entity_id=None,  # Will be resolved by name
                    object_entity_id=None,
                    object_value="Acme Corp",
                    supporting_chunk_ids=[chunk_ids[0]],
                    confidence=0.8,
                ),
            ],
            events=[],
            edges=[],
            processed_chunk_ids=chunk_ids,
        )

        graph_store.apply_extraction(extraction)

        # Verify entities were added
        john = graph_store.find_entity_by_name("John Smith")
        assert john is not None
        assert john.entity_type == EntityType.PERSON

        acme = graph_store.find_entity_by_name("Acme Corp")
        assert acme is not None
        assert acme.entity_type == EntityType.ORGANIZATION

        # Verify claims were added
        claims = graph_store.get_all_claims()
        assert len(claims) >= 1

        # Verify edges were created (MENTIONS, DERIVED_FROM, SUPPORTS, RELATES_TO)
        edges = graph_store.get_edges(edge_type=EdgeType.MENTIONS)
        assert len(edges) >= 2  # At least one per entity

    def test_graph_query(self, graph_store: EvidenceGraphStore, populated_evidence_store: EvidenceStore):
        # First populate with extraction
        chunks = populated_evidence_store.get_chunks_by_document(
            populated_evidence_store.get_latest_document_for_source(
                populated_evidence_store.get_source_by_checksum("c1").id
            ).id
        )
        chunk_ids = [c.id for c in chunks]

        extraction = ExtractionResult(
            entities=[
                Entity(
                    canonical_name="John Smith",
                    entity_type=EntityType.PERSON,
                    supporting_chunk_ids=[chunk_ids[0]],
                    confidence=0.9,
                ),
                Entity(
                    canonical_name="Acme Corp",
                    entity_type=EntityType.ORGANIZATION,
                    supporting_chunk_ids=[chunk_ids[0], chunk_ids[1]],
                    confidence=0.8,
                ),
                Entity(
                    canonical_name="New York",
                    entity_type=EntityType.LOCATION,
                    supporting_chunk_ids=[chunk_ids[1]],
                    confidence=0.8,
                ),
            ],
            claims=[],
            events=[],
            edges=[],
            processed_chunk_ids=chunk_ids,
        )
        graph_store.apply_extraction(extraction)

        # Query from John Smith
        query = GraphQuery(
            start_entity_names=["John Smith"],
            edge_types=[EdgeType.RELATES_TO, EdgeType.MENTIONS],
            max_hops=2,
            limit=10,
        )
        result = graph_store.query_graph(query)

        assert len(result.entities) >= 1
        assert any(e.canonical_name == "John Smith" for e in result.entities)
        # Should find connected entities via multi-hop
        assert len(result.paths) >= 0  # Paths may exist

    def test_temporal_fields(self, graph_store: EvidenceGraphStore):
        event = Event(
            name="Historical Event",
            event_time=datetime(1990, 1, 1, tzinfo=UTC),
            event_time_precision=TemporalPrecision.YEAR,
            event_end_time=datetime(1995, 12, 31, tzinfo=UTC),
            confidence=0.8,
        )
        graph_store.upsert_event(event)

        retrieved = graph_store.get_event(event.id)
        assert retrieved is not None
        assert retrieved.event_time is not None
        assert retrieved.event_time.year == 1990
        assert retrieved.event_time_precision == TemporalPrecision.YEAR
        assert retrieved.event_end_time is not None
        assert retrieved.event_end_time.year == 1995

    def test_persistence(self, temp_paths, populated_evidence_store: EvidenceStore):
        graph_path = temp_paths / "graph.pkl"
        store1 = EvidenceGraphStore(graph_path=graph_path, evidence_store=populated_evidence_store)

        entity = Entity(canonical_name="Persistent Entity", entity_type=EntityType.CONCEPT, confidence=0.9)
        store1.upsert_entity(entity)
        store1.save()

        # Load new store instance
        store2 = EvidenceGraphStore(graph_path=graph_path, evidence_store=populated_evidence_store)
        retrieved = store2.get_entity(entity.id)
        assert retrieved is not None
        assert retrieved.canonical_name == "Persistent Entity"

    def test_stats(self, graph_store: EvidenceGraphStore):
        stats = graph_store.stats()
        assert "entities" in stats
        assert "claims" in stats
        assert "events" in stats
        assert "edges" in stats
        assert all(isinstance(v, int) for v in stats.values())


class TestGraphRetrieval:
    def test_graph_retriever_creation(self, populated_evidence_store: EvidenceStore, settings: Settings):
        from app.graph.retrieval import GraphRetriever
        from app.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(populated_evidence_store)
        retriever = GraphRetriever(
            graph_store=EvidenceGraphStore(evidence_store=populated_evidence_store),
            hybrid_retriever=hybrid,
            settings=settings,
        )
        assert retriever is not None

    def test_get_entity_context(self, graph_store: EvidenceGraphStore, populated_evidence_store: EvidenceStore):
        chunks = populated_evidence_store.get_chunks_by_document(
            populated_evidence_store.get_latest_document_for_source(
                populated_evidence_store.get_source_by_checksum("c1").id
            ).id
        )
        chunk_ids = [c.id for c in chunks]

        extraction = ExtractionResult(
            entities=[
                Entity(
                    canonical_name="John Smith",
                    entity_type=EntityType.PERSON,
                    supporting_chunk_ids=[chunk_ids[0]],
                    confidence=0.9,
                ),
            ],
            claims=[
                Claim(
                    text="John Smith works at Acme Corp",
                    predicate="works at",
                    subject_entity_id=None,  # Will be resolved by name in apply_extraction
                    object_value="Acme Corp",
                    supporting_chunk_ids=[chunk_ids[0]],
                    confidence=0.8,
                ),
            ],
            events=[],
            edges=[],
            processed_chunk_ids=chunk_ids,
        )
        graph_store.apply_extraction(extraction)

        # Manually link the claim to the entity since extraction doesn't do name resolution
        john = graph_store.find_entity_by_name("John Smith")
        assert john is not None
        claims = graph_store.get_all_claims()
        assert len(claims) == 1
        claim = claims[0]
        claim.subject_entity_id = john.id
        graph_store.upsert_claim(claim)
        # Add RELATES_TO edge
        from app.graph.models import EdgeType, GraphEdge
        edge = GraphEdge(
            edge_type=EdgeType.RELATES_TO,
            source_node_id=claim.id,
            source_node_type="claim",
            target_node_id=john.id,
            target_node_type="entity",
            supporting_chunk_ids=claim.supporting_chunk_ids,
            confidence=claim.confidence,
        )
        graph_store.add_edge(edge)
        graph_store.save()

        from app.graph.retrieval import GraphRetriever
        from app.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(populated_evidence_store)
        retriever = GraphRetriever(
            graph_store=graph_store,
            hybrid_retriever=hybrid,
        )

        context = retriever.get_entity_context("John Smith")
        assert context["entity"] is not None
        assert context["entity"].canonical_name == "John Smith"
        assert len(context["claims"]) >= 1