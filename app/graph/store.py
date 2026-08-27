"""NetworkX-backed Evidence Graph Store (Phase 03).

Provides persistent graph storage using NetworkX with pickle serialization.
All graph facts remain traceable to Phase 01 evidence chunks.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import networkx as nx

from app.config import get_settings
from app.evidence.store import EvidenceStore, get_evidence_store
from app.graph.models import (
    Claim,
    EdgeType,
    Entity,
    EntityType,
    Event,
    ExtractionResult,
    GraphEdge,
    GraphQuery,
    GraphQueryResult,
)
from app.logging_config import get_logger

logger = get_logger("argus.graph.store")


class EvidenceGraphStore:
    """NetworkX-backed evidence graph with persistence to disk.

    Nodes: entities, claims, events, chunks, sources, documents
    Edges: typed relationships with provenance and confidence
    """

    def __init__(
        self,
        graph_path: Path | None = None,
        evidence_store: EvidenceStore | None = None,
    ):
        settings = get_settings()
        self.graph_path = graph_path or (settings.data_dir / "graph" / "evidence_graph.pkl")
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_store = evidence_store or get_evidence_store()
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_name_index: dict[str, UUID] = {}
        self._load_graph()

    def _load_graph(self) -> None:
        """Load graph from disk if exists."""
        if self.graph_path.exists():
            try:
                with self.graph_path.open("rb") as f:
                    self._graph = pickle.load(f)
                self._rebuild_entity_name_index()
                logger.info("graph_loaded", path=str(self.graph_path), nodes=self._graph.number_of_nodes(), edges=self._graph.number_of_edges())
            except (pickle.PickleError, OSError, EOFError) as exc:
                logger.warning("graph_load_failed", error=str(exc), path=str(self.graph_path))
                self._graph = nx.MultiDiGraph()
        else:
            logger.info("graph_not_found_creating_new", path=str(self.graph_path))
            self._graph = nx.MultiDiGraph()

    def _rebuild_entity_name_index(self) -> None:
        """Rebuild the entity name lookup index from the graph."""
        self._entity_name_index.clear()
        for node_key in self._graph.nodes:
            node_data = self._graph.nodes[node_key]
            if node_data.get("node_type") == "entity":
                data = json.loads(node_data.get("data", "{}"))
                name = data.get("canonical_name", "")
                if name:
                    self._entity_name_index[name.lower()] = UUID(data["id"])

    def save(self) -> None:
        """Persist graph to disk."""
        try:
            with self.graph_path.open("wb") as f:
                pickle.dump(self._graph, f)
            logger.info("graph_saved", path=str(self.graph_path), nodes=self._graph.number_of_nodes(), edges=self._graph.number_of_edges())
        except Exception as exc:
            logger.error("graph_save_failed", error=str(exc), path=str(self.graph_path))
            raise

    # -- Node operations -----------------------------------------------------

    def _node_key(self, node_type: str, node_id: UUID) -> str:
        """Generate unique node key for NetworkX."""
        return f"{node_type}:{node_id}"

    def _add_node(self, node_type: str, node_id: UUID, data: dict[str, Any]) -> None:
        """Add or update a node in the graph."""
        key = self._node_key(node_type, node_id)
        self._graph.add_node(key, node_type=node_type, node_id=str(node_id), data=json.dumps(data, default=str), updated_at=datetime.now(UTC).isoformat())

    def _get_node_data(self, node_type: str, node_id: UUID) -> dict[str, Any] | None:
        """Get node data from graph."""
        key = self._node_key(node_type, node_id)
        if self._graph.has_node(key):
            node_data = self._graph.nodes[key]
            return json.loads(node_data.get("data", "{}"))
        return None

    def _node_exists(self, node_type: str, node_id: UUID) -> bool:
        """Check if node exists."""
        return self._graph.has_node(self._node_key(node_type, node_id))

    # -- Entity operations ---------------------------------------------------

    def upsert_entity(self, entity: Entity) -> Entity:
        """Insert or update an entity. Merges aliases and supporting chunks."""
        # First check if entity with same canonical name already exists
        existing_by_name = self.find_entity_by_name(entity.canonical_name, entity.entity_type)
        if existing_by_name:
            # Merge with existing entity
            existing = existing_by_name
            # Merge aliases
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            # Merge supporting chunks
            for chunk_id in entity.supporting_chunk_ids:
                if chunk_id not in existing.supporting_chunk_ids:
                    existing.supporting_chunk_ids.append(chunk_id)
            # Update confidence (max of existing and new)
            existing.confidence = max(existing.confidence, entity.confidence)
            # Update description if new one is longer/more informative
            if entity.description and (not existing.description or len(entity.description) > len(existing.description)):
                existing.description = entity.description
            existing.updated_at = datetime.now(UTC)
            entity = existing
        else:
            # Check if entity with same ID exists (for updates by ID)
            existing_data = self._get_node_data("entity", entity.id)
            if existing_data:
                existing = Entity(**existing_data)
                # Merge aliases
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                # Merge supporting chunks
                for chunk_id in entity.supporting_chunk_ids:
                    if chunk_id not in existing.supporting_chunk_ids:
                        existing.supporting_chunk_ids.append(chunk_id)
                existing.confidence = max(existing.confidence, entity.confidence)
                if entity.description and (not existing.description or len(entity.description) > len(existing.description)):
                    existing.description = entity.description
                existing.updated_at = datetime.now(UTC)
                entity = existing

        self._add_node("entity", entity.id, entity.model_dump(mode="json"))
        self._entity_name_index[entity.canonical_name.lower()] = entity.id
        return entity

    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID."""
        data = self._get_node_data("entity", entity_id)
        return Entity(**data) if data else None

    def find_entity_by_name(self, name: str, entity_type: EntityType | None = None) -> Entity | None:
        """Find entity by canonical name (and optionally type)."""
        name_lower = name.lower()
        entity_id = self._entity_name_index.get(name_lower)
        if entity_id is None:
            return None
        entity = self.get_entity(entity_id)
        if entity is None:
            return None
        if entity_type is not None and entity.entity_type != entity_type:
            return None
        return entity

    def find_entities_by_alias(self, alias: str) -> list[Entity]:
        """Find entities that have this alias."""
        results = []
        alias_lower = alias.lower()
        for node_key in self._graph.nodes:
            node_data = self._graph.nodes[node_key]
            if node_data.get("node_type") == "entity":
                data = json.loads(node_data.get("data", "{}"))
                for a in data.get("aliases", []):
                    if a.lower() == alias_lower:
                        results.append(Entity(**data))
                        break
        return results

    def get_all_entities(self) -> list[Entity]:
        """Get all entities in the graph."""
        results = []
        for node_key in self._graph.nodes:
            node_data = self._graph.nodes[node_key]
            if node_data.get("node_type") == "entity":
                data = json.loads(node_data.get("data", "{}"))
                results.append(Entity(**data))
        return results

    # -- Claim operations ----------------------------------------------------

    def upsert_claim(self, claim: Claim) -> Claim:
        """Insert or update a claim. Merges supporting/contradicting chunks."""
        existing_data = self._get_node_data("claim", claim.id)
        if existing_data:
            existing = Claim(**existing_data)
            # Merge supporting chunks
            for chunk_id in claim.supporting_chunk_ids:
                if chunk_id not in existing.supporting_chunk_ids:
                    existing.supporting_chunk_ids.append(chunk_id)
            # Merge contradicting chunks
            for chunk_id in claim.contradicting_chunk_ids:
                if chunk_id not in existing.contradicting_chunk_ids:
                    existing.contradicting_chunk_ids.append(chunk_id)
            # Merge subject/object entity IDs (take non-None values)
            if claim.subject_entity_id is not None:
                existing.subject_entity_id = claim.subject_entity_id
            if claim.object_entity_id is not None:
                existing.object_entity_id = claim.object_entity_id
            if claim.object_value is not None:
                existing.object_value = claim.object_value
            # Update confidence (weighted by evidence count)
            total_support = len(existing.supporting_chunk_ids)
            total_contra = len(existing.contradicting_chunk_ids)
            if total_support + total_contra > 0:
                existing.confidence = total_support / (total_support + total_contra)
            existing.updated_at = datetime.now(UTC)
            claim = existing

        self._add_node("claim", claim.id, claim.model_dump(mode="json"))
        return claim

    def get_claim(self, claim_id: UUID) -> Claim | None:
        """Get claim by ID."""
        data = self._get_node_data("claim", claim_id)
        return Claim(**data) if data else None

    def get_all_claims(self) -> list[Claim]:
        """Get all claims in the graph."""
        results = []
        for node_key in self._graph.nodes:
            node_data = self._graph.nodes[node_key]
            if node_data.get("node_type") == "claim":
                data = json.loads(node_data.get("data", "{}"))
                results.append(Claim(**data))
        return results

    # -- Event operations ----------------------------------------------------

    def upsert_event(self, event: Event) -> Event:
        """Insert or update an event."""
        existing_data = self._get_node_data("event", event.id)
        if existing_data:
            existing = Event(**existing_data)
            # Merge supporting chunks
            for chunk_id in event.supporting_chunk_ids:
                if chunk_id not in existing.supporting_chunk_ids:
                    existing.supporting_chunk_ids.append(chunk_id)
            # Merge participants
            for pid in event.participant_entity_ids:
                if pid not in existing.participant_entity_ids:
                    existing.participant_entity_ids.append(pid)
            existing.confidence = max(existing.confidence, event.confidence)
            existing.updated_at = datetime.now(UTC)
            event = existing

        self._add_node("event", event.id, event.model_dump(mode="json"))
        return event

    def get_event(self, event_id: UUID) -> Event | None:
        """Get event by ID."""
        data = self._get_node_data("event", event_id)
        return Event(**data) if data else None

    def get_all_events(self) -> list[Event]:
        """Get all events in the graph."""
        results = []
        for node_key in self._graph.nodes:
            node_data = self._graph.nodes[node_key]
            if node_data.get("node_type") == "event":
                data = json.loads(node_data.get("data", "{}"))
                results.append(Event(**data))
        return results

    # -- Edge operations -----------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge to the graph. Handles duplicate detection."""
        source_key = self._node_key(edge.source_node_type, edge.source_node_id)
        target_key = self._node_key(edge.target_node_type, edge.target_node_id)

        # Check for existing edge of same type between same nodes
        if self._graph.has_edge(source_key, target_key):
            for edge_key, edge_data in self._graph[source_key][target_key].items():
                if edge_data.get("edge_type") == edge.edge_type.value:
                    # Merge with existing edge
                    existing_edge = GraphEdge(**json.loads(edge_data.get("data", "{}")))
                    # Merge supporting chunks
                    for chunk_id in edge.supporting_chunk_ids:
                        if chunk_id not in existing_edge.supporting_chunk_ids:
                            existing_edge.supporting_chunk_ids.append(chunk_id)
                    existing_edge.confidence = max(existing_edge.confidence, edge.confidence)
                    existing_edge.updated_at = datetime.now(UTC)
                    # Update edge data
                    self._graph[source_key][target_key][edge_key]["data"] = json.dumps(existing_edge.model_dump(mode="json"), default=str)
                    self._graph[source_key][target_key][edge_key]["updated_at"] = existing_edge.updated_at.isoformat()
                    return existing_edge

        # Add new edge
        self._graph.add_edge(
            source_key,
            target_key,
            edge_type=edge.edge_type.value,
            data=json.dumps(edge.model_dump(mode="json"), default=str),
            created_at=edge.created_at.isoformat(),
            updated_at=edge.updated_at.isoformat(),
        )
        return edge

    def get_edges(
        self,
        source_node_id: UUID | None = None,
        source_node_type: str | None = None,
        target_node_id: UUID | None = None,
        target_node_type: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[GraphEdge]:
        """Query edges with optional filters."""
        results = []
        for source_key, target_key, edge_key, edge_data in self._graph.edges(keys=True, data=True):
            # Filter by source
            if source_node_id:
                expected_source = self._node_key(source_node_type or "", source_node_id)
                if source_key != expected_source:
                    continue
            elif source_node_type:
                if not source_key.startswith(f"{source_node_type}:"):
                    continue

            # Filter by target
            if target_node_id:
                expected_target = self._node_key(target_node_type or "", target_node_id)
                if target_key != expected_target:
                    continue
            elif target_node_type:
                if not target_key.startswith(f"{target_node_type}:"):
                    continue

            # Filter by edge type
            if edge_type and edge_data.get("edge_type") != edge_type.value:
                continue

            edge = GraphEdge(**json.loads(edge_data.get("data", "{}")))
            results.append(edge)

        return results

    # -- Graph query operations ----------------------------------------------

    def query_graph(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a multi-hop graph query."""
        result = GraphQueryResult()

        # Find starting nodes
        start_keys = []
        for entity_id in query.start_entity_ids:
            key = self._node_key("entity", entity_id)
            if self._graph.has_node(key):
                start_keys.append(key)

        for name in query.start_entity_names:
            entity = self.find_entity_by_name(name)
            if entity:
                key = self._node_key("entity", entity.id)
                if key not in start_keys:
                    start_keys.append(key)

        if not start_keys:
            return result

        # Multi-hop traversal
        visited = set()
        visited_edges: set[tuple[str, str, str]] = set()
        current_frontier = set(start_keys)
        paths: dict[str, list[str]] = {k: [k] for k in start_keys}

        for hop in range(query.max_hops):
            next_frontier = set()
            for node_key in current_frontier:
                if node_key in visited:
                    continue
                visited.add(node_key)

                # Collect node data
                node_data = self._graph.nodes[node_key]
                node_type = node_data.get("node_type")
                data = json.loads(node_data.get("data", "{}"))

                if node_type == "entity":
                    result.entities.append(Entity(**data))
                elif node_type == "claim":
                    result.claims.append(Claim(**data))
                elif node_type == "event":
                    result.events.append(Event(**data))

                # Traverse outgoing edges
                for _, target_key, edge_key, edge_data in self._graph.out_edges(node_key, keys=True, data=True):
                    edge_tuple = (node_key, target_key, edge_key)
                    if edge_tuple in visited_edges:
                        continue
                    visited_edges.add(edge_tuple)
                    edge = GraphEdge(**json.loads(edge_data.get("data", "{}")))
                    if query.edge_types and edge.edge_type not in query.edge_types:
                        continue

                    result.edges.append(edge)

                    # Check temporal filter
                    if query.time_window_start and edge.valid_from and edge.valid_from < query.time_window_start:
                        continue
                    if query.time_window_end and edge.valid_to and edge.valid_to > query.time_window_end:
                        continue

                    if target_key not in visited:
                        next_frontier.add(target_key)
                        paths[target_key] = paths[node_key] + [target_key]

            current_frontier = next_frontier
            if not current_frontier:
                break

        # Collect paths
        for path_keys in paths.values():
            if len(path_keys) > 1:
                result.paths.append([UUID(k.split(":", 1)[1]) for k in path_keys])

        # Collect evidence refs from supporting chunks
        all_chunk_ids = set()
        for entity in result.entities:
            all_chunk_ids.update(entity.supporting_chunk_ids)
        for claim in result.claims:
            all_chunk_ids.update(claim.supporting_chunk_ids)
            all_chunk_ids.update(claim.contradicting_chunk_ids)
        for event in result.events:
            all_chunk_ids.update(event.supporting_chunk_ids)
        for edge in result.edges:
            all_chunk_ids.update(edge.supporting_chunk_ids)

        if all_chunk_ids:
            # Get evidence refs from evidence store
            chunks = self.evidence_store.get_chunks_by_ids(list(all_chunk_ids))
            # Build minimal evidence refs (without scores since we don't have query context)
            from app.evidence.models import EvidenceRef
            for chunk in chunks:
                doc = self.evidence_store.get_document(chunk.document_id)
                if doc:
                    source = self.evidence_store.get_source(doc.source_id)
                    if source:
                        result.evidence_refs.append(EvidenceRef(
                            chunk_id=chunk.id,
                            document_id=doc.id,
                            source_id=source.id,
                            source_path=source.path,
                            source_type=source.type,
                            text=chunk.text,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            section_path=chunk.section_path,
                            score=1.0,  # placeholder
                            rank=1,
                        ))

        # Limit results
        result.entities = result.entities[:query.limit]
        result.claims = result.claims[:query.limit]
        result.events = result.events[:query.limit]
        result.edges = result.edges[:query.limit]

        return result

    # -- Extraction integration ----------------------------------------------

    def apply_extraction(self, extraction: ExtractionResult) -> None:
        """Apply extraction results to the graph."""
        # Add entities
        for entity in extraction.entities:
            self.upsert_entity(entity)
            # Add MENTIONS edges from chunks to entity
            for chunk_id in entity.supporting_chunk_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.MENTIONS,
                    source_node_id=chunk_id,
                    source_node_type="chunk",
                    target_node_id=entity.id,
                    target_node_type="entity",
                    supporting_chunk_ids=[chunk_id],
                    confidence=entity.confidence,
                )
                self.add_edge(edge)

        # Add claims
        for claim in extraction.claims:
            self.upsert_claim(claim)
            # Add DERIVED_FROM edges from chunks to claim
            for chunk_id in claim.supporting_chunk_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.DERIVED_FROM,
                    source_node_id=chunk_id,
                    source_node_type="chunk",
                    target_node_id=claim.id,
                    target_node_type="claim",
                    supporting_chunk_ids=[chunk_id],
                    confidence=claim.confidence,
                )
                self.add_edge(edge)
            # Add SUPPORTS edges from chunks to claim
            for chunk_id in claim.supporting_chunk_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.SUPPORTS,
                    source_node_id=chunk_id,
                    source_node_type="chunk",
                    target_node_id=claim.id,
                    target_node_type="claim",
                    supporting_chunk_ids=[chunk_id],
                    confidence=claim.confidence,
                )
                self.add_edge(edge)
            # Add CONTRADICTS edges
            for chunk_id in claim.contradicting_chunk_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.CONTRADICTS,
                    source_node_id=chunk_id,
                    source_node_type="chunk",
                    target_node_id=claim.id,
                    target_node_type="claim",
                    supporting_chunk_ids=[chunk_id],
                    confidence=claim.confidence,
                )
                self.add_edge(edge)

            # Link claim to subject/object entities
            if claim.subject_entity_id:
                edge = GraphEdge(
                    edge_type=EdgeType.RELATES_TO,
                    source_node_id=claim.id,
                    source_node_type="claim",
                    target_node_id=claim.subject_entity_id,
                    target_node_type="entity",
                    supporting_chunk_ids=claim.supporting_chunk_ids,
                    confidence=claim.confidence,
                )
                self.add_edge(edge)
            if claim.object_entity_id:
                edge = GraphEdge(
                    edge_type=EdgeType.RELATES_TO,
                    source_node_id=claim.id,
                    source_node_type="claim",
                    target_node_id=claim.object_entity_id,
                    target_node_type="entity",
                    supporting_chunk_ids=claim.supporting_chunk_ids,
                    confidence=claim.confidence,
                )
                self.add_edge(edge)

        # Add events
        for event in extraction.events:
            self.upsert_event(event)
            # Add DERIVED_FROM edges
            for chunk_id in event.supporting_chunk_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.DERIVED_FROM,
                    source_node_id=chunk_id,
                    source_node_type="chunk",
                    target_node_id=event.id,
                    target_node_type="event",
                    supporting_chunk_ids=[chunk_id],
                    confidence=event.confidence,
                )
                self.add_edge(edge)
            # Link event to participants
            for participant_id in event.participant_entity_ids:
                edge = GraphEdge(
                    edge_type=EdgeType.RELATES_TO,
                    source_node_id=event.id,
                    source_node_type="event",
                    target_node_id=participant_id,
                    target_node_type="entity",
                    supporting_chunk_ids=event.supporting_chunk_ids,
                    confidence=event.confidence,
                )
                self.add_edge(edge)
            if event.location_entity_id:
                edge = GraphEdge(
                    edge_type=EdgeType.RELATES_TO,
                    source_node_id=event.id,
                    source_node_type="event",
                    target_node_id=event.location_entity_id,
                    target_node_type="entity",
                    supporting_chunk_ids=event.supporting_chunk_ids,
                    confidence=event.confidence,
                )
                self.add_edge(edge)

        # Add explicit edges from extraction
        for edge in extraction.edges:
            self.add_edge(edge)

        # Save after batch update
        self.save()

    # -- Stats ---------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        counts = {"entities": 0, "claims": 0, "events": 0, "chunks": 0, "sources": 0, "documents": 0, "edges": 0}
        for node_key in self._graph.nodes:
            node_type = self._graph.nodes[node_key].get("node_type")
            if node_type in counts:
                counts[node_type] += 1
        counts["edges"] = self._graph.number_of_edges()
        return counts


# Singleton instance
_graph_store: EvidenceGraphStore | None = None


def get_graph_store() -> EvidenceGraphStore:
    """Get or create the singleton graph store."""
    global _graph_store
    if _graph_store is None:
        _graph_store = EvidenceGraphStore()
    return _graph_store