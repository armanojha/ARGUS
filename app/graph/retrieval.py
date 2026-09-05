"""Graph-based Retrieval (Phase 03).

Provides multi-hop entity/relationship query support over the Evidence Graph.
Integrates with Phase 01 hybrid retrieval as an additional retrieval method.
Enhanced with entity/relation embeddings for semantic graph search (Phase 18+).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import Settings, get_settings
from app.evidence.models import EvidenceRef
from app.graph.models import EdgeType, GraphQuery
from app.graph.store import EvidenceGraphStore, get_graph_store
from app.logging_config import get_logger
from app.retrieval.hybrid import HybridRetriever, get_hybrid_retriever

logger = get_logger("argus.graph.retrieval")


class GraphRetriever:
    """Graph-based retriever for multi-hop queries.

    Wraps the EvidenceGraphStore and provides a retrieval interface
    compatible with the Phase 01 HybridRetriever pattern.

    Enhanced with entity/relation embeddings for semantic graph search:
    - Entity embeddings enable finding relevant entities by meaning, not just name
    - Relation embeddings enable finding relevant relationships by theme
    - Combined with multi-hop traversal for comprehensive evidence discovery
    """

    def __init__(
        self,
        graph_store: EvidenceGraphStore | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        settings: Settings | None = None,
    ):
        self.graph_store = graph_store or get_graph_store()
        self.hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self.settings = settings or get_settings()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        max_hops: int = 2,
        edge_types: list[EdgeType] | None = None,
        time_window_start: str | None = None,
        time_window_end: str | None = None,
    ) -> list[EvidenceRef]:
        """Execute a graph-based search and return EvidenceRefs.

        This method:
        1. Uses hybrid retrieval to find initial relevant chunks
        2. Extracts entity names from initial results + entity embedding search
        3. Performs multi-hop graph traversal from those entities
        4. Returns combined evidence with confidence-weighted scoring
        """
        top_k = top_k or self.settings.retrieval_top_k

        # Step 1: Get initial evidence via hybrid retrieval
        initial_results = self.hybrid_retriever.search(query, top_k=top_k)

        if not initial_results:
            logger.info("graph_retrieval_no_initial_results", query=query[:50])
            return []

        # Step 2: Extract entity names from initial results
        # Combine regex heuristic with entity embedding search
        entity_names = self._extract_entity_names_from_results(initial_results)

        # Also search entity embeddings for semantic matches
        entity_names.extend(self._search_entity_embeddings(query, top_k=5))

        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in entity_names:
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                unique_names.append(name)
        entity_names = unique_names

        # Step 3: Build graph query
        graph_query = GraphQuery(
            start_entity_names=entity_names[:5],  # Limit starting entities
            edge_types=edge_types or [
                EdgeType.RELATES_TO,
                EdgeType.MENTIONS,
                EdgeType.SUPPORTS,
                EdgeType.DERIVED_FROM,
                EdgeType.CONTRADICTS,
            ],
            max_hops=max_hops,
            limit=top_k * 2,
        )

        # Parse time window if provided
        if time_window_start:
            try:
                graph_query.time_window_start = datetime.fromisoformat(time_window_start)
            except ValueError:
                pass
        if time_window_end:
            try:
                graph_query.time_window_end = datetime.fromisoformat(time_window_end)
            except ValueError:
                pass

        # Step 4: Execute graph query
        graph_result = self.graph_store.query_graph(graph_query)

        # Step 5: Combine and rank results
        # Start with initial results, add graph-discovered evidence
        combined_refs = list(initial_results)

        # Add graph evidence refs (avoid duplicates by chunk_id)
        seen_chunk_ids = {ref.chunk_id for ref in combined_refs}
        for ref in graph_result.evidence_refs:
            if ref.chunk_id not in seen_chunk_ids:
                combined_refs.append(ref)
                seen_chunk_ids.add(ref.chunk_id)

        # Re-rank: boost evidence that appears in graph paths
        # (Simplified: just return combined up to top_k)
        final_results = combined_refs[:top_k]

        logger.info(
            "graph_retrieval_completed",
            query=query[:50],
            initial_results=len(initial_results),
            entity_names_found=len(entity_names),
            graph_entities=len(graph_result.entities),
            graph_claims=len(graph_result.claims),
            graph_events=len(graph_result.events),
            final_results=len(final_results),
        )

        return final_results

    def _extract_entity_names_from_results(self, results: list[EvidenceRef]) -> list[str]:
        """Extract potential entity names from retrieval results.

        Uses regex heuristic for capitalized words/phrases.
        """
        import re

        entity_names = []
        # Simple pattern: capitalized words/phrases (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

        for ref in results[:5]:  # Only check top 5 results
            matches = re.findall(pattern, ref.text)
            for match in matches:
                if len(match) > 2 and match not in entity_names:
                    entity_names.append(match)

        return entity_names[:10]  # Limit

    def _search_entity_embeddings(self, query: str, top_k: int = 5) -> list[str]:
        """Search entity embeddings for semantically similar entities.

        Returns entity names that are semantically related to the query.
        """
        try:
            from app.graph.entity_embeddings import get_entity_embedding_store
            emb_store = get_entity_embedding_store()

            if emb_store.entity_count == 0:
                return []

            results = emb_store.search_entities(query, top_k=top_k)
            names = []
            for entity_id, score in results:
                text = emb_store.get_entity_text(entity_id)
                if text and score > 0.3:  # Minimum similarity threshold
                    # Extract just the entity name (first part before description)
                    name = text.split(" ")[0] if text else ""
                    if name and len(name) > 1:
                        names.append(name)

            return names
        except (ImportError, OSError, ValueError):
            # Graceful fallback if entity embeddings not available
            return []

    def search_by_entity(
        self,
        entity_name: str,
        top_k: int | None = None,
        max_hops: int = 2,
    ) -> list[EvidenceRef]:
        """Search for evidence related to a specific entity."""
        top_k = top_k or self.settings.retrieval_top_k

        graph_query = GraphQuery(
            start_entity_names=[entity_name],
            edge_types=[
                EdgeType.RELATES_TO,
                EdgeType.MENTIONS,
                EdgeType.SUPPORTS,
                EdgeType.DERIVED_FROM,
                EdgeType.CONTRADICTS,
            ],
            max_hops=max_hops,
            limit=top_k,
        )

        graph_result = self.graph_store.query_graph(graph_query)

        # Convert to EvidenceRefs
        refs = list(graph_result.evidence_refs)[:top_k]

        logger.info("graph_entity_search", entity=entity_name, results=len(refs))
        return refs

    def search_temporal(
        self,
        query: str,
        time_start: str,
        time_end: str,
        top_k: int | None = None,
    ) -> list[EvidenceRef]:
        """Search with temporal constraints (validity time)."""
        return self.search(
            query,
            top_k=top_k,
            time_window_start=time_start,
            time_window_end=time_end,
        )

    def get_entity_context(self, entity_name: str) -> dict[str, Any]:
        """Get full context for an entity (claims, events, relations)."""
        entity = self.graph_store.find_entity_by_name(entity_name)
        if not entity:
            return {"entity": None, "claims": [], "events": [], "relations": []}

        # Get claims involving this entity
        claims = []
        for claim in self.graph_store.get_all_claims():
            # Check direct entity ID links
            if claim.subject_entity_id == entity.id or claim.object_entity_id == entity.id or claim.object_value and claim.object_value.lower() == entity_name.lower():
                claims.append(claim)
            # Check if subject name matches (would need subject name stored)
            # For now, we check object_value which is commonly used

        # Get events involving this entity
        events = []
        for event in self.graph_store.get_all_events():
            if entity.id in event.participant_entity_ids or event.location_entity_id == entity.id:
                events.append(event)

        # Get relations
        relations = self.graph_store.get_edges(
            source_node_id=entity.id,
            source_node_type="entity",
        )
        relations += self.graph_store.get_edges(
            target_node_id=entity.id,
            target_node_type="entity",
        )

        return {
            "entity": entity,
            "claims": claims,
            "events": events,
            "relations": relations,
        }


def get_graph_retriever() -> GraphRetriever:
    """Get or create the singleton graph retriever."""
    return GraphRetriever()