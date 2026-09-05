"""Entity and Relation Embeddings (Phase 18+).

Provides vector search over graph entities and relations.
Embeds entity names + descriptions and relation descriptions,
enabling semantic search over the knowledge graph structure.

Inspired by LightRAG's dual-level retrieval: entity-focused (local)
and theme-focused (global) search paths.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from app.config import Settings, get_settings
from app.graph.models import Claim, Entity, GraphEdge
from app.logging_config import get_logger

logger = get_logger("argus.graph.entity_embeddings")


class EntityEmbeddingStore:
    """FAISS-backed vector store for entity and relation embeddings.

    Stores entity names + descriptions and relation descriptions as
    vectors, enabling semantic similarity search over graph nodes.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._dim = 384  # Matches all-MiniLM-L6-v2

        # FAISS index for entities
        self._entity_index: faiss.IndexFlatIP | None = None
        self._entity_ids: list[UUID] = []  # Parallel list mapping FAISS index -> entity UUID

        # FAISS index for relations (edges)
        self._relation_index: faiss.IndexFlatIP | None = None
        self._relation_edge_ids: list[UUID] = []  # Parallel list mapping FAISS index -> edge UUID

        # In-memory text cache for display/debug
        self._entity_texts: dict[UUID, str] = {}
        self._relation_texts: dict[UUID, str] = {}

        self._initialized = False

    def _ensure_indexes(self) -> None:
        """Create FAISS indexes if they don't exist."""
        if self._initialized:
            return
        self._entity_index = faiss.IndexFlatIP(self._dim)
        self._relation_index = faiss.IndexFlatIP(self._dim)
        self._initialized = True

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the shared embedding model."""
        from app.retrieval.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        return gen.generate_embedding(text)

    def _entity_to_text(self, entity: Entity) -> str:
        """Convert entity to embeddable text: name + description."""
        parts = [entity.canonical_name]
        if entity.description:
            parts.append(entity.description)
        if entity.aliases:
            parts.append(", ".join(entity.aliases[:3]))
        return " ".join(parts)

    def _relation_to_text(self, edge: GraphEdge, source_name: str = "", target_name: str = "") -> str:
        """Convert relation to embeddable text: source + predicate + target."""
        parts = []
        if source_name:
            parts.append(source_name)
        if edge.edge_type.value:
            parts.append(edge.edge_type.value.replace("_", " ").lower())
        if target_name:
            parts.append(target_name)
        return " ".join(parts) if parts else f"edge {edge.id}"

    def _claim_to_text(self, claim: Claim) -> str:
        """Convert claim to embeddable text: subject + predicate + object."""
        parts = []
        if claim.predicate:
            parts.append(claim.predicate)
        if claim.object_value:
            parts.append(claim.object_value)
        return " ".join(parts) if parts else f"claim {claim.id}"

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity embedding."""
        self._ensure_indexes()
        text = self._entity_to_text(entity)
        embedding = self._get_embedding(text)

        with self._lock:
            # Check if entity already exists
            if entity.id in self._entity_texts:
                # Update: find and replace
                idx = self._entity_ids.index(entity.id) if entity.id in self._entity_ids else -1
                if idx >= 0:
                    # FAISS IndexFlatIP doesn't support update; we'd need to rebuild
                    # For now, just update the text cache
                    self._entity_texts[entity.id] = text
                    return

            # Add new
            faiss.normalize_L2(embedding.reshape(1, -1))
            self._entity_index.add(embedding.reshape(1, -1))
            self._entity_ids.append(entity.id)
            self._entity_texts[entity.id] = text

    def add_relation(self, edge: GraphEdge, source_name: str = "", target_name: str = "") -> None:
        """Add or update a relation embedding."""
        self._ensure_indexes()
        text = self._relation_to_text(edge, source_name, target_name)
        embedding = self._get_embedding(text)

        with self._lock:
            if edge.id in self._relation_texts:
                return  # Already exists

            faiss.normalize_L2(embedding.reshape(1, -1))
            self._relation_index.add(embedding.reshape(1, -1))
            self._relation_edge_ids.append(edge.id)
            self._relation_texts[edge.id] = text

    def add_claim(self, claim: Claim) -> None:
        """Add or update a claim embedding (stored in relation index)."""
        self._ensure_indexes()
        text = self._claim_to_text(claim)
        if not text.strip():
            return
        embedding = self._get_embedding(text)

        with self._lock:
            if claim.id in self._relation_texts:
                return

            faiss.normalize_L2(embedding.reshape(1, -1))
            self._relation_index.add(embedding.reshape(1, -1))
            self._relation_edge_ids.append(claim.id)
            self._relation_texts[claim.id] = text

    def search_entities(
        self, query: str, top_k: int = 10
    ) -> list[tuple[UUID, float]]:
        """Find entities most similar to query text.

        Returns list of (entity_id, similarity_score) tuples.
        """
        self._ensure_indexes()
        if self._entity_index.ntotal == 0:
            return []

        embedding = self._get_embedding(query)
        faiss.normalize_L2(embedding.reshape(1, -1))

        with self._lock:
            k = min(top_k, self._entity_index.ntotal)
            scores, indices = self._entity_index.search(embedding.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entity_ids):
                continue
            results.append((self._entity_ids[idx], float(score)))

        return results

    def search_relations(
        self, query: str, top_k: int = 10
    ) -> list[tuple[UUID, float]]:
        """Find relations/claims most similar to query text.

        Returns list of (edge_or_claim_id, similarity_score) tuples.
        """
        self._ensure_indexes()
        if self._relation_index.ntotal == 0:
            return []

        embedding = self._get_embedding(query)
        faiss.normalize_L2(embedding.reshape(1, -1))

        with self._lock:
            k = min(top_k, self._relation_index.ntotal)
            scores, indices = self._relation_index.search(embedding.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._relation_edge_ids):
                continue
            results.append((self._relation_edge_ids[idx], float(score)))

        return results

    def get_entity_text(self, entity_id: UUID) -> str | None:
        """Get the embedded text for an entity."""
        return self._entity_texts.get(entity_id)

    def get_relation_text(self, edge_id: UUID) -> str | None:
        """Get the embedded text for a relation."""
        return self._relation_texts.get(edge_id)

    @property
    def entity_count(self) -> int:
        return self._entity_index.ntotal if self._entity_index else 0

    @property
    def relation_count(self) -> int:
        return self._relation_index.ntotal if self._relation_index else 0

    def save(self, path: Path | None = None) -> None:
        """Persist entity embeddings to disk."""
        path = path or (self.settings.graph_path.parent / "entity_embeddings")
        path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._entity_index and self._entity_index.ntotal > 0:
                faiss.write_index(self._entity_index, str(path / "entities.faiss"))
                ids_data = [str(uid) for uid in self._entity_ids]
                (path / "entity_ids.json").write_text(json.dumps(ids_data))
                texts_data = {str(k): v for k, v in self._entity_texts.items()}
                (path / "entity_texts.json").write_text(json.dumps(texts_data))

            if self._relation_index and self._relation_index.ntotal > 0:
                faiss.write_index(self._relation_index, str(path / "relations.faiss"))
                ids_data = [str(uid) for uid in self._relation_edge_ids]
                (path / "relation_ids.json").write_text(json.dumps(ids_data))
                texts_data = {str(k): v for k, v in self._relation_texts.items()}
                (path / "relation_texts.json").write_text(json.dumps(texts_data))

        logger.info("entity_embeddings_saved", path=str(path),
                     entities=self.entity_count, relations=self.relation_count)

    def load(self, path: Path | None = None) -> bool:
        """Load entity embeddings from disk. Returns True if loaded."""
        path = path or (self.settings.graph_path.parent / "entity_embeddings")

        entity_faiss = path / "entities.faiss"
        entity_ids = path / "entity_ids.json"
        if not entity_faiss.exists() or not entity_ids.exists():
            return False

        with self._lock:
            self._entity_index = faiss.read_index(str(entity_faiss))
            self._entity_ids = [UUID(uid) for uid in json.loads(entity_ids.read_text())]

            texts_path = path / "entity_texts.json"
            if texts_path.exists():
                raw = json.loads(texts_path.read_text())
                self._entity_texts = {UUID(k): v for k, v in raw.items()}

            relation_faiss = path / "relations.faiss"
            relation_ids = path / "relation_ids.json"
            if relation_faiss.exists() and relation_ids.exists():
                self._relation_index = faiss.read_index(str(relation_faiss))
                self._relation_edge_ids = [UUID(uid) for uid in json.loads(relation_ids.read_text())]

                r_texts_path = path / "relation_texts.json"
                if r_texts_path.exists():
                    raw = json.loads(r_texts_path.read_text())
                    self._relation_texts = {UUID(k): v for k, v in raw.items()}

            self._initialized = True

        logger.info("entity_embeddings_loaded", path=str(path),
                     entities=self.entity_count, relations=self.relation_count)
        return True


_store: EntityEmbeddingStore | None = None


def get_entity_embedding_store() -> EntityEmbeddingStore:
    """Get or create the singleton entity embedding store."""
    global _store
    if _store is None:
        _store = EntityEmbeddingStore()
    return _store
