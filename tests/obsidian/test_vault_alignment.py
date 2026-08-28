"""Phase 09.3 vault-graph alignment + vault memory tests.

Verifies canonical entity ids across wikilinks (no graph fragmentation),
claim-class tagging, and the Phase 08 VAULT_MEMORY layer coordination.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.evidence.store import EvidenceStore
from app.graph.models import EdgeType, EntityType
from app.graph.store import EvidenceGraphStore
from app.integrations.obsidian.alignment import VaultGraphAligner, VaultMemoryCoordinator
from app.integrations.obsidian.parser import parse_obsidian_note
from app.memory.interfaces import DefaultMemoryFactory, set_memory_factory
from app.memory.store import MemoryStore


@pytest.fixture
def vault() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "vault"
        (root / "Sub").mkdir(parents=True)
        (root / "Alpha.md").write_text("# Alpha\n\nthinking about [[Beta]].\n", encoding="utf-8")
        (root / "Beta.md").write_text("# Beta\n\nBeta is the shared page.\n", encoding="utf-8")
        (root / "Sub" / "Gamma.md").write_text("# Gamma\n\nlinks to [[Beta]] too.\n", encoding="utf-8")
        yield root


@pytest.fixture
def store(vault: Path) -> EvidenceStore:
    return EvidenceStore(
        db_path=vault / "evidence.db",
        bm25_index_path=vault / "bm25.pkl",
        faiss_index_path=vault / "faiss.index",
    )


@pytest.fixture
def graph_store(vault: Path, store: EvidenceStore) -> EvidenceGraphStore:
    return EvidenceGraphStore(graph_path=vault / "graph.pkl", evidence_store=store)


class TestVaultGraphAligner:
    def test_note_creates_canonical_entity_and_claim(self, vault: Path, graph_store: EvidenceGraphStore):
        note = parse_obsidian_note(vault / "Alpha.md", vault)
        result = VaultGraphAligner(graph_store).align_note(
            note,
            chunk_ids=[],
            knowledge_class="knowledge_note",
            treatment_rule="personalization_only",
        )
        assert result.file_stem == "Alpha"
        assert result.entity_id
        assert result.claim_id

        entity = graph_store.get_entity(result.entity_id)
        assert entity.canonical_name == "Alpha"
        assert entity.entity_type == EntityType.CONCEPT
        assert graph_store.get_claim(result.claim_id).metadata["claim_class_tag"] == "knowledge_note"

    def test_same_wikilink_target_shares_canonical_entity(self, vault: Path, graph_store: EvidenceGraphStore):
        """Links to Beta from two folders resolve to ONE graph entity."""
        aligner = VaultGraphAligner(graph_store)
        alpha = parse_obsidian_note(vault / "Alpha.md", vault)
        gamma = parse_obsidian_note(vault / "Sub" / "Gamma.md", vault)

        r_alpha = aligner.align_note(alpha, chunk_ids=[], knowledge_class="knowledge_note")
        r_gamma = aligner.align_note(gamma, chunk_ids=[], knowledge_class="knowledge_note")

        # Both notes linked a wikilink entity for Beta.
        assert len(r_alpha.wikilink_entity_ids) == 1
        assert len(r_gamma.wikilink_entity_ids) == 1
        # Canonical silence: the exact same entity id is reused across folders.
        assert r_alpha.wikilink_entity_ids[0] == r_gamma.wikilink_entity_ids[0]

        beta = graph_store.find_entity_by_name("Beta", EntityType.CONCEPT)
        assert beta is not None
        assert beta.id == r_alpha.wikilink_entity_ids[0]
        # And both notes relate to Beta.
        edges = graph_store.get_edges(edge_type=EdgeType.RELATES_TO)
        betas = [e for e in edges if e.target_node_id == beta.id]
        assert len(betas) == 2

    def test_align_vault_summary_counts(self, vault: Path, graph_store: EvidenceGraphStore):
        notes = [
            parse_obsidian_note(vault / "Alpha.md", vault),
            parse_obsidian_note(vault / "Beta.md", vault),
            parse_obsidian_note(vault / "Sub" / "Gamma.md", vault),
        ]
        summary = VaultGraphAligner(graph_store).align_vault(notes, chunk_map={})
        assert summary["aligned"] == 3
        assert summary["claims"] == 3
        assert summary["edges"] >= 2


class TestVaultMemoryCoordinator:
    async def test_sync_loads_vault_memory_layer(self, vault: Path, store: EvidenceStore, graph_store: EvidenceGraphStore):
        memory_store = MemoryStore(db_path=vault / "memory.db")
        coordinator = VaultMemoryCoordinator(
            memory_store=memory_store,
            aligner=VaultGraphAligner(graph_store),
        )
        stats = await coordinator.sync_vault_memory(str(vault))
        assert stats["synced"] == 3
        assert stats["aligned"] == 3

        entities = await coordinator.get_vault_entities(str(vault))
        assert len(entities) >= 3  # Alpha, Beta, Gamma (+ their wikilink targets)

        record = await coordinator.get_note_memory("Alpha.md")
        assert record is not None
        assert record.subject == "Alpha.md"
        metadata = record.metadata
        assert metadata["claim_ids"]
        assert metadata.get("note_type") in {"personal_context", "knowledge_note"}

        await coordinator.link_note_to_entities("Sub/Gamma.md", ["entity-1", "entity-2"])
        memory_store.close()

    async def test_no_memory_store_raises_on_sync(self, vault: Path, graph_store: EvidenceGraphStore):
        set_memory_factory(DefaultMemoryFactory())  # no memory infrastructure
        coordinator = VaultMemoryCoordinator(memory_store=None, aligner=VaultGraphAligner(graph_store))
        with pytest.raises(RuntimeError):
            await coordinator.sync_vault_memory(str(vault))