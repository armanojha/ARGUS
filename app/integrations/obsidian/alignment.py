"""Phase 09.3: Vault-graph alignment and vault memory coordination.

Maps Obsidian notes onto the Evidence Graph (V3 §6):
  * each note page   -> a CONCEPT entity with the page name as canonical id
  * each wikilink    -> a RELATES_TO edge (targets normalized to a canonical
                        name so the same page never fragments the graph)
  * each note body   -> a Claim tagged with its knowledge class
Additionally coordinates the Phase 08 VAULT_MEMORY layer so notes are
retrievable alongside graph entities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.graph.models import Claim, EdgeType, Entity, EntityType, GraphEdge
from app.graph.store import EvidenceGraphStore, get_graph_store
from app.logging_config import get_logger
from app.memory.interfaces import (
    MemoryLayer,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    VaultMemoryInterface,
)

logger = get_logger("argus.obsidian.alignment")


class GraphAlignmentResult(BaseModel):
    """Result of aligning a single note with the Evidence Graph."""

    model_config = ConfigDict(extra="forbid")

    note_path: str
    file_stem: str
    entity_id: UUID
    claim_id: UUID
    wikilink_entity_ids: list[UUID] = Field(default_factory=list)
    edges_added: int = 0
    knowledge_class: str | None = None
    treatment_rule: str | None = None


def _note_description(note: Any, limit: int = 200) -> str | None:
    """Derive a short description from the note body."""
    title = getattr(getattr(note, "frontmatter", None), "title", None)
    body = (note.content_without_frontmatter or "").strip()
    first_paragraph = next((p.strip() for p in body.split("\n\n") if p.strip()), "")
    text = " ".join((first_paragraph or title or "").split())
    if not text:
        return None
    return text[:limit]


def _first_content_line(note: Any) -> str:
    """First meaningful content line (used as claim text)."""
    title = getattr(getattr(note, "frontmatter", None), "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()[:300]
    for line in (note.content_without_frontmatter or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:300]
    return (getattr(note, "file_stem", "") or "")[:300]


class VaultGraphAligner:
    """Aligns vault notes with canonical graph entities and claims.

    Entities are deduplicated by `canonical_name` inside the graph store
    (`upsert_entity`), which prevents graph fragmentation when multiple
    notes reference the same page across folders.
    """

    def __init__(self, graph_store: EvidenceGraphStore | None = None) -> None:
        self.graph_store = graph_store or get_graph_store()

    def align_note(
        self,
        note: Any,
        chunk_ids: list[UUID],
        knowledge_class: str | None = None,
        treatment_rule: str | None = None,
        confidence: float = 0.9,
    ) -> GraphAlignmentResult:
        """Align one parsed note with the evidence graph."""
        canonical = note.file_stem
        note_path = note.vault_relative_path

        aliases = list(getattr(getattr(note, "frontmatter", None), "aliases", []) or [])
        if isinstance(aliases, list):
            aliases = [str(a) for a in aliases if isinstance(a, str) and a.strip()]
        else:
            aliases = []
        if note_path not in aliases:
            aliases.append(note_path)

        note_entity = Entity(
            canonical_name=canonical,
            entity_type=EntityType.CONCEPT,
            aliases=aliases,
            description=_note_description(note),
            supporting_chunk_ids=list(chunk_ids),
            confidence=confidence,
            metadata={
                "note_path": note_path,
                "knowledge_class": knowledge_class,
                "treatment_rule": treatment_rule,
            },
        )
        note_entity = self.graph_store.upsert_entity(note_entity)

        claim = Claim(
            text=_first_content_line(note),
            predicate="obsidian_note",
            subject_entity_id=note_entity.id,
            supporting_chunk_ids=list(chunk_ids),
            confidence=confidence,
            metadata={
                "note_path": note_path,
                "note_id": note_path,
                "knowledge_class": knowledge_class,
                "claim_class_tag": knowledge_class,
                "treatment_rule": treatment_rule,
            },
        )
        claim = self.graph_store.upsert_claim(claim)

        edges_added = 0
        seen_targets: set[str] = set()
        wikilink_entity_ids: list[UUID] = []
        for wikilink in note.wikilinks or []:
            target_stem, target_alias = self._canonical_target(wikilink)
            if not target_stem or target_stem.lower() == canonical.lower():
                continue
            if target_stem.lower() in seen_targets:
                continue
            seen_targets.add(target_stem.lower())

            target_entity = Entity(
                canonical_name=target_stem,
                entity_type=EntityType.CONCEPT,
                aliases=[target_alias] if target_alias else [target_stem],
                metadata={"note_paths": [note_path]},
            )
            target_entity = self.graph_store.upsert_entity(target_entity)
            wikilink_entity_ids.append(target_entity.id)

            self.graph_store.add_edge(GraphEdge(
                edge_type=EdgeType.RELATES_TO,
                source_node_id=note_entity.id,
                source_node_type="entity",
                target_node_id=target_entity.id,
                target_node_type="entity",
                supporting_chunk_ids=list(chunk_ids),
                confidence=confidence,
                metadata={"note_path": note_path, "wikilink": getattr(wikilink, "target", "")},
            ))
            edges_added += 1

        return GraphAlignmentResult(
            note_path=note_path,
            file_stem=canonical,
            entity_id=note_entity.id,
            claim_id=claim.id,
            wikilink_entity_ids=wikilink_entity_ids,
            edges_added=edges_added,
            knowledge_class=knowledge_class,
            treatment_rule=treatment_rule,
        )

    def align_vault(self, notes: list[Any], chunk_map: dict[str, list[UUID]]) -> dict[str, Any]:
        """Align a whole vault. `chunk_map` maps vault_relative_path -> chunk ids."""
        results: list[GraphAlignmentResult] = []
        failed = 0
        for note in notes:
            chunk_ids = chunk_map.get(note.vault_relative_path, [])
            try:
                results.append(self.align_note(note, chunk_ids))
            except Exception as exc:  # noqa: BLE001
                logger.warning("vault_note_alignment_failed", note=note.vault_relative_path, error=str(exc))
                failed += 1
        return {
            "aligned": len(results),
            "failed": failed,
            "entities": len({r.entity_id for r in results}),
            "claims": sum(1 for r in results if r.claim_id),
            "edges": sum(r.edges_added for r in results),
            "results": results,
        }

    @staticmethod
    def _canonical_target(wikilink: Any) -> tuple[str, str | None]:
        """Normalize a wikilink target to (canonical leaf name, full path alias).

        Strips heading anchors and `.md`, and reduces across folders to the
        leaf page name so the same page always maps to one graph entity.
        """
        target = (getattr(wikilink, "target", "") or "").strip()
        alias = (getattr(wikilink, "alias", None) or target or None)
        if alias is not None:
            alias = alias.strip() or None
        stem = target.split("#", 1)[0].strip()
        if stem.lower().endswith(".md"):
            stem = stem[:-3]
        parts = [p for p in stem.replace("\\", "/").split("/") if p]
        canonical = parts[-1] if parts else stem
        return canonical, alias


class VaultMemoryCoordinator(VaultMemoryInterface):
    """Coordinates vault notes with the Phase 08 VAULT_MEMORY layer.

    Each note gets a VAULT_MEMORY MemoryRecord pointing back to its graph
    entity/claim ids, an aligned entity set, and note provenance. Retrieval
    uses Phase 08's SQLite memory store.
    """

    def __init__(
        self,
        memory_store: Any | None = None,
        aligner: VaultGraphAligner | None = None,
    ) -> None:
        self.aligner = aligner or VaultGraphAligner()
        self.memory_store = memory_store
        if self.memory_store is None:
            # Resolve lazily so importing the module never has side effects.
            from app.memory.interfaces import get_memory_factory

            factory = get_memory_factory()
            if hasattr(factory, "create_memory_store"):
                self.memory_store = factory.create_memory_store()

    async def sync_vault_memory(
        self,
        vault_path: str,
        chunk_map: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Sync vault notes into the VAULT_MEMORY layer. Returns sync stats."""
        from app.integrations.obsidian.scanner import VaultScanner

        if self.memory_store is None:
            raise RuntimeError("vault memory is disabled: no memory store available")

        root = Path(vault_path)
        chunk_map = chunk_map or {}
        notes = VaultScanner(root).scan()
        synced = 0
        aligned = 0
        for note in notes:
            try:
                chunk_ids = [UUID(c) for c in chunk_map.get(note.vault_relative_path, []) if _is_uuid(c)]
                result = self.aligner.align_note(note, chunk_ids)
                aligned += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("vault_note_sync_failed", note=note.vault_relative_path, error=str(exc))
                continue

            entity_ids = [str(result.entity_id), *[str(e) for e in result.wikilink_entity_ids]]
            record_id = uuid4()
            await self.memory_store.store(MemoryRecord(
                id=record_id,
                layer=MemoryLayer.VAULT_MEMORY,
                scope=MemoryScope.VAULT,
                content=f"Vault note {note.vault_relative_path} linked to {len(entity_ids)} graph entities",
                subject=note.vault_relative_path,
                supporting_chunk_ids=[str(c) for c in chunk_ids],
                source_query=str(root),
                confidence=0.9,
                tags=["vault", result.knowledge_class or "note"],
                metadata={
                    "note_path": note.vault_relative_path,
                    "note_id": str(record_id),
                    "entity_ids": entity_ids,
                    "claim_ids": [str(result.claim_id)],
                    "note_type": result.knowledge_class or "personal_context",
                    "file_modified": note.file_modified.isoformat(),
                },
            ))
            synced += 1

        logger.info("vault_memory_synced", vault=str(root), synced=synced, notes=len(notes))
        return {"synced": synced, "aligned": aligned, "notes": len(notes)}

    async def get_vault_entities(self, vault_path: str) -> list[Any]:
        """Get all graph entity ids linked from vault memory records."""
        if self.memory_store is None:
            return []
        records = await self.memory_store.retrieve(MemoryQuery(
            query_text=vault_path,
            layers=[MemoryLayer.VAULT_MEMORY],
            scope=MemoryScope.VAULT,
            limit=100,
        ))
        entity_ids: list[str] = []
        for record in records:
            metadata = getattr(record, "metadata", {}) or {}
            entity_ids.extend(metadata.get("entity_ids", []) or [])
        return list(dict.fromkeys(entity_ids))

    async def link_note_to_entities(self, note_path: str, entity_ids: list[str]) -> None:
        """Store a VAULT_MEMORY record linking a note to graph entities."""
        if self.memory_store is None:
            raise RuntimeError("vault memory is disabled: no memory store available")
        record_id = uuid4()
        await self.memory_store.store(MemoryRecord(
            id=record_id,
            layer=MemoryLayer.VAULT_MEMORY,
            scope=MemoryScope.VAULT,
            content=f"Vault note {note_path} linked to {len(entity_ids)} graph entities",
            subject=note_path,
            source_query=note_path,
            confidence=0.9,
            tags=["vault"],
            metadata={
                "note_path": note_path,
                "note_id": str(record_id),
                "entity_ids": list(entity_ids),
                "claim_ids": [],
                "note_type": "personal_context",
            },
        ))

    async def get_note_memory(self, note_path: str) -> Any | None:
        """Get the vault memory record for a specific note."""
        if self.memory_store is None:
            return None
        records = await self.memory_store.retrieve(MemoryQuery(
            query_text=note_path,
            layers=[MemoryLayer.VAULT_MEMORY],
            scope=MemoryScope.VAULT,
            limit=10,
        ))
        for record in records:
            if getattr(record, "subject", None) == note_path:
                return record
        return records[0] if records else None


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except (ValueError, TypeError):
        return False