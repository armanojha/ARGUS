"""Tests for Memory -> Obsidian Brain selective promotion.

Promotion must be selectitve (only PROMOTED long-term records with provenance
and sufficient confidence), provenance-preserving, and idempotent.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from app.integrations.obsidian.promotion import promote_eligible_memories
from app.memory.interfaces import MemoryLayer, MemoryPromotionStatus, MemoryRecord, MemoryScope
from app.memory.store import MemoryStore


@pytest.fixture
def memory_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def memory_store(memory_db):
    store = MemoryStore(memory_db)
    yield store
    store.close()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path / "brain_vault"


def _run_coro(coro):
    return asyncio.run(coro)


def _make_record(
    *, layer: MemoryLayer = MemoryLayer.LONG_TERM_KNOWLEDGE,
    status: MemoryPromotionStatus = MemoryPromotionStatus.PROMOTED,
    confidence: float = 0.95,
    with_provenance: bool = True,
    subject: str | None = "Acme",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        layer=layer,
        scope=MemoryScope.GLOBAL,
        content="Verified fact about Acme from the corpus.",
        subject=subject if with_provenance else None,
        predicate="acquired" if with_provenance else None,
        object="Beta" if with_provenance else None,
        supporting_chunk_ids=[str(uuid4())] if with_provenance else [],
        # source_query is a provenance signal; blank it for the no-provenance case.
        source_query="what did Acme acquire?" if with_provenance else None,
        confidence=confidence,
        promotion_status=status,
        tags=["company"],
    )


class TestSelectiveEligibility:
    def test_promotes_only_promoted_provenanced_long_term_records(self, memory_store, vault):
        vault.mkdir(parents=True)
        eligible = _make_record()
        not_promoted_status = _make_record(status=MemoryPromotionStatus.PROVISIONAL)
        not_long_term = _make_record(layer=MemoryLayer.RESEARCH_HISTORY)
        no_provenance = _make_record(with_provenance=False)

        _run_coro(memory_store.store(eligible))
        _run_coro(memory_store.store(not_promoted_status))
        _run_coro(memory_store.store(not_long_term))
        _run_coro(memory_store.store(no_provenance))

        result = promote_eligible_memories(memory_store, vault_root=vault)
        assert result.notes_created == 1
        # Only the fully-provenanced, PROMOTED, long-term record is promoted;
        # the PROVISIONAL and non-long-term records are filtered out entirely,
        # and the no-provenance record is counted as skipped.
        assert result.notes_skipped >= 1
        assert len(result.created_paths) == 1

        # The note carries provenance + derived-knowledge markers.
        note = Path(result.created_paths[0]).read_text(encoding="utf-8")
        assert "type: argus-knowledge" in note
        assert "memory_id:" in note
        assert "sources:" in note
        assert "confidence: 0.95" in note
        assert "Derived knowledge" in note or "derived knowledge" in note

    def test_promotion_is_idempotent(self, memory_store, vault):
        vault.mkdir(parents=True)
        _run_coro(memory_store.store(_make_record()))

        first = promote_eligible_memories(memory_store, vault_root=vault)
        assert first.notes_created == 1

        second = promote_eligible_memories(memory_store, vault_root=vault)
        assert second.notes_created == 0
        assert second.notes_skipped == 1

    def test_missing_vault_is_skipped(self, memory_store, vault):
        # Vault does not exist yet -> nothing written, no error.
        result = promote_eligible_memories(memory_store, vault_root=vault)
        assert result.notes_created == 0
        assert result.notes_skipped == 0


class TestProvenance:
    def test_note_keeps_supporting_chunk_ids_and_source_query(self, memory_store, vault):
        vault.mkdir(parents=True)
        rec = _make_record(subject="Alpha")
        chunk_id = rec.supporting_chunk_ids[0]
        _run_coro(memory_store.store(rec))

        result = promote_eligible_memories(memory_store, vault_root=vault)
        assert result.notes_created == 1
        note = Path(result.created_paths[0]).read_text(encoding="utf-8")
        assert chunk_id in note
        assert "source_query" in note
        assert "what did Acme acquire?" in note