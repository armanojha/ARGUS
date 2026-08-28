"""Phase 09 acceptance: vault backup independence (V3 §14).

The Obsidian vault must be backed up separately from the ARGUS indexes.
Deleting / recreating the ARGUS stores (evidence.db, memory.db, graph.pkl)
must never lose vault content.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.evidence.store import EvidenceStore
from app.graph.store import EvidenceGraphStore
from app.integrations.obsidian.alignment import VaultGraphAligner, VaultMemoryCoordinator
from app.integrations.obsidian.classifier import RuleBasedObsidianClassifier
from app.integrations.obsidian.ingestion import ObsidianIngestionPipeline
from app.memory.store import MemoryStore

ARGUS_ARTEFACTS = [
    "evidence.db",
    "bm25.pkl",
    "faiss.index",
    "memory.db",
    "graph.pkl",
    "manifest.pkl",
]


def _write_vault(vault: Path) -> None:
    (vault / "Personal").mkdir(parents=True)
    (vault / "Personal" / "source.md").write_text(
        "---\nurl: https://example.com/report\nauthor: Jane Doe\n---\n\n# Source\n\nPrimary source data.\n",
        encoding="utf-8",
    )
    (vault / "Personal" / "hypothesis.md").write_text(
        "# Hunch\n\nI hypothesize that evidence density accelerates insight.\n",
        encoding="utf-8",
    )
    (vault / "Personal" / "plain.md").write_text(
        "# Notes\n\nJust a personal note about the vault.\n",
        encoding="utf-8",
    )


def _assert_argus_artefacts(data_dir: Path, expected: bool) -> None:
    for name in ARGUS_ARTEFACTS:
        assert (data_dir / name).exists() is expected, f"{name} exists={expected}"


class TestVaultBackupIndependence:
    async def test_vault_independent_of_argus_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vault = root / "vault"
            backup = root / "vault-backup"
            data_dir = root / "argus"
            data_dir.mkdir()

            _write_vault(vault)
            note_files = {p.name for p in vault.rglob("*.md")}

            # 1. Ingest into ARGUS stores + align graph + sync vault memory.
            store = EvidenceStore(
                db_path=data_dir / "evidence.db",
                bm25_index_path=data_dir / "bm25.pkl",
                faiss_index_path=data_dir / "faiss.index",
            )
            graph_store = EvidenceGraphStore(
                graph_path=data_dir / "graph.pkl",
                evidence_store=store,
            )
            memory_store = MemoryStore(db_path=data_dir / "memory.db")

            pipeline = ObsidianIngestionPipeline(
                vault,
                store=store,
                manifest_path=data_dir / "manifest.pkl",
                classifier=RuleBasedObsidianClassifier(),
                aligner=VaultGraphAligner(graph_store),
                enable_hypothesis_objectives=True,
            )
            result = pipeline.ingest_vault(incremental=False)
            assert result.notes_new == 3

            coordinator = VaultMemoryCoordinator(
                memory_store=memory_store,
                aligner=VaultGraphAligner(graph_store),
            )
            stats = await coordinator.sync_vault_memory(str(vault))
            assert stats["synced"] == 3

            store.close()
            memory_store.close()
            _assert_argus_artefacts(data_dir, expected=True)
            assert {p.name for p in vault.rglob("*.md")} == note_files

            # 2. Back up the vault independently of the ARGUS indexes.
            shutil.copytree(vault, backup)
            assert {p.name for p in backup.rglob("*.md")} == note_files

            # 3. Delete ALL ARGUS indexes: vault content and backup survive.
            for name in ARGUS_ARTEFACTS:
                (data_dir / name).unlink(missing_ok=True)
            _assert_argus_artefacts(data_dir, expected=False)
            assert {p.name for p in vault.rglob("*.md")} == note_files
            assert {p.name for p in backup.rglob("*.md")} == note_files

            # 4. Recreating the ARGUS indexes must not need vault content.
            fresh_store = EvidenceStore(
                db_path=data_dir / "evidence.db",
                bm25_index_path=data_dir / "bm25.pkl",
                faiss_index_path=data_dir / "faiss.index",
            )
            fresh_pipeline = ObsidianIngestionPipeline(
                backup,
                store=fresh_store,
                manifest_path=data_dir / "manifest.pkl",
                classifier=RuleBasedObsidianClassifier(),
                enable_hypothesis_objectives=True,
            )
            re = fresh_pipeline.ingest_vault(incremental=False)
            assert re.notes_new == 3
            assert {p.name for p in backup.rglob("*.md")} == note_files
            fresh_store.close()
            _assert_argus_artefacts(data_dir, expected=True)