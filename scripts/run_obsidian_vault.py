#!/usr/bin/env python
"""Obsidian vault integration CLI (Phase 05 + Phase 08/09).

Runs the full-Obsidian runtime paths that Phase 12 would otherwise expose:
vault ingestion (Phase 05/09 classification), vault-memory sync (Phase 08/09),
and hypothesis research + write-back (Phase 09.2 / Phase 04 verification cross-check).

Usage:
    python scripts/run_obsidian_vault.py <vault_root> [--ingest] [--sync-memory] [--research]
    python scripts/run_obsidian_vault.py <vault_root> --all

Flags default to off; ``--all`` is equivalent to all three.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import get_settings
from app.evidence.store import get_evidence_store
from app.integrations.obsidian.contracts import set_obsidian_extension_factory
from app.integrations.obsidian.factory import ObsidianExtensionFactory
from app.integrations.obsidian.ingestion import ObsidianIngestionResult, ingest_obsidian_vault
from app.logging_config import configure_logging, get_logger

logger = get_logger("argus.run_obsidian_vault")


async def _sync_vault_memory(vault_root: Path) -> dict[str, int]:
    """Sync vault notes into the Phase 08 VAULT_MEMORY layer (Phase 09 alignment)."""
    from app.memory.factory import initialize_memory_system
    from app.memory.store import get_memory_store

    await initialize_memory_system()
    from app.integrations.obsidian.alignment import VaultMemoryCoordinator

    coordinator = VaultMemoryCoordinator(memory_store=get_memory_store())
    return await coordinator.sync_vault_memory(str(vault_root))


async def _run_research(vault_root: Path) -> list[object]:
    """Classify -> convert -> research -> write back for every hypothesis note."""
    from app.integrations.obsidian.research import ObsidianResearchCoordinator

    coordinator = ObsidianResearchCoordinator(vault_root)
    outcomes = await coordinator.process_vault()
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Obsidian vault ingestion / memory sync / hypothesis research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vault_root", type=Path, help="Path to the Obsidian vault.")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest the vault (classification + graph alignment, Phase 05/09).",
    )
    parser.add_argument(
        "--sync-memory",
        action="store_true",
        help="Sync vault notes into the VAULT_MEMORY layer (Phase 08/09).",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run hypothesis research with Phase 04 verification cross-check (Phase 09.2).",
    )
    parser.add_argument("--all", action="store_true", help="Run ingest + sync-memory + research.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    vault_root = args.vault_root
    if not vault_root.exists() or not vault_root.is_dir():
        logger.error("vault_root_not_found", path=str(vault_root))
        return 1

    want_ingest = args.ingest or args.all
    want_sync = args.sync_memory or args.all
    want_research = args.research or args.all
    if not (want_ingest or want_sync or want_research):
        parser.error("at least one of --ingest / --sync-memory / --research (or --all) is required")

    settings = get_settings()
    settings.log_level = args.log_level
    settings.obsidian_enabled = True
    settings.obsidian_full_enabled = True
    settings.memory_enabled = True

    configure_logging(settings)

    # Wire the concrete Phase 09 factory so shared-interface consumers get
    # real components instead of the no-op default.
    set_obsidian_extension_factory(
        ObsidianExtensionFactory(vault_root, settings.obsidian_write_back_root)
    )

    if want_ingest:
        result: ObsidianIngestionResult = ingest_obsidian_vault(
            vault_root,
            incremental=settings.obsidian_incremental_sync,
            store=get_evidence_store(),
        )
        logger.info(
            "vault_ingested",
            new=result.notes_new,
            updated=result.notes_updated,
            unchanged=result.notes_unchanged,
            chunks_created=result.chunks_created,
            classified=result.notes_classified,
        )

    if want_sync:
        stats = asyncio.run(_sync_vault_memory(vault_root))
        logger.info("vault_memory_sync_complete", **stats)

    if want_research:
        outcomes = asyncio.run(_run_research(vault_root))
        statuses: dict[str, int] = {}
        for o in outcomes:
            statuses[o.status] = statuses.get(o.status, 0) + 1
        logger.info("hypothesis_research_complete", outcomes=len(outcomes), statuses=statuses)

    print("\nVault integration complete!")
    print(f"  Vault: {vault_root}")
    print(f"  Ingest: {want_ingest}  |  Memory sync: {want_sync}  |  Research: {want_research}")
    return 0


if __name__ == "__main__":
    sys.exit(main())