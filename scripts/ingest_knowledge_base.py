#!/usr/bin/env python
"""Clean entry point to sync the user Knowledge Base into ARGUS.

Recursively ingests the configured knowledge-base directory
(``settings.knowledge_base_path`` or ``--knowledge-base``) into the
EvidenceStore using the existing IngestionPipeline, then refreshes the
retrieval indexes. Idempotent: unchanged files are skipped via the
content-checksum dedup already built into the pipeline.

Usage:
    python scripts/ingest_knowledge_base.py [--knowledge-base PATH]
        [--db-path PATH] [--no-rebuild-indexes] [--log-level LEVEL]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.knowledge_base import ingest_knowledge_base, supported_extensions
from app.logging_config import configure_logging, get_logger

logger = get_logger("argus.ingest_knowledge_base")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Sync the user Knowledge Base into the ARGUS evidence store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=settings.knowledge_base_path,
        help="Root directory of the user knowledge base (overrides config).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to SQLite evidence database (overrides config).",
    )
    parser.add_argument(
        "--no-rebuild-indexes",
        action="store_true",
        help="Skip BM25/FAISS index refresh after ingestion.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    if args.db_path:
        settings.evidence_db_path = args.db_path
    settings.log_level = args.log_level
    configure_logging(settings)

    root = args.knowledge_base
    if not root.exists():
        logger.error("kb_dir_not_found", path=str(root))
        print(f"\nKnowledge base directory not found: {root}")
        return 1
    if not root.is_dir():
        logger.error("kb_dir_not_directory", path=str(root))
        print(f"\nNot a directory: {root}")
        return 1

    print(f"\nSyncing knowledge base: {root}")
    print(f"Supported types: {', '.join(supported_extensions())}")

    result = ingest_knowledge_base(
        root=root,
        rebuild_indexes=not args.no_rebuild_indexes,
    )

    print("\nIngestion complete!")
    print(f"  Ingested:     {result.ingested}")
    print(f"  Unchanged:    {result.unchanged}")
    print(f"  Errors:       {result.errors}")
    print(f"  Indexed:      {result.indexed} ({result.indexed_chunks} chunks)")
    print(f"  Duration:     {result.duration_s}s")

    for doc in result.documents_ingested:
        print(f"    + {doc['filename']} (v{doc['version']})")
    for path in result.error_paths:
        print(f"    ! {path}")
    if result.errors > 0:
        print(f"\n{result.errors} file(s) failed; see logs for details.")

    return 0


if __name__ == "__main__":
    sys.exit(main())