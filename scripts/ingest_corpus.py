#!/usr/bin/env python
"""Ingest/sync a document corpus into the ARGUS evidence store.

Drops PDFs/TXT/Markdown/spreadsheets into a directory and runs one command to
sync them into the EvidenceStore. Idempotent: unchanged files are skipped via
content checksums, so repeated runs are incremental and safe.

Usage:
    python scripts/ingest_corpus.py [corpus_directory] [--db-path PATH] [--bm25-index PATH] [--faiss-index PATH] [--rebuild-indexes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.evidence.store import EvidenceStore
from app.ingestion.pipeline import ingest_corpus_directory
from app.logging_config import configure_logging, get_logger
from app.retrieval.bm25 import BM25Retriever, assign_bm25_doc_ids
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vector import FAISSVectorStore, assign_embedding_indices

logger = get_logger("argus.ingest_corpus")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a corpus of documents into the ARGUS evidence store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "corpus_dir",
        nargs="?",
        type=Path,
        default=Path("./knowledge_base"),
        help="Path to directory containing documents to ingest (.pdf, .txt, .md). "
        "Defaults to ./knowledge_base.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to SQLite evidence database",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        help="Path to BM25 index file",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        help="Path to FAISS vector index file",
    )
    parser.add_argument(
        "--rebuild-indexes",
        action="store_true",
        help="Force rebuild of BM25 and FAISS indexes after ingestion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Validate corpus directory
    corpus_dir = args.corpus_dir
    if not corpus_dir.exists():
        logger.error("corpus_dir_not_found", path=str(corpus_dir))
        return 1
    if not corpus_dir.is_dir():
        logger.error("corpus_dir_not_directory", path=str(corpus_dir))
        return 1

    # Load settings and override with CLI args
    settings = get_settings()
    if args.db_path:
        settings.evidence_db_path = args.db_path
    if args.bm25_index:
        settings.bm25_index_path = args.bm25_index
    if args.faiss_index:
        settings.faiss_index_path = args.faiss_index
    settings.log_level = args.log_level

    # Configure logging
    configure_logging(settings)

    logger.info("ingestion_started", corpus_dir=str(corpus_dir))

    # Initialize store
    store = EvidenceStore(
        db_path=settings.evidence_db_path,
        bm25_index_path=settings.bm25_index_path,
        faiss_index_path=settings.faiss_index_path,
    )

    # Run ingestion
    documents = ingest_corpus_directory(corpus_dir, store)

    if not documents:
        logger.warning("no_documents_ingested", corpus_dir=str(corpus_dir))
        return 0

    logger.info("ingestion_completed", document_count=len(documents))

    # Rebuild indexes if requested
    if args.rebuild_indexes:
        logger.info("rebuilding_indexes")
        assign_bm25_doc_ids(store)
        assign_embedding_indices(store)

        # Get all chunks
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks ORDER BY document_id, ordinal"
            ).fetchall()
        if rows:
            chunk_ids = [row["id"] for row in rows]
            chunks = store.get_chunks_by_ids([__import__("uuid").UUID(cid) for cid in chunk_ids])

            # Build BM25 index
            bm25 = BM25Retriever(store, index_path=settings.bm25_index_path)
            bm25.build_index(chunks)

            # Generate embeddings and build FAISS index
            embedder = EmbeddingGenerator()
            embeddings = embedder.embed_chunks(chunks)

            vector_store = FAISSVectorStore(store, index_path=settings.faiss_index_path)
            vector_store.build_index(embeddings, [__import__("uuid").UUID(cid) for cid in chunk_ids])

            logger.info("indexes_rebuilt", chunk_count=len(chunks))

    # Print summary
    total_chunks = sum(
        len(store.get_chunks_by_document(doc.id)) for doc in documents
    )
    logger.info(
        "ingestion_summary",
        documents=len(documents),
        total_chunks=total_chunks,
        db_path=str(settings.evidence_db_path),
        bm25_index=str(settings.bm25_index_path),
        faiss_index=str(settings.faiss_index_path),
    )

    print("\nIngestion complete!")
    print(f"  Documents: {len(documents)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Database: {settings.evidence_db_path}")
    print(f"  BM25 index: {settings.bm25_index_path}")
    print(f"  FAISS index: {settings.faiss_index_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())