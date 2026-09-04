"""Tests for ARGUS Brain API surface (graph + document resolution).

Covers the read-only document-resolution endpoint added for the Brain UI's
"Open Document" action — resolving a graph node (chunk / document / source) to
its underlying text, reusing the existing evidence store (no redesign).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.evidence.models import Chunk, Document, Source, SourceType
from app.evidence.store import EvidenceStore


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def populated_store(temp_db):
    """Evidence store with one source, one document and two chunks."""
    store = EvidenceStore(temp_db)

    source = Source(
        type=SourceType.TEXT,
        path="/test/corpus.txt",
        checksum="brain_corpus_checksum",
    )
    store.upsert_source(source)

    doc = Document(
        source_id=source.id,
        version=1,
        checksum="brain_doc_checksum",
        chunking_strategy="semantic_v1",
    )
    store.insert_document(doc)

    chunks = [
        Chunk(
            document_id=doc.id,
            ordinal=0,
            text="ARGUS is a hybrid retrieval research system built for evidence-grounded answers.",
            token_count=12,
            page_start=1,
            section_path="Overview",
        ),
        Chunk(
            document_id=doc.id,
            ordinal=1,
            text="The Brain visualizes the evidence graph as an interactive knowledge space.",
            token_count=11,
            page_start=2,
            section_path="Brain",
        ),
    ]
    store.insert_chunks(chunks)
    return store


@pytest.fixture
def client(populated_store):
    import app.evidence.store as store_module

    original = store_module.get_evidence_store
    store_module.get_evidence_store = lambda: populated_store

    app = create_app()
    with TestClient(app) as c:
        yield c

    store_module.get_evidence_store = original


def test_document_resolves_chunk(client, populated_store):
    chunk_id = populated_store.get_chunks_by_document(
        populated_store.list_documents()[0].id
    )[0].id

    res = client.get(f"/api/v1/brain/document?node_id={chunk_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["node_type"] == "chunk"
    assert body["node_id"] == str(chunk_id)
    assert "evidence-grounded answers" in body["content"]
    assert body["path"] == "/test/corpus.txt"
    assert body["page_start"] == 1


def test_document_resolves_full_document(client, populated_store):
    doc_id = populated_store.list_documents()[0].id

    res = client.get(f"/api/v1/brain/document?node_id={doc_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["node_type"] == "document"
    assert "ARGUS is a hybrid retrieval" in body["content"]
    assert "interactive knowledge space" in body["content"]
    assert body["path"] == "/test/corpus.txt"


def test_document_resolves_source(client, populated_store):
    source_id = populated_store.list_sources()[0].id

    res = client.get(f"/api/v1/brain/document?node_id={source_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["node_type"] == "source"
    assert body["path"] == "/test/corpus.txt"


def test_document_unknown_node_returns_404(client):
    res = client.get("/api/v1/brain/document?node_id=00000000-0000-0000-0000-000000000000")
    assert res.status_code == 404


def test_document_missing_param_returns_422(client):
    res = client.get("/api/v1/brain/document")
    assert res.status_code == 422


def test_node_label_claims_are_short():
    from app.api.brain import _node_label

    label = _node_label("claim", {"text": "The core idea is to turn city buses into mobile, AI-powered urban sensing units."})
    assert len(label) <= 48
    assert label.endswith("…")
    assert label == "The core idea is to turn city…"


def test_node_label_chunk_readable_snippet_and_runtogether():
    from app.api.brain import _node_label

    readable = _node_label("chunk", {"text": "Buses are turned into sensing units.", "ordinal": 3})
    assert readable == "#4 · Buses are turned into sensing…"

    run_together = _node_label("chunk", {"text": "Runtogethertextwithoutspaces", "ordinal": 0})
    assert run_together == "#1"


def test_short_label_ellipsis_only_when_truncated():
    from app.api.brain import _short_label

    assert _short_label("Edge AI") == "Edge AI"
    assert not _short_label("Edge AI").endswith("…")
    assert _short_label("") is None
    assert _short_label(None) is None

