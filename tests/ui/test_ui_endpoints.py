"""Lightweight smoke tests for the Phase 12.1 verification endpoint.

Deliberately minimal (per Phase 12 testing guidance): verifies the
endpoint is registered and that the UI-facing verify seam degrades
gracefully when no verifier is available — both cheap and deterministic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.evidence.store import EvidenceStore
from app.graph.store import EvidenceGraphStore
from app.llm_gateway.providers.exceptions import ProviderUnavailableError
from app.llm_gateway.routing.router import LLMRouter
from app.verification.models import VerificationStatus
from tests.mocks.mock_provider import MockProvider


def _make_app_client(tmp: Path, failing_provider: MockProvider) -> TestClient:
    from app.api.main import create_app

    evidence_store = EvidenceStore(
        db_path=tmp / "evidence.db",
        bm25_index_path=tmp / "bm25.pkl",
        faiss_index_path=tmp / "faiss.index",
    )
    graph_store = EvidenceGraphStore(graph_path=tmp / "graph.pkl", evidence_store=evidence_store)

    import app.api.verification as api_verify

    api_verify.get_evidence_store = lambda: evidence_store  # type: ignore[assignment]
    api_verify.get_graph_store = lambda: graph_store  # type: ignore[assignment]
    api_verify.get_router = lambda: LLMRouter(failing_provider)  # type: ignore[assignment]

    return TestClient(create_app())


def _all_route_paths(app) -> set[str]:
    """Collect route paths, descending into included routers.

    Newer FastAPI/Starlette versions keep ``include_router()`` entries as
    opaque ``_IncludedRouter`` markers in ``app.routes`` — they have no
    ``.path`` themselves. Their real routes live in
    ``.original_router.routes`` (prefixes already applied at definition).
    Blindly assuming ``.path`` raises AttributeError.
    """
    paths: set[str] = set()

    def _walk(routes) -> None:
        for r in routes:
            p = getattr(r, "path", None)
            if p is not None:
                paths.add(p)
            nested = getattr(r, "routes", None)
            if nested:
                _walk(nested)
            original = getattr(r, "original_router", None)
            if original is not None:
                _walk(getattr(original, "routes", []))

    _walk(app.routes)
    return paths


def test_verify_route_registered():
    from app.api.main import create_app

    client = TestClient(create_app())
    paths = _all_route_paths(client.app)
    assert "/api/v1/verify" in paths


def test_verify_gracefully_degrades_without_verifier():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        failing = MockProvider(should_fail=True, fail_with=ProviderUnavailableError, fail_message="no provider")
        client = _make_app_client(tmp, failing)

        response = client.post(
            "/api/v1/verify",
            json={
                "claim_text": "Acme acquired Beta in 2024.",
                "supporting_chunk_ids": [],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == VerificationStatus.ERROR.value
    assert body["confidence"] == 0.0
    assert "verification call failed" in body["reasoning"]