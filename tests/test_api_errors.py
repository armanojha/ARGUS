"""Tests for the Phase 00.2 consistent error envelope."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_404_returns_error_envelope_with_request_id():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/nonexistent")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "HTTP_ERROR"
    assert body["error"]["request_id"] is not None
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]


def test_405_method_not_allowed_returns_error_envelope():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/health", json={})

    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == "HTTP_ERROR"
    assert body["error"]["request_id"] is not None


def test_validation_error_returns_422_envelope():
    app = create_app()

    @app.get("/_test/validated")
    async def _validated(required_param: int):  # pragma: no cover - trivial handler
        return {"required_param": required_param}

    with TestClient(app) as client:
        response = client.get("/_test/validated")

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"] == "Request validation failed"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["request_id"] is not None


def test_unhandled_exception_returns_500_envelope():
    app = create_app()

    @app.get("/_test/boom")
    async def _boom():  # pragma: no cover - trivial handler
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "An unexpected error occurred"
    assert body["error"]["request_id"] is not None
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]


def test_error_envelope_request_id_matches_response_header_on_404():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/nonexistent", headers={"X-Request-ID": "req_fixed123"})

    assert response.headers["X-Request-ID"] == "req_fixed123"
    assert response.json()["error"]["request_id"] == "req_fixed123"
