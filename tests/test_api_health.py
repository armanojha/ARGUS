"""Tests for the Phase 00.2 /health endpoint."""

from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_health_returns_200_with_expected_shape():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == "0.1.0"
    assert body["environment"] == "development"
    # Timestamp should be ISO 8601 and parseable.
    datetime.fromisoformat(body["timestamp"])


def test_health_environment_matches_settings(monkeypatch):
    monkeypatch.setenv("ARGUS_ENV", "test")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["environment"] == "test"


def test_health_includes_request_id_header():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req_")


def test_health_no_auth_required():
    """/health must be reachable without any credentials (pure liveness probe)."""
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code != 401
    assert response.status_code != 403
