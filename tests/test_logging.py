"""Tests for the Phase 00.2 structured JSON logging + request ID binding."""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from app.api.main import create_app


def _parse_json_lines(raw: str) -> list[dict]:
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def test_request_produces_json_log_with_request_id(capsys):
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    request_id = response.headers["X-Request-ID"]

    captured = capsys.readouterr()
    records = _parse_json_lines(captured.out)

    assert records, "expected at least one JSON log record on stdout"
    request_records = [r for r in records if r.get("request_id") == request_id]
    assert request_records, "expected a log record carrying the request's request_id"

    finished = [r for r in request_records if r.get("event") == "request_finished"]
    assert finished, "expected a request_finished log record"
    assert finished[0]["status_code"] == 200
    assert finished[0]["path"] == "/health"


def test_concurrent_requests_have_distinct_request_ids(capsys):
    app = create_app()
    results: list[str] = []
    lock = threading.Lock()

    def _make_request(client: TestClient) -> None:
        response = client.get("/health")
        with lock:
            results.append(response.headers["X-Request-ID"])

    with TestClient(app) as client:
        threads = [threading.Thread(target=_make_request, args=(client,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(results) == 5
    assert len(set(results)) == 5, "each request should get a distinct request_id"


def test_log_records_are_valid_json_with_iso_timestamp(capsys):
    app = create_app()
    with TestClient(app) as client:
        client.get("/health")

    captured = capsys.readouterr()
    records = _parse_json_lines(captured.out)

    assert records
    for record in records:
        assert "timestamp" in record
        assert "level" in record
        assert "event" in record
