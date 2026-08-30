"""Thin HTTP client for the ARGUS API (Phase 12.1 evidence-explorer UI).

Presentation-layer convenience: wraps the existing API endpoints
(`/api/v1/query`, `/api/v1/verify`, `/health`) so the Streamlit app never
touches HTTP/JSON plumbing. No business logic lives here — retrieval,
orchestration, verification, routing, and memory all stay server-side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_API_BASE_URL = os.getenv("ARGUS_API_BASE_URL", "http://localhost:8000")


class ARGUSAPIClientError(Exception):
    """Raised when the ARGUS API returns a non-2xx response or is unreachable."""


@dataclass
class ARGUSAPIClient:
    """Small synchronous client for the ARGUS API (one session, reused)."""

    base_url: str = DEFAULT_API_BASE_URL
    timeout: float = 120.0
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def query(self, query_text: str, *, user_early_stop: bool = False) -> dict[str, Any]:
        """Run one query through the agentic loop (`POST /api/v1/query`)."""
        response = self._session.post(
            f"{self.base_url}/api/v1/query",
            json={"query": query_text, "user_early_stop": user_early_stop},
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def verify(
        self,
        claim_text: str,
        supporting_chunk_ids: list[str],
        *,
        contradicting_chunk_ids: list[str] | None = None,
        entity_names: list[str] | None = None,
        temporal_context: str | None = None,
    ) -> dict[str, Any]:
        """Verify a claim against the evidence store (`POST /api/v1/verify`)."""
        response = self._session.post(
            f"{self.base_url}/api/v1/verify",
            json={
                "claim_text": claim_text,
                "supporting_chunk_ids": supporting_chunk_ids,
                "contradicting_chunk_ids": contradicting_chunk_ids or [],
                "entity_names": entity_names or [],
                "temporal_context": temporal_context,
            },
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def health(self) -> dict[str, Any]:
        """Return the API health payload (`GET /health`)."""
        response = self._session.get(f"{self.base_url}/health", timeout=10.0)
        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
            message = (payload.get("error") or {}).get("message") or response.text
        except ValueError:
            message = response.text
        raise ARGUSAPIClientError(f"{response.status_code}: {message}")