"""Health check endpoints (Phase 00.2).

Liveness probe: pure liveness, no dependency checks.
Readiness probe: verifies downstream dependencies (evidence store, vector index).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

router = APIRouter()

APP_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    version: str = Field(default=APP_VERSION)
    environment: str
    timestamp: datetime


class ReadinessCheck(BaseModel):
    name: str
    status: Literal["ok", "error"]
    message: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    version: str = Field(default=APP_VERSION)
    timestamp: datetime
    checks: list[ReadinessCheck]


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness probe: returns healthy if the process is running."""
    return HealthResponse(
        environment=settings.env,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def ready(settings: Settings = Depends(get_settings)) -> ReadinessResponse:
    """Readiness probe: verifies downstream dependencies are accessible."""
    checks: list[ReadinessCheck] = []

    # Check evidence store
    try:
        from app.evidence.store import EvidenceStore

        store = EvidenceStore()
        stats = store.stats()
        checks.append(ReadinessCheck(
            name="evidence_store",
            status="ok",
            message=f"{stats.get('chunk_count', 0)} chunks indexed",
        ))
    except Exception as e:  # noqa: BLE001 - readiness check must not crash
        checks.append(ReadinessCheck(
            name="evidence_store",
            status="error",
            message=str(e),
        ))

    # Check vector index
    try:
        from app.retrieval.vector import get_embedding_model

        get_embedding_model()
        checks.append(ReadinessCheck(
            name="vector_index",
            status="ok",
            message="Embedding model loaded",
        ))
    except Exception as e:  # noqa: BLE001 - readiness check must not crash
        checks.append(ReadinessCheck(
            name="vector_index",
            status="error",
            message=str(e),
        ))

    # Check LLM providers
    try:
        from app.config import get_router

        router_instance = get_router()
        providers = list(router_instance.providers.keys()) if hasattr(router_instance, "providers") else []
        checks.append(ReadinessCheck(
            name="llm_providers",
            status="ok",
            message=f"{len(providers)} providers configured",
        ))
    except Exception as e:  # noqa: BLE001 - readiness check must not crash
        checks.append(ReadinessCheck(
            name="llm_providers",
            status="error",
            message=str(e),
        ))

    all_ok = all(c.status == "ok" for c in checks)
    return ReadinessResponse(
        status="ready" if all_ok else "not_ready",
        timestamp=datetime.now(UTC),
        checks=checks,
    )
