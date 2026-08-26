"""Liveness health check endpoint (Phase 00.2).

Pure liveness probe: no dependency on the database, vector store, or any
external service. That kind of "readiness" check (verifying downstream
dependencies) belongs to a later phase once those dependencies exist.
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


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        environment=settings.env,
        timestamp=datetime.now(UTC),
    )
