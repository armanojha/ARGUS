"""FastAPI application factory (Phase 00.2 + 01 + 02).

Boots the FastAPI app used by the ARGUS API surface.

Phase 01 adds the retrieval endpoint. Phase 02 adds the agentic query
endpoint (plan/retrieve/synthesize loop over Phase 01 retrieval).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestIDMiddleware
from app.api.orchestration import router as orchestration_router
from app.api.retrieval import router as retrieval_router
from app.api.telemetry import router as telemetry_router
from app.api.verification import router as verification_router
from app.config import get_settings
from app.llm_gateway.telemetry import set_telemetry_persistence_dir
from app.logging_config import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    set_telemetry_persistence_dir(settings.data_dir / "telemetry")
    logger = get_logger("argus.startup")
    logger.info("app_startup", environment=settings.env, log_level=settings.log_level)
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    """Build and return the ARGUS FastAPI application."""
    app = FastAPI(title="ARGUS", version="0.1.0", lifespan=lifespan)

    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(retrieval_router)
    app.include_router(orchestration_router)
    app.include_router(verification_router)
    app.include_router(telemetry_router)

    return app


# Default app instance for `uvicorn app.api.main:app`. The `--factory` form
# (`uvicorn app.api.main:create_app --factory`) is preferred for tests.
app = create_app()