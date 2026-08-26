"""FastAPI application factory (Phase 00.2).

Boots the FastAPI app used by the ARGUS API surface. Scope for 00.2 is
deliberately minimal: app factory, structured logging, request ID
middleware, consistent error handling, and the `/health` liveness
endpoint.

Explicitly NOT wired up here (deferred to later phases): LLM gateway,
retrieval, evidence graph, orchestration, authentication, rate limiting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestIDMiddleware
from app.config import get_settings
from app.logging_config import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
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

    return app


# Default app instance for `uvicorn app.api.main:app`. The `--factory` form
# (`uvicorn app.api.main:create_app --factory`) is preferred for tests.
app = create_app()
