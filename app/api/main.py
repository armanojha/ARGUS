"""FastAPI application factory (Phase 00.2 + 01 + 02).

Boots the FastAPI app used by the ARGUS API surface.

Phase 01 adds the retrieval endpoint. Phase 02 adds the agentic query
endpoint (plan/retrieve/synthesize loop over Phase 01 retrieval).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.brain import router as brain_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.knowledge_base import router as knowledge_base_router
from app.api.middleware import RequestIDMiddleware
from app.api.obsidian import router as obsidian_router
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
    if settings.memory_enabled:
        from app.memory import initialize_memory_system

        await initialize_memory_system()
    yield
    logger.info("app_shutdown")
    # HARDEN-06.5.6: release providers/HTTP clients, quota state, and memory.
    from app.runtime import shutdown_runtime

    await shutdown_runtime()


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
    app.include_router(knowledge_base_router)
    app.include_router(brain_router)
    app.include_router(obsidian_router)

    # ARGUS Brain knowledge-graph UI (standalone page served by the API).
    # Mounted at /brain; static assets under app/ui/brain.
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    brain_dir = Path(__file__).resolve().parent.parent / "ui" / "brain"

    @app.get("/brain", include_in_schema=False)
    def brain_index() -> FileResponse:
        return FileResponse(str(brain_dir / "index.html"))

    if brain_dir.exists():
        app.mount("/brain/static", StaticFiles(directory=str(brain_dir)), name="brain-static")

    return app


# Default app instance for `uvicorn app.api.main:app`. The `--factory` form
# (`uvicorn app.api.main:create_app --factory`) is preferred for tests.
app = create_app()