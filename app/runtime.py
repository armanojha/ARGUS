"""Runtime lifecycle management (HARDEN-06.5.6).

Central, single-responsibility shutdown orchestration for the long-lived
resources the API server owns. The FastAPI lifespan (and any process that
embeds ARGUS) calls ``shutdown_runtime()`` on exit so HTTP clients, quota
state, and the memory system are released deterministically instead of
relying on interpreter teardown.

Rules:
- Graceful and idempotent: every component is nulled from its module-level
  singleton before being closed, so a second call is a no-op and concurrent
  re-creation after shutdown is safe (all getters are lazy).
- Non-fatal: a failure in one component never blocks the rest of shutdown.
- Scope limited: only resources created through this package's public
  singletons are closed. The evidence store / retrievers are excluded
  deliberately because they are wired as lazy singletons that tests reuse
  across requests and closing them mid-process is surprising.
"""

from __future__ import annotations

from app.logging_config import get_logger

logger = get_logger("argus.runtime")


async def shutdown_runtime() -> None:
    """Release all long-lived ARGUS runtime resources (idempotent, non-fatal)."""
    try:
        await _shutdown_llm()
    except Exception as exc:  # noqa: BLE001 - shutdown must never propagate
        logger.warning("runtime_shutdown_llm_failed", error=str(exc))

    try:
        await _shutdown_memory()
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_shutdown_memory_failed", error=str(exc))


async def _shutdown_llm() -> None:
    """Close the LLM router (=> all providers/HTTP clients) and quota tracker."""
    try:
        from app.llm_gateway import close_router

        await close_router()
    except Exception as exc:  # noqa: BLE001 - shutdown must never propagate
        logger.warning("runtime_llm_router_close_failed", error=str(exc))

    try:
        from app.llm_gateway.quota import close_quota_tracker

        await close_quota_tracker()
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_quota_close_failed", error=str(exc))


async def _shutdown_memory() -> None:
    """Close the memory system if it was initialized (optional, non-fatal)."""
    try:
        from app.memory import shutdown_memory_system

        # Shutdown is best-effort; the memory factory clears its own singleton.
        await shutdown_memory_system()
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_memory_shutdown_failed", error=str(exc))


__all__ = ["shutdown_runtime"]