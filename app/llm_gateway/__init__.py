"""LLM Gateway public interface (Phase 00.3).

Application code imports from here, never from concrete providers.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm_gateway.providers.factory import create_provider
from app.llm_gateway.routing.router import LLMRouter

_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Get or create the singleton LLM router.

    The router wraps the configured provider (from `configs/providers.yaml`
    + `ARGUS_LLM_PROVIDER` env var). Created on first call.
    """
    global _router
    if _router is None:
        provider = create_provider(get_settings())
        _router = LLMRouter(provider)
    return _router


async def close_router() -> None:
    """Close the router and release provider resources (HTTP client)."""
    global _router
    if _router is not None:
        await _router.aclose()
        _router = None


__all__ = ["close_router", "get_router"]