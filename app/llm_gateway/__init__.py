"""LLM Gateway public interface (Phase 00.3 + Phase 07).

Application code imports from here, never from concrete providers.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm_gateway.providers.factory import create_provider
from app.llm_gateway.routing.multi_model_router import MultiModelRouter
from app.llm_gateway.routing.router import LLMRouter

_router: LLMRouter | MultiModelRouter | None = None


def get_router() -> LLMRouter | MultiModelRouter:
    """Get or create the singleton LLM router.

    Phase 07: Returns MultiModelRouter if `multimodel_enabled` is True,
    otherwise returns the legacy LLMRouter for backward compatibility.
    """
    global _router
    if _router is None:
        settings = get_settings()
        if settings.multimodel_enabled:
            _router = MultiModelRouter(settings)
        else:
            provider = create_provider(settings)
            _router = LLMRouter(provider)
    return _router


async def close_router() -> None:
    """Close the router and release provider resources (HTTP client)."""
    global _router
    if _router is not None:
        await _router.aclose()
        _router = None


__all__ = ["close_router", "get_router"]