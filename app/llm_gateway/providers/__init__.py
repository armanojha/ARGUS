"""Provider registry (Phase 00.3).

A simple name -> class registry. Concrete provider modules (e.g.
`groq.py`) call `register_provider()` at import time. The factory
(`factory.py`) looks providers up here by the `name` field from
`configs/providers.yaml` / `ARGUS_LLM_PROVIDER`.

Adding a new provider = write the class + call `register_provider()` +
add a `configs/providers.yaml` entry. No changes to the factory,
router, or gateway public interface are required (see the "Future
providers must be addable without changing the gateway's public
interface" rule in the Phase 00.3 brief).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

ProviderFactory = Callable[..., "OpenAICompatibleProvider"]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider constructor under `name`. Overwrites silently on re-import."""
    _REGISTRY[name] = factory


def get_provider_factory(name: str) -> ProviderFactory | None:
    return _REGISTRY.get(name)


def registered_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = ["get_provider_factory", "register_provider", "registered_providers"]
