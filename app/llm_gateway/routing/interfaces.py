"""LLM Router Interface Contracts (Phase 07).

Defines the abstract interface for the LLM router that Phase 07 will implement.
Phase 06, 08, 10, 11, 12 depend on this interface for LLM access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import (
    CompletionResponse,
    Message,
    Tool,
    ToolChoice,
)


class LLMRouterInterface(ABC):
    """Abstract interface for the LLM Router.

    Phase 07 implements this. Phases 06, 08, 10, 11, 12 depend on this interface.
    """

    @property
    @abstractmethod
    def provider(self) -> Any:
        """Return the underlying provider instance."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        timeout: float = 30.0,
        call_type: str = "general",
        request_id: str | None = None,
    ) -> CompletionResponse:
        """Run a completion call through the router.

        The router handles provider selection, fallback, and routing logic.
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release router and provider resources."""
        ...

    @abstractmethod
    def get_capabilities(self, model: str | None = None) -> ProviderCapabilities:
        """Get capabilities for a specific model or the default model."""
        ...


class RouterFactoryInterface(ABC):
    """Factory interface for creating routers.

    Allows phases to create routers without depending on concrete implementation.
    """

    @abstractmethod
    def create_router(self) -> LLMRouterInterface:
        """Create a new router instance."""
        ...

    @abstractmethod
    def get_default_router(self) -> LLMRouterInterface:
        """Get the default singleton router."""
        ...


# Re-export the existing LLMRouter as the default implementation
# This maintains backward compatibility while allowing Phase 07 to replace it
from app.llm_gateway import get_router as get_default_router
from app.llm_gateway.routing.router import LLMRouter as DefaultLLMRouter

__all__ = [
    "DefaultLLMRouter",
    "LLMRouterInterface",
    "RouterFactoryInterface",
    "get_default_router",
]