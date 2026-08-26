"""LLM Router (Phase 00.3).

00.3 scope: single-provider passthrough. Phase 07 will extend this with
capability/quota-aware routing, call-type policy execution, and
provider-level fallback chains.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.llm_gateway.providers.base import LLMProvider
from app.llm_gateway.providers.models import CompletionResponse, Message, Tool, ToolChoice


class LLMRouter:
    """Routes completion calls to the configured provider.

    Phase 00.3: wraps a single provider. Phase 07 will add:
    - call-type -> model policy resolution
    - provider capability/quota awareness
    - primary/fallback execution chains
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

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
        """Delegate to the underlying provider."""
        return await self._provider.complete(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            call_type=call_type,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        """Release provider resources."""
        await self._provider.aclose()

    @property
    def provider(self) -> LLMProvider:
        return self._provider