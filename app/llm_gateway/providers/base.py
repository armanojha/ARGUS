"""Gateway-facing provider interface (Phase 00.3).

`LLMProvider` is the one interface every layer above "concrete
provider" is allowed to depend on. Application code must never import
a concrete provider (e.g. `GroqProvider`) directly — only this
Protocol, via the router / `get_router()`.

A `Protocol` (rather than an ABC) is used deliberately: it allows
structural subtyping, so `tests/mocks/mock_provider.py` can satisfy the
interface for unit tests without inheriting from anything, and
`isinstance()` checks work via `@runtime_checkable`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.models import (
    CompletionResponse,
    Message,
    Tool,
    ToolChoice,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic interface. Implementations translate to/from native wire formats."""

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. `"groq"`. Matches `configs/providers.yaml` entry name."""
        ...

    @property
    def default_model(self) -> str:
        """Model used when `complete(..., model=None)`."""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """What this provider/model combination actually supports."""
        ...

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
        """Run a single completion call.

        `call_type` is accepted and passed through today but unused for
        routing decisions in Phase 00.3 — it exists so Phase 07's
        capability/quota-aware router can be introduced without
        changing this interface. Same for `request_id`: accepted for
        propagation into logs, not used for behavior in 00.3.
        """
        ...

    async def aclose(self) -> None:
        """Release any held resources (e.g. the underlying HTTP client)."""
        ...
