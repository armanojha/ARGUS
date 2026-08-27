"""Deterministic mock provider for unit tests (Phase 00.3).

Zero network calls, fully controllable responses. Satisfies the
`LLMProvider` protocol structurally.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.base import LLMProvider
from app.llm_gateway.providers.exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMProviderError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutOrNetworkError,
)
from app.llm_gateway.providers.models import (
    CompletionResponse,
    Message,
    Tool,
    ToolChoice,
    Usage,
)


class MockProvider(LLMProvider):
    """In-memory mock provider for unit tests.

    Allows pre-programming responses per model, simulating errors,
    and inspecting call history.
    """

    def __init__(
        self,
        name: str = "mock",
        default_model: str = "mock-model",
        responses: dict[str, CompletionResponse] | None = None,
        should_fail: bool = False,
        fail_with: type[LLMProviderError] | None = None,
        fail_message: str = "mock failure",
    ) -> None:
        self._name = name
        self._default_model = default_model
        self._responses = responses or {}
        self._should_fail = should_fail
        self._fail_with = fail_with
        self._fail_message = fail_message
        self._call_log: list[dict[str, Any]] = []
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_output=True,
            tool_calling=True,
            streaming=False,
        )

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
        self._call_log.append({
            "messages": [m.model_dump() for m in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format.__name__ if response_format else None,
            "tools": [t.model_dump() for t in tools] if tools else None,
            "tool_choice": tool_choice.model_dump() if tool_choice else None,
            "timeout": timeout,
            "call_type": call_type,
            "request_id": request_id,
        })

        if self._should_fail and self._fail_with:
            raise self._fail_with(self._fail_message)

        key = model or self.default_model
        if key in self._responses:
            resp = self._responses[key]
            if request_id:
                resp = resp.model_copy(update={"request_id": request_id})
            return resp

        # Default response
        if response_format:
            # Return valid JSON for the schema (minimal valid instance)
            return CompletionResponse(
                content='{"result": "mock structured output"}',
                model=key,
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                provider=self._name,
                request_id=request_id,
            )

        return CompletionResponse(
            content="Mock response",
            model=key,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider=self._name,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        self._closed = True

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self._call_log

    @property
    def closed(self) -> bool:
        return self._closed


# Convenience exception factories for tests
def mock_auth_error() -> AuthenticationError:
    return AuthenticationError("invalid api key", provider="mock", status_code=401)


def mock_rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limit exceeded", provider="mock", status_code=429)


def mock_model_not_found_error() -> ModelNotFoundError:
    return ModelNotFoundError("model not found", provider="mock", status_code=404)


def mock_provider_unavailable_error() -> ProviderUnavailableError:
    return ProviderUnavailableError("service unavailable", provider="mock", status_code=503)


def mock_timeout_error() -> TimeoutOrNetworkError:
    return TimeoutOrNetworkError("request timed out", provider="mock")


def mock_config_error() -> ConfigurationError:
    return ConfigurationError("missing api key", provider="mock")