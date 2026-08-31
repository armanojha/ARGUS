"""Base class for OpenAI-compatible HTTP providers (Phase 00.3).

Handles everything that is common across providers exposing an
OpenAI-shaped `/chat/completions` endpoint: HTTP client lifecycle,
payload construction (including Pydantic -> JSON Schema conversion for
structured output), response parsing, HTTP-error -> normalized-error
mapping, and retry with exponential backoff + jitter.

Concrete providers (e.g. `groq.py`) only need to supply connection
details (base URL, API key, default model, capabilities) — they should
not need to override any of the request/response handling below
unless a provider has a genuine wire-format quirk.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
from pydantic import BaseModel, TypeAdapter

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers.exceptions import (
    AuthenticationError,
    ContextLengthError,
    LLMProviderError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutOrNetworkError,
)
from app.llm_gateway.providers.models import (
    CompletionResponse,
    FunctionCall,
    Message,
    Tool,
    ToolCall,
    ToolChoice,
    Usage,
)
from app.logging_config import get_logger

logger = get_logger("argus.llm_gateway.provider")

CHAT_COMPLETIONS_PATH = "/chat/completions"


class OpenAICompatibleProvider:
    """Base for providers exposing an OpenAI-compatible `/chat/completions` endpoint.

    Implements the `LLMProvider` protocol structurally (see
    `app.llm_gateway.providers.base.LLMProvider`) — it does not inherit
    from it, since `LLMProvider` is a `Protocol` and structural typing
    is sufficient here.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        name: str,
        capabilities: ProviderCapabilities | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            # Never construct a provider with an empty key silently; the
            # factory is responsible for raising ConfigurationError with
            # a clearer message before we ever get here, but this guards
            # against direct construction too.
            raise ValueError("api_key must not be empty")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._name = name
        self._capabilities = capabilities or ProviderCapabilities()
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=5.0, read=self._timeout, write=10.0, pool=5.0),
            )
        return self._client

    # -- request construction -------------------------------------------------

    def _build_payload(
        self,
        messages: list[Message],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        response_format: type[BaseModel] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": [m.model_dump(exclude_none=True, mode="json") for m in messages],
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if response_format is not None:
            if not self._capabilities.structured_output:
                logger.warning(
                    "structured_output_not_supported",
                    provider=self._name,
                    model=payload["model"],
                )
            else:
                schema = TypeAdapter(response_format).json_schema()
                # Strict JSON-schema mode (used by most OpenAI-compatible
                # providers, including Groq and Gemini) requires every
                # property to be listed as required and additionalProperties
                # to be false. Pydantic only marks truly-required fields and
                # doesn't set additionalProperties by default, so enforce both
                # here rather than relying on every caller's model config.
                props = schema.get("properties", {})
                if props:
                    schema["required"] = sorted(props.keys())
                schema.setdefault("additionalProperties", False)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                }

        if tools is not None:
            if not self._capabilities.tool_calling:
                logger.warning(
                    "tool_calling_not_supported", provider=self._name, model=payload["model"]
                )
            else:
                payload["tools"] = [t.model_dump(exclude_none=True, mode="json") for t in tools]
                if tool_choice is not None:
                    tc = tool_choice.model_dump(exclude_none=True, mode="json")
                    # OpenAI-compatible APIs take tool_choice as either a
                    # bare string ("auto"/"none"/"required") or an object
                    # {"type": "function", "function": {"name": ...}}.
                    payload["tool_choice"] = (
                        tc["type"] if tool_choice.function is None else tc
                    )

        return payload

    # -- response parsing -------------------------------------------------

    def _parse_response(self, data: dict[str, Any], model: str) -> CompletionResponse:
        # Some OpenAI-compatible gateways (e.g. OpenCode Zen) occasionally
        # return HTTP 200 with a non-standard body that has no "choices"
        # (e.g. an upstream error envelope). Normalize that instead of
        # crashing with a raw KeyError so the router can fall back properly.
        if not isinstance(data, dict) or not data.get("choices"):
            raise ProviderUnavailableError(
                "Provider returned a malformed response: body missing 'choices'",
                status_code=200,
                provider=self._name,
            )
        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls: list[ToolCall] | None = None
        if msg.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in msg["tool_calls"]
            ]

        usage = None
        if data.get("usage"):
            u = data["usage"]
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        return CompletionResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            model=data.get("model", model),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            provider=self._name,
        )

    # -- error normalization -------------------------------------------------

    def _normalize_error(self, response: httpx.Response) -> LLMProviderError:
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except (ValueError, TypeError, AttributeError):
            error_msg = response.text

        status = response.status_code
        lower_msg = error_msg.lower() if isinstance(error_msg, str) else ""

        if status in (401, 403):
            return AuthenticationError(error_msg, status_code=status, provider=self._name)
        if status == 429:
            return RateLimitError(error_msg, status_code=status, provider=self._name)
        if status == 404:
            return ModelNotFoundError(error_msg, status_code=status, provider=self._name)
        if "context length" in lower_msg or "context_length" in lower_msg or "maximum context" in lower_msg:
            return ContextLengthError(error_msg, status_code=status, provider=self._name)
        if 500 <= status < 600:
            return ProviderUnavailableError(error_msg, status_code=status, provider=self._name)
        return LLMProviderError(
            "PROVIDER_ERROR", error_msg, status_code=status, provider=self._name
        )

    # -- retry -------------------------------------------------

    async def _request_with_retry(
        self, payload: dict[str, Any], *, timeout: float
    ) -> httpx.Response:
        client = await self._ensure_client()
        last_error: LLMProviderError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    CHAT_COMPLETIONS_PATH, json=payload, timeout=timeout
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = TimeoutOrNetworkError(str(exc), provider=self._name)
                if attempt >= self._max_retries:
                    raise last_error from exc
                wait = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "llm_retry_network_error",
                    provider=self._name,
                    attempt=attempt + 1,
                    wait_seconds=round(wait, 2),
                )
                await asyncio.sleep(wait)
                continue

            if (response.status_code == 429 or 500 <= response.status_code < 600) and attempt < self._max_retries:
                wait = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "llm_retry_http_error",
                    provider=self._name,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                    wait_seconds=round(wait, 2),
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code >= 400:
                raise self._normalize_error(response)

            return response

        # Only reachable if every retry hit a retryable HTTP status.
        if last_error is not None:
            raise last_error
        raise LLMProviderError(
            "MAX_RETRIES_EXCEEDED",
            "All retry attempts failed",
            retryable=False,
            provider=self._name,
        )

    # -- public API -------------------------------------------------

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
        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        logger.info(
            "llm_request",
            provider=self._name,
            model=resolved_model,
            call_type=call_type,
            request_id=request_id,
            has_tools=tools is not None,
            has_response_format=response_format is not None,
        )

        response = await self._request_with_retry(payload, timeout=timeout)
        data = response.json()
        result = self._parse_response(data, resolved_model)
        result = result.model_copy(update={"request_id": request_id})

        logger.info(
            "llm_response",
            provider=self._name,
            model=result.model,
            finish_reason=result.finish_reason,
            usage=result.usage.model_dump() if result.usage else None,
            request_id=request_id,
        )

        return result

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
