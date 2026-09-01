"""Unit tests for the OpenAI-compatible provider base (Phase 00.3)."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app.llm_gateway.providers.exceptions import (
    ProviderUnavailableError,
    TimeoutOrNetworkError,
)
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider


class DummyProvider(OpenAICompatibleProvider):
    """Concrete subclass for testing the base class without a network call."""


def _make_provider() -> DummyProvider:
    return DummyProvider(
        name="dummy",
        base_url="https://example.invalid/v1",
        default_model="test-model",
        api_key="sk-test-not-a-real-key",
    )


def test_parse_response_valid_body():
    provider = _make_provider()
    data = {
        "id": "x",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    result = provider._parse_response(data, "test-model")
    assert result.content == "hi"
    assert result.provider == "dummy"
    assert result.model == "test-model"
    assert result.usage is not None and result.usage.total_tokens == 3


def test_parse_response_malformed_200_raises_provider_unavailable():
    """A 200 body without 'choices' (e.g. an upstream error envelope) must
    normalize to ProviderUnavailableError instead of crashing with KeyError,
    so routing fallback works."""
    provider = _make_provider()
    with pytest.raises(ProviderUnavailableError) as excinfo:
        provider._parse_response({"id": "x", "model": "test-model"}, "test-model")
    assert "choices" in str(excinfo.value)


def test_parse_response_non_dict_raises_provider_unavailable():
    provider = _make_provider()
    with pytest.raises(ProviderUnavailableError):
        provider._parse_response(["not", "a", "dict"], "test-model")


def test_request_retry_bounded_by_attempt_ceiling():
    """A hanging provider must be abandoned after the per-attempt ceiling,
    not after timeout * (retries + 1), and must surface a retryable
    TimeoutOrNetworkError so the router can fall back to another provider."""

    async def _hang(request: httpx.Request) -> httpx.Response:
        # Server hangs longer than the bounded read deadline.
        await asyncio.sleep(0.5)
        raise httpx.ReadError("mock server hung")

    provider = _make_provider()
    provider._attempt_ceiling_s = 0.3
    provider._max_retries = 2  # naively up to 3 attempts
    provider._client = httpx.AsyncClient(
        base_url="https://example.invalid/v1",
        transport=httpx.MockTransport(_hang),
        timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0),
    )

    start = time.monotonic()
    with pytest.raises(TimeoutOrNetworkError):
        asyncio.run(provider._request_with_retry({}, timeout=10.0))
    elapsed = time.monotonic() - start

    # Bounded well under the naive 3 x 10s = 30s, roughly the ceiling.
    assert elapsed < 2.0, f"attempt not bounded, took {elapsed:.2f}s"