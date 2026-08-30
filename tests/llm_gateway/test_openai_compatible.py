"""Unit tests for the OpenAI-compatible provider base (Phase 00.3)."""

from __future__ import annotations

import pytest

from app.llm_gateway.providers.exceptions import ProviderUnavailableError
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