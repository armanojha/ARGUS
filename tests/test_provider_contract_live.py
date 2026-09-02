"""HARDEN-06.5.7: live provider-contract suite.

An explicit, opt-in suite that validates the ``LLMProvider`` contract
(see ``app/llm_gateway/providers/base.py``) uniformly against *every*
configured provider whose API key is present, rather than a hand-written
per-provider test.

It is NOT part of the normal commit-time test run. Execute it deliberately:

    RUN_LIVE_LLM_TESTS=1 pytest tests/test_provider_contract_live.py -v

Each provider that is enabled in ``configs/providers.yaml`` and has its
``api_key_env`` variable exported becomes a parameter. Providers without a
key are skipped (not failed) so the matrix adapts to what is configured.

What it verifies per provider:
  * structural conformance to ``LLMProvider`` (a ``runtime_checkable`` Protocol);
  * ``name`` / ``default_model`` / ``capabilities`` surface;
  * a real ``complete()`` call returns a normalized ``CompletionResponse``
    carrying usage, a provider string, and a model string — the cross-provider
    wire contract the gateway's routing/fusion layers depend on.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from app.config import load_providers_config, get_settings
from app.llm_gateway.providers.base import LLMProvider
from app.llm_gateway.providers.models import CompletionResponse, Message, MessageRole

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_LLM_TESTS"),
    reason="Set RUN_LIVE_LLM_TESTS=1 to run the live provider-contract suite",
)


def _configured_provider_params():
    """(name, id) params for every enabled provider that has its API key set."""
    data = load_providers_config(get_settings())
    params = []
    for entry in (data.get("providers", []) or []):
        if not entry.get("enabled", False):
            continue
        name = entry.get("name")
        env_var = str(entry.get("api_key_env") or f"{name.upper()}_API_KEY")
        if not env_var or not os.getenv(env_var):
            params.append(pytest.param(
                name,
                marks=pytest.mark.skip(reason=f"no {env_var} set"),
            ))
        else:
            params.append(name)
    return params


# Build the parametrization at import time (config + env are static here).
_PROVIDERS = _configured_provider_params()


class _JsonResponse(BaseModel):
    answer: str
    confidence: float


@pytest.mark.parametrize("provider_name", _PROVIDERS)
class TestProviderContract:
    """Validate the LLMProvider contract against each configured live provider."""

    def test_structural_conformance(self, provider_name):
        """The built provider structurally satisfies LLMProvider (Protocol)."""
        provider = self._build(provider_name)
        assert isinstance(provider, LLMProvider), f"{provider_name} does not conform to LLMProvider"
        assert provider.name == provider_name
        assert provider.default_model

    def test_completion_returns_normalized_response(self, provider_name):
        """A live completion returns the gateway's canonical CompletionResponse."""
        import asyncio

        provider = self._build(provider_name)
        resp = asyncio.run(provider.complete(
            [Message(role=MessageRole.USER, content="Reply with exactly: pong")],
            temperature=0.0,
            max_tokens=16,
            timeout=max(30.0, provider.capabilities.max_output_tokens or 8192),
        ))
        assert isinstance(resp, CompletionResponse)
        assert resp.content, "expected non-empty content from provider"
        assert resp.provider == provider_name
        assert resp.model
        assert resp.usage is not None and resp.usage.total_tokens > 0

    def test_structured_output_contract(self, provider_name):
        """Providers advertising structured output honor the response_format contract."""
        import asyncio

        provider = self._build(provider_name)
        if not provider.capabilities.structured_output:
            pytest.skip("provider does not advertise structured output")
        resp = asyncio.run(provider.complete(
            [Message(role=MessageRole.USER, content='Return JSON: {"answer": "pong", "confidence": 0.9}')],
            response_format=_JsonResponse,
            max_tokens=32,
        ))
        parsed = _JsonResponse.model_validate_json(resp.content)
        assert parsed.answer.strip().lower() == "pong"
        assert parsed.confidence == 0.9

    @staticmethod
    def _build(provider_name: str) -> LLMProvider:
        """Build the provider exactly as the registry would (shared path)."""
        from app.llm_gateway.providers import get_provider_factory

        data = load_providers_config(get_settings())
        for entry in (data.get("providers", []) or []):
            if entry.get("name") == provider_name:
                cfg = entry
                break
        else:
            raise AssertionError(f"provider {provider_name} not in config")

        from app.config import Settings

        settings = Settings(_env_file=None, llm_timeout=float(cfg.get("timeout", 30.0)))
        api_key = os.getenv(str(cfg.get("api_key_env") or f"{provider_name.upper()}_API_KEY"))
        factory = get_provider_factory(provider_name)
        assert factory is not None, f"no factory registered for {provider_name}"
        return factory(
            api_key=api_key,
            model=cfg.get("default_model"),
            timeout=float(cfg.get("timeout", 30.0)),
            max_retries=int(cfg.get("max_retries", 2)),
        )