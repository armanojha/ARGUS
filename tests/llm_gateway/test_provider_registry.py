"""Tests for ProviderRegistry (HARDEN-06.5.1).

Verifies the separation of provider lifecycle from routing:
- injected providers take precedence and need no config
- config-driven providers are built lazily / tolerate missing keys
- unknown providers return None (router can skip them)
- close_all() releases every provider and is idempotent
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm_gateway.providers.registry import ProviderRegistry
from tests.mocks.mock_provider import MockProvider


@pytest.fixture
def settings() -> Settings:
    return Settings(
        multimodel_enabled=True,
        multimodel_providers_config_path="configs/model_policy.yaml",
    )


def test_injected_providers_available_and_not_from_config(settings):
    injected = {
        "groq": MockProvider(name="groq", default_model="openai/gpt-oss-120b"),
    }
    registry = ProviderRegistry(settings=settings, providers=injected)

    prov = registry.get("groq")
    assert prov is injected["groq"]


def test_unknown_provider_returns_none(settings):
    registry = ProviderRegistry(settings=settings)
    # No provider qualifies as this name via config (no keys set, and not
    # injected) -> get() returns None instead of raising.
    assert registry.get("nonexistent_provider") is None


def test_get_is_lazy_does_not_eagerly_build(settings):
    """Config-driven providers are reserved but not instantiated until get()."""
    registry = ProviderRegistry(settings=settings)
    # Empty cache before any get() — no eager constructor side effects.
    assert registry.all() == []


@pytest.mark.asyncio
async def test_close_all_releases_and_is_idempotent(settings):
    providers = {
        "groq": MockProvider(name="groq"),
        "gemini": MockProvider(name="gemini"),
    }
    registry = ProviderRegistry(settings=settings, providers=providers)

    # Force both into the cache.
    assert registry.get("groq") is not None
    assert registry.get("gemini") is not None

    await registry.close_all()
    assert providers["groq"].closed is True
    assert providers["gemini"].closed is True
    assert registry.all() == []

    # Second close is a no-op (no providers cached), does not raise.
    await registry.close_all()


@pytest.mark.asyncio
async def test_missing_api_key_returns_none_without_raising(settings, monkeypatch):
    """A config provider with no API key yields None (router skips it)."""
    # 'cerebras' is enabled in providers.yaml and has no key in this env.
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    registry = ProviderRegistry(settings=settings)
    got = registry.get("cerebras")
    assert got is None