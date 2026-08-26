"""Shared pytest fixtures for ARGUS tests.

`pythonpath = ["."]` in pyproject.toml makes the `app` package importable
without installation, so this file currently only needs to exist as the
test-session anchor for future fixtures (DB, vault, provider mocks, etc.
added in later phases).
"""

from __future__ import annotations

import pytest

from app.llm_gateway.routing.router import LLMRouter
from tests.mocks.mock_provider import MockProvider


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure get_settings() cache doesn't leak state between tests."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_provider() -> MockProvider:
    """Provide a fresh MockProvider for each test."""
    return MockProvider()


@pytest.fixture
def mock_router(mock_provider: MockProvider) -> LLMRouter:
    """Provide an LLMRouter wrapping the mock provider."""
    return LLMRouter(mock_provider)


@pytest.fixture(autouse=True)
def _reset_router():
    """Reset the global router singleton before/after each test."""
    import app.llm_gateway as gateway

    gateway._router = None
    yield
    gateway._router = None