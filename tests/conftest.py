"""Shared pytest fixtures for ARGUS tests.

`pythonpath = ["."]` in pyproject.toml makes the `app` package importable
without installation, so this file currently only needs to exist as the
test-session anchor for future fixtures (DB, vault, provider mocks, etc.
added in later phases).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure get_settings() cache doesn't leak state between tests."""
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
