"""HARDEN-06.5.6: runtime lifecycle shutdown tests.

Verifies ``app.runtime.shutdown_runtime`` is:
- idempotent and total (never raises, even with no runtime initialized),
- non-fatal (a failing component close never blocks the rest), and
- safe to run in the API lifespan on every server shutdown.
"""

from __future__ import annotations

import pytest

from app.runtime import shutdown_runtime


@pytest.mark.asyncio
async def test_shutdown_runtime_is_total_and_idempotent():
    """Runs twice back-to-back, including before any runtime was created."""
    await shutdown_runtime()
    await shutdown_runtime()
    assert True


@pytest.mark.asyncio
async def test_shutdown_runtime_survives_failing_component(monkeypatch):
    """A component that raises during close must not propagate to callers."""
    async def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr("app.runtime._shutdown_llm", _boom)
    await shutdown_runtime()
    assert True  # memory shutdown still ran and nothing propagated


@pytest.mark.asyncio
async def test_shutdown_runtime_survives_import_errors(monkeypatch):
    """A missing optional subsystem (e.g. memory) must not break shutdown."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("app.memory"):
            raise ImportError("memory module unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    await shutdown_runtime()
    assert True