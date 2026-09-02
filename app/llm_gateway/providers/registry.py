"""Provider registry — owns provider instantiation and lifecycle (HARDEN-06.5.1).

Separates two concerns that historically lived together in
``MultiModelRouter``:

* **Routing / selection** (``MultiModelRouter``) — pick the best
  provider/model for a call type, run the fallback chain, record telemetry.
* **Provider lifecycle** (this module) — instantiate configured providers
  (or accept injected ones), cache them lazily, and cleanly release their
  resources (HTTP clients, connections) on shutdown.

Nothing outside this module should call provider-construction or
per-provider ``aclose()``. A router receives a registry and asks it for
providers by name; the router never touches provider creation or teardown.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings, load_providers_config, get_settings
from app.llm_gateway.providers.base import LLMProvider
from app.llm_gateway.providers.exceptions import ConfigurationError
from app.logging_config import get_logger

logger = get_logger("argus.llm_gateway.providers.registry")


class ProviderRegistry:
    """Lazily creates, caches, and closes LLM provider instances.

    Provider instances are created on first use and cached by name. Injected
    providers (tests, embedding pipeline, single-provider path) take
    precedence over config-driven creation. Call ``close_all()`` once when the
    owning process/router shuts down to release every provider's resources.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: dict[str, LLMProvider] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Injected providers (tests / explicit wiring) win over config. Copied
        # so clearing/closing the registry never mutates the caller's dict.
        self._injected: dict[str, LLMProvider] = dict(providers or {})
        # Fully-built (or injected) instances, keyed by provider name.
        self._cache: dict[str, LLMProvider] = {}
        # Config-driven provider names that are enabled but not yet built.
        self._known_unbuilt: set[str] = set()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Register injected + enabled config-driven providers (no eager build)."""
        if self._initialized:
            return

        for name, provider in self._injected.items():
            self._cache.setdefault(name, provider)

        data = load_providers_config(self._settings)
        for entry in data.get("providers", []) or []:
            if not entry.get("enabled", False):
                continue
            provider_name = entry.get("name")
            if not provider_name or provider_name in self._cache:
                continue
            self._known_unbuilt.add(provider_name)

        self._initialized = True

    def get(self, provider_name: str) -> LLMProvider | None:
        """Return a cached/injected provider, or None if it is unknown/unusable.

        Config-driven providers are created on first access (lazy). Creation
        failure (missing key, unknown provider, init error) is logged and
        returns None so the router can simply skip that candidate.
        """
        self._ensure_initialized()

        cached = self._cache.get(provider_name)
        if cached is not None:
            return cached

        if provider_name in self._known_unbuilt:
            provider = self._build_from_config(provider_name)
            if provider is not None:
                self._cache[provider_name] = provider
                self._known_unbuilt.discard(provider_name)
            else:
                # Do not retry a broken provider on every request.
                self._known_unbuilt.discard(provider_name)
            return provider

        return None

    def _build_from_config(self, provider_name: str) -> LLMProvider | None:
        """Instantiate a config-driven provider by name (best-effort, lazy)."""
        from app.llm_gateway.providers import get_provider_factory

        factory = get_provider_factory(provider_name)
        if factory is None:
            logger.warning("registry_unknown_provider", provider=provider_name)
            return None

        data = load_providers_config(self._settings)
        config: dict[str, Any] = {}
        for entry in data.get("providers", []) or []:
            if entry.get("name") == provider_name:
                config = entry
                break

        api_key_env = str(config.get("api_key_env") or f"{provider_name.upper()}_API_KEY")
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            logger.warning(
                "registry_missing_key",
                provider=provider_name,
                env_var=api_key_env,
            )
            return None

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.get("timeout", self._settings.llm_timeout),
            "max_retries": config.get("max_retries", self._settings.llm_max_retries),
            "attempt_ceiling_s": self._settings.llm_attempt_ceiling_s,
        }
        if config.get("default_model"):
            kwargs["model"] = config["default_model"]
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]

        try:
            return factory(**kwargs)  # type: ignore[return-value]
        except (ConfigurationError, ValueError, OSError) as exc:
            logger.warning(
                "registry_init_failed",
                provider=provider_name,
                error=str(exc),
            )
            return None

    def all(self) -> list[LLMProvider]:
        """All currently-cached provider instances (built/injected only)."""
        self._ensure_initialized()
        return list(self._cache.values())

    async def close_all(self) -> None:
        """Close every provider and fully reset the registry (shutdown).

        Both injected and config-built providers are released and the registry
        is cleared. After this call ``get()``/``all()`` return nothing/empty
        (a fresh registry is needed to rebuild). Safe to call multiple times;
        one provider's teardown failure never prevents the rest from closing.
        """
        for provider in self._cache.values():
            try:
                await provider.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "registry_provider_close_failed",
                    provider=provider.name,
                    error=str(exc),
                )
        self._cache.clear()
        self._known_unbuilt.clear()
        self._injected.clear()
        self._initialized = False