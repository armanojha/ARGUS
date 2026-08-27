"""Provider factory (Phase 00.3).

`create_provider(settings)` is the single place that turns configuration
(`configs/providers.yaml` + environment variables via `app.config.Settings`)
into a live `LLMProvider` instance. This is explicit-config execution
only — see the CRITICAL MODEL-POLICY RULE in the Phase 00.3 brief: no
provider/model is chosen here based on capability, quota, or ranking.
The active provider name comes from `Settings.llm_provider`
(`ARGUS_LLM_PROVIDER`), which defaults to `"groq"`.
"""

from __future__ import annotations

import os

from app.config import Settings, load_providers_config
from app.llm_gateway.providers import cerebras as _cerebras  # noqa: F401
from app.llm_gateway.providers import gemini as _gemini  # noqa: F401
from app.llm_gateway.providers import get_provider_factory

# Import side effect: registers providers in the provider registry. Add
# further `import app.llm_gateway.providers.<name>` lines here as new
# providers are implemented (never remove the explicit-registration
# pattern in favor of auto-discovery — see model-policy rule).
from app.llm_gateway.providers import groq as _groq  # noqa: F401
from app.llm_gateway.providers.exceptions import ConfigurationError
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider


def _find_provider_config(settings: Settings, name: str) -> dict[str, object]:
    """Look up `name` in `configs/providers.yaml`'s `providers` list. Returns {} if absent."""
    data = load_providers_config(settings)
    for entry in data.get("providers", []) or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return {}


def create_provider(settings: Settings | None = None) -> OpenAICompatibleProvider:
    """Instantiate the configured LLM provider.

    Resolution order for the API key: the `api_key_env` named in
    `configs/providers.yaml` for this provider (falls back to
    `<PROVIDER>_API_KEY` if the config entry doesn't specify one), read
    from the process environment. Never read from YAML directly — keys
    must never be committed to config files.
    """
    from app.config import get_settings

    settings = settings or get_settings()
    provider_name = settings.llm_provider

    factory = get_provider_factory(provider_name)
    if factory is None:
        raise ConfigurationError(
            f"Unknown LLM provider '{provider_name}'. Registered providers: "
            f"{', '.join(_registered_names()) or '(none)'}"
        )

    config_entry = _find_provider_config(settings, provider_name)
    api_key_env = str(config_entry.get("api_key_env") or f"{provider_name.upper()}_API_KEY")
    api_key = os.getenv(api_key_env, "")

    if not api_key:
        raise ConfigurationError(
            f"Missing API key for provider '{provider_name}': environment variable "
            f"'{api_key_env}' is not set. Copy .env.example to .env and fill it in."
        )

    kwargs: dict[str, object] = {
        "api_key": api_key,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_model:
        kwargs["model"] = settings.llm_model

    return factory(**kwargs)  # type: ignore[return-value]


def _registered_names() -> list[str]:
    from app.llm_gateway.providers import registered_providers

    return registered_providers()
