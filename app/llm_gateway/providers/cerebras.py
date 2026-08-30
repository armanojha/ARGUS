"""Cerebras provider (Phase 07).

Cerebras exposes an OpenAI-compatible endpoint at
`https://api.cerebras.ai/v1`.

As of 2026-08, the free trial requires a credit card ($5/30 days).
Fast inference on custom wafer-scale hardware for GPT-OSS models.

Model IDs verified at implementation time against Cerebras docs.
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_DEFAULT_MODEL = "gpt-oss-120b"


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras: fast inference on wafer-scale hardware, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = CEREBRAS_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or CEREBRAS_BASE_URL,
            default_model=model,
            name="cerebras",
            capabilities=ProviderCapabilities(
                structured_output=True,
                tool_calling=True,
                streaming=False,
                max_context_tokens=131_072,
                max_output_tokens=8_192,
            ),
            timeout=timeout,
            max_retries=max_retries,
        )


register_provider("cerebras", CerebrasProvider)