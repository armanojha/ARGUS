"""Z.ai (Zhipu AI) provider (Phase 21).

Z.ai provides OpenAI-compatible API access to the GLM model family.
Base URL: https://api.z.ai/api/paas/v4

Free models verified 2026-09-06 via live API testing:

- GLM-4.5-Flash -> `glm-4.5-flash`  (free, tool calling, structured output)
- GLM-4.6V-Flash -> `glm-4.6v-flash` (free, vision, structured output, no tool calling)
- GLM-4.7-Flash -> `glm-4.7-flash`  (free, tool calling, aggressive rate limits)

All three are permanently free ($0). Paid models (glm-5.x) require balance.

NOTE: glm-4.7-flash has aggressive rate limits and ~33s latency under load.
glm-4.5-flash is the recommended free model (5.5s avg, tool calling + structured output).
glm-4.6v-flash is a vision model with fast latency (4.9s) but no tool calling support.

Auth: API key from https://z.ai (Z.AI Open Platform) supplied via
the `ZAI_API_KEY` environment variable (`api_key_env` in configs/providers.yaml).
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_DEFAULT_MODEL = "glm-4.5-flash"


class ZAIProvider(OpenAICompatibleProvider):
    """Z.ai (Zhipu AI): OpenAI-compatible API with GLM models."""

    def __init__(
        self,
        api_key: str,
        model: str = ZAI_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        attempt_ceiling_s: float = 15.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or ZAI_BASE_URL,
            default_model=model,
            name="zai",
            capabilities=ProviderCapabilities(
                structured_output=True,
                tool_calling=True,
                streaming=False,
                max_context_tokens=128_000,
                max_output_tokens=4_096,
            ),
            timeout=timeout,
            max_retries=max_retries,
            attempt_ceiling_s=attempt_ceiling_s,
        )


register_provider("zai", ZAIProvider)
