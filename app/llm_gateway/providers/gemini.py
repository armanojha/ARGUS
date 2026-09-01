"""Gemini provider (Phase 07).

Google's Gemini API exposes an OpenAI-compatible endpoint at
`https://generativelanguage.googleapis.com/v1beta/openai/`.

Free tier limits (as of 2026-08):
- gemini-3.5-flash-lite: 15 RPM, 1,000 RPD
- gemini-3.5-flash: 10 RPM, 1,000 RPD
- Data used for training on free tier (important for privacy-sensitive use cases)
- Note: 1M token context window is a major advantage for evidence-heavy workloads

Model IDs verified live 2026-08-31 against the Gemini OpenAI-compatible
endpoint: `gemini-2.5-flash-lite` and `gemini-2.5-flash` are no longer
available to new users (HTTP 404) and Google directs users to the
`gemini-3.5-flash-lite` / `gemini-3.5-flash` generation.
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini: large context window, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        attempt_ceiling_s: float = 15.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or GEMINI_BASE_URL,
            default_model=model,
            name="gemini",
            capabilities=ProviderCapabilities(
                structured_output=True,
                tool_calling=True,
                streaming=False,
                max_context_tokens=1_048_576,
                max_output_tokens=8_192,
            ),
            timeout=timeout,
            max_retries=max_retries,
            attempt_ceiling_s=attempt_ceiling_s,
        )


register_provider("gemini", GeminiProvider)