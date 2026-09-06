"""NVIDIA NIM provider (Phase 07).

NVIDIA Inference Microservices (NIM) exposes an OpenAI-compatible endpoint at
`https://integrate.api.nvidia.com` with 46+ free models. Key advantage for
ARGUS: 1M context window on Nemotron 3 Ultra, enabling synthesis to see all
evidence at once instead of top-K only.

Model IDs verified 2026-09-06 against
https://docs.api.nvidia.com/nim/reference/llm-apis.

- Default model: `nvidia/nemotron-3-ultra-550b-a55b` (550B total / 55B active,
  1M context, 32K output, tool calling, reasoning modes).
- Fallback: `nvidia/nemotron-3-super-120b-a12b` (120B total / 12B active,
  1M context, 32K output).

CRITICAL LIMITATION: NIM does NOT support `response_format` (structured output
with JSON schema enforcement). The `structured_output` capability is therefore
set to False. NIM is suitable only for call types that return free text
(synthesis, general) — NOT for call types requiring Pydantic-based structured
output (query_analysis, research_planning, evidence_extraction, verification).

Free-tier limits: ~40 RPM, no credit card required.
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

NIM_BASE_URL = "https://integrate.api.nvidia.com"
NIM_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NIM_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NVIDIA NIM: free models with 1M context, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = NIM_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        attempt_ceiling_s: float = 15.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or NIM_BASE_URL,
            default_model=model,
            name="nvidia_nim",
            capabilities=ProviderCapabilities(
                structured_output=False,  # NIM does NOT support response_format
                tool_calling=True,
                streaming=False,
                max_context_tokens=1_000_000,  # 1M context
                max_output_tokens=32_768,
            ),
            timeout=timeout,
            max_retries=max_retries,
            attempt_ceiling_s=attempt_ceiling_s,
        )


register_provider("nvidia_nim", NvidiaNimProvider)
