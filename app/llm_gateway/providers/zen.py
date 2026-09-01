"""OpenCode Zen provider (Phase 07 extension).

OpenCode Zen is OpenCode's curated AI gateway (`https://opencode.ai/docs/zen`).
It exposes an OpenAI-compatible `https://opencode.ai/zen/v1/chat/completions`
endpoint (base URL `https://opencode.ai/zen/v1` + `/chat/completions`), which
the `OpenAICompatibleProvider` base class drives directly.

Model IDs verified at implementation time (2026-08-30) against the official
OpenCode Zen docs / pricing page:

- MiMo-V2.5 Free         -> `mimo-v2.5-free`
- Nemotron 3 Ultra Free  -> `nemotron-3-ultra-free`
- Nemotron 3.5 Lightning Free -> `nemotron-3.5-lightning-free`

All three are free (`$0` per 1M tokens) "for a limited time" as of 2026-08-30.
The Nemotron models are NVIDIA free-hosted endpoints: trial use only, do not
submit personal or confidential data; session data may be logged for
security/improvement purposes (per NVIDIA API Trial Terms of Service).

Auth: the gateway expects an API key from `https://opencode.ai/auth` supplied
via the `OPENCODE_ZEN_API_KEY` environment variable (`api_key_env` in
`configs/providers.yaml`).

NOTE (verify with a live key): structured-output / tool-calling support on the
NVIDIA-hosted free endpoints is declared optimistically here (matching the
other free-tier providers). If a live call rejects `response_format` or
`tools`, flip the corresponding capability flag via this class (see D-014).
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_DEFAULT_MODEL = "nemotron-3-ultra-free"


class ZenProvider(OpenAICompatibleProvider):
    """OpenCode Zen: OpenCode's curated gateway, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = ZEN_DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        attempt_ceiling_s: float = 15.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or ZEN_BASE_URL,
            default_model=model,
            name="zen",
            capabilities=ProviderCapabilities(
                structured_output=True,
                tool_calling=True,
                streaming=False,
                max_context_tokens=131_072,
                max_output_tokens=8_192,
            ),
            timeout=timeout,
            max_retries=max_retries,
            attempt_ceiling_s=attempt_ceiling_s,
        )


register_provider("zen", ZenProvider)