"""Groq provider (Phase 00.3).

Groq exposes an OpenAI-compatible endpoint at
`https://api.groq.com/openai/v1` and offers a genuine no-credit-card
free tier. Chosen per D-005 as the first (and, for 00.3, only) real
provider to prove the abstraction end-to-end.

Model IDs verified at implementation time (2026-08-26) against Groq's
own docs (`console.groq.com/docs/models`, `.../docs/deprecations`,
`.../docs/structured-outputs`):

- Default/primary model: `openai/gpt-oss-120b` (note the `openai/`
  prefix — the Architect's plan omitted it; bare `gpt-oss-120b` is not
  a valid Groq model ID).
- Fallback model (config only, not used by the 00.3 router stub):
  `openai/gpt-oss-20b`.
- Both are current **production** models with strict-mode structured
  outputs (`response_format: {type: json_schema, strict: true}`) and
  tool calling support.
- Groq deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
  (announced 2026-06-17); GPT-OSS models are the recommended
  replacements, consistent with the Architect's plan.
- Free-tier limits for GPT-OSS models specifically are **30 RPM /
  1,000 RPD / 8,000 TPM / 200,000 TPD** (per-model, org-level). This is
  materially different from the Architect's plan, which stated "30
  RPM, 14,400 RPD" — that higher RPD figure belongs to the deprecated
  `llama-3.1-8b-instant`, not GPT-OSS. Reported to the Architect/
  Reviewer per the D-005 verification requirement; does not block
  implementation since the provider/model themselves are available and
  capable, only the documented quota number needed correcting.
"""

from __future__ import annotations

from app.llm_gateway.capabilities import ProviderCapabilities
from app.llm_gateway.providers import register_provider
from app.llm_gateway.providers.openai_compatible import OpenAICompatibleProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"


class GroqProvider(OpenAICompatibleProvider):
    """Groq: fast inference on custom LPU hardware, OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_DEFAULT_MODEL,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            default_model=model,
            name="groq",
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


register_provider("groq", GroqProvider)
