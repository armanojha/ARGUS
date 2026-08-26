"""Provider/model capability declarations (Phase 00.3).

A `ProviderCapabilities` instance declares what a given provider/model
actually supports, so the base HTTP provider can degrade gracefully
(warn instead of silently sending unsupported request fields) rather
than assuming every provider supports everything.

This is deliberately data-only: no discovery, no ranking, no
capability-based model selection. Phase 07 may extend this dataclass
with fields like `latency_p50` / `quality_score`, but selection logic
based on those fields is out of scope until Phase 07.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declares what a provider/model combination supports."""

    structured_output: bool = True
    """Supports `response_format: {type: json_schema, ...}` (strict JSON mode)."""

    tool_calling: bool = True
    """Supports OpenAI-style `tools` / `tool_choice`."""

    streaming: bool = False
    """Supports server-sent-event streaming responses. Not used until Phase 07+."""

    vision: bool = False
    """Supports image/multimodal input. Not used until Phase 11."""

    max_context_tokens: int = 131_072
    max_output_tokens: int = 8_192

    supports_parallel_tools: bool = True
    requires_tool_choice_auto: bool = False
