"""Provider/model capability declarations (Phase 00.3 + Phase 07 extension).

A `ProviderCapabilities` instance declares what a given provider/model
actually supports, so the base HTTP provider can degrade gracefully
(warn instead of silently sending unsupported request fields) rather
than assuming every provider supports everything.

Phase 07 extends this with fields for routing decisions:
latency, quality, cost, quota, etc. Selection logic based on these
fields is implemented in Phase 07's router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declares what a provider/model combination supports.

    Phase 00.3 fields (structural capabilities):
    - structured_output, tool_calling, streaming, vision
    - max_context_tokens, max_output_tokens
    - supports_parallel_tools, requires_tool_choice_auto

    Phase 07 fields (routing/selection metadata):
    - latency_p50_ms, quality_score, cost_per_1k_tokens
    - quota_remaining, quota_reset_seconds
    - supported_call_types, preferred_call_types
    """

    # --- Phase 00.3: Structural capabilities ---
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

    # --- Phase 07: Routing/selection metadata ---
    latency_p50_ms: int | None = None
    """Median latency in milliseconds. None = unknown."""

    quality_score: float | None = None
    """Quality score 0.0-1.0. None = unknown."""

    speed_class: str | None = None
    """Coarse speed classification. One of 'fast' | 'medium' | 'slow' | None.

    None = not locally benchmarked (do not treat as a fact). Only set where we
    have a measured or provider-documented basis; this is informational
    metadata, never used for autonomous selection (policy stays explicit).
    """

    cost_per_1k_input_tokens: float | None = None
    """Cost per 1K input tokens in USD. None = unknown/free."""

    cost_per_1k_output_tokens: float | None = None
    """Cost per 1K output tokens in USD. None = unknown/free."""

    quota_remaining: int | None = None
    """Remaining quota (requests or tokens) in current window. None = unknown/unlimited."""

    quota_reset_seconds: int | None = None
    """Seconds until quota resets. None = unknown/no reset."""

    supported_call_types: list[str] = field(default_factory=list)
    """Call types this provider/model is explicitly configured for. Empty = all."""

    preferred_call_types: list[str] = field(default_factory=list)
    """Call types this provider/model is preferred for. Empty = no preference."""

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary additional metadata for future extensions."""


# Default capabilities for the Groq provider (Phase 00.3 default)
GROQ_CAPABILITIES = ProviderCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=True,
    vision=False,
    max_context_tokens=131_072,
    max_output_tokens=8_192,
    supports_parallel_tools=True,
    requires_tool_choice_auto=False,
    # Phase 07 metadata (example values - to be configured per model)
    latency_p50_ms=500,
    quality_score=0.85,
    cost_per_1k_input_tokens=0.0,
    cost_per_1k_output_tokens=0.0,
    quota_remaining=None,
    quota_reset_seconds=None,
    supported_call_types=[
        "general",
        "query_analysis",
        "research_planning",
        "evidence_extraction",
        "reasoning",
        "synthesis",
        "verification",
        "revision",
    ],
    preferred_call_types=["query_analysis", "research_planning", "synthesis"],
)

# Default capabilities for the Gemini provider (Phase 07)
GEMINI_CAPABILITIES = ProviderCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=False,
    vision=False,
    max_context_tokens=1_048_576,
    max_output_tokens=8_192,
    supports_parallel_tools=True,
    requires_tool_choice_auto=False,
    # Phase 07 metadata
    latency_p50_ms=800,
    quality_score=0.88,
    cost_per_1k_input_tokens=0.0,
    cost_per_1k_output_tokens=0.0,
    quota_remaining=None,
    quota_reset_seconds=None,
    supported_call_types=[
        "general",
        "query_analysis",
        "research_planning",
        "evidence_extraction",
        "reasoning",
        "synthesis",
        "verification",
        "revision",
    ],
    preferred_call_types=["query_analysis", "research_planning", "synthesis", "verification", "revision"],
)

# Default capabilities for the Cerebras provider (Phase 07)
CEREBRAS_CAPABILITIES = ProviderCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=False,
    vision=False,
    max_context_tokens=131_072,
    max_output_tokens=8_192,
    supports_parallel_tools=True,
    requires_tool_choice_auto=False,
    # Phase 07 metadata
    latency_p50_ms=300,
    quality_score=0.85,
    cost_per_1k_input_tokens=0.0,
    cost_per_1k_output_tokens=0.0,
    quota_remaining=None,
    quota_reset_seconds=None,
    supported_call_types=[
        "general",
        "query_analysis",
        "research_planning",
        "evidence_extraction",
        "reasoning",
        "synthesis",
        "verification",
        "revision",
    ],
    preferred_call_types=["evidence_extraction", "reasoning", "synthesis"],
)

# Default capabilities for the OpenCode Zen provider (Phase 07 extension, D-014)
# Free NVIDIA-hosted endpoints: Nemotron 3 Ultra / 3.5 Lightning / MiMo-V2.5.
# Latency/quality left as None (not benchmarked locally); costs are $0 (free tier).
ZEN_CAPABILITIES = ProviderCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=False,
    vision=False,
    max_context_tokens=131_072,
    max_output_tokens=8_192,
    supports_parallel_tools=True,
    requires_tool_choice_auto=False,
    cost_per_1k_input_tokens=0.0,
    cost_per_1k_output_tokens=0.0,
    quota_remaining=None,
    quota_reset_seconds=None,
    supported_call_types=[
        "general",
        "query_analysis",
        "research_planning",
        "evidence_extraction",
        "reasoning",
        "synthesis",
        "verification",
        "revision",
    ],
    preferred_call_types=["query_analysis", "research_planning", "reasoning", "synthesis", "verification"],
)

# Capability registry for Phase 07 router
CAPABILITY_REGISTRY: dict[str, ProviderCapabilities] = {
    "groq": GROQ_CAPABILITIES,
    "gemini": GEMINI_CAPABILITIES,
    "cerebras": CEREBRAS_CAPABILITIES,
    "zen": ZEN_CAPABILITIES,
}


def get_capabilities(provider_name: str, model: str | None = None) -> ProviderCapabilities:
    """Get capabilities for a provider/model combination.

    Phase 07 will extend this to support model-specific capabilities.
    """
    return CAPABILITY_REGISTRY.get(provider_name, ProviderCapabilities())


def register_capabilities(provider_name: str, capabilities: ProviderCapabilities) -> None:
    """Register capabilities for a provider. Phase 07 uses this."""
    CAPABILITY_REGISTRY[provider_name] = capabilities