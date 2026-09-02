"""Model policy configuration (Phase 00.3 stub, Phase 07 implementation).

This module defines the explicit model-policy configuration structure.
Phase 00.3 only provides the data structure and loading logic; the
router does not yet consume it for routing decisions. Phase 07 will
wire the router to resolve call_type -> primary/fallback models.

CRITICAL: This is EXPLICIT CONFIGURATION ONLY. No autonomous model
discovery, ranking, or capability-based selection. The project owner
decides the model assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from app.config import Settings, get_settings


@dataclass(frozen=True)
class CallTypePolicy:
    """Policy for a single call type.

    Attributes:
        primary: Primary model ID for this call type.
        fallbacks: Ordered list of fallback model IDs.
        tiers: Optional task-complexity tier -> ordered model chain
            (HARDEN-06.5.2). Keys are one of "fast" / "balanced" / "strong".
            When a tier is requested and present, it fully overrides
            primary+fallbacks. Values are full "provider/model" chains.
    """
    primary: str
    fallbacks: list[str] = field(default_factory=list)
    tiers: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelPolicy:
    """Complete model policy configuration.

    Attributes:
        call_types: Mapping from call_type name to CallTypePolicy.
        provider_fallbacks: Ordered list of provider names to try if
            the primary provider fails entirely. Phase 07 feature.
    """
    call_types: dict[str, CallTypePolicy]
    provider_fallbacks: list[str] = field(default_factory=list)

    def get_policy(self, call_type: str) -> CallTypePolicy:
        """Get policy for a call type, falling back to 'general'."""
        return self.call_types.get(call_type, self.call_types.get("general", CallTypePolicy(primary="")))

    def get_model_chain(self, call_type: str, tier: str | None = None) -> list[str]:
        """Get the full model chain (primary + fallbacks) for a call type.

        If ``tier`` is given and the call type defines a matching tier chain,
        that chain is used instead of the default primary+fallbacks.
        """
        policy = self.get_policy(call_type)
        if tier and tier in policy.tiers:
            return list(policy.tiers[tier])
        return [policy.primary] + policy.fallbacks


DEFAULT_POLICY = ModelPolicy(
    call_types={
        "general": CallTypePolicy(primary="", fallbacks=[]),
        "query_analysis": CallTypePolicy(primary="", fallbacks=[]),
        "research_planning": CallTypePolicy(primary="", fallbacks=[]),
        "evidence_extraction": CallTypePolicy(primary="", fallbacks=[]),
        "reasoning": CallTypePolicy(primary="", fallbacks=[]),
        "synthesis": CallTypePolicy(primary="", fallbacks=[]),
        "verification": CallTypePolicy(primary="", fallbacks=[]),
        "revision": CallTypePolicy(primary="", fallbacks=[]),
    },
    provider_fallbacks=[],
)


def load_model_policy(settings: Settings | None = None) -> ModelPolicy:
    """Load model policy from `configs/model_policy.yaml`.

    Returns DEFAULT_POLICY if the file is missing or empty.
    """
    settings = settings or get_settings()
    policy_path = settings.config_dir / "model_policy.yaml"

    if not policy_path.exists():
        return DEFAULT_POLICY

    with policy_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return DEFAULT_POLICY

    call_types = {}
    for ct_name, ct_data in data.get("call_types", {}).items():
        tiers = {}
        for tier_name, chain in (ct_data.get("tiers", {}) or {}).items():
            if isinstance(chain, list):
                tiers[tier_name] = [str(s) for s in chain]
        call_types[ct_name] = CallTypePolicy(
            primary=ct_data.get("primary", ""),
            fallbacks=ct_data.get("fallbacks", []),
            tiers=tiers,
        )

    return ModelPolicy(
        call_types=call_types,
        provider_fallbacks=data.get("provider_fallbacks", []),
    )