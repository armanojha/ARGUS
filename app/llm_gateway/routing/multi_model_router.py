"""Multi-Model Router (Phase 07).

Extends the single-provider LLMRouter with:
- Call-type -> model policy resolution
- Provider capability/quota awareness
- Primary/fallback execution chains
- Cross-model verification (verifier != synthesizer)
- Telemetry integration
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.llm_gateway.capabilities import get_capabilities
from app.llm_gateway.policies.model_policy import load_model_policy
from app.llm_gateway.providers.base import LLMProvider
from app.llm_gateway.providers.registry import ProviderRegistry
from app.llm_gateway.providers.exceptions import (
    CallCeilingExceededError,
    ConfigurationError,
    LLMProviderError,
    RateLimitError,
)
from app.llm_gateway.providers.models import (
    CompletionResponse,
    Message,
    Tool,
    ToolChoice,
)
from app.llm_gateway.quota import get_quota_tracker
from app.llm_gateway.telemetry import check_call_ceiling, record_routing_decision
from app.logging_config import get_logger

logger = get_logger("argus.llm_gateway.multi_model_router")

# Error codes that indicate a failure at the PROVIDER level (auth, whole
# endpoint down/5xx, network/timeout). When one of these is raised, the
# failed provider should be excluded so we do not waste attempts walking its
# remaining models. Everything else (model-not-found, context-length,
# malformed response, a single model's rate limit) is treated as a MODEL-level
# failure so an intentional intra-provider fallback (e.g. zen/mimo ->
# zen/nemotron) stays reachable.
_PROVIDER_LEVEL_ERROR_CODES = {
    "AUTHENTICATION_ERROR",
    "PROVIDER_UNAVAILABLE",
    "TIMEOUT_OR_NETWORK_ERROR",
}

# Internal error codes that are never worth falling back on (hard config /
# ceiling faults).
_NON_RETRYABLE_CODES = {
    "CONFIGURATION_ERROR",
    "CALL_CEILING_EXCEEDED",
    "CAPABILITY_NOT_SUPPORTED",
}


def _is_provider_level(exc: LLMProviderError) -> bool:
    """Classify a provider error as provider-wide vs model-specific.

    Model 4xx / a single model's rate limit / a malformed single-model response
    are model-level. Auth failures, upstream 5xx and timeouts are provider-wide.
    """
    return exc.code in _PROVIDER_LEVEL_ERROR_CODES


def _is_model_level(exc: LLMProviderError) -> bool:
    return not _is_provider_level(exc) and exc.code not in _NON_RETRYABLE_CODES


@dataclass
class ModelSpec:
    """A provider/model combination."""

    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        """Parse 'provider/model' string."""
        parts = spec.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model spec: {spec}. Expected 'provider/model'")
        return cls(provider=parts[0], model=parts[1])

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class RoutingResult:
    """Result of a routing decision."""

    model_spec: ModelSpec
    provider_instance: LLMProvider
    is_fallback: bool
    fallback_reason: str | None
    attempt: int


class MultiModelRouter:
    """Routes completion calls using call-type policy with fallback chains.

    Features:
    - Explicit call-type configuration (model_policy.yaml)
    - Provider-level fallback chains
    - Quota-aware routing (skips exhausted providers)
    - Capability validation (structured output, tool calling, context length)
    - Cross-model verification support
    - Telemetry integration
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: dict[str, LLMProvider] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._policy = load_model_policy(self._settings)
        # Provider lifecycle is owned by the registry (HARDEN-06.5.1): the
        # router only consumes providers by name and never constructs/tears
        # them down directly.
        self._registry = registry or ProviderRegistry(
            settings=self._settings,
            providers=providers or {},
        )
        # Cache cross-model verification settings to avoid re-reading YAML
        from app.config import load_yaml_config
        policy_path = self._settings.config_dir / "model_policy.yaml"
        policy_data = load_yaml_config(policy_path)
        cross_verification = policy_data.get("cross_model_verification", {})
        self._verifier_must_differ = cross_verification.get("verifier_must_differ_from_synthesizer", True)
        self._allow_same_provider_diff_model = cross_verification.get("allow_same_provider_different_model", True)
        self._preferred_verifiers = cross_verification.get("preferred_verifier_providers", ["gemini", "groq", "cerebras"])

    def _get_provider(self, provider_name: str) -> LLMProvider | None:
        """Get a provider instance from the registry."""
        return self._registry.get(provider_name)

    def _resolve_model_chain(self, call_type: str, tier: str | None = None) -> list[ModelSpec]:
        """Resolve the model chain (primary + fallbacks, or a tier chain) for a call type."""
        model_chain = self._policy.get_model_chain(call_type, tier=tier)
        results = []
        for spec in model_chain:
            if not spec:
                logger.warning(
                    "model_policy_empty_spec",
                    call_type=call_type,
                )
                continue
            try:
                results.append(ModelSpec.parse(spec))
            except ValueError as exc:
                logger.warning(
                    "model_policy_invalid_spec",
                    call_type=call_type,
                    spec=spec,
                    error=str(exc),
                )
        return results

    def _get_provider_fallbacks(self) -> list[str]:
        """Get provider-level fallback chain."""
        return self._policy.provider_fallbacks

    def _validate_capabilities(
        self,
        provider_name: str,
        model: str,
        call_type: str,
        *,
        response_format: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        estimated_tokens: int = 0,
    ) -> tuple[bool, str | None]:
        """Validate that provider/model supports required capabilities."""
        caps = get_capabilities(provider_name, model)

        if response_format and not caps.structured_output:
            return False, "structured_output not supported"

        if tools and not caps.tool_calling:
            return False, "tool_calling not supported"

        if estimated_tokens > caps.max_context_tokens:
            return False, f"estimated tokens ({estimated_tokens}) exceeds max context ({caps.max_context_tokens})"

        # Check if provider supports this call type
        if caps.supported_call_types and call_type not in caps.supported_call_types:
            # This is a soft check - just log, don't block
            logger.debug(
                "provider_call_type_not_explicitly_supported",
                provider=provider_name,
                model=model,
                call_type=call_type,
            )

        return True, None

    def _select_model_for_call_type(
        self,
        call_type: str,
        *,
        response_format: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        estimated_tokens: int = 0,
        exclude_models: set[str] | None = None,
        exclude_providers: set[str] | None = None,
        tier: str | None = None,
    ) -> RoutingResult | None:
        """Select the best available model for a call type.

        ``tier`` (optional) selects a task-complexity-specific model chain
        when the call-type policy defines one (HARDEN-06.5.2); otherwise the
        default primary+fallbacks chain is used.

        Exclusion is tracked at two scopes so that an intentional
        intra-provider fallback stays reachable after a model failure:
          * ``exclude_models`` — "provider/model" keys already failed at the
            model level. The same provider's *other* models remain eligible.
          * ``exclude_providers`` — providers failed at the whole-provider
            level (auth / 5xx / timeout). All of their models are skipped.

        Returns None if no suitable model found.
        """
        exclude_models = exclude_models or set()
        exclude_providers = exclude_providers or set()
        model_chain = self._resolve_model_chain(call_type, tier=tier)
        provider_fallbacks = self._get_provider_fallbacks()

        # First try the call-type specific chain
        for i, model_spec in enumerate(model_chain):
            if model_spec.provider in exclude_providers:
                continue
            if f"{model_spec.provider}/{model_spec.model}" in exclude_models:
                continue

            provider = self._get_provider(model_spec.provider)
            if provider is None:
                continue

            # Check quota (provider-level, and per-model when configured)
            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(model_spec.provider, estimated_tokens, model=model_spec.model):
                logger.debug(
                    "quota_exhausted_skip",
                    provider=model_spec.provider,
                    model=model_spec.model,
                    call_type=call_type,
                )
                continue

            # Check capabilities
            ok, reason = self._validate_capabilities(
                model_spec.provider,
                model_spec.model,
                call_type,
                response_format=response_format,
                tools=tools,
                estimated_tokens=estimated_tokens,
            )
            if not ok:
                logger.debug(
                    "capability_mismatch_skip",
                    provider=model_spec.provider,
                    model=model_spec.model,
                    call_type=call_type,
                    reason=reason,
                )
                continue

            return RoutingResult(
                model_spec=model_spec,
                provider_instance=provider,
                is_fallback=i > 0,
                fallback_reason=None if i == 0 else f"fallback_{i}",
                attempt=i + 1,
            )

        # If call-type chain exhausted, try provider fallbacks with their default models
        for provider_name in provider_fallbacks:
            if provider_name in exclude_providers:
                continue

            provider = self._get_provider(provider_name)
            if provider is None:
                continue

            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(provider_name, estimated_tokens, model=provider.default_model):
                continue

            # Use provider's default model
            model_spec = ModelSpec(provider=provider_name, model=provider.default_model)

            ok, reason = self._validate_capabilities(
                provider_name,
                model_spec.model,
                call_type,
                response_format=response_format,
                tools=tools,
                estimated_tokens=estimated_tokens,
            )
            if not ok:
                continue

            return RoutingResult(
                model_spec=model_spec,
                provider_instance=provider,
                is_fallback=True,
                fallback_reason=f"provider_fallback_{provider_name}",
                attempt=len(model_chain) + 1,
            )

        return None

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        timeout: float = 30.0,
        call_type: str = "general",
        request_id: str | None = None,
        tier: str | None = None,
        query: str | None = None,
    ) -> CompletionResponse:
        """Complete with multi-model routing and fallback.

        If `model` is explicitly provided, it's treated as a provider/model
        override (e.g., "groq/openai/gpt-oss-120b"). Otherwise, the call_type
        policy determines the model chain; `tier` (or auto-classification of
        `query`) selects a complexity-specific chain when configured
        (HARDEN-06.5.2).
        """

        # Task-adaptive tier resolution (HARDEN-06.5.2): an explicit tier wins;
        # otherwise a query is auto-classified to a complexity tier.
        if tier is None:
            from app.llm_gateway.routing.complexity import classify_complexity
            tier = classify_complexity(query or "").value

        # Estimate tokens for quota check
        estimated_tokens = sum(len(m.content or "") for m in messages) // 4
        if max_tokens:
            estimated_tokens += max_tokens

        # If explicit model provided, use it directly (backward compatibility)
        if model:
            model_spec = ModelSpec.parse(model)
            provider = self._get_provider(model_spec.provider)
            if provider is None:
                raise ConfigurationError(f"Provider not available: {model_spec.provider}")

            return await self._execute_with_telemetry(
                provider=provider,
                model_spec=model_spec,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                call_type=call_type,
                request_id=request_id,
                is_fallback=False,
                fallback_reason=None,
            )

        # Route via call-type policy
        routing_result = self._select_model_for_call_type(
            call_type=call_type,
            response_format=response_format,
            tools=tools,
            estimated_tokens=estimated_tokens,
            tier=tier,
        )

        if routing_result is None:
            raise ConfigurationError(
                f"No available model for call_type '{call_type}'. "
                f"Check provider availability, quota, and capabilities."
            )

        last_error: LLMProviderError | None = None

        # Failures tracked at model scope by default so an intentional
        # intra-provider fallback (zen/mimo -> zen/nemotron) stays reachable;
        # a whole-provider failure (auth / 5xx / timeout) additionally excludes
        # every model on that provider to avoid wasting attempts.
        exclude_models: set[str] = set()
        exclude_providers: set[str] = set()

        # Try the selected model, then fall through the ordered fallback chain.
        # The total number of *distinct* routing attempts is bounded by the
        # configured chain + provider fallbacks (never infinite), so a model
        # failure cannot create an unbounded retry storm.
        attempt = 0
        max_routing_attempts = len(self._policy.get_model_chain(call_type)) + len(self._policy.provider_fallbacks) + 1
        max_routing_attempts = max(max_routing_attempts, 3)  # At least 3 attempts
        while attempt < max_routing_attempts:
            attempt += 1
            try:
                return await self._execute_with_telemetry(
                    provider=routing_result.provider_instance,
                    model_spec=routing_result.model_spec,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout=timeout,
                    call_type=call_type,
                    request_id=request_id,
                    is_fallback=routing_result.is_fallback,
                    fallback_reason=routing_result.fallback_reason,
                    attempt=attempt,
                )
            except (RateLimitError, LLMProviderError) as exc:
                last_error = exc
                failed_key = f"{routing_result.model_spec.provider}/{routing_result.model_spec.model}"
                if _is_provider_level(exc):
                    exclude_providers.add(routing_result.model_spec.provider)
                    logger.warning(
                        "routing_attempt_failed_provider",
                        provider=routing_result.model_spec.provider,
                        model=routing_result.model_spec.model,
                        call_type=call_type,
                        attempt=attempt,
                        error=str(exc),
                        error_code=exc.code,
                        scope="provider",
                    )
                else:
                    exclude_models.add(failed_key)
                    logger.warning(
                        "routing_attempt_failed",
                        provider=routing_result.model_spec.provider,
                        model=routing_result.model_spec.model,
                        call_type=call_type,
                        attempt=attempt,
                        error=str(exc),
                        error_code=exc.code,
                        scope="model",
                    )

                # Non-retryable means don't retry the same exact call, but
                # trying a DIFFERENT model/provider via the exclusion-based
                # fallback chain is not a retry—it is the next step.  The
                # bounded loop count prevents infinite attempts.

                # Try next fallback (next model in the chain, or next provider)
                routing_result = self._select_model_for_call_type(
                    call_type=call_type,
                    response_format=response_format,
                    tools=tools,
                    estimated_tokens=estimated_tokens,
                    exclude_models=exclude_models,
                    exclude_providers=exclude_providers,
                    tier=tier,
                )
                if routing_result is None:
                    break

        # All attempts failed
        raise last_error or ConfigurationError(f"All routing attempts failed for call_type '{call_type}'")

    async def _execute_with_telemetry(
        self,
        provider: LLMProvider,
        model_spec: ModelSpec,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: type[BaseModel] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        timeout: float,
        call_type: str,
        request_id: str | None,
        is_fallback: bool,
        fallback_reason: str | None,
        attempt: int = 1,
    ) -> CompletionResponse:
        """Execute completion with telemetry recording.

        Enforces the global hard call ceiling (when a run is active) before
        touching the network: retries and model fallbacks must never push a
        research run past the configured logical-call ceiling.
        """
        if check_call_ceiling():
            raise CallCeilingExceededError(
                "Global LLM call ceiling reached for this run; refusing to "
                "make another model call. Further retries/fallbacks are "
                "suppressed so the ceiling is a true safety bound.",
                provider=model_spec.provider,
                model=model_spec.model,
            )

        start_time = time.time()
        error_code = None
        error_class: str | None = None
        success = False
        response = None

        try:
            response = await provider.complete(
                messages,
                model=model_spec.model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                call_type=call_type,
                request_id=request_id,
            )
            success = True

            # Record quota usage (provider-level, and per-model when tracked)
            if response.usage:
                quota_tracker = get_quota_tracker(self._settings)
                quota_tracker.record_request(
                    provider_name=model_spec.provider,
                    model_name=model_spec.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
                # Calibrate the tracker against the provider's live rate-limit
                # headers (Phase 07 quota awareness): the tracker's local
                # counters are otherwise only advanced by our own requests and
                # drift from the provider's true remaining budget.
                if response.rate_limit_headers:
                    quota_tracker.update_from_headers(
                        model_spec.provider,
                        response.rate_limit_headers,
                    )

            return response

        except LLMProviderError as exc:
            error_code = exc.code
            error_class = type(exc).__name__
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)
            if success and response and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            else:
                prompt_tokens = None
                completion_tokens = None

            record_routing_decision(
                call_type=call_type,
                provider=model_spec.provider,
                model=model_spec.model,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                success=success,
                error_code=error_code,
                error_class=error_class,
                attempt=attempt,
            )

    async def complete_for_verification(
        self,
        messages: list[Message],
        *,
        synthesizer_provider: str,
        synthesizer_model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        timeout: float = 30.0,
        call_type: str = "verification",
        request_id: str | None = None,
    ) -> CompletionResponse:
        """Complete for verification, ensuring different provider from synthesizer.

        This implements the cross-model verification requirement:
        the verifier MUST use a different provider than the synthesizer.
        """

        # Use cached cross-model verification settings
        verifier_must_differ = self._verifier_must_differ
        allow_same_provider_diff_model = self._allow_same_provider_diff_model
        preferred_verifiers = self._preferred_verifiers

        # Determine excluded provider (policy-level: verifier must differ)
        policy_exclude_provider = synthesizer_provider if verifier_must_differ else None

        # If same provider allowed with different model, we can only exclude if
        # the only available model is the same as synthesizer's
        if not verifier_must_differ or not allow_same_provider_diff_model:
            # We still try to use a different provider first
            pass

        estimated_tokens = sum(len(m.content or "") for m in messages) // 4
        if max_tokens:
            estimated_tokens += max_tokens

        # Try preferred verifiers first (excluding synthesizer's provider by policy)
        for pref_provider in preferred_verifiers:
            if policy_exclude_provider and pref_provider == policy_exclude_provider:
                continue

            provider = self._get_provider(pref_provider)
            if provider is None:
                continue

            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(pref_provider, estimated_tokens, model=provider.default_model):
                continue

            # Use the provider's default model
            model_spec = ModelSpec(provider=pref_provider, model=provider.default_model)

            ok, _reason = self._validate_capabilities(
                pref_provider,
                model_spec.model,
                call_type,
                response_format=response_format,
                tools=tools,
                estimated_tokens=estimated_tokens,
            )
            if not ok:
                continue

            return await self._execute_with_telemetry(
                provider=provider,
                model_spec=model_spec,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                call_type=call_type,
                request_id=request_id,
                is_fallback=False,
                fallback_reason=f"cross_model_verification_{pref_provider}",
                attempt=1,
            )

        # If no preferred verifier available, fall back to normal routing
        # but still exclude synthesizer's provider if required by policy.
        # Verification is high-stakes: prefer the STRONG tier.
        routing_result = self._select_model_for_call_type(
            call_type=call_type,
            response_format=response_format,
            tools=tools,
            estimated_tokens=estimated_tokens,
            exclude_models=set(),
            exclude_providers={policy_exclude_provider} if policy_exclude_provider else None,
            tier="strong",
        )

        if routing_result is None:
            raise ConfigurationError(
                f"No available verifier model (must differ from synthesizer: {synthesizer_provider}). "
                f"Check provider availability, quota, and capabilities."
            )

        last_error: LLMProviderError | None = None
        attempt = 0
        exclude_models: set[str] = set()
        # Provider failures during verification additionally exclude the provider,
        # while still honoring the policy exclusion of the synthesizer's provider.
        exclude_providers: set[str] = {policy_exclude_provider} if policy_exclude_provider else set()
        max_verification_attempts = max(
            len(self._policy.get_model_chain(call_type)) + len(self._policy.provider_fallbacks) + 1,
            3,
        )

        while attempt < max_verification_attempts:
            attempt += 1
            try:
                return await self._execute_with_telemetry(
                    provider=routing_result.provider_instance,
                    model_spec=routing_result.model_spec,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout=timeout,
                    call_type=call_type,
                    request_id=request_id,
                    is_fallback=routing_result.is_fallback,
                    fallback_reason=routing_result.fallback_reason,
                    attempt=attempt,
                )
            except (RateLimitError, LLMProviderError) as exc:
                last_error = exc
                if _is_provider_level(exc):
                    exclude_providers.add(routing_result.model_spec.provider)
                else:
                    exclude_models.add(f"{routing_result.model_spec.provider}/{routing_result.model_spec.model}")

                routing_result = self._select_model_for_call_type(
                    call_type=call_type,
                    response_format=response_format,
                    tools=tools,
                    estimated_tokens=estimated_tokens,
                    exclude_models=exclude_models,
                    exclude_providers=exclude_providers,
                    tier="strong",
                )
                if routing_result is None:
                    break

        raise last_error or ConfigurationError("Cross-model verification failed: no available verifier")

    async def aclose(self) -> None:
        """Close all provider connections (delegates to the registry)."""
        await self._registry.close_all()

    def get_available_models(self, call_type: str | None = None) -> list[dict[str, Any]]:
        """Get list of available models for a call type (for debugging/telemetry)."""
        if call_type:
            model_chain = self._resolve_model_chain(call_type)
        else:
            model_chain = []

        results = []
        for model_spec in model_chain:
            provider = self._get_provider(model_spec.provider)
            if provider is None:
                continue
            caps = get_capabilities(model_spec.provider, model_spec.model)
            quota = get_quota_tracker(self._settings).get_quota(model_spec.provider)
            results.append({
                "provider": model_spec.provider,
                "model": model_spec.model,
                "available": provider is not None,
                "quota_remaining": quota.get_status() if quota else None,
                "capabilities": {
                    "structured_output": caps.structured_output,
                    "tool_calling": caps.tool_calling,
                    "vision": caps.vision,
                    "max_context_tokens": caps.max_context_tokens,
                    "max_output_tokens": caps.max_output_tokens,
                    "speed_class": caps.speed_class,
                },
            })
        return results