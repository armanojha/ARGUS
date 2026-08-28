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
from app.llm_gateway.providers.exceptions import (
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
from app.llm_gateway.telemetry import record_routing_decision
from app.logging_config import get_logger

logger = get_logger("argus.llm_gateway.multi_model_router")


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
    ) -> None:
        self._settings = settings or get_settings()
        self._policy = load_model_policy(self._settings)
        self._providers: dict[str, LLMProvider] = providers or {}
        self._provider_cache: dict[str, LLMProvider] = {}
        self._initialized = False
        # Cache cross-model verification settings to avoid re-reading YAML
        from app.config import load_yaml_config
        policy_path = self._settings.config_dir / "model_policy.yaml"
        policy_data = load_yaml_config(policy_path)
        cross_verification = policy_data.get("cross_model_verification", {})
        self._verifier_must_differ = cross_verification.get("verifier_must_differ_from_synthesizer", True)
        self._allow_same_provider_diff_model = cross_verification.get("allow_same_provider_different_model", True)
        self._preferred_verifiers = cross_verification.get("preferred_verifier_providers", ["gemini", "groq", "cerebras"])

    async def _ensure_initialized(self) -> None:
        """Lazy-initialize all configured providers."""
        if self._initialized:
            return

        # Providers injected at construction time (tests, embedding, etc.)
        # take precedence and must be usable by routing.
        for name, provider in self._providers.items():
            self._provider_cache.setdefault(name, provider)

        # Load all enabled providers from config
        data = self._load_providers_config()
        for entry in data.get("providers", []) or []:
            if not entry.get("enabled", False):
                continue
            provider_name = entry.get("name")
            if not provider_name:
                continue
            try:
                provider = await self._create_provider_instance(provider_name, entry)
                self._provider_cache[provider_name] = provider
            except (ConfigurationError, ValueError, OSError) as exc:
                logger.warning(
                    "multi_model_provider_init_failed",
                    provider=provider_name,
                    error=str(exc),
                )

        self._initialized = True

    def _load_providers_config(self) -> dict[str, Any]:
        from app.config import load_providers_config
        return load_providers_config(self._settings)

    async def _create_provider_instance(self, provider_name: str, config: dict[str, Any]) -> LLMProvider:
        """Create a provider instance from config."""
        from app.llm_gateway.providers import get_provider_factory

        factory = get_provider_factory(provider_name)
        if factory is None:
            raise ConfigurationError(f"Unknown provider: {provider_name}")

        api_key_env = str(config.get("api_key_env") or f"{provider_name.upper()}_API_KEY")
        import os
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ConfigurationError(f"Missing API key for provider '{provider_name}': {api_key_env} not set")

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.get("timeout", self._settings.llm_timeout),
            "max_retries": config.get("max_retries", self._settings.llm_max_retries),
        }
        if config.get("default_model"):
            kwargs["model"] = config["default_model"]

        return factory(**kwargs)  # type: ignore[return-value]

    def _get_provider(self, provider_name: str) -> LLMProvider | None:
        """Get a provider instance, creating if necessary."""
        return self._provider_cache.get(provider_name)

    def _resolve_model_chain(self, call_type: str) -> list[ModelSpec]:
        """Resolve the full model chain (primary + fallbacks) for a call type."""
        model_chain = self._policy.get_model_chain(call_type)
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
        exclude_provider: str | None = None,
    ) -> RoutingResult | None:
        """Select the best available model for a call type.

        Returns None if no suitable model found.
        """
        model_chain = self._resolve_model_chain(call_type)
        provider_fallbacks = self._get_provider_fallbacks()

        # First try the call-type specific chain
        for i, model_spec in enumerate(model_chain):
            if exclude_provider and model_spec.provider == exclude_provider:
                continue

            provider = self._get_provider(model_spec.provider)
            if provider is None:
                continue

            # Check quota
            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(model_spec.provider, estimated_tokens):
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
            if exclude_provider and provider_name == exclude_provider:
                continue

            provider = self._get_provider(provider_name)
            if provider is None:
                continue

            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(provider_name, estimated_tokens):
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
    ) -> CompletionResponse:
        """Complete with multi-model routing and fallback.

        If `model` is explicitly provided, it's treated as a provider/model
        override (e.g., "groq/openai/gpt-oss-120b"). Otherwise, the call_type
        policy determines the model chain.
        """
        await self._ensure_initialized()

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
        )

        if routing_result is None:
            raise ConfigurationError(
                f"No available model for call_type '{call_type}'. "
                f"Check provider availability, quota, and capabilities."
            )

        last_error: LLMProviderError | None = None

        # Try the selected model, then fall through fallbacks
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
                )
            except (RateLimitError, LLMProviderError) as exc:
                last_error = exc
                logger.warning(
                    "routing_attempt_failed",
                    provider=routing_result.model_spec.provider,
                    model=routing_result.model_spec.model,
                    call_type=call_type,
                    attempt=attempt,
                    error=str(exc),
                    error_code=exc.code,
                )

                # Don't retry on non-retryable errors
                if not exc.retryable:
                    break

                # Try next fallback
                routing_result = self._select_model_for_call_type(
                    call_type=call_type,
                    response_format=response_format,
                    tools=tools,
                    estimated_tokens=estimated_tokens,
                    exclude_provider=routing_result.model_spec.provider,
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
    ) -> CompletionResponse:
        """Execute completion with telemetry recording."""
        start_time = time.time()
        error_code = None
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

            # Record quota usage
            if response.usage:
                quota_tracker = get_quota_tracker(self._settings)
                quota_tracker.record_request(
                    provider_name=model_spec.provider,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            return response

        except LLMProviderError as exc:
            error_code = exc.code
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
        await self._ensure_initialized()

        # Use cached cross-model verification settings
        verifier_must_differ = self._verifier_must_differ
        allow_same_provider_diff_model = self._allow_same_provider_diff_model
        preferred_verifiers = self._preferred_verifiers

        # Determine excluded provider
        exclude_provider = synthesizer_provider if verifier_must_differ else None

        # If same provider allowed with different model, we can only exclude if
        # the only available model is the same as synthesizer's
        if not verifier_must_differ or not allow_same_provider_diff_model:
            # We still try to use a different provider first
            pass

        estimated_tokens = sum(len(m.content or "") for m in messages) // 4
        if max_tokens:
            estimated_tokens += max_tokens

        # Try preferred verifiers first (excluding synthesizer's provider)
        for pref_provider in preferred_verifiers:
            if exclude_provider and pref_provider == exclude_provider:
                continue

            provider = self._get_provider(pref_provider)
            if provider is None:
                continue

            quota_tracker = get_quota_tracker(self._settings)
            if not quota_tracker.can_make_request(pref_provider, estimated_tokens):
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
            )

        # If no preferred verifier available, fall back to normal routing
        # but still exclude synthesizer's provider if required
        routing_result = self._select_model_for_call_type(
            call_type=call_type,
            response_format=response_format,
            tools=tools,
            estimated_tokens=estimated_tokens,
            exclude_provider=exclude_provider,
        )

        if routing_result is None:
            raise ConfigurationError(
                f"No available verifier model (must differ from synthesizer: {synthesizer_provider}). "
                f"Check provider availability, quota, and capabilities."
            )

        last_error: LLMProviderError | None = None
        attempt = 0

        while True:
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
                )
            except (RateLimitError, LLMProviderError) as exc:
                last_error = exc
                if not exc.retryable:
                    break

                routing_result = self._select_model_for_call_type(
                    call_type=call_type,
                    response_format=response_format,
                    tools=tools,
                    estimated_tokens=estimated_tokens,
                    exclude_provider=routing_result.model_spec.provider,
                )
                if routing_result is None:
                    break

        raise last_error or ConfigurationError("Cross-model verification failed: no available verifier")

    async def aclose(self) -> None:
        """Close all provider connections."""
        for provider in self._provider_cache.values():
            await provider.aclose()
        self._provider_cache.clear()
        self._initialized = False

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
                    "max_context_tokens": caps.max_context_tokens,
                    "max_output_tokens": caps.max_output_tokens,
                },
            })
        return results