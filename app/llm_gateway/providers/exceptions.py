"""Normalized provider error hierarchy (Phase 00.3).

Every provider implementation raises these exceptions — callers
(gateway, router, application code) never see provider-specific
exception types or raw `httpx`/SDK errors. This keeps the "concrete
provider" layer fully swappable per the layering rule in
`E:\\ARGUS_VAULT\\03_ARCHITECTURE_DECISIONS.md` / D-005.
"""

from __future__ import annotations


class LLMProviderError(Exception):
    """Base exception for all normalized provider errors."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.provider = provider
        self.model = model
        super().__init__(f"[{code}] {message}")


class AuthenticationError(LLMProviderError):
    """Invalid or missing API key (HTTP 401/403)."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("AUTHENTICATION_ERROR", message, retryable=False, **kwargs)  # type: ignore[arg-type]


class RateLimitError(LLMProviderError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("RATE_LIMIT_ERROR", message, retryable=True, **kwargs)  # type: ignore[arg-type]


class ModelNotFoundError(LLMProviderError):
    """Requested model is not available (HTTP 404 or equivalent)."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("MODEL_NOT_FOUND", message, retryable=False, **kwargs)  # type: ignore[arg-type]


class ContextLengthError(LLMProviderError):
    """Context window exceeded (HTTP 400 with a context-length message)."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("CONTEXT_LENGTH_EXCEEDED", message, retryable=False, **kwargs)  # type: ignore[arg-type]


class ProviderUnavailableError(LLMProviderError):
    """Provider returned a 5xx or is otherwise unreachable."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("PROVIDER_UNAVAILABLE", message, retryable=True, **kwargs)  # type: ignore[arg-type]


class TimeoutOrNetworkError(LLMProviderError):
    """Request timed out or failed at the network layer."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("TIMEOUT_OR_NETWORK_ERROR", message, retryable=True, **kwargs)  # type: ignore[arg-type]


class ConfigurationError(LLMProviderError):
    """Misconfiguration: missing API key, unknown provider name, invalid config."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("CONFIGURATION_ERROR", message, retryable=False, **kwargs)  # type: ignore[arg-type]


class CapabilityNotSupportedError(LLMProviderError):
    """Requested capability (structured output, tools, ...) not supported by this provider/model."""

    def __init__(self, capability: str, **kwargs: object) -> None:
        super().__init__(
            "CAPABILITY_NOT_SUPPORTED",
            f"Capability '{capability}' is not supported by this provider/model",
            retryable=False,
            **kwargs,  # type: ignore[arg-type]
        )


class CallCeilingExceededError(LLMProviderError):
    """The global hard call ceiling for a research run has been reached.

    Raised by the router before making another actual LLM call when the
    current run's telemetry already records the configured number of
    logical calls. This is a true safety ceiling: model fallbacks and
    retries must not bypass it. Not retryable (further retries would only
    re-hit the ceiling and cost quota for nothing).
    """

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__("CALL_CEILING_EXCEEDED", message, retryable=False, **kwargs)  # type: ignore[arg-type]
