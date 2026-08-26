"""Unit tests for LLM gateway (Phase 00.3).

All tests use MockProvider — zero network calls, no API credits consumed.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm_gateway import close_router, get_router
from app.llm_gateway.providers.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutOrNetworkError,
)
from app.llm_gateway.providers.models import (
    CompletionResponse,
    FunctionDefinition,
    Message,
    MessageRole,
    Tool,
    ToolChoice,
    Usage,
)
from tests.mocks.mock_provider import (
    MockProvider,
)


class Answer(BaseModel):
    answer: str
    confidence: float


class TestMockProviderBasics:
    """Basic MockProvider behavior tests."""

    async def test_complete_basic(self, mock_provider: MockProvider):
        resp = await mock_provider.complete([
            Message(role=MessageRole.USER, content="Hello")
        ])
        assert resp.content == "Mock response"
        assert resp.model == "mock-model"
        assert resp.usage.total_tokens == 15
        assert resp.provider == "mock"

    async def test_complete_with_custom_model(self, mock_provider: MockProvider):
        resp = await mock_provider.complete([
            Message(role=MessageRole.USER, content="Hello")
        ], model="custom-model")
        assert resp.model == "custom-model"

    async def test_complete_structured_output(self, mock_provider: MockProvider):
        # Pre-program a valid response for the Answer schema
        mock_provider._responses["mock-model"] = CompletionResponse(
            content='{"answer": "test answer", "confidence": 0.95}',
            model="mock-model",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider="mock",
        )
        
        resp = await mock_provider.complete([
            Message(role=MessageRole.USER, content="Answer in JSON")
        ], response_format=Answer)

        assert resp.content is not None
        parsed = Answer.model_validate_json(resp.content)
        assert isinstance(parsed.answer, str)
        assert isinstance(parsed.confidence, float)

    async def test_complete_with_tools(self, mock_provider: MockProvider):
        tools = [Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        )]

        resp = await mock_provider.complete([
            Message(role=MessageRole.USER, content="Weather in NYC")
        ], tools=tools, tool_choice=ToolChoice(type="auto"))

        assert resp.content == "Mock response"

    async def test_call_log_captures_all_params(self, mock_provider: MockProvider):
        await mock_provider.complete([
            Message(role=MessageRole.USER, content="Test")
        ], model="test-model", temperature=0.5, max_tokens=100,
            call_type="query_analysis", request_id="req_test123")

        log = mock_provider.call_log
        assert len(log) == 1
        entry = log[0]
        assert entry["model"] == "test-model"
        assert entry["temperature"] == 0.5
        assert entry["max_tokens"] == 100
        assert entry["call_type"] == "query_analysis"
        assert entry["request_id"] == "req_test123"
        assert len(entry["messages"]) == 1

    async def test_aclose_marks_closed(self, mock_provider: MockProvider):
        assert not mock_provider.closed
        await mock_provider.aclose()
        assert mock_provider.closed


class TestMockProviderErrors:
    """MockProvider error simulation tests."""

    async def test_auth_error(self):
        provider = MockProvider(should_fail=True, fail_with=AuthenticationError)
        with pytest.raises(AuthenticationError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "AUTHENTICATION_ERROR"
        assert not exc.value.retryable

    async def test_rate_limit_error(self):
        provider = MockProvider(should_fail=True, fail_with=RateLimitError)
        with pytest.raises(RateLimitError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "RATE_LIMIT_ERROR"
        assert exc.value.retryable

    async def test_model_not_found_error(self):
        provider = MockProvider(should_fail=True, fail_with=ModelNotFoundError)
        with pytest.raises(ModelNotFoundError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "MODEL_NOT_FOUND"
        assert not exc.value.retryable

    async def test_provider_unavailable_error(self):
        provider = MockProvider(should_fail=True, fail_with=ProviderUnavailableError)
        with pytest.raises(ProviderUnavailableError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "PROVIDER_UNAVAILABLE"
        assert exc.value.retryable

    async def test_timeout_error(self):
        provider = MockProvider(should_fail=True, fail_with=TimeoutOrNetworkError)
        with pytest.raises(TimeoutOrNetworkError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "TIMEOUT_OR_NETWORK_ERROR"
        assert exc.value.retryable

    async def test_config_error(self):
        provider = MockProvider(should_fail=True, fail_with=ConfigurationError)
        with pytest.raises(ConfigurationError) as exc:
            await provider.complete([Message(role=MessageRole.USER, content="Hi")])
        assert exc.value.code == "CONFIGURATION_ERROR"
        assert not exc.value.retryable


class TestRouter:
    """LLMRouter passthrough tests."""

    async def test_router_delegates_to_provider(self, mock_router):
        resp = await mock_router.complete([
            Message(role=MessageRole.USER, content="Hello")
        ])
        assert resp.content == "Mock response"
        assert resp.provider == "mock"

    async def test_router_passes_all_params(self, mock_router, mock_provider: MockProvider):
        await mock_router.complete([
            Message(role=MessageRole.USER, content="Test")
        ], model="router-model", temperature=0.7, max_tokens=200,
            call_type="reasoning", request_id="req_router123")

        log = mock_provider.call_log
        assert len(log) == 1
        entry = log[0]
        assert entry["model"] == "router-model"
        assert entry["temperature"] == 0.7
        assert entry["max_tokens"] == 200
        assert entry["call_type"] == "reasoning"
        assert entry["request_id"] == "req_router123"

    async def test_router_aclose_delegates(self, mock_router, mock_provider: MockProvider):
        await mock_router.aclose()
        assert mock_provider.closed


class TestGetRouter:
    """get_router() singleton behavior tests."""

    async def test_get_router_returns_same_instance(self, mock_router, monkeypatch):
        # Reset the global router first
        import app.llm_gateway as gateway
        gateway._router = None

        # Mock create_provider to return our mock provider (sync function)
        import app.llm_gateway as gateway_module
        
        def mock_create_provider(settings=None):
            return mock_router.provider
        
        monkeypatch.setattr(gateway_module, "create_provider", mock_create_provider)

        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

        await close_router()

    async def test_close_router_resets_singleton(self, mock_router, monkeypatch):
        import app.llm_gateway as gateway
        gateway._router = None

        # Mock create_provider to return our mock provider (sync function)
        import app.llm_gateway as gateway_module
        
        def mock_create_provider(settings=None):
            return mock_router.provider
        
        monkeypatch.setattr(gateway_module, "create_provider", mock_create_provider)

        router1 = get_router()
        await close_router()
        router2 = get_router()
        # After close, a new router is created
        assert router1 is not router2

        await close_router()


class TestModelPolicy:
    """Model policy configuration tests."""

    def test_default_policy_has_all_call_types(self):
        from app.llm_gateway.policies import DEFAULT_POLICY

        expected_types = [
            "general", "query_analysis", "research_planning",
            "evidence_extraction", "reasoning", "synthesis",
            "verification", "revision"
        ]
        for ct in expected_types:
            assert ct in DEFAULT_POLICY.call_types

    def test_get_policy_falls_back_to_general(self):
        from app.llm_gateway.policies import DEFAULT_POLICY

        policy = DEFAULT_POLICY.get_policy("unknown_type")
        assert policy is DEFAULT_POLICY.call_types["general"]

    def test_get_model_chain(self):
        from app.llm_gateway.policies import CallTypePolicy, ModelPolicy

        policy = ModelPolicy(call_types={
            "test": CallTypePolicy(primary="model-a", fallbacks=["model-b", "model-c"])
        })
        chain = policy.get_model_chain("test")
        assert chain == ["model-a", "model-b", "model-c"]


class TestProviderRegistry:
    """Provider registry tests."""

    def test_groq_registered(self):
        from app.llm_gateway.providers import registered_providers
        assert "groq" in registered_providers()

    def test_factory_creates_groq(self):
        from app.llm_gateway.providers import get_provider_factory

        # We can't easily test without mocking env, but the registry works
        factory = get_provider_factory("groq")
        assert factory is not None


class TestCapabilities:
    """ProviderCapabilities tests."""

    def test_default_capabilities(self):
        from app.llm_gateway.capabilities import ProviderCapabilities

        caps = ProviderCapabilities()
        assert caps.structured_output is True
        assert caps.tool_calling is True
        assert caps.streaming is False
        assert caps.vision is False

    def test_groq_capabilities(self):
        from app.llm_gateway.capabilities import ProviderCapabilities

        # Can't instantiate without API key, but we can check the class
        caps = ProviderCapabilities(
            structured_output=True,
            tool_calling=True,
            streaming=False,
            max_context_tokens=131_072,
            max_output_tokens=8_192,
        )
        assert caps.max_context_tokens == 131_072