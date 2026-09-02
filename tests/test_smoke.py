"""Phase 00.4 smoke test: health endpoint + LLM gateway call (mocked).

This test verifies the complete Phase 00 foundation works together:
- FastAPI application boots
- /health endpoint returns 200
- LLM gateway can be instantiated and called (with mock provider)
- Request IDs propagate through the stack
- Structured logging works
- Error handling works
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.llm_gateway import close_router, get_router
from app.llm_gateway.providers.models import CompletionResponse, Message, MessageRole, Usage
from app.llm_gateway.routing.router import LLMRouter
from tests.mocks.mock_provider import MockProvider


class TestPhase00Smoke:
    """End-to-end smoke test for Phase 00 foundation."""

    def test_health_endpoint_works(self):
        """Verify /health endpoint returns 200 with expected shape."""
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == "0.1.0"
        assert body["environment"] == "development"
        # Timestamp should be ISO 8601 and parseable
        from datetime import datetime
        datetime.fromisoformat(body["timestamp"])

        # Request ID should be present
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"].startswith("req_")

    def test_health_endpoint_no_auth_required(self):
        """/health must be reachable without any credentials (pure liveness probe)."""
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code != 401
        assert response.status_code != 403

    async def test_llm_gateway_mock_call(self):
        """Verify LLM gateway can be instantiated and called with mock provider."""
        # Create a mock provider with a pre-programmed response
        mock_provider = MockProvider()
        router = LLMRouter(mock_provider)

        response = await router.complete([
            Message(role=MessageRole.USER, content="Say hello")
        ])

        assert response.content == "Mock response"
        assert response.model == "mock-model"
        assert response.provider == "mock"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

        await router.aclose()

    async def test_llm_gateway_structured_output(self):
        """Verify LLM gateway supports structured output with mock provider."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str
            confidence: float

        mock_provider = MockProvider()
        # Pre-program a valid structured response
        mock_provider._responses["mock-model"] = CompletionResponse(
            content='{"answer": "test", "confidence": 0.95}',
            model="mock-model",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            provider="mock",
        )
        router = LLMRouter(mock_provider)

        response = await router.complete([
            Message(role=MessageRole.USER, content="Answer in JSON")
        ], response_format=Answer)

        assert response.content is not None
        parsed = Answer.model_validate_json(response.content)
        assert parsed.answer == "test"
        assert parsed.confidence == 0.95

        await router.aclose()

    async def test_llm_gateway_tool_calling(self):
        """Verify LLM gateway supports tool calling with mock provider."""
        from app.llm_gateway.providers.models import FunctionDefinition, Tool, ToolChoice

        mock_provider = MockProvider()
        router = LLMRouter(mock_provider)

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

        response = await router.complete([
            Message(role=MessageRole.USER, content="Weather in NYC")
        ], tools=tools, tool_choice=ToolChoice(type="auto"))

        assert response.content == "Mock response"

        await router.aclose()

    def test_request_id_propagation(self):
        """Verify request IDs propagate through the stack."""
        app = create_app()
        custom_request_id = "req_smoke_test_12345"
        with TestClient(app) as client:
            response = client.get("/health", headers={"X-Request-ID": custom_request_id})

        assert response.headers["X-Request-ID"] == custom_request_id

    def test_error_envelope_consistency(self):
        """Verify error responses use consistent envelope with request ID."""
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/nonexistent")

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "HTTP_ERROR"
        assert body["error"]["request_id"] is not None
        assert response.headers["X-Request-ID"] == body["error"]["request_id"]

    async def test_gateway_singleton_behavior(self):
        """Verify get_router() returns singleton and close_router() resets it."""
        import app.llm_gateway as gateway_module

        # Reset global state
        gateway_module._router = None

        # Phase 07b: the default is resilient MultiModelRouter; this test
        # exercises the explicit single-provider escape hatch, so opt out to
        # LLMRouter by flipping the flag.
        from app.config import Settings
        original_get_settings = gateway_module.get_settings
        gateway_module.get_settings = lambda: Settings(multimodel_enabled=False)

        # Mock the factory to return our mock provider
        mock_provider = MockProvider()
        mock_router = LLMRouter(mock_provider)

        original_create_provider = gateway_module.create_provider

        def mock_create_provider(settings=None):
            return mock_router.provider

        gateway_module.create_provider = mock_create_provider

        try:
            router1 = get_router()
            router2 = get_router()
            assert router1 is router2

            await close_router()

            router3 = get_router()
            assert router1 is not router3

            await close_router()
        finally:
            gateway_module.create_provider = original_create_provider
            gateway_module.get_settings = original_get_settings


# Allow running this file directly for quick smoke test
if __name__ == "__main__":
    # Run sync tests
    test_instance = TestPhase00Smoke()
    test_instance.test_health_endpoint_works()
    test_instance.test_health_endpoint_no_auth_required()
    test_instance.test_request_id_propagation()
    test_instance.test_error_envelope_consistency()
    print("Sync smoke tests passed!")

    # Run async tests
    async def run_async_tests():
        await test_instance.test_llm_gateway_mock_call()
        await test_instance.test_llm_gateway_structured_output()
        await test_instance.test_llm_gateway_tool_calling()
        await test_instance.test_gateway_singleton_behavior()
        print("Async smoke tests passed!")

    asyncio.run(run_async_tests())
    print("All smoke tests passed!")