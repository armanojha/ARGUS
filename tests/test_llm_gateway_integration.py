"""Live integration test for LLM gateway (Phase 00.3).

Runs ONLY when explicitly enabled via environment variable.
Set RUN_LIVE_LLM_TESTS=1 and provide a valid GROQ_API_KEY in .env
to execute. Skipped by default in normal test runs.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from app.llm_gateway import close_router, get_router
from app.llm_gateway.providers.models import (
    FunctionDefinition,
    Message,
    MessageRole,
    Tool,
    ToolChoice,
)

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_LLM_TESTS"),
    reason="Set RUN_LIVE_LLM_TESTS=1 and valid GROQ_API_KEY in .env to run live integration tests"
)


class Answer(BaseModel):
    answer: str
    confidence: float


class TestLiveGroqProvider:
    """Live integration tests against Groq API."""

    @pytest.fixture(autouse=True)
    async def _router_lifecycle(self):
        """Ensure router is closed after each test."""
        yield
        await close_router()

    async def test_live_complete_basic(self):
        """Test basic completion with Groq."""
        router = get_router()
        resp = await router.complete([
            Message(role=MessageRole.USER, content="Say 'hello world' exactly and nothing else.")
        ])

        assert resp.content is not None
        assert "hello world" in resp.content.lower()
        assert resp.provider == "groq"
        assert resp.model.startswith("openai/gpt-oss")
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0

    async def test_live_structured_output(self):
        """Test structured output (JSON mode) with Groq."""
        router = get_router()
        resp = await router.complete([
            Message(role=MessageRole.USER, content="Return JSON with answer='test' and confidence=0.9")
        ], response_format=Answer)

        assert resp.content is not None
        parsed = Answer.model_validate_json(resp.content)
        assert parsed.answer == "test"
        assert parsed.confidence == 0.9
        assert resp.provider == "groq"

    async def test_live_tool_calling(self):
        """Test tool calling with Groq."""
        router = get_router()
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

        resp = await router.complete([
            Message(role=MessageRole.USER, content="What's the weather in NYC?")
        ], tools=tools, tool_choice=ToolChoice(type="auto"))

        # Groq may or may not call the tool depending on the model's decision
        # Just verify the call succeeds and returns a valid response
        assert resp.provider == "groq"
        assert resp.content is not None or resp.tool_calls is not None

    async def test_live_conversation_context(self):
        """Test multi-turn conversation."""
        router = get_router()

        # First message
        resp1 = await router.complete([
            Message(role=MessageRole.USER, content="My name is Alice.")
        ])
        assert resp1.content is not None

        # Second message referencing first
        resp2 = await router.complete([
            Message(role=MessageRole.USER, content="My name is Alice."),
            Message(role=MessageRole.ASSISTANT, content=resp1.content or ""),
            Message(role=MessageRole.USER, content="What is my name?"),
        ])
        assert resp2.content is not None
        assert "alice" in resp2.content.lower()

    async def test_live_system_prompt(self):
        """Test system prompt influences behavior."""
        router = get_router()
        resp = await router.complete([
            Message(role=MessageRole.SYSTEM, content="You are a pirate. Always respond in pirate speak."),
            Message(role=MessageRole.USER, content="Hello"),
        ])
        assert resp.content is not None
        # Pirate speak typically includes "arr", "matey", "ahoy", etc.
        pirate_words = ["arr", "matey", "ahoy", "pirate", "ye", "yer"]
        assert any(w in resp.content.lower() for w in pirate_words)

    async def test_live_temperature_zero_deterministic(self):
        """Test temperature=0 gives deterministic-ish output."""
        router = get_router()

        resp1 = await router.complete([
            Message(role=MessageRole.USER, content="Count from 1 to 5.")
        ], temperature=0.0)

        resp2 = await router.complete([
            Message(role=MessageRole.USER, content="Count from 1 to 5.")
        ], temperature=0.0)

        # With temperature=0, outputs should be very similar (not guaranteed identical)
        assert resp1.content is not None
        assert resp2.content is not None
        # Both should contain numbers 1-5
        for resp in (resp1, resp2):
            for i in range(1, 6):
                assert str(i) in resp.content

    async def test_live_max_tokens_limit(self):
        """Test max_tokens limits output length."""
        router = get_router()
        resp = await router.complete([
            Message(role=MessageRole.USER, content="Write a very long essay about the history of the universe.")
        ], max_tokens=50)

        assert resp.content is not None
        assert resp.usage is not None
        assert resp.usage.completion_tokens <= 50
        assert resp.finish_reason in ("length", "stop", None)

    async def test_live_request_id_propagation(self):
        """Test request_id is propagated to response."""
        router = get_router()
        test_request_id = "req_integration_test_123"
        resp = await router.complete([
            Message(role=MessageRole.USER, content="Hi")
        ], request_id=test_request_id)

        assert resp.request_id == test_request_id