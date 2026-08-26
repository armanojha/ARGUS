"""Canonical provider-agnostic request/response types (Phase 00.3).

Internal format follows the OpenAI Chat Completions shape, since it is
the lingua franca that most free/cheap providers expose directly or via
an OpenAI-compatible endpoint (see D-005). Concrete provider
implementations are responsible for translating to/from their own
native wire format — application code and the gateway itself only ever
see these normalized types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """A model's request to call a specific function, with JSON-encoded arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str  # JSON-encoded string, not a parsed dict


class ToolCall(BaseModel):
    """A single tool call attached to an assistant message."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # populated for role=tool messages


class FunctionDefinition(BaseModel):
    """A function/tool definition, described as a JSON Schema."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class Tool(BaseModel):
    """A tool made available to the model for a given call."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoice(BaseModel):
    """Controls whether/how the model should call tools."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["auto", "none", "required", "function"] = "auto"
    function: Optional[dict[str, str]] = None  # {"name": "fn_name"} when type == "function"


class Usage(BaseModel):
    """Token usage for a single completion call."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Normalized completion response, regardless of provider."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    model: str
    usage: Usage | None = None
    finish_reason: str | None = None
    provider: str | None = None
    request_id: str | None = None
    """ARGUS-side request ID (propagated from the caller), not the provider's own ID."""
