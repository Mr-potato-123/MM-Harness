"""Provider-neutral Agent Session, Context and Model contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelCapabilities(AgentModel):
    model: str
    text_input: bool = True
    image_input: bool = False
    pdf_input: bool = False
    video_input: bool = False
    audio_input: bool = False
    structured_output: bool = False
    tool_calling: bool = False
    code_interpreter: bool = False
    streaming: bool = True
    max_context_tokens: int = Field(default=128_000, ge=1)
    max_output_tokens: int = Field(default=16_384, ge=1)


class ContextItem(AgentModel):
    source_id: str
    kind: Literal["text", "artifact", "event", "skill", "tool", "state"]
    projection: str
    content: str | None = None
    artifact_id: UUID | None = None
    priority: int = Field(default=0, ge=-1000, le=1000)
    estimated_tokens: int = Field(default=0, ge=0)


class ContextSnapshot(AgentModel):
    snapshot_id: UUID
    session_id: UUID
    items: tuple[ContextItem, ...]
    omitted_source_ids: tuple[str, ...] = ()
    budget_tokens: int = Field(ge=1)
    used_tokens: int = Field(ge=0)
    prompt_template_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class AgentSession(AgentModel):
    session_id: UUID
    activity_id: UUID
    role: str
    provider: str
    model: str
    skill_snapshot_id: UUID | None = None
    tool_grant_digest: str | None = None
    turn_count: int = Field(default=0, ge=0)
    max_turns: int = Field(default=20, ge=1, le=1000)
    terminal: bool = False
    terminal_reason: str | None = None


class ModelToolCall(AgentModel):
    call_id: UUID
    name: str
    arguments: dict[str, Any]


class ModelRequest(AgentModel):
    session: AgentSession
    context: ContextSnapshot
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    response_schema: dict[str, Any] | None = None
    enable_thinking: bool = True
    max_output_tokens: int = Field(default=16_384, ge=1)


class ModelResponse(AgentModel):
    provider: str
    model: str
    text: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    finish_reason: str
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
