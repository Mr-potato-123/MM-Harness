"""Tool descriptors, calls, grants and results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .capability import CapabilityRef


class ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ToolPolicy(ToolModel):
    network: Literal["deny", "allowlisted", "unrestricted"] = "deny"
    filesystem: Literal["none", "workspace-read", "workspace-write"] = "none"
    secrets: Literal["none", "scoped"] = "none"


class ToolDefinition(ToolModel):
    name: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=2000)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_capability: CapabilityRef
    side_effect: Literal["none", "sandboxed-write", "external-write"] = "none"
    idempotency_required: bool = True
    # ``None`` means the local trusted Code Agent tool is observed through
    # probes until it returns. Other tools may retain an explicit operator
    # timeout.
    timeout_seconds: int | None = Field(default=60, ge=1, le=86_400)
    output_limit_bytes: int = Field(default=1_048_576, ge=1, le=100_000_000)
    policy: ToolPolicy = ToolPolicy()


class ToolCall(ToolModel):
    call_id: UUID
    tool_name: str
    tool_version: str
    activity_id: UUID
    session_id: UUID
    idempotency_key: str = Field(min_length=1)
    arguments: dict[str, Any]
    requested_at: datetime


class ToolResult(ToolModel):
    call_id: UUID
    tool_name: str
    ok: bool
    output: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_ids: tuple[UUID, ...] = ()
    completed_at: datetime
    redacted: bool = False
