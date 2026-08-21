"""Versioned event envelope used by reducers and audit storage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    WORKFLOW_STARTED = "workflow.started.v1"
    ACTIVITY_PLANNED = "activity.planned.v1"
    ACTIVITY_STARTED = "activity.started.v1"
    ACTIVITY_COMPLETED = "activity.completed.v1"
    ACTIVITY_FAILED = "activity.failed.v1"
    WORKFLOW_PAUSED = "workflow.paused.v1"
    WORKFLOW_RESUMED = "workflow.resumed.v1"
    WORKFLOW_CANCELLED = "workflow.cancelled.v1"
    WORKFLOW_TERMINAL = "workflow.terminal.v1"
    SKILL_LOADED = "skill.loaded.v1"
    TOOL_CALLED = "tool.called.v1"
    TOOL_COMPLETED = "tool.completed.v1"


class EventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: UUID
    event_type: EventType
    aggregate_type: str
    aggregate_id: UUID
    aggregate_version: int = Field(ge=1)
    correlation_id: UUID
    causation_id: UUID | None = None
    actor_type: str
    actor_id: str
    occurred_at: datetime
    payload: dict[str, Any]
    payload_schema: str
    previous_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DomainEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_type: EventType
    payload: dict[str, Any]
    causation_id: UUID | None = None
    actor_type: str = "system"
    actor_id: str = "workflow"
