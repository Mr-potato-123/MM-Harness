"""Framework-independent workflow state and activity specifications."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .capability import CapabilityRequirement
from .events import DomainEvent


class WorkflowStage(StrEnum):
    INGEST = "ingest"
    MODELING = "modeling"
    CODING = "coding"
    VALIDATION = "validation"
    REVIEW = "review"
    REVISION = "revision"
    FINALIZE = "finalize"


class ActivitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    activity_type: str = Field(min_length=1)
    stage: WorkflowStage
    required_capabilities: tuple[CapabilityRequirement, ...] = ()
    input_artifact_ids: tuple[UUID, ...] = ()
    output_contract: str = Field(min_length=1)
    timeout_seconds: int = Field(default=1800, ge=1, le=86_400)
    max_attempts: int = Field(default=3, ge=1, le=20)
    concurrency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    workflow_id: UUID
    question_id: UUID
    definition: str
    definition_version: str
    stage: WorkflowStage
    revision: int = Field(default=0, ge=0)
    paused: bool = False
    cancelled: bool = False
    terminal: bool = False
    terminal_reason: str | None = None
    version: int = Field(default=1, ge=1)
    facts: dict[str, Any] = Field(default_factory=dict)


class WorkflowDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    next_state: WorkflowState
    activities: tuple[ActivitySpec, ...] = ()
    events: tuple[DomainEvent, ...] = ()
