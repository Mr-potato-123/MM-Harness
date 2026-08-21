"""Typed domain and wire contracts.

The database and event log store JSON representations of these models. Every
external executor response is validated here before it can advance workflow
state or publish an artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QuestionState(StrEnum):
    PENDING = "pending"
    MODELING = "modeling"
    READY_FOR_CODING = "ready_for_coding"
    CODING = "coding"
    READY_FOR_REVIEW = "ready_for_review"
    REVIEWING = "reviewing"
    REVISION_REQUIRED = "revision_required"
    APPROVED = "approved"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class StageKind(StrEnum):
    MODELING = "modeling"
    CODING = "coding"
    REVIEW = "review"
    FINALIZE = "finalize"


class ActivityStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReviewVerdict(StrEnum):
    APPROVE = "approve"
    REVISION_REQUIRED = "revision_required"


class ArtifactKind(StrEnum):
    PROBLEM = "problem"
    DAG_TASK_TABLE = "dag_task_table"
    INPUT = "input"
    MODELING_REPORT = "modeling_report"
    CODING_REPORT = "coding_report"
    REVIEW_REPORT = "review_report"
    FINAL_QUESTION_REPORT = "final_question_report"
    FINAL_LATEX_PAPER = "final_latex_paper"
    SOURCE = "source"
    DATA = "data"
    FIGURE = "figure"
    OUTPUT = "output"
    LOG = "log"
    OTHER = "other"


class HarnessSettings(StrictModel):
    database_path: Path
    artifact_root: Path
    lease_seconds: int = Field(default=300, ge=10, le=86_400)
    activity_timeout_seconds: int = Field(default=1_800, ge=1, le=86_400)
    max_activity_attempts: int = Field(default=3, ge=1, le=20)
    max_revisions: int = Field(default=3, ge=0, le=20)
    # Publication is a first-class output contract for the production report
    # path.  Operators can disable the gate for legacy migrations, but a
    # normal production run must publish both Markdown and compile-ready TeX.
    require_latex_publication: bool = True
    executor_command: tuple[str, ...] | None = None


class ProjectRecord(StrictModel):
    id: UUID
    name: NonEmptyStr
    created_at: datetime
    updated_at: datetime
    version: int = Field(ge=1)


class QuestionRecord(StrictModel):
    id: UUID
    project_id: UUID
    key: NonEmptyStr
    title: NonEmptyStr
    state: QuestionState
    problem_artifact_id: UUID
    revision: int = Field(ge=0)
    failure_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    version: int = Field(ge=1)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None


class ArtifactRecord(StrictModel):
    id: UUID
    project_id: UUID
    question_id: UUID | None
    activity_id: UUID | None
    kind: ArtifactKind
    logical_name: NonEmptyStr
    media_type: NonEmptyStr
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: int = Field(ge=0)
    relative_path: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ProducedArtifact(StrictModel):
    logical_name: NonEmptyStr
    kind: ArtifactKind = ArtifactKind.OTHER
    media_type: NonEmptyStr = "text/plain"
    text: Annotated[str, StringConstraints(max_length=10 * 1024 * 1024)] | None = None
    base64: Annotated[str, StringConstraints(max_length=20 * 1024 * 1024)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exactly_one_content(self) -> "ProducedArtifact":
        if (self.text is None) == (self.base64 is None):
            raise ValueError("exactly one of text or base64 must be supplied")
        return self


class ReportPayload(StrictModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    markdown: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2 * 1024 * 1024)]
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
    claims: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)]] = Field(default_factory=list, max_length=100)
    limitations: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)]] = Field(default_factory=list, max_length=100)
    structured: dict[str, Any] = Field(default_factory=dict)


class ModelingStageOutput(StrictModel):
    stage: Literal[StageKind.MODELING]
    report: ReportPayload
    required_validations: list[NonEmptyStr] = Field(min_length=1)
    expected_outputs: list[NonEmptyStr] = Field(min_length=1)
    downstream_outputs: list[NonEmptyStr] = Field(default_factory=list)
    artifacts: list[ProducedArtifact] = Field(default_factory=list)


class CodingStageOutput(StrictModel):
    stage: Literal[StageKind.CODING]
    report: ReportPayload
    execution_succeeded: bool
    validations: dict[str, bool]
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    artifacts: list[ProducedArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def successful_execution_requires_evidence(self) -> "CodingStageOutput":
        if self.execution_succeeded and not self.artifacts:
            raise ValueError("successful coding execution must include at least one evidence artifact")
        return self


class ReviewStageOutput(StrictModel):
    stage: Literal[StageKind.REVIEW]
    report: ReportPayload
    verdict: ReviewVerdict
    revision_instructions: list[NonEmptyStr] = Field(default_factory=list)
    accepted_claims: list[NonEmptyStr] = Field(default_factory=list)
    artifacts: list[ProducedArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_requires_instructions(self) -> "ReviewStageOutput":
        if self.verdict == ReviewVerdict.REVISION_REQUIRED and not self.revision_instructions:
            raise ValueError("revision_required must include revision_instructions")
        return self


class FinalizeStageOutput(StrictModel):
    """Reviewed report plus the publication artifacts emitted by the final task.

    The Markdown report is materialized by the Harness from ``report``.  The
    model must provide a separate ``final_latex_paper`` artifact when the
    production publication gate is enabled; keeping it as an artifact avoids a
    second provider-specific response field and lets other renderers be added
    without changing this wire contract.
    """
    stage: Literal[StageKind.FINALIZE]
    report: ReportPayload
    downstream_outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ProducedArtifact] = Field(default_factory=list)


StageOutput = Annotated[
    ModelingStageOutput | CodingStageOutput | ReviewStageOutput | FinalizeStageOutput,
    Field(discriminator="stage"),
]


class ActivityRequest(StrictModel):
    protocol_version: Literal[1] = 1
    activity_id: UUID
    idempotency_key: NonEmptyStr
    project: ProjectRecord
    question: QuestionRecord
    stage: StageKind
    attempt: int = Field(ge=1)
    revision: int = Field(ge=0)
    inputs: list[ArtifactRecord]
    instructions: list[NonEmptyStr] = Field(default_factory=list)
    deadline: datetime


class ActivityResponse(StrictModel):
    protocol_version: Literal[1] = 1
    idempotency_key: NonEmptyStr
    output: StageOutput
    executor_metadata: dict[str, Any] = Field(default_factory=dict)


class ActivityRecord(StrictModel):
    id: UUID
    project_id: UUID
    question_id: UUID
    stage: StageKind
    revision: int = Field(ge=0)
    idempotency_key: NonEmptyStr
    status: ActivityStatus
    attempt_count: int = Field(ge=0)
    request_json: dict[str, Any] | None = None
    result_json: dict[str, Any] | None = None
    error: str | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class EventRecord(StrictModel):
    seq: int = Field(ge=1)
    event_id: UUID
    project_id: UUID
    question_id: UUID | None
    activity_id: UUID | None
    event_type: NonEmptyStr
    payload: dict[str, Any]
    occurred_at: datetime
    previous_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    event_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ClaimEvidenceRecord(StrictModel):
    """A published claim linked to the immutable artifacts available to review."""

    id: UUID
    project_id: UUID
    question_id: UUID
    activity_id: UUID
    revision: int = Field(ge=0)
    claim_index: int = Field(ge=0)
    claim: NonEmptyStr
    evidence_artifact_ids: tuple[UUID, ...] = ()
    created_at: datetime


class WorkflowStepResult(StrictModel):
    question: QuestionRecord
    activity: ActivityRecord | None = None
    published_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    terminal: bool = False


def new_uuid() -> UUID:
    return uuid4()
