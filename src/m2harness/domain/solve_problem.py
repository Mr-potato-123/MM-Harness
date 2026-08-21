"""Contracts for the main Harness ``solve_problem`` tool.

The top-level Harness does not dispatch a collection of stage subagents.  It
dispatches one durable tool, ``solve_problem``.  This tool owns the bounded
modeling/code/review loop for one DAG task and communicates with its caller by
reports and immutable artifacts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from m2harness.models import ProducedArtifact, ReportPayload, StrictModel
from m2harness.domain.research import ResearchReport


class ExplorationMode(StrEnum):
    AUTO = "auto"
    SINGLE = "single"
    MULTI = "multi"


class SolveProblemStatus(StrEnum):
    COMPLETED = "completed"
    REVISION_REQUIRED = "revision_required"
    BLOCKED = "blocked"
    FAILED = "failed"


class ModelReviewDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class SolveProblemTask(StrictModel):
    """One node assigned by Main Harness to ``solve_problem``."""

    task_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=200_000)
    dependencies: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    difficulty: int = Field(default=1, ge=1, le=10)
    exploration_mode: ExplorationMode = ExplorationMode.AUTO
    max_branches: int = Field(default=3, ge=1, le=16)
    revision: int = Field(default=0, ge=0, le=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolveProblemContext(StrictModel):
    """Progressively disclosed context passed from Main Harness."""

    source_artifact_ids: tuple[UUID, ...] = ()
    dependency_report_ids: tuple[UUID, ...] = ()
    accepted_report_ids: tuple[UUID, ...] = ()
    instructions: tuple[str, ...] = ()
    research_report: ResearchReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreliminaryModelingReport(StrictModel):
    """One independent modeling route produced inside the solve tool."""

    branch_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9-]*[a-z0-9])?$", min_length=1, max_length=120)
    report: ReportPayload
    candidate_scheme: str = Field(min_length=1, max_length=20_000)
    assumptions: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UnifiedModelingReport(StrictModel):
    """The Model Agent's selected and executable modeling contract."""

    report: ReportPayload
    selected_branch_ids: tuple[str, ...] = Field(min_length=1)
    main_scheme: str = Field(min_length=1, max_length=30_000)
    required_validations: tuple[str, ...] = Field(min_length=1)
    expected_outputs: tuple[str, ...] = Field(min_length=1)
    coding_instructions: tuple[str, ...] = ()
    rejected_branch_reasons: dict[str, str] = Field(default_factory=dict)


class CodingHarnessReport(StrictModel):
    """Code Harness output returned to the Model Agent for review."""

    report: ReportPayload
    execution_succeeded: bool
    validations: dict[str, bool]
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    deviations: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    artifacts: list[ProducedArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def successful_execution_requires_evidence(self) -> "CodingHarnessReport":
        if self.execution_succeeded and not self.artifacts:
            raise ValueError("successful solve_problem execution must include evidence artifacts")
        return self


class SolveProblemReview(StrictModel):
    """Model Agent decision after inspecting the Coding Report."""

    decision: ModelReviewDecision
    rationale: str = Field(min_length=1, max_length=30_000)
    revision_instructions: tuple[str, ...] = ()
    accepted_claims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def revision_requires_instructions(self) -> "SolveProblemReview":
        if self.decision in {ModelReviewDecision.REVISE, ModelReviewDecision.REJECT} and not self.revision_instructions:
            raise ValueError("revise/reject decisions must include revision_instructions")
        return self


class SolveProblemIteration(StrictModel):
    """Durable snapshot of one model → code → model-review iteration."""

    iteration: int = Field(ge=1)
    preliminary_reports: tuple[PreliminaryModelingReport, ...] = ()
    modeling_report: UnifiedModelingReport
    coding_report: CodingHarnessReport
    review: SolveProblemReview


class SolveProblemReport(StrictModel):
    """The only result Main Harness receives from ``solve_problem``."""

    protocol_version: Literal[1] = 1
    task_id: str
    status: SolveProblemStatus
    iteration_count: int = Field(ge=0)
    iterations: tuple[SolveProblemIteration, ...] = ()
    final_report: ReportPayload | None = None
    revision_instructions: tuple[str, ...] = ()
    artifacts: list[ProducedArtifact] = Field(default_factory=list)
    research_report: ResearchReport | None = None
    error: str | None = None

    @model_validator(mode="after")
    def completed_requires_final_report(self) -> "SolveProblemReport":
        if self.status == SolveProblemStatus.COMPLETED and self.final_report is None:
            raise ValueError("completed solve_problem result must include final_report")
        if self.status == SolveProblemStatus.REVISION_REQUIRED and not self.revision_instructions:
            raise ValueError("revision_required solve_problem result must include instructions")
        return self
