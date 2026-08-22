"""Contracts for the main Harness ``solve_problem`` tool.

The top-level Harness does not dispatch a collection of stage subagents.  It
dispatches one durable tool, ``solve_problem``.  This tool owns the bounded
modeling/code/review loop for one DAG task and communicates with its caller by
reports and immutable artifacts.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from m2harness.models import ProducedArtifact, ReportPayload, StrictModel
from m2harness.domain.research import ResearchReport
from m2harness.domain.media import MultimodalInput


# Initial implementation is not counted as a revision.  This hard workflow
# policy permits at most two review-driven return rounds for one task.
MAX_REVISION_ROUNDS = 2


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


class RevisionTarget(StrEnum):
    """Compatibility values; active Code→Model repair routing always uses CODE."""

    CODE = "code"
    MODEL = "model"
    FULL = "full"


class ReadOnlyFileRole(StrEnum):
    PROBLEM = "problem"
    INPUT = "input"
    DEPENDENCY_SOLUTION = "dependency_solution"
    DEPENDENCY_OUTPUT = "dependency_output"
    GENERATED = "generated"
    REFERENCE = "reference"


class ReadOnlyFileReference(StrictModel):
    """A workspace-relative allowlist entry, not a general filesystem grant."""

    relative_path: str = Field(min_length=1, max_length=1_000)
    purpose: str = Field(min_length=1, max_length=2_000)
    role: ReadOnlyFileRole
    owner_task_id: str | None = Field(default=None, max_length=120)
    media_type: str = Field(default="application/octet-stream", min_length=1, max_length=200)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def workspace_relative_path_only(self) -> "ReadOnlyFileReference":
        normalized = self.relative_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("read-only file path must be a normalized workspace-relative path")
        if ":" in path.parts[0]:
            raise ValueError("read-only file path cannot contain a drive prefix")
        object.__setattr__(self, "relative_path", path.as_posix())
        return self


class DisclosedTextFile(StrictModel):
    """Verified text disclosed after an agent requests an allowlisted path."""

    relative_path: str
    purpose: str
    content: str = Field(max_length=512_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    truncated: bool = False


class DependencySolutionContext(StrictModel):
    """Compressed, high-value projection of one accepted upstream solution."""

    task_id: str
    report_id: UUID
    title: str
    summary: str = Field(max_length=30_000)
    claims: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    downstream_outputs: dict[str, Any] = Field(default_factory=dict)
    solution_report_path: str | None = None
    solve_files: tuple[ReadOnlyFileReference, ...] = ()
    estimated_tokens: int = Field(default=0, ge=0)


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
    multimodal_inputs: tuple[MultimodalInput, ...] = ()
    dependency_report_ids: tuple[UUID, ...] = ()
    accepted_report_ids: tuple[UUID, ...] = ()
    dependency_solutions: tuple[DependencySolutionContext, ...] = ()
    readonly_files: tuple[ReadOnlyFileReference, ...] = ()
    disclosed_text_files: tuple[DisclosedTextFile, ...] = ()
    instructions: tuple[str, ...] = ()
    research_report: ResearchReport | None = None
    context_budget_tokens: int = Field(default=32_000, ge=1_000, le=1_000_000)
    compression: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplorationMaterial(StrictModel):
    """Internal route material produced by the exploration pass."""

    branch_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9-]*[a-z0-9])?$", min_length=1, max_length=120)
    report: ReportPayload
    candidate_scheme: str = Field(min_length=1, max_length=20_000)
    assumptions: tuple[str, ...] = ()
    required_validations: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    expected_figures: tuple[str, ...] = ()
    coding_instructions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    requested_file_paths: tuple[str, ...] = ()


class PreliminaryModelingReport(StrictModel):
    """The one report formed after exploration has been summarized."""

    branch_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9-]*[a-z0-9])?$", min_length=1, max_length=120)
    report: ReportPayload
    candidate_scheme: str = Field(min_length=1, max_length=20_000)
    assumptions: tuple[str, ...] = ()
    required_validations: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    expected_figures: tuple[str, ...] = ()
    coding_instructions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    requested_file_paths: tuple[str, ...] = ()
    exploration_summary: ReportPayload | None = None


class UnifiedModelingReport(StrictModel):
    """The Model Agent's selected and executable modeling contract."""

    report: ReportPayload
    selected_branch_ids: tuple[str, ...] = Field(min_length=1)
    main_scheme: str = Field(min_length=1, max_length=30_000)
    required_validations: tuple[str, ...] = Field(min_length=1)
    expected_outputs: tuple[str, ...] = Field(min_length=1)
    expected_figures: tuple[str, ...] = ()
    coding_instructions: tuple[str, ...] = ()
    rejected_branch_reasons: dict[str, str] = Field(default_factory=dict)


class CodingHarnessReport(StrictModel):
    """Code Harness output returned to the Model Agent for review."""

    report: ReportPayload
    execution_succeeded: bool
    validations: dict[str, bool]
    validation_evidence: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    deviations: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    artifacts: list[ProducedArtifact] = Field(default_factory=list)
    generated_files: tuple[ReadOnlyFileReference, ...] = ()

    @model_validator(mode="after")
    def successful_execution_requires_evidence(self) -> "CodingHarnessReport":
        if self.execution_succeeded and not (self.artifacts or self.generated_files):
            raise ValueError("successful solve_problem execution must include evidence artifacts")
        return self


class SolveProblemReview(StrictModel):
    """Model Agent decision after inspecting the Coding Report."""

    decision: ModelReviewDecision
    rationale: str = Field(min_length=1, max_length=30_000)
    revision_instructions: tuple[str, ...] = ()
    accepted_claims: tuple[str, ...] = ()
    revision_target: RevisionTarget = RevisionTarget.CODE
    requested_file_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def revision_requires_instructions(self) -> "SolveProblemReview":
        if self.decision in {ModelReviewDecision.REVISE, ModelReviewDecision.REJECT} and not self.revision_instructions:
            raise ValueError("revise/reject decisions must include revision_instructions")
        return self


class SolveProblemIteration(StrictModel):
    """Durable snapshot of one model → code → model-review iteration."""

    iteration: int = Field(ge=1)
    preliminary_reports: tuple[PreliminaryModelingReport, ...] = ()
    # Only the terminal iteration contains the final unified modeling report.
    # Earlier review/revision iterations carry the preliminary contract only;
    # this prevents a premature ``modeling_report.md`` from becoming the
    # source of truth before Code has produced evidence.
    modeling_report: UnifiedModelingReport | None = None
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
    archive_files: tuple[ReadOnlyFileReference, ...] = ()
    research_report: ResearchReport | None = None
    error: str | None = None

    @model_validator(mode="after")
    def completed_requires_final_report(self) -> "SolveProblemReport":
        if self.status == SolveProblemStatus.COMPLETED and self.final_report is None:
            raise ValueError("completed solve_problem result must include final_report")
        if self.status == SolveProblemStatus.REVISION_REQUIRED and not self.revision_instructions:
            raise ValueError("revision_required solve_problem result must include instructions")
        return self
