"""The report-driven ``solve_problem`` tool.

Main Harness owns the process and calls this module as a Tool.  The module is
not a top-level subagent dispatcher.  It is a bounded internal protocol which
lets a Model Agent explore/synthesize a modeling plan, lets a Code Harness
realize that plan, and sends the Coding Report back to the Model Agent until
the model approves or the iteration budget is exhausted.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from m2harness.domain.solve_problem import (
    CodingHarnessReport,
    ExplorationMode,
    PreliminaryModelingReport,
    SolveProblemContext,
    SolveProblemIteration,
    SolveProblemReport,
    SolveProblemStatus,
    SolveProblemReview,
    SolveProblemTask,
    UnifiedModelingReport,
    ModelReviewDecision,
    RevisionTarget,
    DisclosedTextFile,
)
from m2harness.errors import ActivityExecutionError, ConfigurationError
from m2harness.models import ReportPayload
from m2harness.application.research import ResearchAgentPort


class ModelAgentPort(Protocol):
    """Model Agent boundary used inside one solve_problem invocation."""

    def explore(
        self,
        task: SolveProblemTask,
        context: SolveProblemContext,
        *,
        branch_count: int,
        iteration: int,
    ) -> tuple[PreliminaryModelingReport, ...]: ...

    def synthesize(
        self,
        task: SolveProblemTask,
        context: SolveProblemContext,
        preliminary: tuple[PreliminaryModelingReport, ...],
        *,
        iteration: int,
    ) -> UnifiedModelingReport: ...

    def review(
        self,
        task: SolveProblemTask,
        context: SolveProblemContext,
        modeling: UnifiedModelingReport,
        coding: CodingHarnessReport,
        *,
        iteration: int,
    ) -> SolveProblemReview: ...

    def compose_final_report(
        self,
        task: SolveProblemTask,
        context: SolveProblemContext,
        modeling: UnifiedModelingReport,
        coding: CodingHarnessReport,
        review: SolveProblemReview,
        *,
        iteration: int,
    ) -> ReportPayload: ...


class CodeHarnessPort(Protocol):
    """Computational realization boundary used inside solve_problem."""

    def execute(
        self,
        task: SolveProblemTask,
        context: SolveProblemContext,
        modeling: UnifiedModelingReport,
        *,
        iteration: int,
    ) -> CodingHarnessReport: ...


class ReadOnlyFileReaderPort(Protocol):
    """Resolve only paths explicitly allowlisted in SolveProblemContext."""

    def disclose(self, context: SolveProblemContext, paths: Sequence[str]) -> tuple[DisclosedTextFile, ...]: ...


class WorkspaceReadOnlyFileReader:
    """Bounded verified reader used by progressive disclosure."""

    def __init__(self, workspace_root: Path, *, max_file_bytes: int = 256_000, max_total_bytes: int = 1_000_000) -> None:
        self.workspace_root = workspace_root.resolve()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    def disclose(self, context: SolveProblemContext, paths: Sequence[str]) -> tuple[DisclosedTextFile, ...]:
        allowlist = {item.relative_path: item for item in context.readonly_files}
        existing = {item.relative_path for item in context.disclosed_text_files}
        result: list[DisclosedTextFile] = []
        total = 0
        for requested in dict.fromkeys(path.replace("\\", "/") for path in paths):
            if requested in existing:
                continue
            reference = allowlist.get(requested)
            if reference is None:
                raise PermissionError(f"requested file is not disclosed by Main Harness: {requested}")
            target = (self.workspace_root / requested).resolve()
            if self.workspace_root != target and self.workspace_root not in target.parents:
                raise PermissionError(f"requested file escapes workspace: {requested}")
            if not target.is_file() or target.is_symlink():
                raise FileNotFoundError(f"disclosed file is unavailable: {requested}")
            raw = target.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if reference.sha256 is not None and digest != reference.sha256:
                raise ValueError(f"disclosed file digest changed: {requested}")
            if reference.size_bytes is not None and len(raw) != reference.size_bytes:
                raise ValueError(f"disclosed file size changed: {requested}")
            remaining = self.max_total_bytes - total
            if remaining <= 0:
                break
            limit = min(self.max_file_bytes, remaining)
            sample = raw[:limit]
            try:
                content = sample.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"only UTF-8 text can be progressively disclosed: {requested}") from exc
            result.append(DisclosedTextFile(
                relative_path=requested, purpose=reference.purpose, content=content,
                sha256=digest, truncated=len(raw) > limit,
            ))
            total += len(sample)
        return tuple(result)


@dataclass(frozen=True)
class SolveProblemService:
    """Run one bounded report-driven solve loop.

    The service deliberately accepts ports instead of constructing a hidden
    agent tree.  A production deployment can back ``ModelAgentPort`` with a
    Qwen/OpenAI-compatible provider and ``CodeHarnessPort`` with the trusted
    local or container execution backend.  Tests and operators can inspect the
    exact reports exchanged without mocking an invisible orchestration layer.
    """

    model_agent: ModelAgentPort
    code_harness: CodeHarnessPort
    max_iterations: int = 3
    research_agent: ResearchAgentPort | None = None
    file_reader: ReadOnlyFileReaderPort | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_iterations > 20:
            raise ValueError("solve_problem max_iterations must be between 1 and 20")

    @staticmethod
    def branch_count(task: SolveProblemTask) -> int:
        if task.exploration_mode == ExplorationMode.SINGLE:
            return 1
        if task.exploration_mode == ExplorationMode.MULTI:
            return min(task.max_branches, max(2, task.difficulty))
        # AUTO is a policy decision, not a mandatory multi-agent phase.  Easy
        # tasks stay cheap; ambiguous/difficult tasks get a bounded set of
        # independent preliminary routes for Model Agent synthesis.
        return 1 if task.difficulty <= 2 else min(task.max_branches, max(2, (task.difficulty + 1) // 2))

    def solve(self, task: SolveProblemTask, context: SolveProblemContext | None = None) -> SolveProblemReport:
        current_context = context or SolveProblemContext()
        if self.research_agent is not None and current_context.research_report is None:
            research = self.research_agent.research(
                task.problem,
                max_facets=min(8, max(3, task.difficulty)),
                top_k=min(12, max(4, task.max_branches * 2)),
            )
            current_context = current_context.model_copy(update={"research_report": research})
        iterations: list[SolveProblemIteration] = []
        revision_instructions: tuple[str, ...] = ()
        revision_target = RevisionTarget.FULL
        preliminary: tuple[PreliminaryModelingReport, ...] = ()
        modeling: UnifiedModelingReport | None = None
        branch_count = self.branch_count(task)
        for iteration_number in range(1, self.max_iterations + 1):
            if revision_instructions:
                current_context = current_context.model_copy(update={
                    "instructions": (*current_context.instructions, *revision_instructions),
                })
            if modeling is None or revision_target in {RevisionTarget.MODEL, RevisionTarget.FULL}:
                preliminary = tuple(self.model_agent.explore(
                    task, current_context, branch_count=branch_count, iteration=iteration_number,
                ))
                requested = tuple(dict.fromkeys(
                    path for report in preliminary for path in report.requested_file_paths
                ))
                if requested and self.file_reader is not None:
                    disclosed = self.file_reader.disclose(current_context, requested)
                    if disclosed:
                        current_context = current_context.model_copy(update={
                            "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed),
                        })
                        preliminary = tuple(self.model_agent.explore(
                            task, current_context, branch_count=branch_count, iteration=iteration_number,
                        ))
            if not preliminary:
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=iteration_number - 1, iterations=tuple(iterations),
                    research_report=current_context.research_report,
                    error="Model Agent returned no preliminary modeling report",
                )
            branch_ids = [item.branch_id for item in preliminary]
            if len(branch_ids) != len(set(branch_ids)) or len(preliminary) > branch_count:
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=iteration_number - 1, iterations=tuple(iterations),
                    research_report=current_context.research_report,
                    error="Model Agent returned duplicate or excess preliminary branch ids",
                )
            if modeling is None or revision_target in {RevisionTarget.MODEL, RevisionTarget.FULL}:
                modeling = self.model_agent.synthesize(
                    task, current_context, preliminary, iteration=iteration_number,
                )
            if not set(modeling.selected_branch_ids).issubset(set(branch_ids)):
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=iteration_number - 1, iterations=tuple(iterations),
                    research_report=current_context.research_report,
                    error="Unified Modeling Report selected an unknown preliminary branch",
                )
            coding = self.code_harness.execute(
                task, current_context, modeling, iteration=iteration_number,
            )
            review = self.model_agent.review(
                task, current_context, modeling, coding, iteration=iteration_number,
            )
            snapshot = SolveProblemIteration(
                iteration=iteration_number, preliminary_reports=preliminary,
                modeling_report=modeling, coding_report=coding, review=review,
            )
            iterations.append(snapshot)
            if review.decision == ModelReviewDecision.APPROVE:
                if not coding.execution_succeeded:
                    revision_instructions = (
                        "Code Harness execution must succeed and provide validation evidence before approval.",
                    )
                    continue
                final_report = self.model_agent.compose_final_report(
                    task, current_context, modeling, coding, review,
                    iteration=iteration_number,
                )
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.COMPLETED,
                    iteration_count=iteration_number, iterations=tuple(iterations),
                    final_report=final_report, artifacts=list(coding.artifacts),
                    research_report=current_context.research_report,
                )
            revision_instructions = review.revision_instructions
            revision_target = review.revision_target
            if review.requested_file_paths and self.file_reader is not None:
                disclosed = self.file_reader.disclose(current_context, review.requested_file_paths)
                if disclosed:
                    current_context = current_context.model_copy(update={
                        "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed),
                    })

        return SolveProblemReport(
            task_id=task.task_id, status=SolveProblemStatus.REVISION_REQUIRED,
            iteration_count=len(iterations), iterations=tuple(iterations),
            revision_instructions=revision_instructions,
            artifacts=list(iterations[-1].coding_report.artifacts) if iterations else [],
            research_report=current_context.research_report,
            error="solve_problem iteration budget exhausted",
        )


class CommandSolveProblemRunner:
    """Optional external solve_problem adapter using one JSON request/response.

    This is the deployment boundary for a real Model Agent + Code Harness
    implementation.  It is intentionally not enabled by default and never
    uses a shell.  The child process must return a validated
    ``SolveProblemReport`` and the same task id.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int = 3_600,
        cwd: Path | None = None,
        pass_env: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        max_output_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not command or not command[0].strip():
            raise ConfigurationError("solve_problem command cannot be empty")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.pass_env = tuple(pass_env)
        self.environment = dict(environment or {})
        self.max_output_bytes = max_output_bytes

    def execute(self, task: SolveProblemTask, context: SolveProblemContext) -> SolveProblemReport:
        inherited = ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        env = {name: os.environ[name] for name in (*inherited, *self.pass_env) if name in os.environ}
        env.update(self.environment)
        payload = {"task": task.model_dump(mode="json"), "context": context.model_dump(mode="json")}
        try:
            process = subprocess.run(
                self.command, input=json.dumps(payload, ensure_ascii=False), text=True,
                encoding="utf-8", errors="strict", capture_output=True,
                timeout=self.timeout_seconds, cwd=self.cwd, env=env,
                shell=False, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ActivityExecutionError(f"solve_problem adapter failed: {exc}") from exc
        stdout = process.stdout.encode("utf-8")
        if len(stdout) > self.max_output_bytes:
            raise ActivityExecutionError("solve_problem adapter response exceeded byte limit")
        if process.returncode != 0:
            raise ActivityExecutionError(f"solve_problem adapter exited with code {process.returncode}: {process.stderr[-4000:]}")
        try:
            result = SolveProblemReport.model_validate_json(stdout, strict=False)
        except Exception as exc:
            raise ActivityExecutionError(f"solve_problem adapter returned invalid protocol JSON: {exc}") from exc
        if result.task_id != task.task_id:
            raise ActivityExecutionError("solve_problem adapter task id does not match request")
        return result


class UnconfiguredSolveProblemService:
    """Fail-closed marker used by the local runtime until real ports are wired."""

    def solve(self, task: SolveProblemTask, context: SolveProblemContext | None = None) -> SolveProblemReport:
        return SolveProblemReport(
            task_id=task.task_id, status=SolveProblemStatus.BLOCKED,
            iteration_count=0, error="solve_problem Model Agent and Code Harness are not configured",
        )
