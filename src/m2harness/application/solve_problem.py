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
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID

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
    ReadOnlyFileReference,
    ReadOnlyFileRole,
    MAX_REVISION_ROUNDS,
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


class SolveProblemArchivePort(Protocol):
    """Durably archive one live Agent exchange before the next stage runs."""

    def archive_markdown(
        self,
        run_id: UUID,
        *,
        task_id: str,
        attempt: int,
        iteration: int,
        stage: str,
        name: str,
        markdown: str,
        purpose: str,
        role: ReadOnlyFileRole,
    ) -> ReadOnlyFileReference: ...


class SolveProblemProbePort(Protocol):
    """Durable, operator-readable probe for every solve stage boundary.

    Handoff Markdown explains *what* is transferred.  The probe explains
    *when* it happened, which actor was active, and whether the boundary
    succeeded.  Implementations must not persist secrets or full source
    payloads; only bounded metadata and verified paths belong here.
    """

    def record_probe(
        self,
        run_id: UUID,
        *,
        task_id: str,
        attempt: int,
        iteration: int | None,
        event: str,
        actor: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> None: ...


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
                # Binary inputs (PDF/images/data files) are already supplied
                # through ``multimodal_inputs`` or the trusted tool lane.  A
                # Model Agent may still mention their allowlisted path while
                # reasoning, but that request must not abort the whole solve
                # loop merely because the bytes are not UTF-8 text.
                if reference.media_type != "text/plain" and not reference.media_type.startswith("text/"):
                    continue
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
    archive_writer: SolveProblemArchivePort | None = None
    probe_writer: SolveProblemProbePort | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_iterations > 20:
            raise ValueError("solve_problem max_iterations must be between 1 and 20")

    @property
    def revision_round_limit(self) -> int:
        """Return the effective revision budget (initial round excluded)."""

        return min(MAX_REVISION_ROUNDS, max(0, self.max_iterations - 1))

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
        self._probe(current_context, task, None, "solve_start", "Main Harness", "started", {
            "max_iterations": self.max_iterations,
            "revision_round_limit": self.revision_round_limit,
            "readonly_file_count": len(current_context.readonly_files),
            "multimodal_input_count": len(current_context.multimodal_inputs),
        })
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
        resume_iteration_value = current_context.metadata.get("resume_iteration")
        resume_modeling_value = current_context.metadata.get("resume_modeling")
        resume_mode = resume_iteration_value is not None or resume_modeling_value is not None
        if resume_mode:
            if not isinstance(resume_iteration_value, int) or resume_iteration_value < 1:
                raise ValueError("resume_iteration must be a positive integer")
            if not isinstance(resume_modeling_value, dict):
                raise ValueError("resume_modeling must contain a serialized UnifiedModelingReport")
            modeling = UnifiedModelingReport.model_validate(resume_modeling_value, strict=False)
            raw_preliminary = current_context.metadata.get("resume_preliminary", ())
            if not isinstance(raw_preliminary, (list, tuple)):
                raise ValueError("resume_preliminary must be a list of serialized reports")
            preliminary = tuple(
                PreliminaryModelingReport.model_validate(item, strict=False)
                for item in raw_preliminary
                if isinstance(item, dict)
            )
            if not preliminary:
                raise ValueError("resume_preliminary cannot be empty")
            raw_target = current_context.metadata.get("resume_revision_target", RevisionTarget.CODE.value)
            revision_target = RevisionTarget(raw_target)
            start_iteration = resume_iteration_value
            clean_metadata = {
                key: value for key, value in current_context.metadata.items()
                if key not in {"resume_iteration", "resume_modeling", "resume_preliminary", "resume_revision_target"}
            }
            current_context = current_context.model_copy(update={"metadata": clean_metadata})
        else:
            start_iteration = 1
        # ``max_iterations`` is retained as a compatibility/configuration knob,
        # but the workflow never permits more than two review-driven revisions.
        iteration_limit = min(self.max_iterations, MAX_REVISION_ROUNDS + 1)
        if start_iteration > iteration_limit:
            raise ValueError("resume_iteration exceeds the configured solve_problem iteration limit")
        for iteration_number in range(start_iteration, iteration_limit + 1):
            if revision_instructions:
                current_context = current_context.model_copy(update={
                    "instructions": (*current_context.instructions, *revision_instructions),
                })
            if modeling is None or revision_target in {RevisionTarget.MODEL, RevisionTarget.FULL}:
                self._probe(current_context, task, iteration_number, "model_explore_start", "Model Agent", "started", {
                    "branch_count": branch_count,
                    "revision_target": revision_target.value,
                })
                preliminary = tuple(self.model_agent.explore(
                    task, current_context, branch_count=branch_count, iteration=iteration_number,
                ))
                self._probe(current_context, task, iteration_number, "model_explore_complete", "Model Agent", "completed", {
                    "branch_ids": [item.branch_id for item in preliminary],
                    "requested_file_paths": [path for item in preliminary for path in item.requested_file_paths],
                })
                for preliminary_report in preliminary:
                    current_context = self._archive(
                        current_context, task, iteration_number, "modeling",
                        f"preliminary-{preliminary_report.branch_id}",
                        _preliminary_markdown(preliminary_report),
                        purpose=f"Model Agent preliminary route {preliminary_report.branch_id} for {task.task_id}, iteration {iteration_number}.",
                        role=ReadOnlyFileRole.REFERENCE,
                    )
                current_context = self._record_conversation(
                    current_context, task, iteration_number, "Model Agent",
                    "已完成初步路线；路线报告已保存，等待统一建模。",
                )
                requested = tuple(dict.fromkeys(
                    path for report in preliminary for path in report.requested_file_paths
                ))
                if requested and self.file_reader is not None:
                    self._probe(current_context, task, iteration_number, "file_disclosure_requested", "Model Agent", "started", {
                        "paths": list(requested),
                    })
                    try:
                        disclosed = self.file_reader.disclose(current_context, requested)
                    except Exception as exc:
                        self._probe(current_context, task, iteration_number, "file_disclosure_failed", "Main Harness", "failed", {
                            "paths": list(requested), "error": str(exc)[:2_000],
                        })
                        current_context = self._archive(
                            current_context, task, iteration_number, "handoff", "disclosure-failure",
                            _disclosure_failure_markdown(task, iteration_number, requested, exc),
                            purpose=f"Progressive disclosure failure before the next Model Agent boundary for {task.task_id}.",
                            role=ReadOnlyFileRole.REFERENCE,
                        )
                        return SolveProblemReport(
                            task_id=task.task_id, status=SolveProblemStatus.FAILED,
                            iteration_count=max(0, iteration_number - 1), iterations=tuple(iterations),
                            archive_files=self._archive_files(current_context, task),
                            research_report=current_context.research_report,
                            error=f"progressive disclosure failed: {exc}",
                        )
                    self._probe(current_context, task, iteration_number, "file_disclosure_complete", "Main Harness", "completed", {
                        "requested_paths": list(requested),
                        "disclosed_paths": [item.relative_path for item in disclosed],
                        "skipped_binary_paths": [path for path in requested if path not in {item.relative_path for item in disclosed}],
                    })
                    if disclosed:
                        current_context = current_context.model_copy(update={
                            "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed),
                        })
                        preliminary = tuple(self.model_agent.explore(
                            task, current_context, branch_count=branch_count, iteration=iteration_number,
                        ))
                        for preliminary_report in preliminary:
                            current_context = self._archive(
                                current_context, task, iteration_number, "modeling",
                                f"preliminary-{preliminary_report.branch_id}-after-disclosure",
                                _preliminary_markdown(preliminary_report),
                                purpose=f"Model Agent preliminary route {preliminary_report.branch_id} after file disclosure for {task.task_id}, iteration {iteration_number}.",
                                role=ReadOnlyFileRole.REFERENCE,
                            )
            if not preliminary:
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=max(0, iteration_number - 1), iterations=tuple(iterations),
                    archive_files=self._archive_files(current_context, task),
                    research_report=current_context.research_report,
                    error="Model Agent returned no preliminary modeling report",
                )
            branch_ids = [item.branch_id for item in preliminary]
            if len(branch_ids) != len(set(branch_ids)) or len(preliminary) > branch_count:
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=max(0, iteration_number - 1), iterations=tuple(iterations),
                    archive_files=self._archive_files(current_context, task),
                    research_report=current_context.research_report,
                    error="Model Agent returned duplicate or excess preliminary branch ids",
                )
            if modeling is None or revision_target in {RevisionTarget.MODEL, RevisionTarget.FULL}:
                self._probe(current_context, task, iteration_number, "model_synthesize_start", "Model Agent", "started", {})
                modeling = self.model_agent.synthesize(
                    task, current_context, preliminary, iteration=iteration_number,
                )
                self._probe(current_context, task, iteration_number, "model_synthesize_complete", "Model Agent", "completed", {
                    "selected_branch_ids": list(modeling.selected_branch_ids),
                    "required_validations": list(modeling.required_validations),
                    "expected_outputs": list(modeling.expected_outputs),
                })
                current_context = self._archive(
                    current_context, task, iteration_number, "modeling", "modeling_report",
                    _modeling_markdown(modeling),
                    purpose=f"Unified Model Agent modeling plan for {task.task_id}, iteration {iteration_number}.",
                    role=ReadOnlyFileRole.REFERENCE,
                )
                current_context = self._record_conversation(
                    current_context, task, iteration_number, "Model Agent",
                    f"统一建模已完成。主方案：{modeling.main_scheme[:1800]}；验证项：{', '.join(modeling.required_validations[:12])}。",
                )
            # Make the exact Model -> Code contract an explicit, durable
            # handoff.  The Code Agent still receives the typed modeling
            # object, but the artifact makes the boundary inspectable and
            # gives it a verified path it may read through the tool lane.
            current_context = self._archive(
                current_context, task, iteration_number, "handoff", "model-to-code",
                _compact_model_to_code_handoff(
                    task, current_context, preliminary, modeling, iteration_number,
                    max_revision_rounds=self.revision_round_limit,
                ),
                purpose=f"Explicit Model Agent to Code Agent handoff for {task.task_id}, iteration {iteration_number}.",
                role=ReadOnlyFileRole.REFERENCE,
            )
            self._probe(current_context, task, iteration_number, "model_to_code_handoff", "Model Agent", "completed", {
                "handoff": "model-to-code.md",
                "required_validations": list(modeling.required_validations),
                "expected_outputs": list(modeling.expected_outputs),
            })
            if not set(modeling.selected_branch_ids).issubset(set(branch_ids)):
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.FAILED,
                    iteration_count=iteration_number - 1, iterations=tuple(iterations),
                    archive_files=self._archive_files(current_context, task),
                    research_report=current_context.research_report,
                    error="Unified Modeling Report selected an unknown preliminary branch",
                )
            self._probe(current_context, task, iteration_number, "code_execute_start", "Code Agent", "started", {
                "revision_target": revision_target.value,
            })
            try:
                coding = self.code_harness.execute(
                task, current_context, modeling, iteration=iteration_number,
                )
            except Exception as exc:
                self._probe(current_context, task, iteration_number, "code_execute_failed", "Code Agent", "failed", {"error": str(exc)[:2_000]})
                raise
            self._probe(current_context, task, iteration_number, "code_execute_complete", "Code Agent", "completed", {
                "execution_succeeded": coding.execution_succeeded,
                "validations": coding.validations,
                "generated_files": [item.relative_path for item in coding.generated_files],
                "artifact_names": [item.logical_name for item in coding.artifacts],
            })
            # Generated source/output files become reviewable evidence only
            # after the Code Harness has materialized and hashed them.  Merge
            # those references into the read-only manifest before asking the
            # Review Agent to inspect a path; this preserves progressive
            # disclosure without granting arbitrary workspace access.
            if coding.generated_files:
                known = {item.relative_path: item for item in current_context.readonly_files}
                known.update({item.relative_path: item for item in coding.generated_files})
                current_context = current_context.model_copy(update={"readonly_files": tuple(known.values())})
            current_context = self._archive(
                current_context, task, iteration_number, "coding", "coding_report",
                _coding_markdown(coding),
                purpose=f"Code Harness execution report for {task.task_id}, iteration {iteration_number}.",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT,
            )
            current_context = self._record_conversation(
                current_context, task, iteration_number, "Code Agent",
                f"代码执行返回：成功={coding.execution_succeeded}；验证={coding.validations}；生成文件={', '.join(item.relative_path for item in coding.generated_files)}。",
                channel="code",
            )
            current_context = self._archive(
                current_context, task, iteration_number, "handoff", "code-to-review",
                _compact_code_to_review_handoff(
                    task, current_context, modeling, coding, iteration_number,
                    max_revision_rounds=self.revision_round_limit,
                ),
                purpose=f"Explicit Code Harness to Review Agent handoff for {task.task_id}, iteration {iteration_number}.",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT,
            )
            self._probe(current_context, task, iteration_number, "code_to_review_handoff", "Code Agent", "completed", {
                "handoff": "code-to-review.md",
                "execution_succeeded": coding.execution_succeeded,
                "generated_files": [item.relative_path for item in coding.generated_files],
            })
            self._probe(current_context, task, iteration_number, "review_start", "Review Agent", "started", {})
            review = self.model_agent.review(
                task, current_context, modeling, coding, iteration=iteration_number,
            )
            self._probe(current_context, task, iteration_number, "review_complete", "Review Agent", "completed", {
                "decision": review.decision.value,
                "revision_target": review.revision_target.value,
                "revision_instruction_count": len(review.revision_instructions),
                "requested_file_paths": list(review.requested_file_paths),
            })
            current_context = self._archive(
                current_context, task, iteration_number, "review", "review_report",
                _review_markdown(review),
                purpose=f"Review Agent decision for {task.task_id}, iteration {iteration_number}.",
                role=ReadOnlyFileRole.REFERENCE,
            )
            self._probe(current_context, task, iteration_number, "review_to_next_stage_handoff", "Review Agent", "completed", {
                "handoff": "review-to-next-stage.md",
                "decision": review.decision.value,
                "revision_target": review.revision_target.value,
                "revision_instructions": list(review.revision_instructions),
            })
            current_context = self._record_conversation(
                current_context, task, iteration_number, "Review Agent",
                f"审查决定：{review.decision.value}；返修目标：{review.revision_target.value}；指令：{'；'.join(review.revision_instructions[:8])}。",
            )
            current_context = self._archive(
                current_context, task, iteration_number, "handoff", "review-to-next-stage",
                _compact_review_to_next_stage_handoff(
                    task, current_context, review, iteration_number,
                    max_revision_rounds=self.revision_round_limit,
                ),
                purpose=f"Explicit Review Agent decision handoff for {task.task_id}, iteration {iteration_number}.",
                role=ReadOnlyFileRole.REFERENCE,
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
                self._probe(current_context, task, iteration_number, "final_report_composed", "Model Agent", "completed", {
                    "title": final_report.title,
                    "markdown_chars": len(final_report.markdown),
                })
                return SolveProblemReport(
                    task_id=task.task_id, status=SolveProblemStatus.COMPLETED,
                    iteration_count=iteration_number, iterations=tuple(iterations),
                    final_report=final_report, artifacts=list(coding.artifacts),
                    archive_files=self._archive_files(current_context, task),
                    research_report=current_context.research_report,
                )
            revision_instructions = review.revision_instructions
            revision_target = review.revision_target
            if revision_instructions:
                current_context = self._archive(
                    current_context, task, iteration_number, "review", "revision_instructions",
                    "# Review Agent 返修指令\n\n" + "\n".join(f"- {item}" for item in revision_instructions) + "\n",
                    purpose=f"Review Agent revision instructions for {task.task_id}, after iteration {iteration_number}.",
                    role=ReadOnlyFileRole.REFERENCE,
                )
            if review.requested_file_paths and self.file_reader is not None:
                disclosed = self.file_reader.disclose(current_context, review.requested_file_paths)
                if disclosed:
                    current_context = current_context.model_copy(update={
                        "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed),
                    })

        return SolveProblemReport(
            task_id=task.task_id, status=SolveProblemStatus.REVISION_REQUIRED,
            iteration_count=(start_iteration - 1) + len(iterations), iterations=tuple(iterations),
            revision_instructions=revision_instructions,
            artifacts=list(iterations[-1].coding_report.artifacts) if iterations else [],
            archive_files=self._archive_files(current_context, task),
            research_report=current_context.research_report,
            error="solve_problem iteration budget exhausted",
        )

    def _probe(
        self,
        context: SolveProblemContext,
        task: SolveProblemTask,
        iteration: int | None,
        event: str,
        actor: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self.probe_writer is None:
            return
        raw_run_id = context.metadata.get("run_id")
        if not raw_run_id:
            return
        self.probe_writer.record_probe(
            UUID(str(raw_run_id)), task_id=task.task_id,
            attempt=task.revision + 1, iteration=iteration, event=event,
            actor=actor, status=status, details=details or {},
        )

    def _archive(
        self,
        context: SolveProblemContext,
        task: SolveProblemTask,
        iteration: int,
        stage: str,
        name: str,
        markdown: str,
        *,
        purpose: str,
        role: ReadOnlyFileRole,
    ) -> SolveProblemContext:
        if self.archive_writer is None:
            return context
        raw_run_id = context.metadata.get("run_id")
        if not raw_run_id:
            return context
        reference = self.archive_writer.archive_markdown(
            UUID(str(raw_run_id)), task_id=task.task_id, attempt=task.revision + 1,
            iteration=iteration, stage=stage, name=name, markdown=markdown,
            purpose=purpose, role=role,
        )
        known = {item.relative_path: item for item in context.readonly_files}
        known[reference.relative_path] = reference
        return context.model_copy(update={"readonly_files": tuple(known.values())})

    @staticmethod
    def _record_conversation(
        context: SolveProblemContext,
        task: SolveProblemTask,
        iteration: int,
        actor: str,
        message: str,
        channel: str = "model",
    ) -> SolveProblemContext:
        """Carry one of the two independent Agent conversation ledgers."""

        metadata = dict(context.metadata)
        if channel not in {"model", "code"}:
            raise ValueError("conversation channel must be model or code")
        history_key = f"{channel}_conversation"
        history = metadata.get(history_key, ())
        if not isinstance(history, (list, tuple)):
            history = ()
        entries = [item for item in history if isinstance(item, dict)]
        entries.append({
            "task_id": task.task_id, "iteration": iteration,
            "actor": actor, "message": message,
        })
        metadata[history_key] = entries[-24:]
        return context.model_copy(update={"metadata": metadata})

    @staticmethod
    def _archive_files(context: SolveProblemContext, task: SolveProblemTask) -> tuple[ReadOnlyFileReference, ...]:
        marker = f"/tasks/{task.task_id}/"
        return tuple(
            item for item in context.readonly_files
            if marker in ("/" + item.relative_path.replace("\\", "/")) and "/exchanges/" in item.relative_path.replace("\\", "/")
        )


def _preliminary_markdown(report: PreliminaryModelingReport) -> str:
    lines = [
        f"# Preliminary Modeling Route: {report.branch_id}",
        "", "## Candidate Scheme", "", report.candidate_scheme,
        "", "## Assumptions", "", *(f"- {item}" for item in report.assumptions),
        "", "## Expected Outputs", "", *(f"- {item}" for item in report.expected_outputs),
        "", "## Risks", "", *(f"- {item}" for item in report.risks),
        "", "## Model Agent Report", "", report.report.markdown,
    ]
    if report.requested_file_paths:
        lines.extend(["", "## Requested Read-only Files", "", *(f"- `{item}`" for item in report.requested_file_paths)])
    return "\n".join(lines).strip() + "\n"


def _modeling_markdown(report: UnifiedModelingReport) -> str:
    return "\n".join([
        f"# Unified Modeling Report: {report.report.title}", "", report.report.markdown,
        "", "## Main Scheme", "", report.main_scheme,
        "", "## Selected Branches", "", *(f"- `{item}`" for item in report.selected_branch_ids),
        "", "## Required Validations", "", *(f"- {item}" for item in report.required_validations),
        "", "## Expected Outputs", "", *(f"- {item}" for item in report.expected_outputs),
        "", "## Coding Instructions", "", *(f"- {item}" for item in report.coding_instructions),
    ]).strip() + "\n"


def _coding_markdown(report: CodingHarnessReport) -> str:
    lines = [
        f"# Coding Harness Report: {report.report.title}", "", report.report.markdown,
        "", f"- Execution succeeded: `{report.execution_succeeded}`",
        "", "## Validations", "", *(f"- `{key}`: `{value}`" for key, value in report.validations.items()),
        "", "## Validation Evidence", "", *(f"- `{key}`: {value}" for key, value in report.validation_evidence.items()),
        "", "## Metrics", "", *(f"- `{key}`: `{value}`" for key, value in report.metrics.items()),
        "", "## Issues", "", *(f"- {item}" for item in report.issues),
        "", "## Generated Files", "", *(f"- `{item.relative_path}` — {item.purpose}" for item in report.generated_files),
    ]
    return "\n".join(lines).strip() + "\n"


def _review_markdown(review: SolveProblemReview) -> str:
    lines = [
        f"# Review Agent 审查决定：{review.decision.value}", "", "## 理由", "", review.rationale,
        "", "## 已接受声明", "", *(f"- {item}" for item in review.accepted_claims),
        "", f"## 返修目标\n\n`{review.revision_target.value}`",
        "", "## 返修指令", "", *(f"- {item}" for item in review.revision_instructions),
        "", "## 请求读取的只读文件", "", *(f"- `{item}`" for item in review.requested_file_paths),
    ]
    return "\n".join(lines).strip() + "\n"


def _disclosure_failure_markdown(
    task: SolveProblemTask,
    iteration: int,
    requested_paths: Sequence[str],
    error: Exception,
) -> str:
    """Keep a visible report even when a boundary fails before Code runs."""

    return "\n".join([
        "# 交接前失败报告：Model Agent → 文件披露",
        "",
        f"- 题目：`{task.task_id}` — {task.title}",
        f"- 迭代：`{iteration}`",
        "- 结果：`failed`",
        "",
        "## 转接理由",
        "",
        "Model Agent 请求了只读路径，但文件披露阶段未能完成；因此没有把不完整上下文转给 Code Agent。",
        "",
        "## 请求路径",
        "",
        *(f"- `{path}`" for path in requested_paths),
        "",
        "## 失败原因",
        "",
        f"`{str(error)[:2_000]}`",
        "",
        "## 下一步",
        "",
        "若路径是 PDF、图片或其他二进制输入，应通过 multimodal_inputs 或受限工具读取；只有 UTF-8 文本才进入渐进式文本披露。",
    ]).strip() + "\n"


def _handoff_paths(context: SolveProblemContext, *, task_id: str | None = None) -> list[str]:
    """Return stable, workspace-relative paths visible at an Agent boundary."""

    marker = f"/tasks/{task_id}/" if task_id else "/tasks/"
    return sorted({
        item.relative_path for item in context.readonly_files
        if marker in ("/" + item.relative_path.replace("\\", "/"))
        and "/exchanges/" in item.relative_path.replace("\\", "/")
    })


def _handoff_file_table(context: SolveProblemContext, task_id: str) -> list[str]:
    rows = []
    for item in context.readonly_files:
        if f"/tasks/{task_id}/" not in ("/" + item.relative_path.replace("\\", "/")):
            continue
        rows.append(
            f"- `{item.relative_path}` — role=`{item.role.value}`, purpose={item.purpose}, "
            f"sha256=`{item.sha256 or 'not-recorded'}`, size={item.size_bytes if item.size_bytes is not None else 'unknown'} bytes"
        )
    return sorted(rows)


def _model_to_code_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    preliminary: tuple[PreliminaryModelingReport, ...],
    modeling: UnifiedModelingReport,
    iteration: int,
) -> str:
    lines = [
        "# Handoff: Model Agent → Code Agent", "",
        f"- Task: `{task.task_id}` — {task.title}", f"- Iteration: `{iteration}`",
        "- Boundary: Model Agent has selected a typed modeling contract; Code Agent must realize and validate it.",
        "", "## What was sent", "",
        "### Unified modeling contract", "", "```json",
        json.dumps(modeling.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```", "", "### Preliminary routes considered", "", "```json",
        json.dumps([item.model_dump(mode="json") for item in preliminary], ensure_ascii=False, indent=2),
        "```", "", "## Read-only files available at this boundary", "",
        *_handoff_file_table(context, task.task_id),
        "", "## Disclosed text already in context", "",
        *(f"- `{item.relative_path}` (truncated={item.truncated}, sha256=`{item.sha256}`)" for item in context.disclosed_text_files),
        "", "## Code Agent obligations", "",
        "- Implement the main scheme and every required validation.",
        "- Use only allowlisted workspace paths and record generated source/output paths.",
        "- Return reproducible validation evidence; do not report success without executable evidence.",
        "", "## Exchange paths", "", *_handoff_paths(context, task_id=task.task_id),
    ]
    return "\n".join(lines).strip() + "\n"


def _code_to_review_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    modeling: UnifiedModelingReport,
    coding: CodingHarnessReport,
    iteration: int,
) -> str:
    lines = [
        "# Handoff: Code Harness → Review Agent", "",
        f"- Task: `{task.task_id}` — {task.title}", f"- Iteration: `{iteration}`",
        "- Boundary: Code Harness has executed the selected model and is returning evidence for independent review.",
        "", "## Modeling contract received", "", "```json",
        json.dumps(modeling.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```", "", "## Coding report returned", "", "```json",
        json.dumps(coding.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```", "", "## Allowlisted evidence files", "",
        *_handoff_file_table(context, task.task_id),
        "", "## Review obligations", "",
        "- Check each required validation against reproducible evidence.",
        "- Check generated source and outputs through the disclosed paths where needed.",
        "- Approve only claims supported by the returned evidence; otherwise give a narrow revision target and concrete instructions.",
        "", "## Exchange paths", "", *_handoff_paths(context, task_id=task.task_id),
    ]
    return "\n".join(lines).strip() + "\n"


def _review_to_next_stage_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    review: SolveProblemReview,
    iteration: int,
) -> str:
    instruction_lines = [f"- {item}" for item in review.revision_instructions] or [
        "- No revision instructions; proceed to final composition if execution succeeded."
    ]
    requested_lines = [f"- `{item}`" for item in review.requested_file_paths] or ["- None"]
    lines = [
        "# Handoff: Review Agent → Next Stage", "",
        f"- Task: `{task.task_id}` — {task.title}", f"- Iteration: `{iteration}`",
        f"- Decision: `{review.decision.value}`", f"- Revision target: `{review.revision_target.value}`",
        "", "## Review decision", "", "```json",
        json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "```", "", "## Instructions delivered", "",
        *instruction_lines,
        "", "## Requested disclosures", "",
        *requested_lines,
        "", "## Files still visible to the next stage", "",
        *_handoff_file_table(context, task.task_id),
    ]
    return "\n".join(lines).strip() + "\n"


def _conversation_lines(context: SolveProblemContext, channel: str = "model") -> list[str]:
    """Render only the latest events from one independent Agent context."""

    history = context.metadata.get(f"{channel}_conversation", ())
    if not isinstance(history, (list, tuple)):
        return ["- （暂无共享会话事件）"]
    result: list[str] = []
    for item in history[-12:]:
        if isinstance(item, dict):
            result.append(
                f"- 迭代 {item.get('iteration', '?')} / {item.get('actor', 'Agent')}：{str(item.get('message', ''))[:1200]}"
            )
    return result or ["- （暂无共享会话事件）"]


def _handoff_budget_lines(iteration: int, max_revision_rounds: int) -> list[str]:
    """Render the visible revision budget on every stage boundary."""

    used = max(0, iteration - 1)
    remaining = max(0, max_revision_rounds - used)
    return [
        f"- 返修上限：最多 {max_revision_rounds} 轮（初始实现轮不计入返修）。",
        f"- 当前返修序号：第 {used} 轮；本次交接后剩余返修额度：{remaining} 轮。",
    ]


def _compact_model_to_code_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    preliminary: tuple[PreliminaryModelingReport, ...],
    modeling: UnifiedModelingReport,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    """Short, human-readable index; typed modeling remains in the call context."""

    lines = [
        "# 交接：Model Agent → Code Agent", "",
        f"- 题目：`{task.task_id}` — {task.title}", f"- 迭代：`{iteration}`",
        "- 同一共享会话中的建模→编码交接；完整结构化对象通过调用参数传递。",
        "- 转接理由：统一建模已完成，主方案、验证项和期望输出已锁定；现在转给 Code Agent 实现并产生可复现证据。",
        *_handoff_budget_lines(iteration, max_revision_rounds),
        "", "## 本轮编码契约", "",
        f"- 主方案：{modeling.main_scheme[:2200]}",
        f"- 选中路线：{', '.join(modeling.selected_branch_ids)}",
        f"- 必须验证：{'; '.join(modeling.required_validations[:16])}",
        f"- 期望输出：{'; '.join(modeling.expected_outputs[:12])}",
        f"- 编码指令：{'; '.join(modeling.coding_instructions[:16])}",
        f"- 初步路线数：{len(preliminary)}",
        "", "## 只读文件索引", "", *_handoff_file_table(context, task.task_id),
        "", "## Model Agent 上下文最近事件", "", *_conversation_lines(context, "model"),
        "", "## Code Agent 必须完成", "",
        "- 只实现本题范围；使用受限工具写入、读取和验证源代码。",
        "- 每个验证项返回可复现证据；不得只报 true。",
    ]
    return "\n".join(lines).strip() + "\n"


def _compact_code_to_review_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    modeling: UnifiedModelingReport,
    coding: CodingHarnessReport,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    lines = [
        "# 交接：Code Harness → Review Agent", "",
        f"- 题目：`{task.task_id}` — {task.title}", f"- 迭代：`{iteration}`",
        "- 同一共享会话中的执行→审查交接；本文件只列可核验摘要和路径。",
        "- 转接理由：Code Agent 已返回本轮执行结果；Review Agent 现在只负责核验声明、验证证据和产物完整性，并决定批准或指定最窄返修目标。",
        *_handoff_budget_lines(iteration, max_revision_rounds),
        "", "## 执行摘要", "",
        f"- 执行成功：`{coding.execution_succeeded}`",
        f"- 预期输出：{'; '.join(modeling.expected_outputs[:12])}",
        f"- 必须验证：{'; '.join(modeling.required_validations[:16])}",
        f"- 验证：{json.dumps(coding.validations, ensure_ascii=False)}",
        *(f"- 证据：`{key}` — {value}" for key, value in list(coding.validation_evidence.items())[:16]),
        *(f"- 指标：`{key}` = `{value}`" for key, value in list(coding.metrics.items())[:16]),
        *(f"- 问题：{item}" for item in coding.issues[:12]),
        *(f"- 偏差：{item}" for item in coding.deviations[:12]),
        "", "## 产物索引", "",
        *(f"- `{item.logical_name}` ({item.kind.value})" for item in coding.artifacts[:16]),
        *(f"- `{item.relative_path}` ({item.role.value})" for item in coding.generated_files),
        "", "## 只读文件索引", "", *_handoff_file_table(context, task.task_id),
        "", "## Code Agent 上下文最近事件", "", *_conversation_lines(context, "code"),
        "", "## Review Agent 必须完成", "",
        "- 逐项核验验证证据；只批准有可复现证据的声明。",
        "- 需要源代码或输出时，只请求清单中的路径。",
        f"- 建模主方案摘要：{modeling.main_scheme[:1000]}",
    ]
    return "\n".join(lines).strip() + "\n"


def _compact_review_to_next_stage_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    review: SolveProblemReview,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    instruction_lines = [f"- {item}" for item in review.revision_instructions] or [
        "- 无返修指令；若执行成功，进入最终报告。"
    ]
    requested_lines = [f"- `{item}`" for item in review.requested_file_paths] or ["- 无"]
    if review.decision == ModelReviewDecision.APPROVE:
        transfer_reason = "审查已批准且证据满足门槛；交给最终报告阶段，只能使用已审查声明。"
    else:
        transfer_reason = (
            f"审查未批准当前产物：{review.rationale[:1200]}；"
            f"按最窄目标 `{review.revision_target.value}` 转回返修，不重新打开未受影响的阶段。"
        )
    lines = [
        "# 交接：Review Agent → 下一阶段 / 返修", "",
        f"- 题目：`{task.task_id}` — {task.title}", f"- 迭代：`{iteration}`",
        f"- 决定：`{review.decision.value}`", f"- 返修目标：`{review.revision_target.value}`",
        f"- 转接理由：{transfer_reason}",
        *_handoff_budget_lines(iteration, max_revision_rounds),
        "", "## 审查结论", "", f"- 理由：{review.rationale[:2400]}",
        "", "## 必须执行的返修指令", "", *instruction_lines,
        "", "## 请求披露的文件", "", *requested_lines,
        "", "## 下轮 Model Agent 上下文", "", *_conversation_lines(context, "model"),
        "", "## 仍可见的文件索引", "", *_handoff_file_table(context, task.task_id),
    ]
    return "\n".join(lines).strip() + "\n"


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
