"""The report-driven ``solve_problem`` tool.

Main Harness owns the process and calls this module as a Tool.  The module is
not a top-level subagent dispatcher.  It is a bounded internal protocol which
lets a Model Agent explore/synthesize a modeling plan, lets a Code Harness
realize that plan, and sends the Coding Report back to the same Model Agent
through a Code→Model handoff. The Model Agent then gives Code-only repair
instructions; it does not start a second modeling pass during revisions.
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
from m2harness.human_control import HumanControlStore, HumanInterruptRequested
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
            if reference.media_type.lower() == "application/pdf" or requested.lower().endswith(".pdf"):
                # The original PDF remains a binary, read-only allowlisted
                # input and is also sent through the multimodal lane.  For a
                # textual handoff, disclose a local extraction of the same
                # original bytes instead of silently dropping the request.
                # If PyMuPDF is unavailable or the fixture is not a valid PDF,
                # return an explicit path/digest marker rather than pretending
                # that the original was inaccessible.
                extracted = self._extract_pdf_text(raw, limit)
                content = extracted or (
                    f"[原始 PDF 二进制题面未做文本抽取；请通过只读路径打开原文件。]\\n"
                    f"path={requested}\\nmedia_type={reference.media_type}\\n"
                    f"sha256={digest}\\nsize_bytes={len(raw)}"
                )
                result.append(DisclosedTextFile(
                    relative_path=requested, purpose=reference.purpose + "；原始 PDF 只读输入及其本地文本披露。",
                    content=content[:limit], sha256=digest, truncated=len(content.encode("utf-8")) > limit,
                ))
                total += min(len(content.encode("utf-8")), limit)
                continue
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

    @staticmethod
    def _extract_pdf_text(raw: bytes, limit: int) -> str:
        """Extract bounded text locally while preserving the original PDF path."""

        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz  # type: ignore[no-redef]
            document = fitz.open(stream=raw, filetype="pdf")
            try:
                chunks: list[str] = []
                used = 0
                for page_number, page in enumerate(document, start=1):
                    text = str(page.get_text("text") or "")
                    if not text:
                        continue
                    chunk = f"\\n--- PDF 原文第 {page_number} 页 ---\\n{text}"
                    remaining = limit - used
                    if remaining <= 0:
                        break
                    encoded = chunk.encode("utf-8")
                    if len(encoded) > remaining:
                        chunks.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                        break
                    chunks.append(chunk)
                    used += len(encoded)
                return "".join(chunks).strip()
            finally:
                document.close()
        except Exception:
            return ""


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
    human_control: HumanControlStore | None = None

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
        try:
            return self._solve(task, context)
        except HumanInterruptRequested as exc:
            interrupted_context = exc.context if isinstance(exc.context, SolveProblemContext) else (context or SolveProblemContext())
            self._probe(interrupted_context, task, exc.iteration, "human_interrupt", "Human Operator", "blocked", {
                "reason": exc.reason[:2_000],
            })
            return SolveProblemReport(
                task_id=task.task_id,
                status=SolveProblemStatus.BLOCKED,
                iteration_count=max(0, exc.iteration or 0),
                iterations=(),
                archive_files=self._archive_files(interrupted_context, task),
                research_report=interrupted_context.research_report,
                error=f"human interrupt: {exc.reason[:2_000]}",
            )

    def _solve(self, task: SolveProblemTask, context: SolveProblemContext | None = None) -> SolveProblemReport:
        current_context = self._prepare_context(context or SolveProblemContext(), task)
        self._probe(current_context, task, None, "solve_start", "Main Harness", "started", {
            "max_iterations": self.max_iterations,
            "revision_round_limit": self.revision_round_limit,
            "readonly_file_count": len(current_context.readonly_files),
            "multimodal_input_count": len(current_context.multimodal_inputs),
        })
        current_context = self._control_checkpoint(current_context, task, None, "solve_start")
        if self.research_agent is not None and current_context.research_report is None:
            research = self.research_agent.research(
                task.problem,
                max_facets=min(8, max(3, task.difficulty)),
                top_k=min(12, max(4, task.max_branches * 2)),
            )
            current_context = current_context.model_copy(update={"research_report": research})
        # The original problem PDF is always part of the boundary contract,
        # not an optional binary that disappears when an Agent asks for
        # progressive disclosure.  Keep the original allowlisted path and
        # proactively add a bounded local text extraction to the same context
        # so every detailed report/handoff can show both forms of the source.
        pdf_paths = tuple(dict.fromkeys(
            item.relative_path for item in current_context.readonly_files
            if item.role == ReadOnlyFileRole.PROBLEM
            and (item.media_type.lower() == "application/pdf" or item.relative_path.lower().endswith(".pdf"))
        ))
        if pdf_paths and self.file_reader is not None:
            self._probe(current_context, task, None, "problem_pdf_disclosure_start", "Main Harness", "started", {
                "paths": list(pdf_paths),
            })
            try:
                disclosed_pdf = self.file_reader.disclose(current_context, pdf_paths)
            except Exception as exc:
                self._probe(current_context, task, None, "problem_pdf_disclosure_failed", "Main Harness", "failed", {
                    "paths": list(pdf_paths), "error": str(exc)[:2_000],
                })
                raise
            if disclosed_pdf:
                current_context = current_context.model_copy(update={
                    "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed_pdf),
                })
            self._probe(current_context, task, None, "problem_pdf_disclosure_complete", "Main Harness", "completed", {
                "requested_paths": list(pdf_paths),
                "disclosed_paths": [item.relative_path for item in disclosed_pdf],
            })
        iterations: list[SolveProblemIteration] = []
        revision_instructions: tuple[str, ...] = ()
        # The modeling contract is created once per solve invocation. A
        # Code→Model repair round never reopens explore/synthesize.
        revision_target = RevisionTarget.CODE
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
            # Older checkpoints may contain MODEL/FULL. A resumed run is
            # already past modeling, so normalize the effective route to CODE.
            raw_target = current_context.metadata.get("resume_revision_target", RevisionTarget.CODE.value)
            RevisionTarget(raw_target)  # validate the serialized value
            revision_target = RevisionTarget.CODE
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
            current_context = self._control_checkpoint(current_context, task, iteration_number, "iteration_start")
            if revision_instructions:
                current_context = current_context.model_copy(update={
                    "instructions": (*current_context.instructions, *revision_instructions),
                })
            # Only the first pass may explore. Revision instructions are fed
            # to Code with the existing modeling contract intact.
            if modeling is None:
                self._probe(current_context, task, iteration_number, "model_explore_start", "Model Agent", "started", {
                    "branch_count": branch_count,
                    "revision_target": revision_target.value,
                })
                try:
                    preliminary = tuple(self.model_agent.explore(
                        task, current_context, branch_count=branch_count, iteration=iteration_number,
                    ))
                except Exception as exc:
                    self._probe(current_context, task, iteration_number, "model_explore_failed", "Model Agent", "failed", {
                        "error": str(exc)[:2_000],
                    })
                    raise
                self._probe(current_context, task, iteration_number, "model_explore_complete", "Model Agent", "completed", {
                    "branch_ids": [item.branch_id for item in preliminary],
                    "requested_file_paths": [path for item in preliminary for path in item.requested_file_paths],
                })
                current_context = self._record_context_marker(
                    current_context, task, iteration_number, "Model Agent",
                    "已完成内部初步路线比较；不对外另存路线报告，等待统一建模。",
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
                            purpose=f"题目 {task.task_id} 下一次 Model Agent 交接前的渐进式披露失败记录。",
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
                        try:
                            preliminary = tuple(self.model_agent.explore(
                                task, current_context, branch_count=branch_count, iteration=iteration_number,
                            ))
                        except Exception as exc:
                            self._probe(current_context, task, iteration_number, "model_explore_failed", "Model Agent", "failed", {
                                "error": str(exc)[:2_000], "after_disclosure": True,
                            })
                            raise
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
            if modeling is None:
                self._probe(current_context, task, iteration_number, "model_synthesize_start", "Model Agent", "started", {})
                try:
                    modeling = self.model_agent.synthesize(
                        task, current_context, preliminary, iteration=iteration_number,
                    )
                except Exception as exc:
                    self._probe(current_context, task, iteration_number, "model_synthesize_failed", "Model Agent", "failed", {
                        "error": str(exc)[:2_000],
                    })
                    raise
                self._probe(current_context, task, iteration_number, "model_synthesize_complete", "Model Agent", "completed", {
                    "selected_branch_ids": list(modeling.selected_branch_ids),
                    "required_validations": list(modeling.required_validations),
                    "expected_outputs": list(modeling.expected_outputs),
                })
                current_context = self._archive(
                    current_context, task, iteration_number, "modeling", "modeling_report",
                    _modeling_markdown(modeling),
                    purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Model Agent 统一建模方案。",
                    role=ReadOnlyFileRole.REFERENCE,
                )
                current_context = self._record_context_marker(
                    current_context, task, iteration_number, "Model Agent",
                    f"统一建模已完成。主方案：{modeling.main_scheme[:1800]}；验证项：{', '.join(modeling.required_validations[:12])}。",
                )
            # The full Model→Code contract is written exactly once.  On a
            # repair iteration the previous iteration's delta handoff is
            # already in the allowlist; re-emitting the modeling report here
            # only creates a second full copy and makes the Code context
            # drift.  Code receives the same typed ``modeling`` object plus
            # the latest delta instructions.
            if iteration_number == start_iteration and not resume_mode:
                current_context = self._archive(
                    current_context, task, iteration_number, "handoff", "model-to-code",
                    _compact_model_to_code_handoff(
                        task, current_context, preliminary, modeling, iteration_number,
                        max_revision_rounds=self.revision_round_limit,
                    ),
                    purpose=f"题目 {task.task_id} 的初始 Model Agent→Code Agent 建模契约。",
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
            current_context = self._control_checkpoint(current_context, task, iteration_number, "code_execute_start")
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
            current_context = self._control_checkpoint(current_context, task, iteration_number, "code_execute_complete")
            # Generated source/output files become Model Agent evidence only
            # after the Code Harness has materialized and hashed them. Merge
            # those references into the read-only manifest before the
            # Code→Model repair boundary; this preserves progressive
            # disclosure without granting arbitrary workspace access.
            if coding.generated_files:
                known = {item.relative_path: item for item in current_context.readonly_files}
                known.update({item.relative_path: item for item in coding.generated_files})
                current_context = current_context.model_copy(update={"readonly_files": tuple(known.values())})
            current_context = self._record_context_marker(
                current_context, task, iteration_number, "Code Agent",
                f"Code Agent 已提交本轮 Markdown 报告；验证索引仅作定位；生成文件：{', '.join(item.relative_path for item in coding.generated_files)}。",
                channel="code",
            )
            current_context = self._archive(
                current_context, task, iteration_number, "handoff", "code-to-model",
                _compact_code_to_model_handoff(
                    task, current_context, modeling, coding, iteration_number,
                    max_revision_rounds=self.revision_round_limit,
                ),
                purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Code Harness→Model Agent 返修交接。",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT,
            )
            # Compatibility indexes keep legacy inspectors pointed at the one
            # canonical report without writing a second copy of its body.
            current_context = self._archive(
                current_context, task, iteration_number, "coding", "coding_report",
                _pointer_markdown(
                    "Code→Model 报告索引",
                    _current_exchange_path(current_context, task, iteration_number, "code-to-model.md"),
                ),
                purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Code→Model 报告正文索引。",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT,
            )
            self._probe(current_context, task, iteration_number, "code_to_model_handoff", "Code Agent", "completed", {
                "handoff": "code-to-model.md",
                "execution_succeeded": coding.execution_succeeded,
                "generated_files": [item.relative_path for item in coding.generated_files],
            })
            self._probe(current_context, task, iteration_number, "model_code_review_start", "Model Agent", "started", {})
            current_context = self._control_checkpoint(current_context, task, iteration_number, "model_code_review_start")
            try:
                review = self.model_agent.review(
                    task, current_context, modeling, coding, iteration=iteration_number,
                )
            except Exception as exc:
                self._probe(current_context, task, iteration_number, "model_code_review_failed", "Model Agent", "failed", {
                    "error": str(exc)[:2_000],
                })
                raise
            # The provider schema remains SolveProblemReview for compatibility,
            # but this is the same Model Agent in its Code→Model repair phase.
            # Any non-approval is always a Code repair; the modeling contract
            # is never reopened by this loop.
            original_revision_target = review.revision_target
            if review.revision_target != RevisionTarget.CODE:
                updates: dict[str, Any] = {"revision_target": RevisionTarget.CODE}
                if review.decision != ModelReviewDecision.APPROVE:
                    updates["revision_instructions"] = (
                        "不要重新探索或综合建模；保留本轮统一建模契约，只按 Code→Model 证据修复并重新执行。",
                        *review.revision_instructions,
                    )
                review = review.model_copy(update=updates)
            self._probe(current_context, task, iteration_number, "model_code_review_complete", "Model Agent", "completed", {
                "decision": review.decision.value,
                "revision_target": review.revision_target.value,
                "provider_revision_target": original_revision_target.value,
                "revision_instruction_count": len(review.revision_instructions),
                "requested_file_paths": list(review.requested_file_paths),
            })
            self._probe(current_context, task, iteration_number, "model_to_code_revision_handoff", "Model Agent", "completed", {
                "handoff": "model-to-code-revision.md",
                "decision": review.decision.value,
                "revision_target": review.revision_target.value,
                "revision_instructions": list(review.revision_instructions),
            })
            current_context = self._record_context_marker(
                current_context, task, iteration_number, "Model Agent",
                f"审查决定：{review.decision.value}；返修目标：{review.revision_target.value}；指令：{'；'.join(review.revision_instructions[:8])}。",
            )
            current_context = self._archive(
                current_context, task, iteration_number, "handoff", "model-to-code-revision",
                _compact_model_to_code_revision_handoff(
                    task, current_context, modeling, coding, review, iteration_number,
                    max_revision_rounds=self.revision_round_limit,
                ),
                purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Model Agent→Code Agent 返修交接。",
                role=ReadOnlyFileRole.REFERENCE,
            )
            current_context = self._archive(
                current_context, task, iteration_number, "review", "review_report",
                _pointer_markdown(
                    "Model Agent 返修意见索引",
                    _current_exchange_path(current_context, task, iteration_number, "model-to-code-revision.md"),
                ),
                purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Model→Code 返修交接正文索引。",
                role=ReadOnlyFileRole.REFERENCE,
            )
            if review.revision_instructions:
                current_context = self._archive(
                    current_context, task, iteration_number, "review", "revision_instructions",
                    _pointer_markdown(
                        "Model Agent→Code Agent 返修指令索引",
                        _current_exchange_path(current_context, task, iteration_number, "model-to-code-revision.md"),
                    ),
                    purpose=f"题目 {task.task_id} 第 {iteration_number} 轮 Code 返修指令正文索引。",
                    role=ReadOnlyFileRole.REFERENCE,
                )
            snapshot = SolveProblemIteration(
                iteration=iteration_number,
                # A revision snapshot keeps only a compact pointer to the
                # already accepted model.  The first snapshot is the single
                # full modeling record; later snapshots must not serialize it
                # again into every report/handoff.
                preliminary_reports=(preliminary if iteration_number == start_iteration else ()),
                modeling_report=_modeling_snapshot(modeling, iteration_number, first_iteration=start_iteration),
                coding_report=coding, review=review,
            )
            iterations.append(snapshot)
            # A successful review, or the final allowed iteration, closes this
            # one-shot solve call.  At the cap the Model Agent still writes a
            # total question report with explicit limitations; Main Harness
            # must not call solve again to manufacture another review loop.
            if review.decision == ModelReviewDecision.APPROVE or iteration_number == iteration_limit:
                current_context = self._control_checkpoint(current_context, task, iteration_number, "final_report_start")
                try:
                    final_report = self.model_agent.compose_final_report(
                        task, current_context, modeling, coding, review,
                        iteration=iteration_number,
                    )
                except Exception as exc:
                    self._probe(current_context, task, iteration_number, "final_report_failed", "Model Agent", "failed", {
                        "error": str(exc)[:2_000],
                    })
                    raise
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
                    # The cap is a normal terminal outcome of this atomic
                    # solve call, not an outer failure/retry signal.  Any
                    # unresolved item belongs in the Chinese final report.
                    error=None,
                )
            revision_instructions = review.revision_instructions
            revision_target = review.revision_target
            if review.requested_file_paths and self.file_reader is not None:
                disclosed = self.file_reader.disclose(current_context, review.requested_file_paths)
                if disclosed:
                    current_context = current_context.model_copy(update={
                        "disclosed_text_files": (*current_context.disclosed_text_files, *disclosed),
                    })

        # The repair budget belongs to solve_problem.  Never leak a
        # ``revision_required`` state to Main Harness: that would make the
        # outer DAG dispatch the same problem again and duplicate the whole
        # Model↔Code exchange.  The final internal snapshot and instructions
        # remain available in the archived handoff for operator recovery.
        return SolveProblemReport(
            task_id=task.task_id, status=SolveProblemStatus.FAILED,
            iteration_count=(start_iteration - 1) + len(iterations), iterations=tuple(iterations),
            revision_instructions=revision_instructions,
            artifacts=list(iterations[-1].coding_report.artifacts) if iterations else [],
            archive_files=self._archive_files(current_context, task),
            research_report=current_context.research_report,
            error="solve_problem 内部返修额度已耗尽；主 Harness 不会再次调用本题。",
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
        raw_run_id = context.metadata.get("run_id")
        if not raw_run_id:
            return
        payload = details or {}
        if self.human_control is not None:
            self.human_control.publish_status(
                str(raw_run_id), task_id=task.task_id, attempt=task.revision + 1,
                iteration=iteration, event=event, actor=actor, status=status,
                details=dict(payload),
            )
        if self.probe_writer is not None:
            self.probe_writer.record_probe(
                UUID(str(raw_run_id)), task_id=task.task_id,
                attempt=task.revision + 1, iteration=iteration, event=event,
                actor=actor, status=status, details=payload,
            )

    def _control_checkpoint(
        self,
        context: SolveProblemContext,
        task: SolveProblemTask,
        iteration: int | None,
        stage: str,
    ) -> SolveProblemContext:
        """Consume operator commands only at explicit workflow safe points."""

        if self.human_control is None:
            return context
        raw_run_id = context.metadata.get("run_id")
        if not raw_run_id:
            return context
        run_id = str(raw_run_id)
        cursor_value = context.metadata.get("human_control_cursor", 0)
        cursor = int(cursor_value) if isinstance(cursor_value, int) and cursor_value >= 0 else 0
        commands = self.human_control.pending(run_id, after_sequence=cursor)
        if not commands:
            if self.human_control.is_interrupted(run_id):
                raise HumanInterruptRequested(run_id, task.task_id, iteration, "operator requested interrupt", context=context)
            return context
        instructions = list(context.instructions)
        next_cursor = cursor
        updated_context = context
        for command in commands:
            next_cursor = max(next_cursor, command.sequence)
            if command.kind == "suggestion":
                instructions.append(f"[Human operator suggestion at {stage}] {command.message}")
                self._probe(updated_context, task, iteration, "human_suggestion", "Human Operator", "received", {
                    "stage": stage, "sequence": command.sequence, "message": command.message,
                })
            else:
                self._probe(updated_context, task, iteration, "human_interrupt_requested", "Human Operator", "received", {
                    "stage": stage, "sequence": command.sequence, "reason": command.message,
                })
        updated_context = updated_context.model_copy(update={
            "instructions": tuple(instructions),
            "metadata": {**updated_context.metadata, "human_control_cursor": next_cursor},
        })
        interrupt = next((item for item in commands if item.kind == "interrupt"), None)
        if interrupt is not None:
            raise HumanInterruptRequested(run_id, task.task_id, iteration, interrupt.message, context=updated_context)
        return updated_context

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
        self._probe(context, task, iteration, "context_projection_updated", "Main Harness", "recorded", {
            "stage": stage,
            "path": reference.relative_path,
            "role": reference.role.value,
        })
        known = {item.relative_path: item for item in context.readonly_files}
        known[reference.relative_path] = reference
        return context.model_copy(update={"readonly_files": tuple(known.values())})

    @staticmethod
    def _prepare_context(context: SolveProblemContext, task: SolveProblemTask) -> SolveProblemContext:
        """Bind one typed context projection to the Main Harness event stream.

        The Main Harness does not own a second transcript. Durable probes and
        archived handoffs are the event log; this immutable Pydantic object is
        only the current projection passed to the next stage. Remove legacy
        conversation tails so resumed runs cannot accidentally replay them.
        """

        metadata = dict(context.metadata)
        metadata.pop("model_conversation", None)
        metadata.pop("code_conversation", None)
        raw_run_id = metadata.get("run_id")
        if raw_run_id:
            run_id = str(raw_run_id)
            run_name = str(metadata.get("run_name", run_id))
            attempt = int(metadata.get("task_attempt", task.revision + 1))
            metadata.update({
                "context_owner": "main-harness",
                "context_model": "durable probes and handoff artifacts; typed projection, not transcript replay",
                "context_session_id": f"m2h-main-{run_id}-{task.task_id}-attempt-{attempt}",
                "context_event_log": f"reports/runs/{run_name}/probe.ndjson",
                "context_policy": "read current typed fields and allowlisted exchange paths; do not expect conversation history in metadata",
            })
        return context.model_copy(update={"metadata": metadata})

    def _record_context_marker(
        self,
        context: SolveProblemContext,
        task: SolveProblemTask,
        iteration: int,
        actor: str,
        message: str,
        channel: str = "model",
    ) -> SolveProblemContext:
        """Record a bounded context marker without storing transcript text.

        The call sites remain because they mark the existing workflow
        boundaries. The message body is intentionally discarded: the
        corresponding handoff is archived separately and the probe stream is
        the durable event source.
        """

        if channel not in {"model", "code"}:
            raise ValueError("conversation channel must be model or code")
        self._probe(context, task, iteration, "context_marker", actor, "recorded", {
            "channel": channel,
            "message_chars": len(message),
            "source_of_truth": "handoff-artifact-and-probe-stream",
        })
        return context

    @staticmethod
    def _archive_files(context: SolveProblemContext, task: SolveProblemTask) -> tuple[ReadOnlyFileReference, ...]:
        marker = f"/tasks/{task.task_id}/"
        return tuple(
            item for item in context.readonly_files
            if marker in ("/" + item.relative_path.replace("\\", "/")) and "/exchanges/" in item.relative_path.replace("\\", "/")
        )


def _preliminary_markdown(report: PreliminaryModelingReport) -> str:
    lines = [
        f"# 初步建模路线：{report.branch_id}",
        "", "## 候选方案", "", report.candidate_scheme,
        "", "## 假设", "", *(f"- {item}" for item in report.assumptions),
        "", "## 预期输出", "", *(f"- {item}" for item in report.expected_outputs),
        "", "## 风险", "", *(f"- {item}" for item in report.risks),
        "", "## Model Agent 报告", "", report.report.markdown,
    ]
    if report.requested_file_paths:
        lines.extend(["", "## 请求的只读文件", "", *(f"- `{item}`" for item in report.requested_file_paths)])
    return "\n".join(lines).strip() + "\n"


def _modeling_markdown(report: UnifiedModelingReport) -> str:
    return "\n".join([
        f"# 统一建模报告：{report.report.title}", "", report.report.markdown,
        "", "## 主方案", "", report.main_scheme,
        "", "## 选中路线", "", *(f"- `{item}`" for item in report.selected_branch_ids),
        "", "## 必须验证", "", *(f"- {item}" for item in report.required_validations),
        "", "## 预期输出", "", *(f"- {item}" for item in report.expected_outputs),
        "", "## 预期图", "", *(f"- {item}" for item in report.expected_figures),
        "", "## 编码指令", "", *(f"- {item}" for item in report.coding_instructions),
    ]).strip() + "\n"


def _modeling_snapshot(
    modeling: UnifiedModelingReport,
    iteration: int,
    *,
    first_iteration: int,
) -> UnifiedModelingReport:
    """Return the full model once, then a typed path-only continuation view."""

    if iteration == first_iteration:
        return modeling
    return modeling.model_copy(update={
        "report": ReportPayload(
            title="统一建模契约索引",
            summary="本轮沿用首轮已锁定的统一建模契约；未重新建模。",
            markdown=(
                "# 统一建模契约索引\n\n"
                "本轮为 Code 返修延续。完整建模报告只在首轮交接保存；"
                "请按白名单路径读取首轮 `model-to-code.md`。\n"
            ),
            limitations=["本对象是返修快照索引，不是新的建模报告。"],
        ),
        "main_scheme": "（沿用首轮统一建模契约；详见首轮 model-to-code.md）",
        "coding_instructions": (),
    })


def _coding_markdown(report: CodingHarnessReport) -> str:
    lines = [
        f"# Code Harness 执行报告：{report.report.title}", "", report.report.markdown,
        "", "## 本轮观察证据", "",
        "- 本文件是 Code→Model 的主交接报告；Model Agent 应以 Markdown、源代码、输出文件和探针为准。",
        "- 结构化验证索引仅用于定位，不作为独立成功/失败裁决。",
        "", "## 验证索引（辅助）", "", *(f"- `{key}`：`{value}`" for key, value in report.validations.items()),
        "", "## 验证证据索引（辅助）", "", *(f"- `{key}`：{value}" for key, value in report.validation_evidence.items()),
        "", "## 原始 Markdown 报告", "",
        "代码 stdout 已由 Code Harness 原样嵌入上面的 `report.markdown`；Model Agent 应以该 Markdown、源代码、输出和探针为主，不得因孤立的 false 字段直接否决。",
        "", "## 指标", "", *(f"- `{key}`：`{value}`" for key, value in report.metrics.items()),
        "", "## 问题", "", *(f"- {item}" for item in report.issues),
        "", "## 生成文件", "", *(f"- `{item.relative_path}` — {item.purpose}" for item in report.generated_files),
    ]
    return "\n".join(lines).strip() + "\n"


def _review_markdown(review: SolveProblemReview) -> str:
    lines = [
        f"# Model Agent 代码返修意见：{review.decision.value}", "",
        "本报告来自同一 Model Agent 读取 Code→Model 交接后的返修阶段；不是独立 Review Agent，也不触发重新建模。",
        "", "## 理由", "", review.rationale,
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


def _current_exchange_path(context: SolveProblemContext, task: SolveProblemTask, iteration: int, filename: str) -> str:
    marker = f"/tasks/{task.task_id}/"
    candidates = [
        item.relative_path for item in context.readonly_files
        if marker in ("/" + item.relative_path.replace("\\", "/"))
        and f"/exchanges/iteration-{iteration}/handoff/{filename}" in item.relative_path.replace("\\", "/")
    ]
    if candidates:
        return sorted(candidates)[-1]
    return f"exchanges/iteration-{iteration}/handoff/{filename}"


def _pointer_markdown(title: str, target: str) -> str:
    return f"# {title}\n\n正文已归档于 `{target}`；本文件仅作索引，不复制正文。\n"


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


def _conversation_lines(context: SolveProblemContext, channel: str = "model") -> list[str]:
    """Compatibility projection for legacy renderers; never read transcript metadata."""

    del context, channel
    return ["- Main Harness context is event-sourced; read current allowlisted handoff paths and the probe ledger."]


def _handoff_budget_lines(iteration: int, max_revision_rounds: int) -> list[str]:
    """Render the visible revision budget on every stage boundary."""

    used = max(0, iteration - 1)
    remaining = max(0, max_revision_rounds - used)
    return [
        f"- 返修上限：最多 {max_revision_rounds} 轮（初始实现轮不计入返修）。",
        f"- 当前返修序号：第 {used} 轮；本次交接后剩余返修额度：{remaining} 轮。",
    ]


def _problem_pdf_lines(context: SolveProblemContext) -> list[str]:
    """Render the original PDF as an explicit read-only handoff input."""

    files = [
        item for item in context.readonly_files
        if item.role == ReadOnlyFileRole.PROBLEM
        and (item.media_type.lower() == "application/pdf" or item.relative_path.lower().endswith(".pdf"))
    ]
    multimodal_names = {item.logical_name for item in context.multimodal_inputs}
    if not files:
        return ["- 未提供原始 PDF；请勿假设题面已被读取。"]
    lines: list[str] = []
    for item in files:
        name = Path(item.relative_path).name
        lines.extend([
            f"- 原始文件：`{item.relative_path}`",
            f"  - 作用：{item.purpose}",
            f"  - 只读角色：`{item.role.value}`；媒体类型：`{item.media_type}`",
            f"  - sha256：`{item.sha256 or '运行时核验'}`；大小：`{item.size_bytes if item.size_bytes is not None else '运行时核验'} bytes`",
            f"  - 多模态输入：`{'已提供' if name in multimodal_names else '未单独提供'}`",
            "  - 披露规则：原始 PDF 不复制进交接 Markdown；Agent 只能按此白名单路径只读打开，文本抽取作为同一上下文的辅助证据。",
        ])
    return lines


def _disclosed_text_lines(context: SolveProblemContext) -> list[str]:
    """Include the exact disclosed text in detailed handoffs, when present."""

    if not context.disclosed_text_files:
        return ["- （暂无已披露文本；二进制原文仍以只读路径提供。）"]
    lines: list[str] = []
    for item in context.disclosed_text_files:
        lines.extend([
            f"### `{item.relative_path}`",
            f"- 作用：{item.purpose}",
            f"- sha256（原始文件）：`{item.sha256}`；是否截断：`{item.truncated}`",
            "",
            "```text",
            item.content,
            "```",
        ])
    return lines


def _compact_model_to_code_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    preliminary: tuple[PreliminaryModelingReport, ...],
    modeling: UnifiedModelingReport,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    """Write a reference-only first Model→Code handoff.

    The accepted model is a single canonical ``modeling_report.md``.  This
    boundary must not copy its body (or a preliminary route) into a second
    report; it only tells Code Agent which allowlisted files to read.
    """

    del preliminary, iteration, max_revision_rounds
    available = [item for item in context.readonly_files if "/exchanges/" in item.relative_path.replace("\\", "/")]
    modeling_paths = [item.relative_path for item in available if item.relative_path.endswith("/modeling/modeling_report.md")]
    problem_paths = [
        item.relative_path for item in context.readonly_files
        if getattr(getattr(item, "role", None), "value", getattr(item, "role", "")) == "problem"
        or item.relative_path.lower().endswith(".pdf")
    ]
    lines = [
        "# Model → Code", "",
        f"题目：{task.title}（{task.task_id}）", "",
        "建模报告（唯一建模来源）：",
        *(f"- `{path}`" for path in dict.fromkeys(modeling_paths)),
        "" if modeling_paths else "- 建模报告未在白名单中找到；不得自行补写模型。",
        "",
        "原始题面（只读来源）：",
        *(f"- `{path}`" for path in dict.fromkeys(problem_paths)),
        "" if problem_paths else "- 原始题面未在白名单中找到。",
        "",
        "交接要求：",
        "- Code Agent 先按白名单读取上述建模报告和题面；本文件不重复建模正文。",
        "- 只处理当前题目，按建模报告实现、执行和验证。",
        "- 生成文件、图片和执行结果由 Code→Model 交接直接汇报；不在本文件重复。",
    ]
    return "\n".join(lines).strip() + "\n"


def _compact_code_to_model_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    modeling: UnifiedModelingReport,
    coding: CodingHarnessReport,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    """Write only generated paths and the Code Agent's direct result."""

    del context, modeling, iteration, max_revision_rounds
    paths = [item.relative_path for item in coding.generated_files]
    paths.extend(item.logical_name for item in coding.artifacts if item.logical_name)
    result = (coding.report.markdown or coding.report.summary or "Code Agent 没有返回可读执行结果。").strip()
    lines = [
        "# Code → Model", "",
        f"题目：{task.title}（{task.task_id}）", "",
        "产生的文件/图片：",
        *(f"- {path}" for path in dict.fromkeys(paths)),
        "（本轮没有产生文件或图片。）" if not paths else "",
        "",
        "结果：", result,
    ]
    return "\n".join(lines).strip() + "\n"


def _compact_model_to_code_revision_handoff(
    task: SolveProblemTask,
    context: SolveProblemContext,
    modeling: UnifiedModelingReport,
    coding: CodingHarnessReport,
    review: SolveProblemReview,
    iteration: int,
    *,
    max_revision_rounds: int = MAX_REVISION_ROUNDS,
) -> str:
    """Write the natural-language repair request only."""

    del context, modeling, coding, iteration, max_revision_rounds
    reason = review.rationale.strip()
    instructions = "；".join(item.strip() for item in review.revision_instructions if item.strip())
    requested = "；".join(review.requested_file_paths)
    lines = [
        "# Model → Code（返修）", "",
        f"题目：{task.title}（{task.task_id}）", "",
        "请修改：", reason or "请根据上一轮结果修正实现。",
        instructions or "请重新执行并汇报结果。",
    ]
    if requested:
        lines.extend(["", f"需要查看的文件：{requested}"])
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
