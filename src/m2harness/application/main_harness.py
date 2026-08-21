"""Main Harness process owner for DAG/TODO-driven ``solve_problem`` calls.

The Main Harness is the top-level Agent's durable process boundary.  It owns
the task graph, dependency unlocking, retry/revision state and report index.
It does not perform mathematical modeling itself and it does not create a
subagent tree.  Each executable problem node is dispatched through the single
registered ``solve_problem`` Tool.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import json
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field

from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.tools import ToolRuntime
from m2harness.application.compact import compact_text, estimate_tokens
from m2harness.application.report_store import ReportFileRecord, RunReportStore
from m2harness.domain.capability import CapabilityRequirement
from m2harness.domain.dag import DAGTaskKind, DAGTaskTable
from m2harness.domain.solve_problem import (
    ExplorationMode,
    DependencySolutionContext,
    ReadOnlyFileRole,
    SolveProblemContext,
    SolveProblemReport,
    SolveProblemStatus,
    SolveProblemTask,
    MAX_REVISION_ROUNDS,
    ModelReviewDecision,
)
from m2harness.domain.tool import ToolCall
from m2harness.models import ArtifactKind, ProducedArtifact, ReportPayload, StrictModel, utc_now
from m2harness.publication import validate_publication_artifacts


class MainTaskStatus(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    REVISION_REQUIRED = "revision_required"
    FAILED = "failed"


class MainHarnessTask(StrictModel):
    task_id: str
    status: MainTaskStatus
    attempts: int = Field(default=0, ge=0)
    last_report_id: UUID | None = None
    last_error: str | None = None


class MainHarnessDecision(StrictModel):
    kind: str
    reason: str = Field(min_length=1, max_length=10_000)
    affected_task_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class MainHarnessState(StrictModel):
    run_id: UUID
    problem: str = Field(min_length=1, max_length=200_000)
    dag: DAGTaskTable
    tasks: tuple[MainHarnessTask, ...]
    report_ids: tuple[UUID, ...] = ()
    reports: tuple[SolveProblemReport, ...] = ()
    report_files: tuple[ReportFileRecord, ...] = ()
    decisions: tuple[MainHarnessDecision, ...] = ()
    final_report: ReportPayload | None = None
    final_latex_paper: ProducedArtifact | None = None
    terminal: bool = False
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MainHarnessRepository(Protocol):
    """Durable state port for Main Harness runs."""

    def create(self, state: MainHarnessState) -> MainHarnessState: ...
    def get(self, run_id: UUID) -> MainHarnessState: ...
    def save(self, state: MainHarnessState, expected_version: int) -> MainHarnessState: ...


class PaperComposerPort(Protocol):
    """Main-Harness-owned final report/LaTeX composition boundary."""

    def compose(self, problem: str, reports: tuple[SolveProblemReport, ...]) -> tuple[ReportPayload, ProducedArtifact]: ...


class MainHarness:
    """Enforce DAG/TODO semantics around one ``solve_problem`` Tool."""

    def __init__(self, tool_runtime: ToolRuntime, capabilities: CapabilityRegistry, repository: MainHarnessRepository | None = None, report_store: RunReportStore | None = None) -> None:
        self.tool_runtime = tool_runtime
        self.capabilities = capabilities
        self.repository = repository
        self.report_store = report_store

    def start(self, problem: str, dag: DAGTaskTable, *, run_id: UUID | None = None) -> MainHarnessState:
        """Create a process state from a planner-provided DAG.

        The planner can choose a serial graph, a parallel graph, or a simple
        TODO chain.  The Main Harness validates it and marks only root tasks
        ready; no downstream task is visible as executable prematurely.
        """

        statuses = []
        for node in dag.tasks:
            status = MainTaskStatus.READY if not node.depends_on else MainTaskStatus.BLOCKED
            statuses.append(MainHarnessTask(task_id=node.id, status=status))
        now = utc_now()
        state = MainHarnessState(
            run_id=run_id or uuid4(), problem=problem, dag=dag,
            tasks=tuple(statuses), created_at=now, updated_at=now,
        )
        if self.repository is not None:
            return self.repository.create(state)
        return state

    def ready_tasks(self, state: MainHarnessState) -> tuple[str, ...]:
        by_id = {item.task_id: item for item in state.tasks}
        result: list[str] = []
        for node in state.dag.tasks:
            record = by_id[node.id]
            if record.status not in {MainTaskStatus.READY, MainTaskStatus.REVISION_REQUIRED}:
                continue
            if all(by_id[dependency].status == MainTaskStatus.COMPLETED for dependency in node.depends_on):
                result.append(node.id)
        return tuple(result)

    def todo_view(self, state: MainHarnessState) -> dict[str, object]:
        """Return the intentionally small operator-facing TODO projection.

        Durable state still retains typed reports and file indexes for audit,
        but the Main Harness display is a plan/status view rather than a dump
        of every Model/Code/Review payload.
        """

        by_id = {item.task_id: item for item in state.tasks}
        return {
            "run_id": str(state.run_id),
            "todo": [
                {
                    "task_id": node.id,
                    "title": node.title,
                    "kind": node.kind.value,
                    "status": by_id[node.id].status.value,
                    "depends_on": list(node.depends_on),
                    "scope": node.metadata.get("scope"),
                    "ready": node.id in self.ready_tasks(state),
                }
                for node in state.dag.tasks
            ],
            "next_ready": list(self.ready_tasks(state)),
            "report_file_count": len(state.report_files),
            "terminal": state.terminal,
            "version": state.version,
        }

    def dispatch(
        self,
        state: MainHarnessState,
        task_id: str,
        *,
        context: SolveProblemContext | None = None,
        max_iterations: int = 3,
    ) -> MainHarnessState:
        """Dispatch exactly one ready problem node to ``solve_problem``."""

        node = next((item for item in state.dag.tasks if item.id == task_id), None)
        if node is None:
            raise ValueError(f"unknown DAG task: {task_id}")
        if node.kind != DAGTaskKind.SOLVE_PROBLEM:
            raise ValueError("only solve_problem DAG tasks are executable by dispatch()")
        by_id = {item.task_id: item for item in state.tasks}
        record = by_id[task_id]
        if record.status not in {MainTaskStatus.READY, MainTaskStatus.REVISION_REQUIRED}:
            raise ValueError(f"DAG task {task_id} is not ready: {record.status.value}")
        if record.status == MainTaskStatus.REVISION_REQUIRED and self._revision_rounds_consumed(state, task_id) >= MAX_REVISION_ROUNDS:
            raise ValueError(
                f"DAG task {task_id} exhausted the maximum of {MAX_REVISION_ROUNDS} review-driven revision rounds"
            )
        if not all(by_id[dependency].status == MainTaskStatus.COMPLETED for dependency in node.depends_on):
            raise ValueError(f"DAG task {task_id} has incomplete dependencies")
        definition = self.tool_runtime.registry.get("solve_problem")
        if definition is None:
            raise RuntimeError("solve_problem tool is not registered")
        resolution = self.capabilities.resolve([
            CapabilityRequirement(capability=definition.required_capability, reason="Main Harness problem dispatch"),
        ])
        if not resolution.complete:
            raise PermissionError("Main Harness lacks problem.solve capability")

        attempt = record.attempts + 1
        node_problem = node.metadata.get("problem", state.problem)
        if not isinstance(node_problem, str) or not node_problem.strip():
            node_problem = state.problem
        task = SolveProblemTask(
            task_id=node.id, title=node.title, problem=node_problem,
            dependencies=node.depends_on, requested_outputs=(node.output_contract,),
            difficulty=int(node.metadata.get("difficulty", 1)) if isinstance(node.metadata.get("difficulty", 1), int) else 1,
            exploration_mode=ExplorationMode(node.metadata.get("exploration_mode", "auto")) if isinstance(node.metadata.get("exploration_mode", "auto"), str) else ExplorationMode.AUTO,
            max_branches=int(node.metadata.get("max_branches", 3)) if isinstance(node.metadata.get("max_branches", 3), int) else 3,
            revision=attempt - 1, metadata=node.metadata,
        )
        solve_context = self._context_for_dispatch(state, node.id, node.depends_on, context)
        solve_context = solve_context.model_copy(update={
            "metadata": {
                **solve_context.metadata,
                "run_id": str(state.run_id),
                "task_attempt": attempt,
            },
        })
        call = ToolCall(
            call_id=uuid4(), tool_name=definition.name, tool_version=definition.version,
            activity_id=uuid4(), session_id=uuid4(),
            idempotency_key=f"main-harness:{state.run_id}:{task_id}:attempt-{attempt}",
            arguments={"task": task.model_dump(mode="json"), "context": solve_context.model_dump(mode="json"), "max_iterations": max_iterations},
            requested_at=utc_now(),
        )
        running_record = record.model_copy(update={"status": MainTaskStatus.RUNNING, "attempts": attempt, "last_error": None})
        working = self._replace_task(state, running_record)
        result = self.tool_runtime.execute(call, resolution)
        if not result.ok:
            failed = running_record.model_copy(update={"status": MainTaskStatus.FAILED, "last_error": result.error_message})
            return self._commit(self._replace_task(working, failed), state.version)
        try:
            report = SolveProblemReport.model_validate(result.output or {}, strict=False)
        except Exception as exc:
            failed = running_record.model_copy(update={"status": MainTaskStatus.FAILED, "last_error": f"invalid solve_problem report: {exc}"})
            return self._commit(self._replace_task(working, failed), state.version)
        report_id = uuid4()
        try:
            persisted_files = self.report_store.persist(state.run_id, report, attempt=attempt) if self.report_store is not None else ()
        except Exception as exc:
            failed = running_record.model_copy(update={"status": MainTaskStatus.FAILED, "last_error": f"durable report persistence failed: {exc}"})
            return self._commit(self._replace_task(working, failed), state.version)
        if report.status == SolveProblemStatus.COMPLETED:
            status = MainTaskStatus.COMPLETED
            error = None
        elif report.status == SolveProblemStatus.REVISION_REQUIRED:
            status = MainTaskStatus.REVISION_REQUIRED
            error = report.error
        elif report.status == SolveProblemStatus.BLOCKED:
            status = MainTaskStatus.BLOCKED
            error = report.error
        else:
            status = MainTaskStatus.FAILED
            error = report.error
        updated = running_record.model_copy(update={"status": status, "last_report_id": report_id, "last_error": error})
        state_with_report = working.model_copy(update={
            "reports": (*working.reports, report), "report_ids": (*working.report_ids, report_id),
            "report_files": (*working.report_files, *persisted_files),
            "updated_at": utc_now(),
        })
        state_with_report = self._replace_task(state_with_report, updated)
        return self._commit(self._unlock_dependents(state_with_report), state.version)

    def publish(
        self,
        state: MainHarnessState,
        *,
        final_report: ReportPayload,
        final_latex_paper: ProducedArtifact,
    ) -> MainHarnessState:
        """Close the Main Harness terminal with the reviewed report and TeX.

        The actual bytes should be persisted through ``report_render`` and the
        ArtifactStore by the deployment adapter.  This method enforces the
        process-level contract before a caller marks the run terminal.
        """

        terminal = next(item for item in state.dag.tasks if item.id == state.dag.terminal_task_id)
        if terminal.kind != DAGTaskKind.PUBLISH_LATEX:
            raise ValueError("DAG terminal is not publish_latex")
        by_id = {item.task_id: item for item in state.tasks}
        if any(item.status != MainTaskStatus.COMPLETED for item in state.tasks if item.task_id != terminal.id):
            raise ValueError("cannot publish before every solve_problem task is completed")
        if not all(by_id[dependency].status == MainTaskStatus.COMPLETED for dependency in terminal.depends_on):
            raise ValueError("cannot publish before terminal dependencies are completed")
        if final_latex_paper.kind != ArtifactKind.FINAL_LATEX_PAPER:
            raise ValueError("final publication artifact must have kind final_latex_paper")
        validate_publication_artifacts([final_latex_paper])
        publication_files = self.report_store.persist_publication(
            state.run_id, final_report, final_latex_paper,
        ) if self.report_store is not None else ()
        completed_terminal = by_id[terminal.id].model_copy(update={"status": MainTaskStatus.COMPLETED})
        published_state = self._replace_task(state, completed_terminal)
        return self._commit(published_state.model_copy(update={
            "final_report": final_report,
            "final_latex_paper": final_latex_paper,
            "report_files": (*published_state.report_files, *publication_files),
            "terminal": True,
            "updated_at": utc_now(),
        }), state.version)

    def generate_paper(self, state: MainHarnessState, composer: PaperComposerPort) -> MainHarnessState:
        """Generate and publish the terminal paper after all solve nodes finish.

        The terminal DAG node is deliberately not dispatched through
        ``solve_problem``.  It is a Main Harness operation with global report
        context, so a paper composer cannot accidentally see only one task or
        close the run before the dependency graph is complete.
        """
        terminal = next(item for item in state.dag.tasks if item.id == state.dag.terminal_task_id)
        if terminal.kind != DAGTaskKind.PUBLISH_LATEX:
            raise ValueError("DAG terminal is not publish_latex")
        if any(item.status != MainTaskStatus.COMPLETED for item in state.tasks if item.task_id != terminal.id):
            raise ValueError("cannot compose paper before every solve_problem task is completed")
        final_report, final_latex = composer.compose(state.problem, state.reports)
        return self.publish(state, final_report=final_report, final_latex_paper=final_latex)

    def rollback(self, state: MainHarnessState, task_id: str, *, reason: str) -> MainHarnessState:
        """Invalidate one task and all descendants while retaining audit history."""

        if state.terminal:
            raise ValueError("cannot rollback a terminal run without starting a new revision run")
        if not any(node.id == task_id for node in state.dag.tasks):
            raise ValueError(f"unknown DAG task: {task_id}")
        affected = {task_id}
        changed = True
        while changed:
            changed = False
            for node in state.dag.tasks:
                if node.id not in affected and any(dependency in affected for dependency in node.depends_on):
                    affected.add(node.id)
                    changed = True
        tasks = []
        for record in state.tasks:
            if record.task_id not in affected:
                tasks.append(record)
            elif record.task_id == task_id:
                tasks.append(record.model_copy(update={"status": MainTaskStatus.READY, "last_error": reason}))
            else:
                tasks.append(record.model_copy(update={"status": MainTaskStatus.BLOCKED, "last_error": f"upstream rollback: {task_id}"}))
        decision = MainHarnessDecision(kind="rollback", reason=reason, affected_task_ids=tuple(sorted(affected)))
        return self._commit(state.model_copy(update={
            "tasks": tuple(tasks), "decisions": (*state.decisions, decision),
            "final_report": None, "final_latex_paper": None, "terminal": False,
        }), state.version)

    def replan(self, state: MainHarnessState, dag: DAGTaskTable, *, reason: str) -> MainHarnessState:
        """Replace the remaining DAG while preserving unchanged completed work."""

        if state.terminal:
            raise ValueError("cannot replan a terminal run")
        old_nodes = {node.id: node for node in state.dag.tasks}
        old_tasks = {task.task_id: task for task in state.tasks}
        completed: set[str] = set()
        for node in dag.tasks:
            previous = old_nodes.get(node.id)
            record = old_tasks.get(node.id)
            if previous == node and record is not None and record.status == MainTaskStatus.COMPLETED:
                completed.add(node.id)
        tasks: list[MainHarnessTask] = []
        for node in dag.tasks:
            previous_record = old_tasks.get(node.id)
            if node.id in completed and previous_record is not None:
                tasks.append(previous_record)
            else:
                status = MainTaskStatus.READY if all(dependency in completed for dependency in node.depends_on) else MainTaskStatus.BLOCKED
                tasks.append(MainHarnessTask(task_id=node.id, status=status))
        changed_ids = tuple(sorted(set(old_nodes) ^ {node.id for node in dag.tasks} | {
            node.id for node in dag.tasks if old_nodes.get(node.id) != node
        }))
        decision = MainHarnessDecision(kind="replan", reason=reason, affected_task_ids=changed_ids)
        return self._commit(state.model_copy(update={
            "dag": dag, "tasks": tuple(tasks), "decisions": (*state.decisions, decision),
            "final_report": None, "final_latex_paper": None, "terminal": False,
        }), state.version)

    def _commit(self, state: MainHarnessState, expected_version: int) -> MainHarnessState:
        committed = state.model_copy(update={"version": expected_version + 1, "updated_at": utc_now()})
        if self.repository is not None:
            return self.repository.save(committed, expected_version=expected_version)
        return committed

    def _unlock_dependents(self, state: MainHarnessState) -> MainHarnessState:
        by_id = {item.task_id: item for item in state.tasks}
        changed = False
        for node in state.dag.tasks:
            current = by_id[node.id]
            if current.status != MainTaskStatus.BLOCKED:
                continue
            if all(by_id[dependency].status == MainTaskStatus.COMPLETED for dependency in node.depends_on):
                by_id[node.id] = current.model_copy(update={"status": MainTaskStatus.READY})
                changed = True
        if not changed:
            return state
        return state.model_copy(update={"tasks": tuple(by_id[item.id] for item in state.dag.tasks), "updated_at": utc_now()})

    @staticmethod
    def _revision_rounds_consumed(state: MainHarnessState, task_id: str) -> int:
        """Count durable non-approval review decisions across resumed dispatches."""

        consumed = 0
        for report in state.reports:
            if report.task_id != task_id:
                continue
            consumed += sum(
                1 for iteration in report.iterations
                if iteration.review.decision in {ModelReviewDecision.REVISE, ModelReviewDecision.REJECT}
            )
        return consumed

    @staticmethod
    def _replace_task(state: MainHarnessState, task: MainHarnessTask) -> MainHarnessState:
        tasks = tuple(task if item.task_id == task.task_id else item for item in state.tasks)
        return state.model_copy(update={"tasks": tasks, "updated_at": utc_now()})

    @staticmethod
    def _context_for_dispatch(
        state: MainHarnessState,
        task_id: str,
        dependencies: tuple[str, ...],
        supplied: SolveProblemContext | None,
    ) -> SolveProblemContext:
        """Build progressive-disclosure context from durable report snapshots."""
        context = supplied or SolveProblemContext()
        report_pairs = tuple(zip(state.report_ids, state.reports))
        dependency_ids: list[UUID] = list(context.dependency_report_ids)
        accepted_ids: list[UUID] = list(context.accepted_report_ids)
        dependency_solutions: list[DependencySolutionContext] = list(context.dependency_solutions)
        for report_id, report in report_pairs:
            if report.task_id in dependencies and report_id not in dependency_ids:
                dependency_ids.append(report_id)
                if report.status == SolveProblemStatus.COMPLETED and report_id not in accepted_ids:
                    accepted_ids.append(report_id)
        existing_dependency_tasks = {item.task_id for item in dependency_solutions}
        per_dependency_budget = max(1_000, context.context_budget_tokens // max(1, len(dependencies)))
        for dependency in dependencies:
            if dependency in existing_dependency_tasks:
                continue
            match = next((
                (report_id, report) for report_id, report in reversed(report_pairs)
                if report.task_id == dependency and report.status == SolveProblemStatus.COMPLETED and report.final_report is not None
            ), None)
            if match is None:
                continue
            report_id, report = match
            assert report.final_report is not None
            dependency_task = next(item for item in state.tasks if item.task_id == dependency)
            files = tuple(
                item.as_readonly() for item in state.report_files
                if item.task_id == dependency
                and item.attempt == dependency_task.attempts
                and (
                    item.role == ReadOnlyFileRole.DEPENDENCY_SOLUTION
                    or item.iteration == report.iteration_count
                    or "/exchanges/" in item.relative_path.replace("\\", "/")
                )
                and item.role in {
                    ReadOnlyFileRole.DEPENDENCY_SOLUTION,
                    ReadOnlyFileRole.DEPENDENCY_OUTPUT,
                    ReadOnlyFileRole.GENERATED,
                }
            )
            solution_path = next((item.relative_path for item in files if item.role == ReadOnlyFileRole.DEPENDENCY_SOLUTION), None)
            requested_keys = node_dependency_outputs(state.dag, task_id, dependency)
            structured = report.final_report.structured
            downstream = structured.get("downstream_outputs", structured) if isinstance(structured, dict) else {}
            if requested_keys and isinstance(downstream, dict):
                downstream = {key: downstream[key] for key in requested_keys if key in downstream}
            summary = compact_text(report.final_report.summary, max(500, per_dependency_budget // 2))
            if estimate_tokens(downstream) > max(500, per_dependency_budget // 3):
                downstream = {"_compacted": compact_text(
                    json.dumps(downstream, ensure_ascii=False, sort_keys=True, default=str),
                    max(500, per_dependency_budget // 3),
                )}
            dependency_solutions.append(DependencySolutionContext(
                task_id=dependency, report_id=report_id, title=report.final_report.title,
                summary=summary,
                claims=tuple(compact_text(item, 500) for item in report.final_report.claims[:12]),
                limitations=tuple(compact_text(item, 400) for item in report.final_report.limitations[:8]),
                downstream_outputs=downstream if isinstance(downstream, dict) else {},
                solution_report_path=solution_path, solve_files=files,
                estimated_tokens=estimate_tokens(summary) + estimate_tokens(downstream),
            ))
        instructions = list(context.instructions)
        research_report = context.research_report
        for _, report in reversed(report_pairs):
            if report.task_id != task_id:
                continue
            if report.status == SolveProblemStatus.REVISION_REQUIRED:
                instructions.extend(item for item in report.revision_instructions if item not in instructions)
            if research_report is None and report.research_report is not None:
                research_report = report.research_report
            break
        if estimate_tokens(instructions) > 4_000:
            recent_instructions = instructions[-10:]
            older = "\n".join(f"- {item}" for item in instructions[:-10])
            instructions = ["Compacted earlier revision history:\n" + compact_text(older, 2_000), *recent_instructions]
        return context.model_copy(update={
            "dependency_report_ids": tuple(dependency_ids),
            "accepted_report_ids": tuple(accepted_ids),
            "dependency_solutions": tuple(dependency_solutions),
            "readonly_files": _merge_readonly_files(
                context.readonly_files,
                tuple(file for solution in dependency_solutions for file in solution.solve_files),
            ),
            "instructions": tuple(instructions),
            "research_report": research_report,
            "compression": {
                **context.compression,
                "strategy": "compact-v1",
                "dependency_count": len(dependency_solutions),
                "budget_tokens": context.context_budget_tokens,
                "estimated_dependency_tokens": sum(item.estimated_tokens for item in dependency_solutions),
            },
        })


def _merge_readonly_files(*groups):
    merged = {}
    for group in groups:
        for item in group:
            merged[item.relative_path] = item
    return tuple(merged.values())


def node_dependency_outputs(dag: DAGTaskTable, task_id: str, dependency: str) -> tuple[str, ...]:
    node = next(item for item in dag.tasks if item.id == task_id)
    return tuple(node.dependency_outputs.get(dependency, ()))
