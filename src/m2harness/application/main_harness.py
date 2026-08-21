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
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field

from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.tools import ToolRuntime
from m2harness.domain.capability import CapabilityRequirement
from m2harness.domain.dag import DAGTaskKind, DAGTaskTable
from m2harness.domain.solve_problem import (
    ExplorationMode,
    SolveProblemContext,
    SolveProblemReport,
    SolveProblemStatus,
    SolveProblemTask,
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


class MainHarnessState(StrictModel):
    run_id: UUID
    problem: str = Field(min_length=1, max_length=200_000)
    dag: DAGTaskTable
    tasks: tuple[MainHarnessTask, ...]
    report_ids: tuple[UUID, ...] = ()
    reports: tuple[SolveProblemReport, ...] = ()
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


class MainHarness:
    """Enforce DAG/TODO semantics around one ``solve_problem`` Tool."""

    def __init__(self, tool_runtime: ToolRuntime, capabilities: CapabilityRegistry, repository: MainHarnessRepository | None = None) -> None:
        self.tool_runtime = tool_runtime
        self.capabilities = capabilities
        self.repository = repository

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
        task = SolveProblemTask(
            task_id=node.id, title=node.title, problem=state.problem,
            dependencies=node.depends_on, requested_outputs=(node.output_contract,),
            difficulty=int(node.metadata.get("difficulty", 1)) if isinstance(node.metadata.get("difficulty", 1), int) else 1,
            exploration_mode=ExplorationMode(node.metadata.get("exploration_mode", "auto")) if isinstance(node.metadata.get("exploration_mode", "auto"), str) else ExplorationMode.AUTO,
            max_branches=int(node.metadata.get("max_branches", 3)) if isinstance(node.metadata.get("max_branches", 3), int) else 3,
            revision=attempt - 1, metadata=node.metadata,
        )
        solve_context = context or SolveProblemContext()
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
        return self._commit(state.model_copy(update={
            "final_report": final_report,
            "final_latex_paper": final_latex_paper,
            "terminal": True,
            "updated_at": utc_now(),
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
    def _replace_task(state: MainHarnessState, task: MainHarnessTask) -> MainHarnessState:
        tasks = tuple(task if item.task_id == task.task_id else item for item in state.tasks)
        return state.model_copy(update={"tasks": tasks, "updated_at": utc_now()})
