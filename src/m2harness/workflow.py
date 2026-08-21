"""Durable, deterministic workflow for one production question report."""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from m2harness.artifacts import ArtifactStore
from m2harness.errors import ActivityExecutionError, ConflictError
from m2harness.executor import ActivityExecutor
from m2harness.models import (
    ActivityRequest,
    ActivityResponse,
    ActivityStatus,
    ArtifactKind,
    ArtifactRecord,
    CodingStageOutput,
    FinalizeStageOutput,
    HarnessSettings,
    ModelingStageOutput,
    ProducedArtifact,
    QuestionRecord,
    QuestionState,
    ReviewStageOutput,
    ReviewVerdict,
    StageKind,
    WorkflowStepResult,
    utc_now,
)
from m2harness.store import HarnessStore
from m2harness.infrastructure.db.sqlite_artifacts import SQLiteArtifactRegistry
from m2harness.publication import validate_publication_artifacts
from m2harness.domain.dag import DAGTaskTable, canonical_single_question_dag


REPORT_KINDS = {
    StageKind.MODELING: ArtifactKind.MODELING_REPORT,
    StageKind.CODING: ArtifactKind.CODING_REPORT,
    StageKind.REVIEW: ArtifactKind.REVIEW_REPORT,
    StageKind.FINALIZE: ArtifactKind.FINAL_QUESTION_REPORT,
}
MAX_INPUT_FILE_BYTES = 200 * 1024 * 1024


class _LeaseHeartbeat:
    def __init__(self, store: HarnessStore, question_id: UUID, activity_id: UUID, worker_id: str, seconds: int, attempt_count: int) -> None:
        self.store, self.question_id, self.activity_id = store, question_id, activity_id
        self.worker_id, self.seconds, self.attempt_count = worker_id, seconds, attempt_count
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name=f"m2h-lease-{activity_id}", daemon=True)

    def __enter__(self) -> "_LeaseHeartbeat":
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.wait(max(1.0, self.seconds / 3)):
            try:
                self.store.renew_activity_lease(self.activity_id, self.worker_id, self.seconds, self.attempt_count)
                self.store.renew_question_lease(self.question_id, self.worker_id, self.seconds)
            except BaseException as exc:
                self.error = exc
                self.stop_event.set()

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.error:
            raise self.error


class SingleQuestionWorkflow:
    """Owns orchestration; agents implement activities, never workflow state."""

    def __init__(self, settings: HarnessSettings, executor: ActivityExecutor, *, artifact_registry: SQLiteArtifactRegistry | None = None) -> None:
        self.settings = settings
        self.store = HarnessStore(settings.database_path)
        self.artifacts = ArtifactStore(settings.artifact_root)
        self.executor = executor
        self.artifact_registry = artifact_registry
        self.store.initialize()

    def create_project(self, name: str):
        return self.store.create_project(name)

    def add_question(self, project_id: UUID, key: str, title: str, problem: str) -> QuestionRecord:
        encoded_problem = problem.encode("utf-8")
        if len(encoded_problem) > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"problem artifact exceeds the {MAX_INPUT_FILE_BYTES} byte Harness budget")
        artifact = self.artifacts.put_bytes(
            encoded_problem, project_id=project_id, question_id=None,
            activity_id=None, kind=ArtifactKind.PROBLEM,
            logical_name=f"{key}-problem.md", media_type="text/markdown",
        )
        question = self.store.create_question(project_id, key, title, artifact)
        self._materialize_dag_task_table(question)
        return question

    def add_question_bytes(
        self, project_id: UUID, key: str, title: str, problem: bytes,
        *, logical_name: str, media_type: str,
    ) -> QuestionRecord:
        if len(problem) > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"problem artifact exceeds the {MAX_INPUT_FILE_BYTES} byte Harness budget")
        artifact = self.artifacts.put_bytes(
            problem, project_id=project_id, question_id=None, activity_id=None,
            kind=ArtifactKind.PROBLEM, logical_name=logical_name, media_type=media_type,
        )
        question = self.store.create_question(project_id, key, title, artifact)
        self._materialize_dag_task_table(question)
        return question

    def _materialize_dag_task_table(self, question: QuestionRecord) -> ArtifactRecord:
        """Persist the canonical DAG as a first-class, auditable input."""

        table = canonical_single_question_dag()
        artifact = self.artifacts.put_bytes(
            table.model_dump_json(indent=2).encode("utf-8"),
            project_id=question.project_id, question_id=question.id,
            activity_id=None, kind=ArtifactKind.DAG_TASK_TABLE,
            logical_name=f"{question.key}-dag-task-table.json",
            media_type="application/json",
            metadata={"schema_version": table.schema_version, "terminal_task_id": table.terminal_task_id, "topological_order": table.topological_order()},
        )
        registered = self.store.register_artifact(artifact)
        if self.artifact_registry is not None:
            self.artifact_registry.register(registered)
        return registered

    def _ensure_valid_dag_task_table(self, question: QuestionRecord) -> DAGTaskTable:
        """Ensure every executable question has exactly one valid mainline DAG.

        Questions created by older database/CLI versions may predate the
        first-class DAG artifact.  Backfilling that immutable artifact is safe;
        accepting a malformed or substituted graph is not, because it would
        let an activity silently bypass the publication terminal.
        """

        tables = [item for item in self.store.list_artifacts(question.id) if item.kind == ArtifactKind.DAG_TASK_TABLE]
        if not tables:
            self._materialize_dag_task_table(question)
            tables = [item for item in self.store.list_artifacts(question.id) if item.kind == ArtifactKind.DAG_TASK_TABLE]
        if len(tables) != 1:
            raise ConflictError(f"question must have exactly one DAG task table; found {len(tables)}")
        try:
            table = DAGTaskTable.model_validate_json(self.artifacts.read(tables[0]), strict=False)
        except Exception as exc:
            raise ConflictError(f"invalid DAG task table: {exc}") from exc
        if table.terminal_task_id != "publish-paper":
            raise ConflictError("mainline DAG terminal task must be publish-paper")
        return table

    def add_input(
        self, question_id: UUID, data: bytes, *, logical_name: str,
        media_type: str, kind: ArtifactKind = ArtifactKind.INPUT,
    ) -> ArtifactRecord:
        question = self.store.get_question(question_id)
        if len(data) > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"input artifact exceeds the {MAX_INPUT_FILE_BYTES} byte Harness budget")
        artifact = self.artifacts.put_bytes(
            data, project_id=question.project_id, question_id=question.id,
            activity_id=None, kind=kind, logical_name=logical_name, media_type=media_type,
        )
        return self.store.register_artifact(artifact)

    def run_step(self, question_id: UUID, worker_id: str) -> WorkflowStepResult:
        question = self.store.acquire_question_lease(question_id, worker_id, self.settings.lease_seconds)
        result: WorkflowStepResult | None = None
        try:
            if question.state in (QuestionState.COMPLETED, QuestionState.FAILED):
                result = WorkflowStepResult(question=question, terminal=True)
                return result
            self._ensure_valid_dag_task_table(question)
            question, stage = self._enter_stage(question, worker_id)
            activity = self.store.get_or_create_activity(question, stage)
            if activity.status == ActivityStatus.SUCCEEDED:
                response = ActivityResponse.model_validate(activity.result_json, strict=False)
                published = [item for item in self.store.list_artifacts(question.id) if item.activity_id == activity.id]
            else:
                request = self._request(question, activity.id, activity.idempotency_key, activity.attempt_count + 1, stage)
                try:
                    activity = self.store.claim_activity(
                        activity.id, worker_id, self.settings.lease_seconds,
                        self.settings.max_activity_attempts, request.model_dump(mode="json"),
                    )
                except ConflictError as exc:
                    if "leased by" in str(exc):
                        raise
                    failed = self.store.transition(
                        question.id, worker_id, [question.state], QuestionState.FAILED,
                        failure_reason=str(exc),
                    )
                    result = WorkflowStepResult(question=failed, activity=activity, terminal=True)
                    return result
                request = request.model_copy(update={"attempt": activity.attempt_count})
                try:
                    with _LeaseHeartbeat(self.store, question.id, activity.id, worker_id, self.settings.lease_seconds, activity.attempt_count):
                        response = self.executor.execute(request)
                    published = self._materialize_response(question, activity.id, response)
                    activity = self.store.complete_activity(
                        activity.id, worker_id, response.model_dump(mode="json"), published, attempt_count=activity.attempt_count)
                except Exception as exc:
                    try:
                        self.store.fail_activity(activity.id, worker_id, str(exc), attempt_count=activity.attempt_count)
                    except Exception:
                        pass
                    raise ActivityExecutionError(f"{stage.value} activity failed: {exc}") from exc
            question = self._complete_stage(question, worker_id, response)
            result = WorkflowStepResult(
                question=question, activity=activity, published_artifacts=published,
                terminal=question.state in (QuestionState.COMPLETED, QuestionState.FAILED),
            )
            return result
        finally:
            self.store.release_question_lease(question_id, worker_id)
            # The returned object is an operator-facing snapshot.  Refresh it
            # after lease release so a successful step cannot look locked until
            # the next status query.
            if result is not None:
                result.question = self.store.get_question(question_id)

    def run_until_terminal(self, question_id: UUID, worker_id: str, max_steps: int = 100) -> WorkflowStepResult:
        result: WorkflowStepResult | None = None
        for _ in range(max_steps):
            result = self.run_step(question_id, worker_id)
            if result.terminal:
                return result
        raise ConflictError(f"workflow did not terminate within {max_steps} steps")

    def _enter_stage(self, question: QuestionRecord, worker_id: str) -> tuple[QuestionRecord, StageKind]:
        state = question.state
        if state == QuestionState.PENDING:
            question = self.store.transition(question.id, worker_id, [state], QuestionState.MODELING)
            return question, StageKind.MODELING
        if state == QuestionState.MODELING:
            return question, StageKind.MODELING
        if state in (QuestionState.READY_FOR_CODING, QuestionState.REVISION_REQUIRED):
            question = self.store.transition(question.id, worker_id, [state], QuestionState.CODING)
            return question, StageKind.CODING
        if state == QuestionState.CODING:
            return question, StageKind.CODING
        if state == QuestionState.READY_FOR_REVIEW:
            question = self.store.transition(question.id, worker_id, [state], QuestionState.REVIEWING)
            return question, StageKind.REVIEW
        if state == QuestionState.REVIEWING:
            return question, StageKind.REVIEW
        if state == QuestionState.APPROVED:
            question = self.store.transition(question.id, worker_id, [state], QuestionState.FINALIZING)
            return question, StageKind.FINALIZE
        if state == QuestionState.FINALIZING:
            return question, StageKind.FINALIZE
        raise ConflictError(f"state has no executable stage: {state.value}")

    def _request(self, question: QuestionRecord, activity_id: UUID, key: str, attempt: int, stage: StageKind) -> ActivityRequest:
        project = self.store.get_project(question.project_id)
        inputs = self._inputs_for(question, stage)
        instructions = {
            StageKind.MODELING: ["Follow the canonical DAG task table; produce an auditable modeling report and explicit validation contract."],
            StageKind.CODING: ["Implement and execute the model; report evidence, metrics, and generated artifacts."],
            StageKind.REVIEW: ["Independently review assumptions, execution evidence, validations, and claims."],
            StageKind.FINALIZE: ["This is the terminal DAG task: produce the final self-contained single-question report and one polished final_latex_paper artifact using only reviewed evidence."],
        }[stage]
        if question.revision:
            instructions.append(f"This is controlled revision {question.revision}; resolve the latest review instructions.")
            instructions.extend(self._revision_instructions(question))
        return ActivityRequest(
            activity_id=activity_id, idempotency_key=key, project=project,
            question=question, stage=stage, attempt=attempt, revision=question.revision,
            inputs=inputs, instructions=instructions,
            deadline=utc_now() + timedelta(seconds=self.settings.activity_timeout_seconds),
        )

    def _revision_instructions(self, question: QuestionRecord) -> list[str]:
        """Return the last committed review's concrete revision instructions."""
        reviews = [item for item in self.store.list_activities(question.id) if item.stage == StageKind.REVIEW and item.status == ActivityStatus.SUCCEEDED and item.result_json]
        if not reviews:
            return []
        latest = max(reviews, key=lambda item: (item.revision, item.updated_at))
        try:
            output = ActivityResponse.model_validate(latest.result_json, strict=False).output
        except Exception:
            return []
        if not isinstance(output, ReviewStageOutput):
            return []
        return [f"Review instruction: {instruction}" for instruction in output.revision_instructions]

    def _inputs_for(self, question: QuestionRecord, stage: StageKind) -> list[ArtifactRecord]:
        artifacts = self.store.list_artifacts(question.id)
        allowed = {
            StageKind.MODELING: {ArtifactKind.PROBLEM, ArtifactKind.DAG_TASK_TABLE, ArtifactKind.INPUT, ArtifactKind.SOURCE},
            StageKind.CODING: {ArtifactKind.PROBLEM, ArtifactKind.DAG_TASK_TABLE, ArtifactKind.INPUT, ArtifactKind.SOURCE, ArtifactKind.MODELING_REPORT, ArtifactKind.CODING_REPORT, ArtifactKind.REVIEW_REPORT, ArtifactKind.DATA, ArtifactKind.OUTPUT, ArtifactKind.LOG, ArtifactKind.FIGURE},
            StageKind.REVIEW: set(ArtifactKind) - {ArtifactKind.FINAL_QUESTION_REPORT},
            StageKind.FINALIZE: set(ArtifactKind) - {ArtifactKind.FINAL_QUESTION_REPORT},
        }[stage]
        return [artifact for artifact in artifacts if artifact.kind in allowed]

    def _materialize_response(self, question: QuestionRecord, activity_id: UUID, response: ActivityResponse) -> list[ArtifactRecord]:
        output = response.output
        if isinstance(output, FinalizeStageOutput) and self.settings.require_latex_publication:
            # Validate before writing any final bytes.  A malformed publication
            # is therefore retryable and cannot leave a misleading "completed"
            # report in the immutable artifact store.
            try:
                validate_publication_artifacts(output.artifacts)
            except ValueError as exc:
                raise ActivityExecutionError(str(exc)) from exc
        report = ProducedArtifact(
            logical_name=f"{output.stage.value}-r{question.revision}-report.md",
            kind=REPORT_KINDS[output.stage], media_type="text/markdown",
            text=output.report.markdown,
            metadata={"title": output.report.title, "summary": output.report.summary, "revision": question.revision},
        )
        records = [self.artifacts.put_produced(report, project_id=question.project_id, question_id=question.id, activity_id=activity_id)]
        for produced in output.artifacts:
            records.append(self.artifacts.put_produced(produced, project_id=question.project_id, question_id=question.id, activity_id=activity_id))
        if self.artifact_registry is not None:
            for record in records:
                self.artifact_registry.register(record)
        return records

    def _complete_stage(self, question: QuestionRecord, worker_id: str, response: ActivityResponse) -> QuestionRecord:
        output = response.output
        if isinstance(output, ModelingStageOutput):
            return self.store.transition(question.id, worker_id, [QuestionState.MODELING], QuestionState.READY_FOR_CODING)
        if isinstance(output, CodingStageOutput):
            return self.store.transition(question.id, worker_id, [QuestionState.CODING], QuestionState.READY_FOR_REVIEW)
        if isinstance(output, ReviewStageOutput):
            if output.verdict == ReviewVerdict.APPROVE:
                evidence_error = self._approval_evidence_error(question)
                if evidence_error:
                    return self.store.transition(
                        question.id, worker_id, [QuestionState.REVIEWING], QuestionState.FAILED,
                        failure_reason=evidence_error,
                    )
                return self.store.transition(question.id, worker_id, [QuestionState.REVIEWING], QuestionState.APPROVED)
            revision = question.revision + 1
            if revision > self.settings.max_revisions:
                return self.store.transition(
                    question.id, worker_id, [QuestionState.REVIEWING], QuestionState.FAILED,
                    failure_reason=f"revision limit {self.settings.max_revisions} exhausted",
                )
            return self.store.transition(question.id, worker_id, [QuestionState.REVIEWING], QuestionState.REVISION_REQUIRED, revision=revision)
        if isinstance(output, FinalizeStageOutput):
            return self.store.transition(question.id, worker_id, [QuestionState.FINALIZING], QuestionState.COMPLETED)
        raise ActivityExecutionError("unrecognized activity output")

    def _approval_evidence_error(self, question: QuestionRecord) -> str | None:
        activities = self.store.list_activities(question.id)
        modeling = next((item for item in activities if item.stage == StageKind.MODELING and item.status == ActivityStatus.SUCCEEDED), None)
        coding = next((item for item in reversed(activities) if item.stage == StageKind.CODING and item.revision == question.revision and item.status == ActivityStatus.SUCCEEDED), None)
        if not modeling or not coding or not modeling.result_json or not coding.result_json:
            return "review approval lacks committed modeling or coding evidence"
        model_output = ActivityResponse.model_validate(modeling.result_json, strict=False).output
        coding_response = ActivityResponse.model_validate(coding.result_json, strict=False)
        code_output = coding_response.output
        if not isinstance(model_output, ModelingStageOutput) or not isinstance(code_output, CodingStageOutput):
            return "review approval references incompatible evidence stages"
        execution_backend = coding_response.executor_metadata.get("execution_backend")
        # Qwen is a model provider, not an execution authority.  Its claimed
        # execution/validation fields become admissible only after the
        # SandboxedActivityExecutor has replaced them with evidence from the
        # configured local Sandbox.  Keep the legacy process adapter usable for
        # deterministic contract tests, but never allow a Qwen provider or a
        # remote code interpreter to close a production report without this
        # boundary.
        if coding_response.executor_metadata.get("provider") == "qwen" and execution_backend != "local-sandbox":
            return "review approval requires local Sandbox evidence for Qwen coding output"
        if execution_backend == "remote-code-interpreter":
            return "review approval requires local Sandbox evidence; remote code interpreter output is preliminary only"
        if not code_output.execution_succeeded:
            return "review approved an unsuccessful execution"
        failed = [name for name in model_output.required_validations if code_output.validations.get(name) is not True]
        if failed:
            return "review approved missing or failed validations: " + ", ".join(failed)
        return None
