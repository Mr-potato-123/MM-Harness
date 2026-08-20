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


REPORT_KINDS = {
    StageKind.MODELING: ArtifactKind.MODELING_REPORT,
    StageKind.CODING: ArtifactKind.CODING_REPORT,
    StageKind.REVIEW: ArtifactKind.REVIEW_REPORT,
    StageKind.FINALIZE: ArtifactKind.FINAL_QUESTION_REPORT,
}


class _LeaseHeartbeat:
    def __init__(self, store: HarnessStore, question_id: UUID, activity_id: UUID, worker_id: str, seconds: int) -> None:
        self.store, self.question_id, self.activity_id = store, question_id, activity_id
        self.worker_id, self.seconds = worker_id, seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name=f"m2h-lease-{activity_id}", daemon=True)

    def __enter__(self) -> "_LeaseHeartbeat":
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.wait(max(1.0, self.seconds / 3)):
            try:
                self.store.renew_question_lease(self.question_id, self.worker_id, self.seconds)
                self.store.renew_activity_lease(self.activity_id, self.worker_id, self.seconds)
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

    def __init__(self, settings: HarnessSettings, executor: ActivityExecutor) -> None:
        self.settings = settings
        self.store = HarnessStore(settings.database_path)
        self.artifacts = ArtifactStore(settings.artifact_root)
        self.executor = executor
        self.store.initialize()

    def create_project(self, name: str):
        return self.store.create_project(name)

    def add_question(self, project_id: UUID, key: str, title: str, problem: str) -> QuestionRecord:
        artifact = self.artifacts.put_bytes(
            problem.encode("utf-8"), project_id=project_id, question_id=None,
            activity_id=None, kind=ArtifactKind.PROBLEM,
            logical_name=f"{key}-problem.md", media_type="text/markdown",
        )
        return self.store.create_question(project_id, key, title, artifact)

    def add_question_bytes(
        self, project_id: UUID, key: str, title: str, problem: bytes,
        *, logical_name: str, media_type: str,
    ) -> QuestionRecord:
        artifact = self.artifacts.put_bytes(
            problem, project_id=project_id, question_id=None, activity_id=None,
            kind=ArtifactKind.PROBLEM, logical_name=logical_name, media_type=media_type,
        )
        return self.store.create_question(project_id, key, title, artifact)

    def add_input(
        self, question_id: UUID, data: bytes, *, logical_name: str,
        media_type: str, kind: ArtifactKind = ArtifactKind.INPUT,
    ) -> ArtifactRecord:
        question = self.store.get_question(question_id)
        artifact = self.artifacts.put_bytes(
            data, project_id=question.project_id, question_id=question.id,
            activity_id=None, kind=kind, logical_name=logical_name, media_type=media_type,
        )
        return self.store.register_artifact(artifact)

    def run_step(self, question_id: UUID, worker_id: str) -> WorkflowStepResult:
        question = self.store.acquire_question_lease(question_id, worker_id, self.settings.lease_seconds)
        try:
            if question.state in (QuestionState.COMPLETED, QuestionState.FAILED):
                return WorkflowStepResult(question=question, terminal=True)
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
                    failed = self.store.transition(
                        question.id, worker_id, [question.state], QuestionState.FAILED,
                        failure_reason=str(exc),
                    )
                    return WorkflowStepResult(question=failed, activity=activity, terminal=True)
                request = request.model_copy(update={"attempt": activity.attempt_count})
                try:
                    with _LeaseHeartbeat(self.store, question.id, activity.id, worker_id, self.settings.lease_seconds):
                        response = self.executor.execute(request)
                    published = self._materialize_response(question, activity.id, response)
                    activity = self.store.complete_activity(
                        activity.id, worker_id, response.model_dump(mode="json"), published)
                except Exception as exc:
                    try:
                        self.store.fail_activity(activity.id, worker_id, str(exc))
                    except Exception:
                        pass
                    raise ActivityExecutionError(f"{stage.value} activity failed: {exc}") from exc
            question = self._complete_stage(question, worker_id, response)
            return WorkflowStepResult(
                question=question, activity=activity, published_artifacts=published,
                terminal=question.state in (QuestionState.COMPLETED, QuestionState.FAILED),
            )
        finally:
            self.store.release_question_lease(question_id, worker_id)

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
            StageKind.MODELING: ["Produce an auditable modeling report and explicit validation contract."],
            StageKind.CODING: ["Implement and execute the model; report evidence, metrics, and generated artifacts."],
            StageKind.REVIEW: ["Independently review assumptions, execution evidence, validations, and claims."],
            StageKind.FINALIZE: ["Produce the final self-contained single-question report using only reviewed evidence."],
        }[stage]
        if question.revision:
            instructions.append(f"This is controlled revision {question.revision}; resolve the latest review instructions.")
        return ActivityRequest(
            activity_id=activity_id, idempotency_key=key, project=project,
            question=question, stage=stage, attempt=attempt, revision=question.revision,
            inputs=inputs, instructions=instructions,
            deadline=utc_now() + timedelta(seconds=self.settings.activity_timeout_seconds),
        )

    def _inputs_for(self, question: QuestionRecord, stage: StageKind) -> list[ArtifactRecord]:
        artifacts = self.store.list_artifacts(question.id)
        allowed = {
            StageKind.MODELING: {ArtifactKind.PROBLEM, ArtifactKind.INPUT, ArtifactKind.SOURCE},
            StageKind.CODING: {ArtifactKind.PROBLEM, ArtifactKind.INPUT, ArtifactKind.SOURCE, ArtifactKind.MODELING_REPORT, ArtifactKind.CODING_REPORT, ArtifactKind.REVIEW_REPORT, ArtifactKind.DATA, ArtifactKind.OUTPUT},
            StageKind.REVIEW: set(ArtifactKind) - {ArtifactKind.FINAL_QUESTION_REPORT},
            StageKind.FINALIZE: set(ArtifactKind) - {ArtifactKind.FINAL_QUESTION_REPORT},
        }[stage]
        return [artifact for artifact in artifacts if artifact.kind in allowed]

    def _materialize_response(self, question: QuestionRecord, activity_id: UUID, response: ActivityResponse) -> list[ArtifactRecord]:
        output = response.output
        report = ProducedArtifact(
            logical_name=f"{output.stage.value}-r{question.revision}-report.md",
            kind=REPORT_KINDS[output.stage], media_type="text/markdown",
            text=output.report.markdown,
            metadata={"title": output.report.title, "summary": output.report.summary, "revision": question.revision},
        )
        records = [self.artifacts.put_produced(report, project_id=question.project_id, question_id=question.id, activity_id=activity_id)]
        for produced in output.artifacts:
            records.append(self.artifacts.put_produced(produced, project_id=question.project_id, question_id=question.id, activity_id=activity_id))
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
        code_output = ActivityResponse.model_validate(coding.result_json, strict=False).output
        if not isinstance(model_output, ModelingStageOutput) or not isinstance(code_output, CodingStageOutput):
            return "review approval references incompatible evidence stages"
        if not code_output.execution_succeeded:
            return "review approved an unsuccessful execution"
        failed = [name for name in model_output.required_validations if code_output.validations.get(name) is not True]
        if failed:
            return "review approved missing or failed validations: " + ", ".join(failed)
        return None
