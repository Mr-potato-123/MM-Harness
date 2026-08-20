from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from m2harness.artifacts import ArtifactStore
from m2harness.errors import ActivityExecutionError, ArtifactIntegrityError, ConflictError
from m2harness.models import (
    ActivityRequest,
    ActivityResponse,
    ArtifactKind,
    CodingStageOutput,
    FinalizeStageOutput,
    HarnessSettings,
    ModelingStageOutput,
    QuestionState,
    ReportPayload,
    ReviewStageOutput,
    ReviewVerdict,
    StageKind,
    new_uuid,
)
from m2harness.workflow import SingleQuestionWorkflow


def report(title: str) -> ReportPayload:
    return ReportPayload(title=title, markdown=f"# {title}\n\nEvidence.", summary="Evidence summary")


class ScriptedExecutor:
    def __init__(self, revise_once: bool = False, fail_once: bool = False, validation_passes: bool = True) -> None:
        self.revise_once = revise_once
        self.fail_once = fail_once
        self.validation_passes = validation_passes
        self.calls: list[ActivityRequest] = []

    def execute(self, request: ActivityRequest) -> ActivityResponse:
        self.calls.append(request)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("transient agent outage")
        if request.stage == StageKind.MODELING:
            output = ModelingStageOutput(stage=StageKind.MODELING, report=report("Model"), required_validations=["residual"], expected_outputs=["estimate"])
        elif request.stage == StageKind.CODING:
            output = CodingStageOutput(stage=StageKind.CODING, report=report("Execution"), execution_succeeded=True, validations={"residual": self.validation_passes}, metrics={"rmse": 0.1})
        elif request.stage == StageKind.REVIEW:
            verdict = ReviewVerdict.REVISION_REQUIRED if self.revise_once and request.revision == 0 else ReviewVerdict.APPROVE
            output = ReviewStageOutput(
                stage=StageKind.REVIEW, report=report("Review"), verdict=verdict,
                revision_instructions=["tighten tolerance"] if verdict == ReviewVerdict.REVISION_REQUIRED else [],
                accepted_claims=["estimate is supported"] if verdict == ReviewVerdict.APPROVE else [],
            )
        else:
            output = FinalizeStageOutput(stage=StageKind.FINALIZE, report=report("Final Question Report"), downstream_outputs={"status": "reviewed"})
        return ActivityResponse(idempotency_key=request.idempotency_key, output=output)


class HarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = HarnessSettings(database_path=root / "state.db", artifact_root=root / "artifacts", lease_seconds=10, activity_timeout_seconds=30)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def workflow(self, executor: ScriptedExecutor) -> tuple[SingleQuestionWorkflow, object]:
        workflow = SingleQuestionWorkflow(self.settings, executor)
        project = workflow.create_project("production-validation")
        question = workflow.add_question(project.id, "q1", "Question 1", "Estimate the parameter with evidence.")
        return workflow, question

    def test_happy_path_produces_final_report_and_valid_event_chain(self) -> None:
        executor = ScriptedExecutor()
        workflow, question = self.workflow(executor)
        result = workflow.run_until_terminal(question.id, "worker-a")
        self.assertEqual(result.question.state, QuestionState.COMPLETED)
        self.assertEqual([call.stage for call in executor.calls], [StageKind.MODELING, StageKind.CODING, StageKind.REVIEW, StageKind.FINALIZE])
        final = [item for item in workflow.store.list_artifacts(question.id) if item.kind == ArtifactKind.FINAL_QUESTION_REPORT]
        self.assertEqual(len(final), 1)
        self.assertIn(b"Final Question Report", workflow.artifacts.read(final[0]))
        self.assertGreater(workflow.store.verify_event_chain(), 10)

    def test_review_drives_controlled_revision(self) -> None:
        executor = ScriptedExecutor(revise_once=True)
        workflow, question = self.workflow(executor)
        result = workflow.run_until_terminal(question.id, "worker-a")
        self.assertEqual(result.question.state, QuestionState.COMPLETED)
        self.assertEqual(result.question.revision, 1)
        self.assertEqual([call.stage for call in executor.calls], [StageKind.MODELING, StageKind.CODING, StageKind.REVIEW, StageKind.CODING, StageKind.REVIEW, StageKind.FINALIZE])

    def test_review_cannot_approve_failed_validation(self) -> None:
        workflow, question = self.workflow(ScriptedExecutor(validation_passes=False))
        result = workflow.run_until_terminal(question.id, "worker-a")
        self.assertEqual(result.question.state, QuestionState.FAILED)
        self.assertIn("failed validations", result.question.failure_reason or "")
        self.assertNotIn(StageKind.FINALIZE, [item.stage for item in workflow.store.list_activities(question.id)])

    def test_failed_activity_retries_same_idempotency_key(self) -> None:
        executor = ScriptedExecutor(fail_once=True)
        workflow, question = self.workflow(executor)
        with self.assertRaises(ActivityExecutionError):
            workflow.run_step(question.id, "worker-a")
        workflow.run_step(question.id, "worker-a")
        self.assertEqual(executor.calls[0].idempotency_key, executor.calls[1].idempotency_key)
        activity = workflow.store.list_activities(question.id)[0]
        self.assertEqual(activity.attempt_count, 2)

    def test_committed_activity_is_not_reexecuted_after_transition_crash(self) -> None:
        executor = ScriptedExecutor()
        workflow, question = self.workflow(executor)
        original = workflow._complete_stage
        crashed = False

        def crash_once(*args, **kwargs):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("process died after activity commit")
            return original(*args, **kwargs)

        workflow._complete_stage = crash_once  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            workflow.run_step(question.id, "worker-a")
        workflow.run_step(question.id, "worker-a")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(workflow.store.get_question(question.id).state, QuestionState.READY_FOR_CODING)

    def test_attempt_budget_exhaustion_is_terminal(self) -> None:
        executor = ScriptedExecutor(fail_once=True)
        settings = self.settings.model_copy(update={"max_activity_attempts": 1})
        workflow = SingleQuestionWorkflow(settings, executor)
        project = workflow.create_project("attempt-budget")
        question = workflow.add_question(project.id, "q2", "Question 2", "Solve with evidence.")
        with self.assertRaises(ActivityExecutionError):
            workflow.run_step(question.id, "worker-a")
        result = workflow.run_step(question.id, "worker-a")
        self.assertTrue(result.terminal)
        self.assertEqual(result.question.state, QuestionState.FAILED)

    def test_question_lease_excludes_second_worker(self) -> None:
        workflow, question = self.workflow(ScriptedExecutor())
        workflow.store.acquire_question_lease(question.id, "worker-a", 10)
        with self.assertRaises(ConflictError):
            workflow.store.acquire_question_lease(question.id, "worker-b", 10)

    def test_artifact_tampering_is_detected(self) -> None:
        store = ArtifactStore(Path(self.temp.name) / "isolated-artifacts")
        record = store.put_bytes(
            b"original", project_id=new_uuid(), question_id=None, activity_id=None,
            kind=ArtifactKind.OTHER, logical_name="evidence.bin", media_type="application/octet-stream",
        )
        path = store.root / record.relative_path
        path.write_bytes(b"tampered")
        with self.assertRaises(ArtifactIntegrityError):
            store.read(record)


if __name__ == "__main__":
    unittest.main()
