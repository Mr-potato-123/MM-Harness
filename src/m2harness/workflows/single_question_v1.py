"""Own workflow definition for a production single-question report."""

from __future__ import annotations

from uuid import UUID

from m2harness.domain.capability import CapabilityRef, CapabilityRequirement
from m2harness.domain.events import DomainEvent, EventType
from m2harness.domain.workflow import ActivitySpec, WorkflowDecision, WorkflowStage, WorkflowState


class SingleQuestionWorkflowV1:
    name = "single-question-report"
    version = "1.0.0"

    def initial(self, workflow_id: UUID, question_id: UUID, facts: dict) -> WorkflowState:
        return WorkflowState(
            workflow_id=workflow_id, question_id=question_id, definition=self.name,
            definition_version=self.version, stage=WorkflowStage.INGEST, facts=facts,
        )

    def reduce(self, state: WorkflowState, event: DomainEvent) -> WorkflowState:
        if event.event_type == EventType.WORKFLOW_PAUSED:
            return state.model_copy(update={"paused": True, "version": state.version + 1})
        if event.event_type == EventType.WORKFLOW_RESUMED:
            return state.model_copy(update={"paused": False, "version": state.version + 1})
        if event.event_type == EventType.WORKFLOW_CANCELLED:
            return state.model_copy(update={"cancelled": True, "terminal": True, "terminal_reason": "cancelled", "version": state.version + 1})
        if event.event_type != EventType.ACTIVITY_COMPLETED:
            return state
        stage = event.payload.get("stage")
        if stage != state.stage.value:
            return state
        if state.stage == WorkflowStage.INGEST:
            next_stage = WorkflowStage.MODELING
        elif state.stage == WorkflowStage.MODELING:
            next_stage = WorkflowStage.CODING
        elif state.stage == WorkflowStage.CODING:
            next_stage = WorkflowStage.VALIDATION
        elif state.stage == WorkflowStage.VALIDATION:
            next_stage = WorkflowStage.REVIEW
        elif state.stage == WorkflowStage.REVIEW:
            if event.payload.get("verdict") == "revision_required":
                return state.model_copy(update={"stage": WorkflowStage.REVISION, "revision": state.revision + 1, "version": state.version + 1})
            next_stage = WorkflowStage.FINALIZE
        elif state.stage == WorkflowStage.REVISION:
            next_stage = WorkflowStage.CODING
        else:
            return state.model_copy(update={"terminal": True, "terminal_reason": "final report published", "version": state.version + 1})
        return state.model_copy(update={"stage": next_stage, "version": state.version + 1})

    def plan(self, state: WorkflowState) -> tuple[ActivitySpec, ...]:
        if state.terminal or state.paused or state.cancelled:
            return ()
        requirements = {
            WorkflowStage.INGEST: ("document.read", "image.inspect", "workflow.plan"),
            WorkflowStage.MODELING: ("modeling.formulate",),
            WorkflowStage.CODING: ("python.execute", "artifact.write"),
            WorkflowStage.VALIDATION: ("validation.numerical", "python.execute"),
            WorkflowStage.REVIEW: ("report.review",),
            WorkflowStage.REVISION: ("modeling.formulate", "python.execute"),
            WorkflowStage.FINALIZE: ("report.finalize", "report.publish"),
        }[state.stage]
        activity_type = {
            WorkflowStage.INGEST: "media.inventory",
            WorkflowStage.MODELING: "modeling.propose",
            WorkflowStage.CODING: "coding.execute",
            WorkflowStage.VALIDATION: "validation.run",
            WorkflowStage.REVIEW: "review.independent",
            WorkflowStage.REVISION: "revision.execute",
            WorkflowStage.FINALIZE: "report.publish_latex",
        }[state.stage]
        output_contract = (
            "m2harness/final-question-report+final-latex-paper/v1"
            if state.stage == WorkflowStage.FINALIZE
            else f"m2harness/{state.stage.value}-output/v1"
        )
        return (ActivitySpec(
            activity_type=activity_type, stage=state.stage,
            required_capabilities=tuple(CapabilityRequirement(capability=CapabilityRef(name=name), reason=f"required by {activity_type}") for name in requirements),
            output_contract=output_contract,
            timeout_seconds=1800, max_attempts=3,
            metadata={"revision": state.revision, "workflow_definition": self.version, "dag_terminal_task": "publish-paper" if state.stage == WorkflowStage.FINALIZE else None},
        ),)

    def decide(self, state: WorkflowState, event: DomainEvent) -> WorkflowDecision:
        next_state = self.reduce(state, event)
        return WorkflowDecision(next_state=next_state, activities=self.plan(next_state), events=(event,))
