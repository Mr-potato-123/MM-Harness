"""Application-level planner/reducer engine independent of provider frameworks."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID, uuid4

from m2harness.domain.events import DomainEvent, EventType
from m2harness.domain.workflow import ActivitySpec, WorkflowDecision, WorkflowState
from m2harness.application.capabilities import CapabilityRegistry
from m2harness.errors import ConflictError, NotFoundError
from m2harness.ports.workflow import WorkflowDefinition, WorkflowRepository
from m2harness.ports.events import EventStorePort


class WorkflowRepositoryMemory(WorkflowRepository):
    def __init__(self) -> None:
        self._states: dict[UUID, WorkflowState] = {}

    def create(self, state: WorkflowState) -> WorkflowState:
        if state.workflow_id in self._states:
            raise ConflictError(f"workflow already exists: {state.workflow_id}")
        self._states[state.workflow_id] = state
        return state

    def get(self, workflow_id: UUID) -> WorkflowState:
        try:
            return self._states[workflow_id]
        except KeyError as exc:
            raise NotFoundError(f"workflow not found: {workflow_id}") from exc

    def save(self, state: WorkflowState, expected_version: int) -> WorkflowState:
        current = self.get(state.workflow_id)
        if current.version != expected_version:
            raise ConflictError(f"workflow version conflict: {expected_version} != {current.version}")
        if state.version != expected_version + 1:
            raise ConflictError(f"workflow state must advance exactly one version: {state.version} != {expected_version + 1}")
        self._states[state.workflow_id] = state
        return state


class WorkflowEngine:
    def __init__(self, definition: WorkflowDefinition, repository: WorkflowRepository, capabilities: CapabilityRegistry | None = None, events: EventStorePort | None = None) -> None:
        self.definition = definition
        self.repository = repository
        self.capabilities = capabilities
        self.events = events

    def _plan(self, state: WorkflowState) -> tuple[ActivitySpec, ...]:
        activities = self.definition.plan(state)
        if self.capabilities is not None:
            for activity in activities:
                resolution = self.capabilities.resolve(activity.required_capabilities)
                if not resolution.complete:
                    names = ", ".join(item.capability.name for item in resolution.missing)
                    raise ConflictError(f"activity {activity.activity_type} has unresolved capabilities: {names}")
        return activities

    def start(self, question_id: UUID, facts: dict) -> tuple[WorkflowState, tuple[ActivitySpec, ...]]:
        state = self.definition.initial(uuid4(), question_id, facts)
        self.repository.create(state)
        if self.events is not None:
            self.events.append(state.workflow_id, 0, DomainEvent(event_type=EventType.WORKFLOW_STARTED, payload={"question_id": str(question_id), "definition": state.definition, "definition_version": state.definition_version, "facts": facts}))
        return state, self._plan(state)

    def get(self, workflow_id: UUID) -> WorkflowState:
        return self.repository.get(workflow_id)

    def apply_activity_result(self, workflow_id: UUID, *, stage: str, payload: dict, actor_id: str = "worker") -> WorkflowDecision:
        current = self.repository.get(workflow_id)
        if current.terminal:
            return WorkflowDecision(next_state=current, activities=())
        if current.stage.value != stage:
            raise ConflictError(f"activity stage {stage} does not match workflow stage {current.stage.value}")
        event = DomainEvent(event_type=EventType.ACTIVITY_COMPLETED, payload={"stage": stage, **payload}, actor_type="worker", actor_id=actor_id)
        next_state = self.definition.reduce(current, event)
        saved = self.repository.save(next_state, expected_version=current.version)
        if self.events is not None:
            self.events.append(workflow_id, current.version, event)
        return WorkflowDecision(next_state=saved, activities=self._plan(saved), events=(event,))

    def pause(self, workflow_id: UUID, actor_id: str = "operator") -> WorkflowState:
        return self._transition(workflow_id, DomainEvent(event_type=EventType.WORKFLOW_PAUSED, payload={}, actor_type="operator", actor_id=actor_id))

    def resume(self, workflow_id: UUID, actor_id: str = "operator") -> WorkflowState:
        return self._transition(workflow_id, DomainEvent(event_type=EventType.WORKFLOW_RESUMED, payload={}, actor_type="operator", actor_id=actor_id))

    def cancel(self, workflow_id: UUID, actor_id: str = "operator") -> WorkflowState:
        return self._transition(workflow_id, DomainEvent(event_type=EventType.WORKFLOW_CANCELLED, payload={}, actor_type="operator", actor_id=actor_id))

    def _transition(self, workflow_id: UUID, event: DomainEvent) -> WorkflowState:
        current = self.repository.get(workflow_id)
        if current.terminal:
            return current
        saved = self.repository.save(self.definition.reduce(current, event), expected_version=current.version)
        if self.events is not None:
            self.events.append(workflow_id, current.version, event)
        return saved
