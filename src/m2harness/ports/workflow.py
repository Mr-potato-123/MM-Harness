"""Workflow definition and repository ports."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from m2harness.domain.events import DomainEvent
from m2harness.domain.workflow import ActivitySpec, WorkflowDecision, WorkflowState


class WorkflowDefinition(Protocol):
    name: str
    version: str

    def initial(self, workflow_id: UUID, question_id: UUID, facts: dict) -> WorkflowState: ...
    def reduce(self, state: WorkflowState, event: DomainEvent) -> WorkflowState: ...
    def plan(self, state: WorkflowState) -> tuple[ActivitySpec, ...]: ...
    def decide(self, state: WorkflowState, event: DomainEvent) -> WorkflowDecision: ...


class WorkflowRepository(Protocol):
    def create(self, state: WorkflowState) -> WorkflowState: ...
    def get(self, workflow_id: UUID) -> WorkflowState: ...
    def save(self, state: WorkflowState, expected_version: int) -> WorkflowState: ...
