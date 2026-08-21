"""Framework-independent M2Harness domain contracts."""

from .capability import CapabilityRef, CapabilityResolution, CapabilityRequirement
from .events import DomainEvent, EventEnvelope, EventType
from .skill import SkillDefinition, SkillManifest, SkillSummary
from .tool import ToolCall, ToolDefinition, ToolResult
from .workflow import ActivitySpec, WorkflowDecision, WorkflowState
from .agent import AgentSession, ContextItem, ContextSnapshot, ModelCapabilities, ModelRequest, ModelResponse
from .media import MediaObservation, Modality, MultimodalInput
from .dag import DAGTaskKind, DAGTaskNode, DAGTaskTable, canonical_main_harness_dag, canonical_single_question_dag
from .solve_problem import (
    CodingHarnessReport,
    ExplorationMode,
    ModelReviewDecision,
    PreliminaryModelingReport,
    SolveProblemContext,
    SolveProblemIteration,
    SolveProblemReport,
    SolveProblemStatus,
    SolveProblemReview,
    SolveProblemTask,
    UnifiedModelingReport,
)
from .knowledge import KnowledgeEntry, KnowledgeHit, KnowledgeQuery, KnowledgeSearchResult, KnowledgeSourceKind
from .research import ResearchFinding, ResearchReport, ResearchSource, ResearchSourceKind
from .code import CodeProposal

__all__ = [
    "ActivitySpec",
    "CapabilityRef",
    "CapabilityRequirement",
    "CapabilityResolution",
    "DomainEvent",
    "EventEnvelope",
    "EventType",
    "SkillDefinition",
    "SkillManifest",
    "SkillSummary",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "WorkflowDecision",
    "WorkflowState",
    "AgentSession",
    "ContextItem",
    "ContextSnapshot",
    "ModelCapabilities",
    "ModelRequest",
    "ModelResponse",
    "MediaObservation",
    "Modality",
    "MultimodalInput",
    "DAGTaskKind",
    "DAGTaskNode",
    "DAGTaskTable",
    "canonical_single_question_dag",
    "canonical_main_harness_dag",
    "CodingHarnessReport",
    "ExplorationMode",
    "ModelReviewDecision",
    "PreliminaryModelingReport",
    "SolveProblemContext",
    "SolveProblemIteration",
    "SolveProblemReport",
    "SolveProblemStatus",
    "SolveProblemReview",
    "SolveProblemTask",
    "UnifiedModelingReport",
    "KnowledgeEntry",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeSearchResult",
    "KnowledgeSourceKind",
    "ResearchFinding",
    "ResearchReport",
    "ResearchSource",
    "ResearchSourceKind",
    "CodeProposal",
]
