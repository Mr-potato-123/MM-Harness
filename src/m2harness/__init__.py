"""M2Harness public API."""

from m2harness.artifacts import ArtifactStore
from m2harness.executor import ActivityExecutor, CommandActivityExecutor, SandboxedActivityExecutor
from m2harness.models import HarnessSettings, QuestionState, StageKind
from m2harness.store import HarnessStore
from m2harness.workflow import SingleQuestionWorkflow
from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.skills import FilesystemSkillProvider, SkillRegistry
from m2harness.application.tools import ToolRegistry, ToolRuntime
from m2harness.application.workflow_engine import WorkflowEngine, WorkflowRepositoryMemory
from m2harness.infrastructure.db.sqlite_events import SQLiteEventStore
from m2harness.infrastructure.db.tool_runs import SQLiteToolAuditStore, SQLiteToolExecutionStore
from m2harness.infrastructure.db.sqlite_workflows import SQLiteWorkflowRepository
from m2harness.infrastructure.db.sqlite_artifacts import SQLiteArtifactRegistry
from m2harness.infrastructure.db.main_harness import SQLiteMainHarnessRepository
from m2harness.workflows.single_question_v1 import SingleQuestionWorkflowV1
from m2harness.application.runtime_bundle import RuntimeBundle, build_local_runtime, default_skill_root
from m2harness.application.main_harness import MainHarness, MainHarnessDecision, MainHarnessRepository, MainHarnessState, MainHarnessTask, MainTaskStatus, PaperComposerPort
from m2harness.application.solve_problem import CommandSolveProblemRunner, SolveProblemService, UnconfiguredSolveProblemService
from m2harness.application.knowledge import EmptyKnowledgeBase, HMMLKnowledgeBase, KnowledgeBasePort
from m2harness.application.research import DeepResearchService, ResearchAgentPort
from m2harness.application.compact import ContextCompactor, ContextCompactorPort, compact_text, estimate_tokens
from m2harness.application.report_store import ReportFileRecord, RunReportStore
from m2harness.domain.dag import DAGTaskKind, DAGTaskNode, DAGTaskTable, canonical_main_harness_dag, canonical_single_question_dag
from m2harness.domain.solve_problem import (
    CodingHarnessReport,
    ExplorationMode,
    ModelReviewDecision,
    RevisionTarget,
    ReadOnlyFileReference,
    ReadOnlyFileRole,
    DisclosedTextFile,
    DependencySolutionContext,
    PreliminaryModelingReport,
    SolveProblemContext,
    SolveProblemIteration,
    SolveProblemReport,
    SolveProblemReview,
    SolveProblemStatus,
    SolveProblemTask,
    UnifiedModelingReport,
)
from m2harness.domain.knowledge import KnowledgeEntry, KnowledgeHit, KnowledgeQuery, KnowledgeSearchResult, KnowledgeSourceKind
from m2harness.domain.research import ResearchFinding, ResearchReport, ResearchSource, ResearchSourceKind
from m2harness.domain.code import CodeProposal
from m2harness.domain.media import MultimodalInput
from m2harness.infrastructure.code_harness import CodeProposalProvider, LocalPythonCodeHarness

__all__ = [
    "ActivityExecutor",
    "ArtifactStore",
    "CommandActivityExecutor",
    "SandboxedActivityExecutor",
    "HarnessSettings",
    "HarnessStore",
    "QuestionState",
    "SingleQuestionWorkflow",
    "StageKind",
    "CapabilityRegistry",
    "FilesystemSkillProvider",
    "SkillRegistry",
    "SQLiteEventStore",
    "SQLiteToolExecutionStore",
    "SQLiteToolAuditStore",
    "SQLiteWorkflowRepository",
    "SQLiteArtifactRegistry",
    "SQLiteMainHarnessRepository",
    "SingleQuestionWorkflowV1",
    "ToolRegistry",
    "ToolRuntime",
    "WorkflowEngine",
    "WorkflowRepositoryMemory",
    "RuntimeBundle",
    "build_local_runtime",
    "default_skill_root",
    "DAGTaskKind",
    "DAGTaskNode",
    "DAGTaskTable",
    "canonical_single_question_dag",
    "canonical_main_harness_dag",
    "MainHarness",
    "MainHarnessDecision",
    "MainHarnessRepository",
    "MainHarnessState",
    "MainHarnessTask",
    "MainTaskStatus",
    "PaperComposerPort",
    "SolveProblemService",
    "CommandSolveProblemRunner",
    "UnconfiguredSolveProblemService",
    "CodingHarnessReport",
    "ExplorationMode",
    "ModelReviewDecision",
    "RevisionTarget",
    "ReadOnlyFileReference",
    "ReadOnlyFileRole",
    "DisclosedTextFile",
    "DependencySolutionContext",
    "PreliminaryModelingReport",
    "SolveProblemContext",
    "SolveProblemIteration",
    "SolveProblemReport",
    "SolveProblemReview",
    "SolveProblemStatus",
    "SolveProblemTask",
    "UnifiedModelingReport",
    "EmptyKnowledgeBase",
    "HMMLKnowledgeBase",
    "KnowledgeBasePort",
    "DeepResearchService",
    "ResearchAgentPort",
    "ContextCompactor",
    "ContextCompactorPort",
    "compact_text",
    "estimate_tokens",
    "ReportFileRecord",
    "RunReportStore",
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
    "MultimodalInput",
    "CodeProposalProvider",
    "LocalPythonCodeHarness",
]
