"""Explicit local runtime composition; production replaces infrastructure adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.skills import FilesystemSkillProvider, SkillRegistry
from m2harness.application.tools import ResultBudgetMiddleware, ResultRedactionMiddleware, ToolRegistry, ToolRuntime
from m2harness.application.local_tools import LocalToolEnvironment, register_local_tools
from m2harness.artifacts import ArtifactStore
from m2harness.application.workflow_engine import WorkflowEngine
from m2harness.application.main_harness import MainHarness
from m2harness.application.report_store import RunReportStore
from m2harness.application.solve_problem import SolveProblemService, UnconfiguredSolveProblemService
from m2harness.application.knowledge import HMMLKnowledgeBase, EmptyKnowledgeBase, default_hmml_path, KnowledgeBasePort
from m2harness.application.research import AuthorizedWebResearchPort, DeepResearchService
from m2harness.domain.capability import CapabilityRef
from m2harness.workflows.single_question_v1 import SingleQuestionWorkflowV1
from m2harness.application.media import MediaInventory
from m2harness.application.policy import NetworkPolicy
from m2harness.infrastructure.db.tool_runs import SQLiteToolAuditStore, SQLiteToolExecutionStore
from m2harness.infrastructure.db.sqlite_events import SQLiteEventStore
from m2harness.infrastructure.db.sqlite_workflows import SQLiteWorkflowRepository
from m2harness.infrastructure.db.sqlite_artifacts import SQLiteArtifactRegistry
from m2harness.infrastructure.db.main_harness import SQLiteMainHarnessRepository
from m2harness.infrastructure.local_sandbox import DockerSandboxClient, LocalSandboxClient


@dataclass(frozen=True)
class RuntimeBundle:
    capabilities: CapabilityRegistry
    skills: SkillRegistry
    tools: ToolRegistry
    workflows: WorkflowEngine
    media: MediaInventory
    network: NetworkPolicy
    tool_environment: LocalToolEnvironment
    tool_runtime: ToolRuntime
    tool_execution_store: SQLiteToolExecutionStore
    tool_audit_store: SQLiteToolAuditStore
    events: SQLiteEventStore
    sandbox: LocalSandboxClient
    artifact_registry: SQLiteArtifactRegistry
    main_harness: MainHarness
    solve_problem_service: SolveProblemService | UnconfiguredSolveProblemService
    main_harness_repository: SQLiteMainHarnessRepository
    knowledge_base: KnowledgeBasePort
    research_service: DeepResearchService

    def attach_solve_problem_service(self, service: SolveProblemService) -> "RuntimeBundle":
        """Attach a real Model/Code service after sandbox/runtime creation.

        A Qwen Code Harness needs the already-composed local sandbox.  This
        explicit second step keeps provider construction out of the local
        runtime and avoids hidden global configuration.
        """
        self.tool_environment.solve_problem_service = service
        return replace(self, solve_problem_service=service)


def default_skill_root() -> Path:
    """Locate bundled Skills in source, editable, and wheel installations."""
    source_root = Path(__file__).parents[3] / "skills"
    # Compatibility lookup for the historical checkout layout mentioned in
    # the architecture notes.  The current repository-local bundle always
    # wins; the legacy root is used only when an operator actually provides it.
    legacy_root = Path(__file__).parents[3].parent / "MM" / "_Harness" / "skills"
    candidates = (
        source_root,
        legacy_root,
        Path(sys.prefix) / "share" / "m2harness" / "skills",
        Path(sys.base_prefix) / "share" / "m2harness" / "skills",
        Path(__file__).resolve().parents[2] / "share" / "m2harness" / "skills",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return source_root.resolve()


def build_local_runtime(*, skill_root: Path | None = None, workspace_root: Path | None = None, artifact_root: Path | None = None, database_path: Path | None = None, search_endpoint: str | None = None, allow_host_sandbox: bool = True, sandbox_backend: str | None = None, register_workflow_capabilities: bool = True, model_hosts: set[str] | frozenset[str] = frozenset(), search_hosts: set[str] | frozenset[str] = frozenset(), solve_problem_service: SolveProblemService | None = None, hmml_path: Path | None = None, allow_web_research: bool = False, web_research: AuthorizedWebResearchPort | None = None) -> RuntimeBundle:
    capabilities = CapabilityRegistry()
    if register_workflow_capabilities:
        for name in (
            "document.read", "image.inspect", "modeling.formulate", "python.execute",
            "artifact.read", "artifact.write", "workspace.read", "workspace.write",
            "data.profile", "validation.numerical", "report.review", "report.finalize",
            "report.publish",
            "problem.solve",
            "knowledge.search",
            "workflow.plan",
            "web.search",
        ):
            capabilities.register(CapabilityRef(name=name), provider_id="local-unconfigured")
    skills = SkillRegistry()
    root = (skill_root or default_skill_root()).resolve()
    skills.register_provider(FilesystemSkillProvider([root], provider_name="bundled-local"))
    network = NetworkPolicy(model_hosts=model_hosts, search_hosts=search_hosts)
    workspace = (workspace_root or (Path.cwd() / ".m2harness" / "workspace")).resolve()
    artifacts = ArtifactStore((artifact_root or (Path.cwd() / ".m2harness" / "artifacts")).resolve())
    runtime_database = (database_path or (Path.cwd() / ".m2harness" / "runtime.db")).resolve()
    artifact_registry = SQLiteArtifactRegistry(runtime_database)
    resolved_hmml = hmml_path.resolve() if hmml_path is not None else default_hmml_path(Path.cwd())
    knowledge_base: KnowledgeBasePort = HMMLKnowledgeBase(resolved_hmml) if resolved_hmml is not None else EmptyKnowledgeBase()
    research_service = DeepResearchService(knowledge_base, web_search=web_research, allow_web=allow_web_research)
    # The first delivery is explicitly a trusted, single-user local Harness:
    # host execution is the default and Docker/VM is an opt-in deployment
    # backend.  ``allow_host_sandbox=False`` or backend ``none`` restores the
    # fail-closed behavior for operators handling untrusted code.
    requested_backend = (sandbox_backend or os.environ.get("M2HARNESS_SANDBOX_BACKEND", "host")).lower()
    if requested_backend == "docker":
        sandbox = DockerSandboxClient(workspace, image=os.environ.get("M2HARNESS_SANDBOX_IMAGE", "python:3.12-slim"))
    else:
        sandbox = LocalSandboxClient(workspace, allow_host_processes=allow_host_sandbox and requested_backend not in {"none", "disabled", "off"})
    configured_solve_service = solve_problem_service or UnconfiguredSolveProblemService()
    if solve_problem_service is not None and solve_problem_service.research_agent is None:
        configured_solve_service = replace(solve_problem_service, research_agent=research_service)
    tool_environment = LocalToolEnvironment(workspace_root=workspace, artifact_store=artifacts, artifact_registry=artifact_registry, media=MediaInventory(), network=network, search_endpoint=search_endpoint, sandbox=sandbox, solve_problem_service=configured_solve_service, knowledge_base=knowledge_base)
    tools = register_local_tools(ToolRegistry(), tool_environment)
    tool_execution_store = SQLiteToolExecutionStore(runtime_database)
    tool_audit_store = SQLiteToolAuditStore(runtime_database)
    tool_runtime = ToolRuntime(
        tools, max_result_bytes=32 * 1024 * 1024,
        middlewares=(ResultBudgetMiddleware(32 * 1024 * 1024), ResultRedactionMiddleware({"api_key", "access_token", "secret"})),
        execution_store=tool_execution_store, audit_sink=tool_audit_store,
    )
    definition = SingleQuestionWorkflowV1()
    events = SQLiteEventStore(runtime_database)
    engine = WorkflowEngine(definition, SQLiteWorkflowRepository(runtime_database), capabilities, events=events)
    main_harness_repository = SQLiteMainHarnessRepository(runtime_database)
    main_harness = MainHarness(
        tool_runtime, capabilities, repository=main_harness_repository,
        report_store=RunReportStore(workspace),
    )
    return RuntimeBundle(capabilities=capabilities, skills=skills, tools=tools, workflows=engine, media=tool_environment.media or MediaInventory(), network=network, tool_environment=tool_environment, tool_runtime=tool_runtime, tool_execution_store=tool_execution_store, tool_audit_store=tool_audit_store, events=events, sandbox=sandbox, artifact_registry=artifact_registry, main_harness=main_harness, solve_problem_service=configured_solve_service, main_harness_repository=main_harness_repository, knowledge_base=knowledge_base, research_service=research_service)
