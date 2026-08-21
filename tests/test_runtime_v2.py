from __future__ import annotations

import tempfile
import time
import os
import unittest
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.skills import FilesystemSkillProvider, SkillRegistry
from m2harness.application.tools import ResultRedactionMiddleware, ToolRegistry, ToolRuntime
from m2harness.application.workflow_engine import WorkflowEngine, WorkflowRepositoryMemory
from m2harness.application.runtime_bundle import build_local_runtime
from m2harness.application.context import ContextEngine
from m2harness.domain.agent import AgentSession, ContextItem, ModelCapabilities, ModelRequest, ModelResponse, ModelToolCall
from m2harness.runtime.agent_runtime import AgentRuntime
from m2harness.application.media import MediaInventory
from m2harness.application.policy import EgressCapability, NetworkPolicy
from m2harness.artifacts import ArtifactStore
from m2harness.domain.capability import CapabilityRef, CapabilityRequirement
from m2harness.domain.events import DomainEvent, EventType
from m2harness.domain.tool import ToolCall, ToolDefinition, ToolPolicy
from m2harness.domain.dag import DAGTaskTable, canonical_single_question_dag
from m2harness.infrastructure.db.sqlite_events import SQLiteEventStore
from m2harness.infrastructure.db.tool_runs import SQLiteToolExecutionStore
from m2harness.infrastructure.local_sandbox import LocalSandboxClient
from m2harness.ports.sandbox import SandboxSpec
from m2harness.runtime.skill_loader import SkillLoader
from m2harness.workflows.single_question_v1 import SingleQuestionWorkflowV1
from adapters.qwen38_max import _authorize_provider_url, _multimodal_content, _prompt, _skill_context, _system_prompt
from m2harness.models import ActivityRequest, ArtifactKind, ArtifactRecord, ProjectRecord, QuestionRecord, QuestionState, StageKind


class RuntimeV2Test(unittest.TestCase):
    def test_mainline_dag_is_acyclic_and_ends_in_latex_publication(self) -> None:
        table = canonical_single_question_dag()
        self.assertEqual(table.topological_order(), ("ingest", "model", "code", "review", "publish-paper"))
        self.assertEqual(table.tasks[-1].output_contract, "final_question_report+final_latex_paper")
        self.assertEqual(table.terminal_task_id, "publish-paper")

        payload = table.model_dump(mode="json")
        payload["tasks"][1]["depends_on"] = ["publish-paper"]
        with self.assertRaises(ValueError):
            DAGTaskTable.model_validate(payload, strict=False)

    def test_mainline_dag_rejects_non_publication_terminal(self) -> None:
        payload = canonical_single_question_dag().model_dump(mode="json")
        payload["terminal_task_id"] = "review"
        with self.assertRaises(ValueError):
            DAGTaskTable.model_validate(payload, strict=False)

    def test_mainline_dag_rejects_work_after_publication_terminal(self) -> None:
        payload = canonical_single_question_dag().model_dump(mode="json")
        payload["tasks"].append({
            "id": "zzz-after-paper",
            "title": "Illegal post-publication task",
            "kind": "review",
            "depends_on": ["ingest"],
            "output_contract": "illegal",
        })
        with self.assertRaises(ValueError):
            DAGTaskTable.model_validate(payload, strict=False)

    def test_event_store_is_optimistic_outbox_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = SQLiteEventStore(Path(temp) / "events.db")
            aggregate = uuid4()
            event = store.append(aggregate, 0, DomainEvent(event_type=EventType.WORKFLOW_STARTED, payload={"source": "test"}))
            self.assertEqual(event.aggregate_version, 1)
            self.assertEqual(len(store.unpublished()), 1)
            self.assertEqual(store.verify(), 1)
            with self.assertRaises(Exception):
                store.append(aggregate, 0, DomainEvent(event_type=EventType.WORKFLOW_STARTED, payload={}))

    def test_event_store_verifies_interleaved_aggregate_chains(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = SQLiteEventStore(Path(temp) / "events.db")
            first, second = uuid4(), uuid4()
            store.append(first, 0, DomainEvent(event_type=EventType.WORKFLOW_STARTED, payload={"aggregate": "first"}))
            store.append(second, 0, DomainEvent(event_type=EventType.WORKFLOW_STARTED, payload={"aggregate": "second"}))
            store.append(first, 1, DomainEvent(event_type=EventType.ACTIVITY_COMPLETED, payload={"aggregate": "first"}))
            self.assertEqual(store.verify(), 3)

    def test_skill_registry_is_descriptor_first_and_loads_snapshot_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "skills"
            skill_dir = root / "numerical-validation"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: numerical-validation\nversion: 1.0.0\ndescription: Validate numbers\n---\n# Checklist\nUse independent residual checks.",
                encoding="utf-8",
            )
            registry = SkillRegistry()
            registry.register_provider(FilesystemSkillProvider([root]))
            catalog = registry.list()
            self.assertEqual([item.name for item in catalog], ["numerical-validation"])
            self.assertNotIn("Checklist", catalog[0].description)
            snapshot = SkillLoader(registry).load(["numerical-validation"])
            self.assertIn("independent residual", snapshot.loaded[0].content)

    def test_skill_resources_are_snapshotted_and_tamper_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "skills"
            skill_dir = root / "resource-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Skill\nUse the checklist.", encoding="utf-8")
            (skill_dir / "checklist.txt").write_text("check-1", encoding="utf-8")
            (skill_dir / "skill.yaml").write_text("apiVersion: m2harness/v1\nname: resource-skill\nversion: 1.0.0\ndescription: Resource test\nresources: checklist.txt\ncontextTokens: 42\n", encoding="utf-8")
            registry = SkillRegistry()
            registry.register_provider(FilesystemSkillProvider([root]))
            loaded = registry.get("resource-skill")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.manifest.context_tokens, 42)
            self.assertIn("checklist.txt", loaded.resource_digests)
            (skill_dir / "checklist.txt").write_text("tampered", encoding="utf-8")
            self.assertIsNone(registry.get("resource-skill"))

    def test_tool_runtime_enforces_capability_schema_and_idempotency(self) -> None:
        capabilities = CapabilityRegistry()
        capability = CapabilityRef(name="data.profile")
        capabilities.register(capability, "profile-provider")
        resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
        tools = ToolRegistry()
        tools.register(
            ToolDefinition(
                name="data_profile", version="1", description="profile", required_capability=capability,
                input_schema={"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
                output_schema={"type": "object"},
            ),
            lambda arguments: {"name": arguments["name"], "rows": 3},
        )
        runtime = ToolRuntime(tools, middlewares=(ResultRedactionMiddleware({"rows"}),))
        call = ToolCall(
            call_id=uuid4(), tool_name="data_profile", tool_version="1", activity_id=uuid4(), session_id=uuid4(),
            idempotency_key="same-call", arguments={"name": "x"}, requested_at=datetime.now(UTC),
        )
        first = runtime.execute(call, resolution)
        second = runtime.execute(call, resolution)
        self.assertTrue(first.ok)
        self.assertEqual(first, second)
        self.assertEqual(first.output["rows"], "[REDACTED]")
        denied = runtime.execute(call.model_copy(update={"idempotency_key": "denied"}), CapabilityRegistry().resolve([CapabilityRequirement(capability=capability)]))
        self.assertEqual(denied.error_code, "capability_denied")

    def test_tool_idempotency_survives_runtime_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "tool-runs.db"
            capability = CapabilityRef(name="data.profile")
            registry = ToolRegistry()
            calls = []
            registry.register(ToolDefinition(name="data_profile", version="1", description="profile", required_capability=capability, input_schema={"type": "object"}, output_schema={"type": "object"}), lambda args: calls.append(args) or {"rows": 1})
            capabilities = CapabilityRegistry()
            capabilities.register(capability, "profile-provider")
            resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
            call = ToolCall(call_id=uuid4(), tool_name="data_profile", tool_version="1", activity_id=uuid4(), session_id=uuid4(), idempotency_key="durable-key", arguments={}, requested_at=datetime.now(UTC))
            first = ToolRuntime(registry, execution_store=SQLiteToolExecutionStore(database)).execute(call, resolution)
            second = ToolRuntime(registry, execution_store=SQLiteToolExecutionStore(database)).execute(call, resolution)
            self.assertTrue(first.ok)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)

    def test_persistent_tool_lease_fences_reclaimed_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "tool-runs.db"
            store = SQLiteToolExecutionStore(database)
            self.assertTrue(store.reserve("fenced", 3600, "old-token"))
            with store._connection() as connection:
                connection.execute("UPDATE m2_tool_runs SET started_at=? WHERE idempotency_key=?", ("2000-01-01T00:00:00+00:00", "fenced"))
            self.assertTrue(store.reserve("fenced", 3600, "new-token"))
            result = ToolCall(call_id=uuid4(), tool_name="data_profile", tool_version="1", activity_id=uuid4(), session_id=uuid4(), idempotency_key="fenced", arguments={}, requested_at=datetime.now(UTC))
            from m2harness.domain.tool import ToolResult
            completed = ToolResult(call_id=result.call_id, tool_name=result.tool_name, ok=True, output={}, completed_at=datetime.now(UTC))
            with self.assertRaises(RuntimeError):
                store.complete("fenced", completed, "old-token")
            store.complete("fenced", completed, "new-token")
            self.assertEqual(store.lookup("fenced"), completed)

    def test_tool_runtime_enforces_definition_timeout(self) -> None:
        capability = CapabilityRef(name="data.profile")
        capabilities = CapabilityRegistry()
        capabilities.register(capability, "profile-provider")
        resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="slow_tool", version="1", description="slow", required_capability=capability, timeout_seconds=1, input_schema={"type": "object"}, output_schema={"type": "object"}), lambda args: (time.sleep(2), {"ok": True})[1])
        call = ToolCall(call_id=uuid4(), tool_name="slow_tool", tool_version="1", activity_id=uuid4(), session_id=uuid4(), idempotency_key="slow-key", arguments={}, requested_at=datetime.now(UTC))
        result = ToolRuntime(registry).execute(call, resolution)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "tool_execution_error")
        self.assertIn("timeout", result.error_message or "")

    def test_tool_runtime_rejects_unrestricted_network_policy(self) -> None:
        capability = CapabilityRef(name="network.test")
        capabilities = CapabilityRegistry()
        capabilities.register(capability, "test")
        resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="network_test", version="1", description="network", required_capability=capability, input_schema={"type": "object"}, output_schema={"type": "object"}, policy=ToolPolicy(network="unrestricted")), lambda args: {"ok": True})
        call = ToolCall(call_id=uuid4(), tool_name="network_test", tool_version="1", activity_id=uuid4(), session_id=uuid4(), idempotency_key="network-policy", arguments={}, requested_at=datetime.now(UTC))
        result = ToolRuntime(registry).execute(call, resolution)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "policy_denied")

    def test_tool_runtime_validates_nested_schema_constraints(self) -> None:
        capability = CapabilityRef(name="nested.test")
        capabilities = CapabilityRegistry()
        capabilities.register(capability, "test")
        resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="nested_test", version="1", description="nested", required_capability=capability, input_schema={"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string", "minLength": 2}}, "additionalProperties": False}}}, "additionalProperties": False}, output_schema={"type": "object"}), lambda args: {"ok": True})
        definition = registry.get("nested_test")
        assert definition is not None
        call = ToolCall(call_id=uuid4(), tool_name=definition.name, tool_version=definition.version, activity_id=uuid4(), session_id=uuid4(), idempotency_key="nested-schema", arguments={"items": [{"name": "x"}]}, requested_at=datetime.now(UTC))
        result = ToolRuntime(registry).execute(call, resolution)
        self.assertFalse(result.ok)
        self.assertIn("shorter", result.error_message or "")

    def test_local_sandbox_rejects_unrestricted_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sandbox = LocalSandboxClient(Path(temp))
            with self.assertRaises(RuntimeError):
                sandbox.create(SandboxSpec(image_digest="local", network="deny"))
            trusted_host = LocalSandboxClient(Path(temp), allow_host_processes=True)
            with self.assertRaises(PermissionError):
                trusted_host.create(SandboxSpec(image_digest="local", network="unrestricted"))

    def test_workflow_definition_is_versioned_and_revisions_are_targeted(self) -> None:
        definition = SingleQuestionWorkflowV1()
        engine = WorkflowEngine(definition, WorkflowRepositoryMemory())
        state, planned = engine.start(uuid4(), {"problem": "test"})
        self.assertEqual(state.definition_version, "1.0.0")
        self.assertEqual(planned[0].activity_type, "media.inventory")
        for stage, payload in [
            ("ingest", {}), ("modeling", {}), ("coding", {}), ("validation", {}),
            ("review", {"verdict": "revision_required"}), ("revision", {}),
        ]:
            decision = engine.apply_activity_result(state.workflow_id, stage=stage, payload=payload)
            state = decision.next_state
        self.assertEqual(state.stage.value, "coding")
        self.assertEqual(state.revision, 1)

    def test_qwen_projection_sends_pdf_and_image_bytes_as_multimodal_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "artifacts"
            artifacts = ArtifactStore(root)
            project = ProjectRecord(id=uuid4(), name="p", created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1)
            question = QuestionRecord(
                id=uuid4(), project_id=project.id, key="q", title="q", state=QuestionState.MODELING,
                problem_artifact_id=uuid4(), revision=0, created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1,
            )
            pdf = artifacts.put_bytes(b"%PDF-test", project_id=project.id, question_id=question.id, activity_id=None, kind=ArtifactKind.PROBLEM, logical_name="problem.pdf", media_type="application/pdf")
            image = artifacts.put_bytes(b"\x89PNG\r\n", project_id=project.id, question_id=question.id, activity_id=None, kind=ArtifactKind.INPUT, logical_name="figure.png", media_type="image/png")
            request = ActivityRequest(
                activity_id=uuid4(), idempotency_key="projection-test", project=project, question=question,
                stage=StageKind.MODELING, attempt=1, revision=0, inputs=[pdf, image], instructions=["inspect"],
                deadline=datetime.now(UTC),
            )
            content = _multimodal_content(request, root.resolve(), "inspect")
            self.assertEqual(content[0]["type"], "file")
            self.assertTrue(content[0]["file"]["file_data"].startswith("data:application/pdf;base64,"))
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

            duplicate_request = request.model_copy(update={"idempotency_key": "projection-duplicate", "inputs": [pdf, pdf]})
            duplicate_content = _multimodal_content(duplicate_request, root.resolve(), "inspect")
            self.assertEqual(duplicate_content[0]["type"], "file")
            self.assertIn("Duplicate Artifact reference omitted", duplicate_content[1]["text"])

    def test_qwen_projection_sniffs_mismatched_media_and_does_not_embed_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "artifacts"
            artifacts = ArtifactStore(root)
            project = ProjectRecord(id=uuid4(), name="p", created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1)
            question = QuestionRecord(
                id=uuid4(), project_id=project.id, key="q", title="q", state=QuestionState.MODELING,
                problem_artifact_id=uuid4(), revision=0, created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1,
            )
            pdf = artifacts.put_bytes(b"%PDF-test", project_id=project.id, question_id=question.id, activity_id=None, kind=ArtifactKind.PROBLEM, logical_name="problem.bin", media_type="application/octet-stream")
            csv = artifacts.put_bytes(b"x,y\n1,2\n", project_id=project.id, question_id=question.id, activity_id=None, kind=ArtifactKind.DATA, logical_name="data.csv", media_type="text/csv")
            request = ActivityRequest(
                activity_id=uuid4(), idempotency_key="projection-sniff-test", project=project, question=question,
                stage=StageKind.MODELING, attempt=1, revision=0, inputs=[pdf, csv], instructions=["inspect"], deadline=datetime.now(UTC),
            )
            content = _multimodal_content(request, root.resolve(), "inspect")
            self.assertEqual(content[0]["type"], "file")
            self.assertIn("local profiling only", content[1]["text"])
            self.assertNotIn("x,y", content[1]["text"])

    def test_qwen_provider_url_is_fail_closed(self) -> None:
        _authorize_provider_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
        with self.assertRaises(PermissionError):
            _authorize_provider_url("http://dashscope.aliyuncs.com/compatible-mode/v1")
        with self.assertRaises(PermissionError):
            _authorize_provider_url("https://evil.example/compatible-mode/v1")

    def test_qwen_prompt_receives_complete_skill_snapshot_with_stage_focus(self) -> None:
        from unittest.mock import patch
        with patch.dict(os.environ, {"M2HARNESS_SKILL_ROOT": str(Path("skills").resolve())}, clear=False):
            modeling = _skill_context(StageKind.MODELING)
            coding = _skill_context(StageKind.CODING)
        self.assertIn("[modeling-core; FOCUS", modeling)
        self.assertIn("[coding-contract; AVAILABLE", modeling)
        self.assertIn("[coding-contract; FOCUS", coding)
        self.assertIn("[modeling-core; AVAILABLE", coding)
        self.assertIn("model-contract.md", modeling)
        self.assertNotIn("[skill-review;", modeling)
        self.assertIn("sha256=", coding)

    def test_legacy_qwen_roles_share_one_harness_authority_contract(self) -> None:
        for stage in StageKind:
            system = _system_prompt(stage)
            self.assertIn("The Main Harness owns", system)
            self.assertIn("A listed read-only path is a manifest entry", system)
            self.assertIn("Skills improve method selection", system)

    def test_qwen_prompt_carries_harness_owned_dag_and_paper_terminal(self) -> None:
        project = ProjectRecord(id=uuid4(), name="prompt", created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1)
        question = QuestionRecord(id=uuid4(), project_id=project.id, key="q", title="prompt", state=QuestionState.FINALIZING, problem_artifact_id=uuid4(), revision=0, created_at=datetime.now(UTC), updated_at=datetime.now(UTC), version=1)
        request = ActivityRequest(activity_id=uuid4(), idempotency_key="prompt-dag", project=project, question=question, stage=StageKind.FINALIZE, attempt=1, revision=0, inputs=[], instructions=["publish"], deadline=datetime.now(UTC))
        prompt = _prompt(request)
        self.assertIn("ingest -> model -> code -> review -> publish-paper", prompt)
        self.assertIn("final_latex_paper", prompt)

    def test_local_runtime_composition_resolves_workflow_capabilities(self) -> None:
        bundle = build_local_runtime()
        state, activities = bundle.workflows.start(uuid4(), {"problem": "runtime composition"})
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0].activity_type, "media.inventory")
        # The local catalog is a distilled cross-role modeling library rather
        # than a direct mount of third-party Skill packages.
        names = {item.name for item in bundle.skills.list()}
        self.assertEqual(len(names), 43)
        self.assertTrue({
            "modeling-project-orchestration", "statistical-modeling",
            "optimization-modeling", "publication-verification",
            "scientific-figure-design", "systematic-literature-review",
        }.issubset(names))
        with self.assertRaises(PermissionError):
            bundle.network.authorize_model("https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_trusted_host_backend_is_default_and_can_be_disabled(self) -> None:
        self.assertTrue(build_local_runtime().sandbox.available)
        from unittest.mock import patch
        with patch.dict(os.environ, {"M2HARNESS_SANDBOX_BACKEND": "none"}, clear=False):
            self.assertFalse(build_local_runtime().sandbox.available)

    def test_local_tool_catalog_has_required_production_entries(self) -> None:
        bundle = build_local_runtime()
        names = {item.name for item in bundle.tools.catalog()}
        self.assertTrue({"artifact_read", "artifact_write", "artifact_inspect", "pdf_inspect", "image_inspect", "data_profile", "workspace_read", "workspace_write", "python_execute", "validation_run", "report_render", "latex_validate", "dag_task_table", "web_search"}.issubset(names))
        self.assertTrue(bundle.sandbox.available)
        self.assertEqual(bundle.sandbox.image_digest, "local-host-process")

    def test_local_workflow_state_and_events_survive_runtime_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "runtime.db"
            first = build_local_runtime(database_path=database, workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts")
            question_id = uuid4()
            state, _ = first.workflows.start(question_id, {"problem": "restart"})
            first.workflows.apply_activity_result(state.workflow_id, stage="ingest", payload={"artifact_ids": []})
            second = build_local_runtime(database_path=database, workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts")
            restored = second.workflows.get(state.workflow_id)
            self.assertEqual(restored.stage.value, "modeling")
            self.assertEqual(second.events.verify(), 2)

    def test_workflow_repository_rejects_non_monotonic_snapshot_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "runtime.db"
            bundle = build_local_runtime(database_path=database, workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts")
            state, _ = bundle.workflows.start(uuid4(), {"problem": "version"})
            with self.assertRaises(Exception):
                bundle.workflows.repository.save(state, expected_version=state.version)

    def test_context_engine_is_deterministic_and_budgeted(self) -> None:
        session_id = uuid4()
        items = [
            ContextItem(source_id="low", kind="event", projection="summary", estimated_tokens=8, priority=1),
            ContextItem(source_id="high", kind="artifact", projection="native", estimated_tokens=5, priority=10),
        ]
        snapshot = ContextEngine().build(session_id, items, budget_tokens=6, prompt_template="template-v1")
        self.assertEqual([item.source_id for item in snapshot.items], ["high"])
        self.assertEqual(snapshot.omitted_source_ids, ("low",))
        self.assertEqual(snapshot.used_tokens, 5)

    def test_agent_runtime_dispatches_tools_without_provider_specific_workflow_logic(self) -> None:
        class Provider:
            name = "fake"
            def __init__(self):
                self.calls = 0
                self.seen_tools = ()
            def capabilities(self, model):
                return ModelCapabilities(model=model, tool_calling=True)
            async def complete(self, request):
                self.calls += 1
                self.seen_tools = request.tools
                if self.calls == 1:
                    return ModelResponse(provider="fake", model="m", tool_calls=(ModelToolCall(call_id=uuid4(), name="data_profile", arguments={"name": "input"}),), finish_reason="tool_call")
                return ModelResponse(provider="fake", model="m", text="done", finish_reason="stop")
        capabilities = CapabilityRegistry()
        capability = CapabilityRef(name="data.profile")
        capabilities.register(capability, "fake")
        resolution = capabilities.resolve([CapabilityRequirement(capability=capability)])
        tools = ToolRegistry()
        tools.register(ToolDefinition(name="data_profile", version="1", description="profile", required_capability=capability, input_schema={"type": "object", "required": ["name"]}, output_schema={"type": "object"}), lambda args: {"rows": 1})
        from asyncio import run
        session = AgentSession(session_id=uuid4(), activity_id=uuid4(), role="test", provider="fake", model="m")
        from datetime import datetime
        context = ContextEngine().build(session.session_id, [], budget_tokens=10, prompt_template="v1")
        request = ModelRequest(session=session, context=context, messages=({"role": "user", "content": "go"},))
        provider = Provider()
        response = run(AgentRuntime(provider, ToolRuntime(tools), tools).run(request, resolution))
        self.assertEqual(response.text, "done")
        self.assertEqual(provider.seen_tools[0]["function"]["name"], "data_profile")

    def test_compact_middleware_replaces_old_turns_with_continuation_snapshot(self) -> None:
        from asyncio import run
        from m2harness.application.compact import ContextCompactor
        session = AgentSession(session_id=uuid4(), activity_id=uuid4(), role="test", provider="fake", model="m")
        context = ContextEngine().build(session.session_id, [], budget_tokens=1_000, prompt_template="compact-v1")
        messages = tuple(
            {"role": "user" if index % 2 == 0 else "assistant", "content": (f"turn {index} decision file outputs/result-{index}.json " * 80)}
            for index in range(12)
        )
        request = ModelRequest(session=session, context=context, messages=messages)
        compacted = run(ContextCompactor(trigger_ratio=0.5, keep_recent_messages=4).compact(request))
        self.assertEqual(len(compacted.compactions), 1)
        self.assertEqual([item["role"] for item in compacted.messages[-4:]], [item["role"] for item in messages[-4:]])
        self.assertIn("turn 11", compacted.messages[-1]["content"])
        self.assertIn("COMPACTED CONTINUATION STATE", compacted.messages[0]["content"])
        self.assertLess(compacted.compactions[0].estimated_tokens_after, compacted.compactions[0].estimated_tokens_before)

    def test_media_inventory_sniffs_raw_bytes_and_marks_extraction_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = ArtifactStore(Path(temp) / "artifacts")
            artifact = store.put_bytes(b"%PDF-1.7\n", project_id=uuid4(), question_id=None, activity_id=None, kind=ArtifactKind.PROBLEM, logical_name="problem.pdf", media_type="application/octet-stream")
            observation = MediaInventory().inspect(artifact, b"%PDF-1.7\n")
            self.assertEqual(observation.detected_media_type, "application/pdf")
            self.assertEqual(observation.modality.value, "document")
            self.assertTrue(observation.directly_projectable)

    def test_network_policy_allows_only_the_two_explicit_egress_capabilities(self) -> None:
        policy = NetworkPolicy(model_hosts={"dashscope.aliyuncs.com"}, search_hosts={"search.example"})
        policy.authorize(EgressCapability.MODEL_PROVIDER, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        policy.authorize_search("https://search.example/query")
        with self.assertRaises(PermissionError):
            policy.authorize_search("https://dashscope.aliyuncs.com/compatible-mode/v1")
        with self.assertRaises(PermissionError):
            policy.authorize_model("http://dashscope.aliyuncs.com/compatible-mode/v1")


if __name__ == "__main__":
    unittest.main()
