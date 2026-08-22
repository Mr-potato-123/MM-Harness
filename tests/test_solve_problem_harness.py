from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from m2harness import (
    CodingHarnessReport,
    DAGTaskKind,
    DAGTaskNode,
    DAGTaskTable,
    ExplorationMode,
    ModelReviewDecision,
    ReadOnlyFileReference,
    ReadOnlyFileRole,
    PreliminaryModelingReport,
    SolveProblemContext,
    SolveProblemService,
    SolveProblemStatus,
    SolveProblemTask,
    SolveProblemReview,
    RevisionTarget,
    UnifiedModelingReport,
    build_local_runtime,
    canonical_main_harness_dag,
)
from m2harness.models import ArtifactKind, ProducedArtifact, ReportPayload
from m2harness.application.knowledge import HMMLKnowledgeBase, default_hmml_path
from m2harness.application.research import DeepResearchService
from m2harness.domain.knowledge import KnowledgeQuery
from m2harness.domain.code import CodeProposal
from m2harness.infrastructure.code_harness import LocalPythonCodeHarness
from m2harness.application.solve_problem import WorkspaceReadOnlyFileReader
from m2harness.application.report_store import RunReportStore
from adapters.qwen_solve_problem import MainHarnessToolBridge, QwenChatClient, QwenCodeProposalProvider, QwenModelAgent, QwenPaperComposer, _content_parts
from m2harness.domain.media import MultimodalInput
import base64
import hashlib


def _report(title: str, claims: list[str] | None = None) -> ReportPayload:
    return ReportPayload(title=title, summary=title, markdown=f"# {title}\n\nEvidence.", claims=claims or [title])


class FakeModelAgent:
    def __init__(self) -> None:
        self.explorations: list[int] = []
        self.reviews = 0
        self.research_seen = []
        self.instructions_seen = []

    def explore(self, task, context, *, branch_count, iteration):
        self.research_seen.append(context.research_report)
        self.instructions_seen.append(context.instructions)
        self.explorations.append(branch_count)
        return tuple(
            PreliminaryModelingReport(
                branch_id=f"route-{index}", report=_report(f"route {index}"),
                candidate_scheme=f"scheme {index}", assumptions=("bounded",),
                expected_outputs=("result",),
            ) for index in range(branch_count)
        )

    def synthesize(self, task, context, preliminary, *, iteration):
        return UnifiedModelingReport(
            report=_report("unified modeling"), selected_branch_ids=(preliminary[0].branch_id,),
            main_scheme="use selected route", required_validations=("sanity",), expected_outputs=("result",),
            coding_instructions=("emit evidence",),
        )

    def review(self, task, context, modeling, coding, *, iteration):
        self.reviews += 1
        if self.reviews == 1:
            return SolveProblemReview(decision=ModelReviewDecision.REVISE, rationale="add one check", revision_instructions=("add the sanity check",))
        return SolveProblemReview(decision=ModelReviewDecision.APPROVE, rationale="evidence is sufficient", accepted_claims=("result",))

    def compose_final_report(self, task, context, modeling, coding, review, *, iteration):
        return _report("final solution", ["result"])


class FakeCodeHarness:
    def execute(self, task, context, modeling, *, iteration):
        return CodingHarnessReport(
            report=_report("coding report"), execution_succeeded=True,
            validations={"sanity": True}, metrics={"score": 1},
            artifacts=[ProducedArtifact(logical_name="evidence.json", kind=ArtifactKind.OUTPUT, media_type="application/json", text='{"ok": true}')],
        )


class FakeCodeProposalProvider:
    def __init__(self, source: str) -> None:
        self.source = source

    def propose(self, task, context, modeling, *, iteration):
        return CodeProposal(source=self.source, timeout_seconds=60, expected_validations=modeling.required_validations)


class FakeQwenClient:
    def __init__(self) -> None:
        self.calls = []
        self.systems = []

    def json(self, *, system, content, schema):
        self.calls.append(content)
        self.systems.append(system)
        properties = schema.get("properties", {})
        if "branch_id" in properties:
            return {"branch_id": "route-1", "report": _report("route").model_dump(mode="json"), "candidate_scheme": "scheme", "expected_outputs": ["result"]}
        if "selected_branch_ids" in properties:
            return {"report": _report("model").model_dump(mode="json"), "selected_branch_ids": ["route-1"], "main_scheme": "scheme", "required_validations": ["sanity"], "expected_outputs": ["result"]}
        if "decision" in properties:
            return {"decision": "approve", "rationale": "evidence", "accepted_claims": ["result"]}
        if "python_source" in properties or "source" in properties and "logical_name" in properties:
            return {"source": "import json; print(json.dumps({'validations': {'sanity': True}}))", "logical_name": "generated_solution.py", "timeout_seconds": 60}
        return _report("final").model_dump(mode="json")


class SolveProblemHarnessTest(unittest.TestCase):
    def test_main_harness_can_build_serial_question_todos_with_scopes(self) -> None:
        dag = canonical_main_harness_dag(question_count=4, task_problem="BASE")
        self.assertEqual(dag.topological_order(), ("q1", "q2", "q3", "q4", "publish-paper"))
        self.assertEqual(tuple(node.metadata["scope"] for node in dag.tasks[:4]), (
            "question-1", "question-2", "question-3", "question-4",
        ))
        self.assertEqual(dag.tasks[1].depends_on, ("q1",))
        self.assertIn("question-4", dag.tasks[3].metadata["problem"])

    def test_solve_problem_is_a_bounded_report_loop(self) -> None:
        model = FakeModelAgent()
        service = SolveProblemService(model, FakeCodeHarness(), max_iterations=2)
        result = service.solve(SolveProblemTask(
            task_id="q1", title="Question", problem="solve this", difficulty=6,
            exploration_mode=ExplorationMode.AUTO,
        ), SolveProblemContext())
        self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
        self.assertEqual(result.iteration_count, 2)
        # A code-only review revision reuses the accepted modeling contract.
        self.assertEqual(model.explorations, [3])
        self.assertEqual(len(result.iterations[0].preliminary_reports), 3)
        self.assertEqual(result.iterations[0].review.decision, ModelReviewDecision.REVISE)

    def test_main_context_is_a_projection_not_a_transcript_ledger(self) -> None:
        task = SolveProblemTask(task_id="q1", title="Question", problem="solve this")
        context = SolveProblemContext(metadata={
            "run_id": str(uuid4()),
            "run_name": "traceable-run",
            "task_attempt": 1,
            "model_conversation": [{"message": "old model transcript"}],
            "code_conversation": [{"message": "old code transcript"}],
        })
        projection = SolveProblemService._prepare_context(context, task)
        self.assertNotIn("model_conversation", projection.metadata)
        self.assertNotIn("code_conversation", projection.metadata)
        self.assertEqual(projection.metadata["context_owner"], "main-harness")
        self.assertIn("context_session_id", projection.metadata)
        self.assertIn("typed projection", projection.metadata["context_model"])

    def test_code_to_model_repair_forces_code_without_reexploration(self) -> None:
        class OverstrictModel(FakeModelAgent):
            def review(self, task, context, modeling, coding, *, iteration):
                self.reviews += 1
                if self.reviews == 1:
                    return SolveProblemReview(
                        decision=ModelReviewDecision.REVISE,
                        rationale="代码报告缺少一项可复现证据",
                        revision_target=RevisionTarget.MODEL,
                        revision_instructions=("在现有源码中补齐该证据并重新执行",),
                    )
                return SolveProblemReview(decision=ModelReviewDecision.APPROVE, rationale="报告已补齐")

        model = OverstrictModel()
        result = SolveProblemService(model, FakeCodeHarness(), max_iterations=2).solve(
            SolveProblemTask(task_id="q1", title="Q", problem="P", difficulty=4),
        )
        self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
        self.assertEqual(model.explorations, [2])
        self.assertEqual(result.iterations[0].review.revision_target.value, "code")

    def test_main_harness_dispatches_only_solve_problem_nodes_and_unlocks_terminal(self) -> None:
        service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(
                workspace_root=root / "workspace", artifact_root=root / "artifacts",
                database_path=root / "runtime.db", solve_problem_service=service,
            )
            state = bundle.main_harness.start("solve this", canonical_main_harness_dag())
            self.assertEqual(bundle.main_harness.ready_tasks(state), ("q1",))
            next_state = bundle.main_harness.dispatch(state, "q1")
            self.assertEqual(next_state.tasks[0].status.value, "completed")
            self.assertEqual(bundle.main_harness.ready_tasks(next_state), ("publish-paper",))
            self.assertEqual(next_state.reports[0].status, SolveProblemStatus.COMPLETED)

    def test_main_harness_dispatch_strips_transcript_metadata_before_solve(self) -> None:
        class CaptureModel(FakeModelAgent):
            def __init__(self):
                super().__init__()
                self.context = None

            def explore(self, task, context, *, branch_count, iteration):
                self.context = context
                return super().explore(task, context, branch_count=branch_count, iteration=iteration)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = CaptureModel()
            bundle = build_local_runtime(
                workspace_root=root / "workspace", artifact_root=root / "artifacts",
                database_path=root / "runtime.db", solve_problem_service=SolveProblemService(model, FakeCodeHarness(), max_iterations=1),
            )
            state = bundle.main_harness.start("solve", canonical_main_harness_dag())
            supplied = SolveProblemContext(metadata={
                "model_conversation": [{"message": "should not cross boundary"}],
                "code_conversation": [{"message": "should not cross boundary"}],
                "run_name": "main-run",
            })
            bundle.main_harness.dispatch(state, "q1", context=supplied, max_iterations=1)
            assert model.context is not None
            self.assertNotIn("model_conversation", model.context.metadata)
            self.assertNotIn("code_conversation", model.context.metadata)
            self.assertEqual(model.context.metadata["context_owner"], "main-harness")
            self.assertTrue(model.context.metadata["context_session_id"].startswith("m2h-main-"))

    def test_main_harness_does_not_replay_internal_revision_as_outer_dispatch(self) -> None:
        model = FakeModelAgent()
        service = SolveProblemService(model, FakeCodeHarness(), max_iterations=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "workspace", artifact_root=root / "artifacts", database_path=root / "runtime.db", solve_problem_service=service)
            state = bundle.main_harness.start("solve", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1", max_iterations=1)
            self.assertEqual(state.tasks[0].status.value, "completed")
            self.assertEqual(len(state.reports), 1)
            self.assertEqual(state.reports[0].iteration_count, 1)
            with self.assertRaisesRegex(ValueError, "not ready"):
                bundle.main_harness.dispatch(state, "q1", max_iterations=1)

    def test_main_harness_rollback_invalidates_descendants_and_keeps_history(self) -> None:
        service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db", solve_problem_service=service)
            state = bundle.main_harness.start("solve", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1")
            history_count = len(state.reports)
            state = bundle.main_harness.rollback(state, "q1", reason="upstream data correction")
            self.assertEqual(state.tasks[0].status.value, "ready")
            self.assertEqual(state.tasks[-1].status.value, "blocked")
            self.assertEqual(len(state.reports), history_count)
            self.assertEqual(state.decisions[-1].kind, "rollback")

    def test_runtime_registers_solve_problem_as_main_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_local_runtime(
                workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts",
                database_path=Path(temp) / "runtime.db",
            )
            definition = bundle.tools.get("solve_problem")
            self.assertIsNotNone(definition)
            self.assertEqual(definition.required_capability.name, "problem.solve")
            result = bundle.main_harness.dispatch(bundle.main_harness.start("p", canonical_main_harness_dag()), "q1")
            self.assertEqual(result.reports[0].status, SolveProblemStatus.BLOCKED)

    def test_dag_tool_accepts_main_harness_solve_problem_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            definition = bundle.tools.get("dag_task_table")
            assert definition is not None
            from m2harness.domain.tool import ToolCall
            from datetime import UTC, datetime
            from m2harness.domain.capability import CapabilityRequirement
            call = ToolCall(
                call_id=uuid4(), tool_name=definition.name, tool_version=definition.version,
                activity_id=uuid4(), session_id=uuid4(), idempotency_key=str(uuid4()),
                arguments=canonical_main_harness_dag().model_dump(mode="json"), requested_at=datetime.now(UTC),
            )
            result = bundle.tool_runtime.execute(call, bundle.capabilities.resolve([
                CapabilityRequirement(capability=definition.required_capability),
            ]))
            self.assertTrue(result.ok, result.error_message)
            self.assertEqual(tuple(result.output["topological_order"]), ("q1", "publish-paper"))

    def test_main_harness_publication_is_the_terminal_contract(self) -> None:
        service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(
                workspace_root=root / "workspace", artifact_root=root / "artifacts",
                database_path=root / "runtime.db", solve_problem_service=service,
            )
            state = bundle.main_harness.start("solve this", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1")
            latex = ProducedArtifact(
                logical_name="final-paper.tex", kind=ArtifactKind.FINAL_LATEX_PAPER,
                media_type="text/x-tex", text="\\documentclass{article}\n\\title{Final}\n\\begin{document}\n\\begin{abstract}Evidence.\\end{abstract}\n\\section*{Result}Result.\\end{document}",
            )
            published = bundle.main_harness.publish(state, final_report=_report("final"), final_latex_paper=latex)
            self.assertTrue(published.terminal)
            self.assertEqual(published.tasks[-1].status.value, "completed")
            self.assertEqual(published.final_latex_paper.logical_name, "final-paper.tex")

    def test_main_harness_state_survives_runtime_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            state = first.main_harness.start("persist me", canonical_main_harness_dag())
            second = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            restored = second.main_harness_repository.get(state.run_id)
            self.assertEqual(restored.run_id, state.run_id)
            self.assertEqual(restored.dag.terminal_task_id, "publish-paper")
            self.assertEqual(restored.version, 1)

    def test_reports_are_materialized_and_upstream_context_is_compacted(self) -> None:
        service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=2)
        dag = DAGTaskTable(tasks=(
            DAGTaskNode(id="q1", title="First", kind=DAGTaskKind.SOLVE_PROBLEM, output_contract="solution", metadata={"problem": "first subproblem"}),
            DAGTaskNode(id="q2", title="Second", kind=DAGTaskKind.SOLVE_PROBLEM, depends_on=("q1",), dependency_outputs={"q1": ("parameter",)}, output_contract="solution", metadata={"problem": "second subproblem"}),
            DAGTaskNode(id="publish-paper", title="Paper", kind=DAGTaskKind.PUBLISH_LATEX, depends_on=("q2",), output_contract="paper", terminal=True),
        ), terminal_task_id="publish-paper")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "workspace", artifact_root=root / "artifacts", database_path=root / "runtime.db", solve_problem_service=service)
            state = bundle.main_harness.start("whole problem", dag)
            state = bundle.main_harness.dispatch(state, "q1")
            solution = next(item for item in state.report_files if item.relative_path.endswith("solution_report.md"))
            self.assertTrue((root / "workspace" / solution.relative_path).is_file())
            context = bundle.main_harness._context_for_dispatch(state, "q2", ("q1",), None)
            self.assertEqual(context.dependency_solutions[0].task_id, "q1")
            self.assertEqual(context.dependency_solutions[0].summary, "final solution")
            self.assertIn(solution.relative_path, [item.relative_path for item in context.readonly_files])
            self.assertEqual(context.compression["strategy"], "compact-v1")

    def test_progressive_file_disclosure_is_allowlist_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = "accepted upstream result: alpha=0.25".encode()
            path = root / "upstream.md"
            path.write_bytes(data)
            reference = ReadOnlyFileReference(
                relative_path="upstream.md", purpose="Accepted upstream solution",
                role=ReadOnlyFileRole.DEPENDENCY_SOLUTION,
                sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            )
            reader = WorkspaceReadOnlyFileReader(root)
            disclosed = reader.disclose(SolveProblemContext(readonly_files=(reference,)), ("upstream.md",))
            self.assertIn("alpha=0.25", disclosed[0].content)
            with self.assertRaises(PermissionError):
                reader.disclose(SolveProblemContext(readonly_files=(reference,)), ("not-allowed.md",))

    def test_pdf_progressive_disclosure_keeps_original_path_and_text_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = b"%PDF-1.7\x00\xffbinary"
            (root / "problem.pdf").write_bytes(data)
            reference = ReadOnlyFileReference(
                relative_path="problem.pdf", purpose="Original PDF problem statement",
                role=ReadOnlyFileRole.PROBLEM, media_type="application/pdf",
                sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            )
            disclosed = WorkspaceReadOnlyFileReader(root).disclose(
                SolveProblemContext(readonly_files=(reference,)), ("problem.pdf",)
            )
            self.assertEqual(len(disclosed), 1)
            self.assertEqual(disclosed[0].relative_path, "problem.pdf")
            self.assertIn("原始 PDF", disclosed[0].content)

    def test_pdf_is_auto_disclosed_while_agent_handoffs_stay_small(self) -> None:
        class PdfAwareModel(FakeModelAgent):
            def __init__(self):
                super().__init__()
                self.first_context = None

            def explore(self, task, context, *, branch_count, iteration):
                self.first_context = context
                return super().explore(task, context, branch_count=branch_count, iteration=iteration)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"%PDF-1.7\noriginal problem bytes"
            pdf = root / "problem.pdf"
            pdf.write_bytes(raw)
            reference = ReadOnlyFileReference(
                relative_path="problem.pdf", purpose="原始题面 PDF",
                role=ReadOnlyFileRole.PROBLEM, media_type="application/pdf",
                sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
            )
            model = PdfAwareModel()
            run_id = uuid4()
            result = SolveProblemService(
                model, FakeCodeHarness(), max_iterations=2,
                file_reader=WorkspaceReadOnlyFileReader(root),
                archive_writer=RunReportStore(root),
            ).solve(
                SolveProblemTask(task_id="q1", title="Q", problem="P"),
                SolveProblemContext(readonly_files=(reference,), metadata={"run_id": str(run_id)}),
            )
            self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
            assert model.first_context is not None
            self.assertTrue(any(item.relative_path == "problem.pdf" for item in model.first_context.disclosed_text_files))
            exchange_root = root / "reports" / "runs" / str(run_id) / "tasks" / "q1" / "attempt-1" / "exchanges"
            model_to_code = next(exchange_root.rglob("model-to-code.md")).read_text(encoding="utf-8")
            code_to_model = next(exchange_root.rglob("code-to-model.md")).read_text(encoding="utf-8")
            model_to_code_revision = next(exchange_root.rglob("model-to-code-revision.md")).read_text(encoding="utf-8")
            self.assertIn("题目", model_to_code)
            self.assertIn("建模", model_to_code)
            self.assertIn("产生的文件/图片", code_to_model)
            self.assertIn("结果", code_to_model)
            self.assertIn("请修改", model_to_code_revision)
            self.assertNotIn("原始题面 PDF", code_to_model)
            self.assertNotIn("原始题面 PDF", model_to_code_revision)

    def test_probe_ledger_is_written_for_each_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = uuid4()
            store = RunReportStore(root)
            result = SolveProblemService(
                FakeModelAgent(), FakeCodeHarness(), max_iterations=2,
                archive_writer=store, probe_writer=store,
            ).solve(
                SolveProblemTask(task_id="q1", title="Q", problem="P"),
                SolveProblemContext(metadata={"run_id": str(run_id)}),
            )
            self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
            probe = root / "reports" / "runs" / str(run_id) / "probe.ndjson"
            probe_md = root / "reports" / "runs" / str(run_id) / "probe.md"
            self.assertTrue(probe.is_file())
            self.assertTrue(probe_md.is_file())
            events = [json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines()]
            names = {item["event"] for item in events}
            self.assertIn("model_to_code_handoff", names)
            self.assertIn("code_to_model_handoff", names)
            self.assertIn("model_to_code_revision_handoff", names)
            markers = [item for item in events if item["event"] == "context_marker"]
            self.assertTrue(markers)
            self.assertNotIn("message", markers[0]["details"])
            projections = [item for item in events if item["event"] == "context_projection_updated"]
            self.assertTrue(projections)
            self.assertIn("path", projections[0]["details"])

    def test_probe_records_model_failure_reason(self) -> None:
        class FailingModel(FakeModelAgent):
            def explore(self, task, context, *, branch_count, iteration):
                raise RuntimeError("provider returned 401")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = uuid4()
            store = RunReportStore(root)
            with self.assertRaises(RuntimeError):
                SolveProblemService(
                    FailingModel(), FakeCodeHarness(), max_iterations=1,
                    archive_writer=store, probe_writer=store,
                ).solve(
                    SolveProblemTask(task_id="q1", title="Q", problem="P"),
                    SolveProblemContext(metadata={"run_id": str(run_id)}),
                )
            events = [json.loads(line) for line in (root / "reports" / "runs" / str(run_id) / "probe.ndjson").read_text(encoding="utf-8").splitlines()]
            failed = [item for item in events if item["event"] == "model_explore_failed"]
            self.assertEqual(len(failed), 1)
            self.assertIn("401", failed[0]["details"]["error"])

    def test_solve_loop_discloses_requested_file_then_reexplores(self) -> None:
        class RequestingModel(FakeModelAgent):
            def __init__(self):
                super().__init__()
                self.explore_calls = 0
                self.disclosed_seen = False

            def explore(self, task, context, *, branch_count, iteration):
                self.explore_calls += 1
                self.disclosed_seen = self.disclosed_seen or bool(context.disclosed_text_files)
                reports = super().explore(task, context, branch_count=branch_count, iteration=iteration)
                if self.explore_calls == 1:
                    reports = (reports[0].model_copy(update={"requested_file_paths": ("dependency.md",)}),)
                return reports

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = b"verified parameter beta=4"
            (root / "dependency.md").write_bytes(content)
            context = SolveProblemContext(readonly_files=(ReadOnlyFileReference(
                relative_path="dependency.md", purpose="Upstream parameter file",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT,
                sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content),
            ),))
            model = RequestingModel()
            result = SolveProblemService(
                model, FakeCodeHarness(), max_iterations=2,
                file_reader=WorkspaceReadOnlyFileReader(root),
            ).solve(SolveProblemTask(task_id="q2", title="Q2", problem="Use beta"), context)
            self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
            self.assertTrue(model.disclosed_seen)
            self.assertGreaterEqual(model.explore_calls, 2)

    def test_generated_code_becomes_allowlisted_review_evidence(self) -> None:
        class GeneratedCodeHarness(FakeCodeHarness):
            def __init__(self, root):
                self.root = root

            def execute(self, task, context, modeling, *, iteration):
                path = self.root / ".m2harness-code" / task.task_id / f"iteration-{iteration}" / "solve.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                data = b"print('evidence')\n"
                path.write_bytes(data)
                return CodingHarnessReport(
                    report=_report("code"), execution_succeeded=True,
                    validations={"sanity": True},
                    validation_evidence={"sanity": "deterministic"},
                    artifacts=[ProducedArtifact(logical_name="evidence.json", text="{}")],
                    generated_files=(ReadOnlyFileReference(
                        relative_path=path.relative_to(self.root).as_posix(),
                        purpose="Generated source", role=ReadOnlyFileRole.GENERATED,
                        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                    ),),
                )

        class ReviewingModel(FakeModelAgent):
            def __init__(self):
                super().__init__()
                self.review_context = None

            def review(self, task, context, modeling, coding, *, iteration):
                self.review_context = context
                return SolveProblemReview(decision=ModelReviewDecision.APPROVE, rationale="source is reviewable")

            def compose_final_report(self, task, context, modeling, coding, review, *, iteration):
                return _report("final")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = ReviewingModel()
            result = SolveProblemService(
                model, GeneratedCodeHarness(root), max_iterations=1,
                file_reader=WorkspaceReadOnlyFileReader(root),
            ).solve(SolveProblemTask(task_id="q1", title="Q", problem="P"))
            self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
            assert model.review_context is not None
            self.assertTrue(any(item.role == ReadOnlyFileRole.GENERATED for item in model.review_context.readonly_files))

    def test_append_only_code_event_log_is_not_pinned_to_stale_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event = root / ".m2harness-code" / "deepagents-events" / "run-q1.ndjson"
            event.parent.mkdir(parents=True, exist_ok=True)
            event.write_text('{"event":"tool.called"}\n', encoding="utf-8")
            harness = LocalPythonCodeHarness(
                FakeCodeProposalProvider("print('# report')"),
                build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "db.sqlite").sandbox,
                root,
            )
            references = harness._generated_files(
                SolveProblemTask(task_id="q1", title="Q1", problem="P"), 1,
                metadata={"event_log": event.relative_to(root).as_posix()},
            )
            self.assertEqual(len(references), 1)
            self.assertIsNone(references[0].sha256)
            self.assertIsNone(references[0].size_bytes)
            event.write_text('{"event":"tool.called"}\n{"event":"handoff"}\n', encoding="utf-8")
            # The same allowlisted live audit stream remains readable after
            # later tool events append; persistence snapshots its new bytes.
            disclosed = WorkspaceReadOnlyFileReader(root).disclose(
                SolveProblemContext(readonly_files=references), (references[0].relative_path,)
            )
            self.assertIn("handoff", disclosed[0].content)

    def test_each_model_code_review_revision_is_archived_before_next_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = uuid4()
            result = SolveProblemService(
                FakeModelAgent(), FakeCodeHarness(), max_iterations=2,
                archive_writer=RunReportStore(root),
            ).solve(
                SolveProblemTask(task_id="q1", title="Q", problem="P"),
                SolveProblemContext(metadata={"run_id": str(run_id)}),
            )
            self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
            self.assertGreaterEqual(len(result.archive_files), 7)
            exchange_root = root / "reports" / "runs" / str(run_id) / "tasks" / "q1" / "attempt-1" / "exchanges"
            expected = {
                "preliminary-route-0.md", "modeling_report.md", "coding_report.md",
                "review_report.md", "revision_instructions.md",
            }
            names = {path.name for path in exchange_root.rglob("*.md")}
            self.assertTrue(expected.issubset(names), names)
            self.assertGreaterEqual(len(list(exchange_root.rglob("*.md"))), 7)
            model_to_code = next(exchange_root.rglob("model-to-code.md"))
            code_to_model = next(exchange_root.rglob("code-to-model.md"))
            self.assertLess(model_to_code.stat().st_size, 12_000)
            self.assertLess(code_to_model.stat().st_size, 12_000)
            self.assertIn("输出", model_to_code.read_text(encoding="utf-8"))
            self.assertIn("产生的文件/图片", code_to_model.read_text(encoding="utf-8"))
            self.assertIn("结果", code_to_model.read_text(encoding="utf-8"))
            model_to_code_text = model_to_code.read_text(encoding="utf-8")
            code_to_model_text = code_to_model.read_text(encoding="utf-8")
            model_to_code_revision = next(exchange_root.rglob("model-to-code-revision.md")).read_text(encoding="utf-8")
            self.assertIn("建模", model_to_code_text)
            self.assertIn("产生的文件/图片", code_to_model_text)
            self.assertIn("请修改", model_to_code_revision)
            self.assertNotIn("execution_succeeded", code_to_model_text)
            self.assertNotIn("timed_out", code_to_model_text)
            persisted = RunReportStore(root).persist(run_id, result, attempt=1)
            self.assertTrue(any("/exchanges/" in item.relative_path.replace("\\", "/") for item in persisted))

    def test_solve_problem_caps_revisions_at_two_rounds_and_writes_total_report(self) -> None:
        class AlwaysReviseModel(FakeModelAgent):
            def review(self, task, context, modeling, coding, *, iteration):
                self.reviews += 1
                return SolveProblemReview(
                    decision=ModelReviewDecision.REVISE,
                    rationale="evidence remains incomplete",
                    revision_instructions=("produce the missing evidence",),
                )

        model = AlwaysReviseModel()
        result = SolveProblemService(model, FakeCodeHarness(), max_iterations=20).solve(
            SolveProblemTask(task_id="q1", title="Q", problem="P"),
        )
        self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
        self.assertEqual(result.iteration_count, 3)
        self.assertEqual(len(result.iterations), 3)
        self.assertEqual(model.reviews, 3)
        self.assertIsNotNone(result.final_report)
        self.assertIsNone(result.error)

    def test_solve_problem_can_resume_from_serialized_modeling_at_code_stage(self) -> None:
        model = FakeModelAgent()
        task = SolveProblemTask(task_id="q1", title="Q", problem="P")
        preliminary = PreliminaryModelingReport(
            branch_id="route-0", report=_report("route"), candidate_scheme="scheme", expected_outputs=("result",)
        )
        modeling = UnifiedModelingReport(
            report=_report("model"), selected_branch_ids=("route-0",), main_scheme="scheme",
            required_validations=("sanity",), expected_outputs=("result",), coding_instructions=("emit evidence",),
        )
        context = SolveProblemContext(metadata={
            "resume_iteration": 2,
            "resume_modeling": modeling.model_dump(mode="json"),
            "resume_preliminary": [preliminary.model_dump(mode="json")],
            "resume_revision_target": "code",
        })
        result = SolveProblemService(model, FakeCodeHarness(), max_iterations=3).solve(task, context)
        self.assertEqual(result.status, SolveProblemStatus.COMPLETED)
        self.assertEqual(result.iteration_count, 3)
        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(model.explorations, [])

    def test_main_harness_caps_internal_revision_rounds_without_outer_retry(self) -> None:
        class AlwaysReviseModel(FakeModelAgent):
            def review(self, task, context, modeling, coding, *, iteration):
                self.reviews += 1
                return SolveProblemReview(
                    decision=ModelReviewDecision.REVISE,
                    rationale="evidence remains incomplete",
                    revision_instructions=("produce the missing evidence",),
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(
                workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db",
                solve_problem_service=SolveProblemService(AlwaysReviseModel(), FakeCodeHarness(), max_iterations=1),
            )
            state = bundle.main_harness.start("P", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1", max_iterations=1)
            self.assertEqual(state.tasks[0].status.value, "completed")
            self.assertEqual(state.reports[0].iteration_count, 1)
            with self.assertRaisesRegex(ValueError, "not ready"):
                bundle.main_harness.dispatch(state, "q1", max_iterations=1)

    def test_reference_hmml_is_searchable_and_report_first(self) -> None:
        hmml = default_hmml_path(Path(__file__).parents[1])
        self.assertIsNotNone(hmml)
        assert hmml is not None
        index = HMMLKnowledgeBase(hmml)
        result = index.search(KnowledgeQuery(query="linear programming resource allocation", top_k=3))
        self.assertGreater(result.source_count, 0)
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].entry.metadata["source_kind"], "hmml")

    def test_deep_research_is_local_by_default_and_enters_solve_context(self) -> None:
        hmml = default_hmml_path(Path(__file__).parents[1])
        self.assertIsNotNone(hmml)
        assert hmml is not None
        research = DeepResearchService(HMMLKnowledgeBase(hmml))
        model = FakeModelAgent()
        service = SolveProblemService(model, FakeCodeHarness(), max_iterations=1, research_agent=research)
        result = service.solve(SolveProblemTask(task_id="q1", title="Question", problem="resource allocation", difficulty=4))
        self.assertIsNotNone(result.research_report)
        assert result.research_report is not None
        self.assertTrue(result.research_report.local_only)
        self.assertGreaterEqual(len(result.research_report.plan), 3)
        self.assertIsNotNone(model.research_seen[0])

    def test_runtime_registers_local_knowledge_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            definition = bundle.tools.get("knowledge_search")
            self.assertIsNotNone(definition)
            self.assertEqual(definition.required_capability.name, "knowledge.search")
            from datetime import UTC, datetime
            from m2harness.domain.capability import CapabilityRequirement
            from m2harness.domain.tool import ToolCall
            call = ToolCall(
                call_id=uuid4(), tool_name="knowledge_search", tool_version="1", activity_id=uuid4(), session_id=uuid4(),
                idempotency_key=str(uuid4()), arguments={"query": "linear programming", "top_k": 2}, requested_at=datetime.now(UTC),
            )
            result = bundle.tool_runtime.execute(call, bundle.capabilities.resolve([
                CapabilityRequirement(capability=definition.required_capability),  # type: ignore[union-attr]
            ]))
            self.assertTrue(result.ok, result.error_message)
            self.assertEqual(result.output["source"], "local_knowledge_base")

    def test_pi_style_local_code_harness_accepts_markdown_or_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            harness = LocalPythonCodeHarness(
                FakeCodeProposalProvider("import json; print(json.dumps({'validations': {'sanity': True}, 'validation_evidence': {'sanity': 'score=1, asserted from deterministic fixture'}, 'metrics': {'score': 1}}))"),
                bundle.sandbox, root / "w",
            )
            modeling = UnifiedModelingReport(
                report=_report("model"), selected_branch_ids=("route-0",), main_scheme="scheme",
                required_validations=("sanity",), expected_outputs=("result",),
            )
            coding = harness.execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=1)
            self.assertTrue(coding.execution_succeeded)
            self.assertEqual(coding.validations["sanity"], True)
            self.assertTrue(coding.validation_evidence["sanity"])
            self.assertEqual(len(coding.artifacts), 2)

            pretty = LocalPythonCodeHarness(
                FakeCodeProposalProvider("import json; print(json.dumps({'validations': {'sanity': True}, 'validation_evidence': {'sanity': 'pretty-json-evidence'}}, indent=2))"),
                bundle.sandbox, root / "w",
            ).execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=3)
            self.assertTrue(pretty.execution_succeeded)
            self.assertEqual(pretty.validation_evidence["sanity"], "pretty-json-evidence")

            self_reported = LocalPythonCodeHarness(
                FakeCodeProposalProvider("import json; print(json.dumps({'validations': {'sanity': True}}))"),
                bundle.sandbox, root / "w",
            ).execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=2)
            self.assertTrue(self_reported.execution_succeeded)
            self.assertIn("原始报告", self_reported.report.markdown)

            markdown = LocalPythonCodeHarness(
                FakeCodeProposalProvider("print('# 第 1 问执行报告\\n\\n路径已生成，资源平衡证据见本报告。')"),
                bundle.sandbox, root / "w",
            ).execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=4)
            self.assertTrue(markdown.execution_succeeded)
            self.assertEqual(markdown.validations, {})
            self.assertIn("第 1 问执行报告", markdown.report.markdown)

            blocked = LocalPythonCodeHarness(FakeCodeProposalProvider("import socket; print('bad')"), bundle.sandbox, root / "w")
            failed = blocked.execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=1)
            self.assertFalse(failed.execution_succeeded)
            self.assertTrue(failed.issues)

    def test_qwen_solve_adapter_preserves_report_protocol_and_multimodal_parts(self) -> None:
        raw = b"%PDF-1.7\n"
        media = MultimodalInput(logical_name="problem.pdf", media_type="application/pdf", data_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        context = SolveProblemContext(multimodal_inputs=(media,))
        parts = _content_parts(context, "prompt")
        self.assertEqual(parts[0]["type"], "file")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            client = FakeQwenClient()
            agent = QwenModelAgent(client, skills=bundle.skills)
            task = SolveProblemTask(task_id="q1", title="Q", problem="P")
            preliminary = agent.explore(task, context, branch_count=1, iteration=1)
            modeling = agent.synthesize(task, context, preliminary, iteration=1)
            coding = CodingHarnessReport(report=_report("code"), execution_succeeded=True, validations={"sanity": True}, artifacts=[ProducedArtifact(logical_name="evidence.json", text="{}")])
            review = agent.review(task, context, modeling, coding, iteration=1)
            final = agent.compose_final_report(task, context, modeling, coding, review, iteration=1)
            proposal = QwenCodeProposalProvider(client, skills=bundle.skills).propose(task, context, modeling, iteration=1)
            first_prompt = json.loads(client.calls[0][-1]["text"])
            self.assertIn("Skill: problem-intake", first_prompt["skill_context"])
            self.assertIn("Skill: coding-contract", first_prompt["skill_context"])
            code_prompt = json.loads(client.calls[4][-1]["text"])
            self.assertIn("Skill: modeling-core", code_prompt["skill_context"])
            self.assertNotIn("## Explicit omissions", code_prompt["skill_context"])
            self.assertIn("The Main Harness owns", client.systems[0])
            self.assertIn("同一个 Model Agent", json.dumps(client.calls[2], ensure_ascii=False))
            self.assertIn("Code Agent", client.systems[4])
            self.assertEqual(final.title, "final")
            self.assertEqual(proposal.logical_name, "generated_solution.py")
            self.assertGreaterEqual(len(client.calls), 5)

    def test_qwen_client_retries_truncated_json_without_synthetic_repair(self) -> None:
        class Response:
            status_code = 200

            def __init__(self, body):
                self.body = body

            def json(self):
                return self.body

            def raise_for_status(self):
                raise AssertionError("unexpected HTTP error")

        class Client:
            payloads = []
            responses = [
                {"choices": [{"finish_reason": "length", "message": {"content": '{"title":"cut'}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]},
            ]

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, *, headers, json):
                self.payloads.append(json)
                return Response(self.responses.pop(0))

        with patch("adapters.qwen_solve_problem.httpx.Client", Client):
            with patch.dict("os.environ", {"QWEN_MAX_OUTPUT_TOKENS": "4000", "QWEN_STREAM": "0"}, clear=False):
                value = QwenChatClient(api_key="test-key", timeout_seconds=1).json(
                    system="system", content=[{"type": "text", "text": "prompt"}], schema={}
                )
        self.assertEqual(value, {"ok": True})
        self.assertEqual(len(Client.payloads), 2)
        self.assertFalse(Client.payloads[1]["enable_thinking"])
        self.assertEqual(Client.payloads[1]["max_tokens"], 8000)
        self.assertIn("previous response was truncated", Client.payloads[1]["messages"][1]["content"][-1]["text"])

    def test_qwen_client_consumes_sse_stream(self) -> None:
        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}'
                yield 'data: {"choices":[{"delta":{"content":"true}"},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

            def raise_for_status(self):
                raise AssertionError("unexpected HTTP error")

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stream(self, *args, **kwargs):
                return Response()

        with patch("adapters.qwen_solve_problem.httpx.Client", Client):
            with patch.dict("os.environ", {"QWEN_STREAM": "1", "QWEN_STREAM_PROGRESS": "0"}, clear=False):
                value = QwenChatClient(api_key="test-key", timeout_seconds=1).json(
                    system="system", content=[{"type": "text", "text": "prompt"}], schema={}
                )
        self.assertEqual(value, {"ok": True})

    def test_code_agent_tool_bridge_projects_and_audits_main_harness_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "workspace", artifact_root=root / "artifacts", database_path=root / "runtime.db")
            target = root / "workspace" / "sample.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("alpha = 1\n", encoding="utf-8")
            bridge = MainHarnessToolBridge(bundle.tool_runtime, bundle.capabilities)
            names = {item["function"]["name"] for item in bridge.definitions()}
            self.assertTrue({"workspace_read", "workspace_write", "workspace_edit", "workspace_search", "python_execute"}.issubset(names))
            result = bridge.execute("workspace_search", {"pattern": "alpha", "path": "."}, task_id="q1", iteration=1, turn=1)
            self.assertTrue(result["ok"])
            self.assertEqual(result["output"]["matches"][0]["path"], "sample.py")
            first = bridge.execute("workspace_write", {"path": "chunks.txt", "content": "one\n", "overwrite": True}, task_id="q1", iteration=1, turn=1)
            second = bridge.execute("workspace_write", {"path": "chunks.txt", "content": "two\n", "append": True}, task_id="q1", iteration=1, turn=2)
            self.assertTrue(first["ok"] and second["ok"])
            read_back = bridge.execute("workspace_read", {"path": "chunks.txt"}, task_id="q1", iteration=1, turn=3)
            self.assertEqual(read_back["output"]["content"], "one\ntwo\n")
            denied = bridge.execute("report_render", {"markdown": "bad"}, task_id="q1", iteration=1, turn=1)
            self.assertFalse(denied["ok"])
            self.assertEqual(denied["error_code"], "tool_not_allowed")

    def test_runtime_can_attach_provider_after_sandbox_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db")
            service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=1)
            attached = runtime.attach_solve_problem_service(service)
            self.assertIs(attached.solve_problem_service, service)
            self.assertIs(runtime.tool_environment.solve_problem_service, service)

    def test_main_harness_terminal_composer_owns_global_paper_merge(self) -> None:
        class Composer:
            def compose(self, problem, reports):
                return _report("global final"), ProducedArtifact(
                    logical_name="global.tex", kind=ArtifactKind.FINAL_LATEX_PAPER, media_type="text/x-tex",
                    text="\\documentclass{article}\n\\title{Global}\n\\begin{document}\\begin{abstract}Evidence.\\end{abstract}\\section*{Result}Evidence.\\end{document}",
                )

        service = SolveProblemService(FakeModelAgent(), FakeCodeHarness(), max_iterations=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "w", artifact_root=root / "a", database_path=root / "r.db", solve_problem_service=service)
            state = bundle.main_harness.start("P", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1")
            state = bundle.main_harness.generate_paper(state, Composer())
            self.assertTrue(state.terminal)
            self.assertEqual(state.final_report.title, "global final")

    def test_qwen_paper_composer_returns_typed_single_latex_artifact(self) -> None:
        class PaperClient:
            def json(self, *, system, content, schema):
                return {"final_report": _report("paper").model_dump(mode="json"), "latex": "\\documentclass{article}\n\\title{Paper}\n\\begin{document}\\begin{abstract}Evidence.\\end{abstract}\\section*{Result}Evidence.\\end{document}"}

        report, artifact = QwenPaperComposer(PaperClient()).compose("P", ())
        self.assertEqual(report.title, "paper")
        self.assertEqual(artifact.kind, ArtifactKind.FINAL_LATEX_PAPER)


if __name__ == "__main__":
    unittest.main()
