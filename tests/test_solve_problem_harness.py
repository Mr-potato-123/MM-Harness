from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from uuid import uuid4

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
    UnifiedModelingReport,
    build_local_runtime,
    canonical_main_harness_dag,
)
from m2harness.models import ArtifactKind, ProducedArtifact, ReportPayload
from m2harness.application.knowledge import HMMLKnowledgeBase
from m2harness.application.research import DeepResearchService
from m2harness.domain.knowledge import KnowledgeQuery
from m2harness.domain.code import CodeProposal
from m2harness.infrastructure.code_harness import LocalPythonCodeHarness
from m2harness.application.solve_problem import WorkspaceReadOnlyFileReader
from adapters.qwen_solve_problem import QwenCodeProposalProvider, QwenModelAgent, QwenPaperComposer, _content_parts
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
        return CodeProposal(source=self.source, expected_validations=modeling.required_validations)


class FakeQwenClient:
    def __init__(self) -> None:
        self.calls = []

    def json(self, *, system, content, schema):
        self.calls.append(content)
        properties = schema.get("properties", {})
        if "branch_id" in properties:
            return {"branch_id": "route-1", "report": _report("route").model_dump(mode="json"), "candidate_scheme": "scheme", "expected_outputs": ["result"]}
        if "selected_branch_ids" in properties:
            return {"report": _report("model").model_dump(mode="json"), "selected_branch_ids": ["route-1"], "main_scheme": "scheme", "required_validations": ["sanity"], "expected_outputs": ["result"]}
        if "decision" in properties:
            return {"decision": "approve", "rationale": "evidence", "accepted_claims": ["result"]}
        if "python_source" in properties or "source" in properties and "logical_name" in properties:
            return {"source": "import json; print(json.dumps({'validations': {'sanity': True}}))", "logical_name": "generated_solution.py"}
        return _report("final").model_dump(mode="json")


class SolveProblemHarnessTest(unittest.TestCase):
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

    def test_main_harness_reuses_revision_instructions_on_next_dispatch(self) -> None:
        model = FakeModelAgent()
        service = SolveProblemService(model, FakeCodeHarness(), max_iterations=1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "workspace", artifact_root=root / "artifacts", database_path=root / "runtime.db", solve_problem_service=service)
            state = bundle.main_harness.start("solve", canonical_main_harness_dag())
            state = bundle.main_harness.dispatch(state, "q1", max_iterations=1)
            self.assertEqual(state.tasks[0].status.value, "revision_required")
            bundle.main_harness.dispatch(state, "q1", max_iterations=1)
            self.assertIn("add the sanity check", model.instructions_seen[1])

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

    def test_reference_hmml_is_searchable_and_report_first(self) -> None:
        hmml = Path(__file__).parents[1] / "ref_github" / "LLM-MM-Agent" / "MMAgent" / "HMML" / "HMML.json"
        self.assertTrue(hmml.is_file())
        index = HMMLKnowledgeBase(hmml)
        result = index.search(KnowledgeQuery(query="linear programming resource allocation", top_k=3))
        self.assertGreater(result.source_count, 0)
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].entry.metadata["source_kind"], "hmml")

    def test_deep_research_is_local_by_default_and_enters_solve_context(self) -> None:
        hmml = Path(__file__).parents[1] / "ref_github" / "LLM-MM-Agent" / "MMAgent" / "HMML" / "HMML.json"
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

    def test_pi_style_local_code_harness_requires_json_validation_evidence(self) -> None:
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

            self_reported = LocalPythonCodeHarness(
                FakeCodeProposalProvider("import json; print(json.dumps({'validations': {'sanity': True}}))"),
                bundle.sandbox, root / "w",
            ).execute(SolveProblemTask(task_id="q1", title="Q", problem="P"), SolveProblemContext(), modeling, iteration=2)
            self.assertFalse(self_reported.execution_succeeded)

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
            self.assertEqual(final.title, "final")
            self.assertEqual(proposal.logical_name, "generated_solution.py")
            self.assertGreaterEqual(len(client.calls), 5)

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
