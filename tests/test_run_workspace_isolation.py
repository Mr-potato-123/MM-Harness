import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from m2harness.application.code_agent_prompts import CODE_AGENT_SYSTEM_PROMPT, build_code_agent_task_prompt
from m2harness.application.report_store import RunReportStore
from m2harness.application.run_workspace import RunIdentity, RunWorkspace
from m2harness.domain.code import CodeProposal
from m2harness.domain.solve_problem import SolveProblemContext, SolveProblemTask, UnifiedModelingReport
from m2harness.infrastructure.code_harness import LocalPythonCodeHarness
from m2harness.infrastructure.dsh_code_harness import DshCodeProposalProvider, DshRuntimeConfig, DshRuntimeError
from m2harness.infrastructure.local_sandbox import LocalSandboxClient
from m2harness.models import ReportPayload


class _Provider:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def propose(self, task, context, modeling, *, iteration):
        source_path = f".m2harness-code/runs/{self.run_id}/task-{task.task_id}/attempt-1/iteration-{iteration}/solve_{task.task_id}.py"
        return CodeProposal(
            source="from pathlib import Path\nPath('outputs').mkdir(exist_ok=True)\nPath('outputs/result.md').write_text('ok', encoding='utf-8')\nprint('# result')\n",
            logical_name=f"solve_{task.task_id}.py",
            timeout_seconds=10,
            metadata={"source_path": source_path},
        )


def _modeling() -> UnifiedModelingReport:
    return UnifiedModelingReport(
        report=ReportPayload(title="model", summary="model", markdown="model"),
        selected_branch_ids=("route-0",),
        main_scheme="Use the accepted model exactly.",
        required_validations=("sanity",),
        expected_outputs=("result.md",),
    )


class RunWorkspaceIsolationTest(unittest.TestCase):
    def test_run_names_and_manifests_are_unique_and_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            first = RunWorkspace.create(parent, RunIdentity.create(scope="2026B", input_name="2026B.pdf", model="deepseek-v4-flash-vision-exp", code_agent="dsh"))
            second = RunWorkspace.create(parent, RunIdentity.create(scope="2026B", input_name="2026B.pdf", model="deepseek-v4-flash-vision-exp", code_agent="dsh"))
            self.assertNotEqual(first.root, second.root)
            self.assertTrue(first.manifest_path.is_file())
            self.assertIn(first.identity.run_name, first.manifest_path.read_text(encoding="utf-8"))
            self.assertNotEqual(first.workspace_root, second.workspace_root)

    def test_report_store_uses_registered_human_name_without_breaking_uuid_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = RunReportStore(Path(temp))
            run_id = uuid4()
            store.register_run(run_id, "20260822__2026B__deepseek__dsh__abcd1234", metadata={"scope": "2026B"})
            store.record_probe(run_id, task_id="q1", attempt=1, iteration=1, event="start", actor="Main Harness", status="started")
            self.assertTrue((Path(temp) / "reports" / "runs" / "20260822__2026B__deepseek__dsh__abcd1234" / "probe.ndjson").is_file())

    def test_execution_uses_provider_run_lane_and_iteration_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sandbox = LocalSandboxClient(root, allow_host_processes=True)
            harness = LocalPythonCodeHarness(_Provider("run-a"), sandbox, root)
            task = SolveProblemTask(task_id="q1", title="Q1", problem="P")
            result = harness.execute(task, SolveProblemContext(metadata={"run_id": "run-a"}), _modeling(), iteration=1)
            paths = {item.relative_path for item in result.generated_files}
            self.assertIn(".m2harness-code/runs/run-a/task-q1/attempt-1/iteration-1/outputs/result.md", paths)
            self.assertFalse((root / ".m2harness-code" / "q1").exists())

    def test_code_prompt_has_no_domain_solver_policy(self) -> None:
        self.assertNotIn("MILP", CODE_AGENT_SYSTEM_PROMPT)
        self.assertIn("timeout_seconds", CODE_AGENT_SYSTEM_PROMPT)
        prompt = build_code_agent_task_prompt(
            SolveProblemTask(task_id="q1", title="Q1", problem="P"),
            SolveProblemContext(metadata={"run_id": "run-a", "run_name": "traceable-run", "task_attempt": 1}),
            _modeling(), iteration=1, target_relative=".m2harness-code/runs/run-a/task-q1/attempt-1/iteration-1/solve_q1.py",
        )
        self.assertIn("Code Agent 任务信封", prompt)
        self.assertNotIn("locked_model", prompt)
        self.assertNotIn("Skill:", prompt)
        self.assertIn("输出目录", prompt)

    def test_session_managed_prompt_bootstraps_once_then_sends_delta(self) -> None:
        task = SolveProblemTask(task_id="q1", title="Q1", problem="P", requested_outputs=("result.md",))
        context = SolveProblemContext(
            instructions=("修复 result.md 未生成。",),
            metadata={"run_id": "run-a", "run_name": "traceable-run", "task_attempt": 1},
        )
        modeling = _modeling().model_copy(update={"main_scheme": "Use the accepted model exactly and preserve the balance equation."})
        first = build_code_agent_task_prompt(
            task, context, modeling, iteration=1,
            target_relative=".m2harness-code/runs/run-a/task-q1/attempt-1/iteration-1/solve_q1.py",
            session_managed=True,
        )
        later = build_code_agent_task_prompt(
            task, context, modeling, iteration=2,
            target_relative=".m2harness-code/runs/run-a/task-q1/attempt-1/iteration-2/solve_q1.py",
            session_managed=True,
        )
        self.assertNotIn("Use the accepted model exactly and preserve the balance equation.", first)
        self.assertNotIn("Use the accepted model exactly and preserve the balance equation.", later)
        self.assertIn("持久 Code Agent 会话", later)
        self.assertIn("Code 返修", later)

    def test_dsh_session_manifest_and_prompt_index_are_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            provider = DshCodeProposalProvider(root, config=DshRuntimeConfig(cwd=root, model="test-model"))
            prompt_path = root / ".m2harness-code" / "runs" / "run-a" / "task-q1" / "attempt-1" / "iteration-1" / "code-agent-prompt.md"
            prompt_path.parent.mkdir(parents=True)
            event_file = root / ".m2harness-code" / "dsh-events" / "run-a" / "task-q1__attempt-1.ndjson"
            manifest, index = provider._record_session_context(
                prompt_path=prompt_path,
                prompt="bootstrap",
                event_file=event_file,
                session_id="m2h-code-run-a-q1-attempt-1",
                run_id="run-a",
                run_name="run-a",
                task_id="q1",
                attempt=1,
                iteration=1,
            )
            prompt_path.write_text("delta", encoding="utf-8")
            provider._record_session_context(
                prompt_path=prompt_path.parent.parent / "iteration-2" / "code-agent-prompt.md",
                prompt="delta",
                event_file=event_file,
                session_id="m2h-code-run-a-q1-attempt-1",
                run_id="run-a",
                run_name="run-a",
                task_id="q1",
                attempt=1,
                iteration=2,
            )
            self.assertEqual(manifest.name, "session.json")
            self.assertEqual(index.name, "prompt-index.ndjson")
            self.assertIn('"context_owner": "dsh-runtime"', manifest.read_text(encoding="utf-8"))
            records = [line for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 2)
            self.assertIn('"kind":"delta"', records[1])
            with self.assertRaises(DshRuntimeError):
                provider._record_session_context(
                    prompt_path=prompt_path.parent.parent / "iteration-3" / "code-agent-prompt.md",
                    prompt="wrong run",
                    event_file=event_file,
                    session_id="m2h-code-another-run-q1-attempt-1",
                    run_id="another-run",
                    run_name="run-a",
                    task_id="q1",
                    attempt=1,
                    iteration=3,
                )


if __name__ == "__main__":
    unittest.main()
