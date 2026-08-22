import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m2harness.infrastructure.deepagents_code_harness import DeepAgentsCodeProposalProvider
from m2harness.infrastructure.deepagents_code_harness import CodeAgentHandoff
from m2harness.infrastructure.local_sandbox import LocalSandboxClient


class CodeAgentHarnessContractTest(unittest.TestCase):
    def _provider(self, root: Path) -> DeepAgentsCodeProposalProvider:
        # The source-tool contract is independent of the provider model.
        return DeepAgentsCodeProposalProvider(
            root,
            LocalSandboxClient(root, allow_host_processes=True),
            model=object(),
        )

    def test_revision_cannot_reuse_previous_source_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / ".m2harness-code" / "q1" / "iteration-1" / "solve_q1.py"
            previous.parent.mkdir(parents=True)
            previous.write_text("print(1)\n", encoding="utf-8")
            tool = self._provider(root)._write_code_source_tool(
                ".m2harness-code/q1/iteration-2/solve_q1.py"
            )

            rejected = json.loads(tool.invoke({"source": "print(1)\n", "append": False}))
            self.assertFalse(rejected["ok"])
            self.assertIn("identical", rejected["error"])
            self.assertEqual(rejected["previous_path"], ".m2harness-code/q1/iteration-1/solve_q1.py")

            accepted = json.loads(tool.invoke({"source": "print(2)\n", "append": False}))
            self.assertTrue(accepted["ok"])
            self.assertTrue(accepted["complete"])

    def test_python_validation_accepts_agent_selected_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tool = self._provider(root)._python_execute_tool(
                ".m2harness-code/q1/iteration-1/solve_q1.py"
            )
            accepted = json.loads(tool.invoke({"code": "print(1)", "timeout_seconds": 181}))
            self.assertTrue(accepted["ok"])
            rejected = json.loads(tool.invoke({"code": "print(1)", "timeout_seconds": 0}))
            self.assertFalse(rejected["ok"])
            self.assertIn("正整数", rejected["error"])

    def test_forced_handoff_freezes_source_tools_after_validation_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            provider = self._provider(root)
            target = ".m2harness-code/q1/iteration-1/solve_q1.py"
            provider._forced_handoff_targets.add(target)
            tool = provider._write_code_source_tool(target)
            result = json.loads(tool.invoke({"source": "print(1)"}))
            self.assertTrue(result["forced_stop"])
            self.assertIn("return the structured CodeAgentHandoff", result["next_action"])

    def test_todo_write_is_idempotent_and_directs_agent_to_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tool, _ = self._provider(root)._todo_tools(
                ".m2harness-code/q1/iteration-1/solve_q1.py"
            )
            todos = [{"id": "1", "content": "run validation", "status": "in_progress"}]
            first = json.loads(tool.invoke({"todos": todos}))
            second = json.loads(tool.invoke({"todos": todos}))
            self.assertTrue(first["ok"])
            self.assertTrue(second["ok"])
            self.assertTrue(second["unchanged"])
            self.assertIn("python_execute", second["next_action"])

    def test_timeout_requires_source_change_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / ".m2harness-code" / "q1" / "iteration-1" / "solve_q1.py"
            target.parent.mkdir(parents=True)
            target.write_text("while True:\n    pass\n", encoding="utf-8")
            provider = self._provider(root)
            target_relative = ".m2harness-code/q1/iteration-1/solve_q1.py"
            tool = provider._python_execute_tool(target_relative)

            with patch.dict("os.environ", {"M2HARNESS_EXECUTION_CHECKPOINT_SECONDS": "1"}):
                first = json.loads(tool.invoke({"code": "while True: pass", "timeout_seconds": 1}))
                self.assertTrue(first["execution_checkpoint"])
                self.assertTrue(first["still_running"])
                cancel = json.loads(provider._python_cancel_tool(target_relative).invoke({"job_id": first["job_id"]}))
            self.assertTrue(cancel["execution_cancelled"])
            self.assertFalse(cancel["ok"])

    def test_nonzero_execution_requires_source_change_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / ".m2harness-code" / "q1" / "iteration-1" / "solve_q1.py"
            target.parent.mkdir(parents=True)
            target.write_text("print(1)\n", encoding="utf-8")
            tool = self._provider(root)._python_execute_tool(
                ".m2harness-code/q1/iteration-1/solve_q1.py"
            )

            first = json.loads(tool.invoke({"code": "raise SystemExit(3)", "timeout_seconds": 10}))
            self.assertFalse(first["ok"])
            self.assertFalse(first["timed_out"])
            self.assertIn("targeted correction", first["next_action"])
            second = json.loads(tool.invoke({"code": "print(1)", "timeout_seconds": 10}))
            self.assertFalse(second["ok"])
            self.assertIn("rewrite the target source", second["error"])

    def test_structured_handoff_accepts_json_validation_array(self) -> None:
        handoff = CodeAgentHandoff.model_validate({
            "source_path": ".m2harness-code/q1/iteration-1/solve_q1.py",
            "logical_name": "solve_q1.py",
            "timeout_seconds": 120,
            "expected_validations": ["V1-初始条件", "V2-资源非负"],
        })
        self.assertEqual(handoff.expected_validations, ["V1-初始条件", "V2-资源非负"])


if __name__ == "__main__":
    unittest.main()
