from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from m2harness import build_local_runtime
from m2harness.domain.capability import CapabilityRequirement
from m2harness.domain.tool import ToolCall
from m2harness.domain.dag import canonical_single_question_dag


class LocalToolsTest(unittest.TestCase):
    def test_local_catalog_executes_multimodal_adjacent_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_local_runtime(workspace_root=root / "workspace", artifact_root=root / "artifacts", database_path=root / "runtime.db", allow_host_sandbox=True)
            requirements = [CapabilityRequirement(capability=item.required_capability) for item in bundle.tools.catalog()]
            resolution = bundle.capabilities.resolve(requirements)
            self.assertTrue(resolution.complete)

            def call(name: str, arguments: dict, *, expect_ok: bool = True):
                definition = bundle.tools.get(name)
                assert definition is not None
                tool_call = ToolCall(
                    call_id=uuid4(), tool_name=name, tool_version=definition.version,
                    activity_id=uuid4(), session_id=uuid4(), idempotency_key=str(uuid4()),
                    arguments=arguments, requested_at=datetime.now(UTC),
                )
                result = bundle.tool_runtime.execute(tool_call, resolution)
                if expect_ok:
                    self.assertTrue(result.ok, result.error_message)
                return result.output or {}

            call("workspace_write", {"path": "input/data.csv", "content": "x,y\n1,2\n3,4\n"})
            call("workspace_edit", {"path": "input/data.csv", "old_text": "3,4", "new_text": "3,5"})
            ambiguous = call("workspace_edit", {"path": "input/data.csv", "old_text": ",", "new_text": ";"}, expect_ok=False)
            self.assertEqual(ambiguous, {})
            (root / "workspace" / "problem.pdf").write_bytes(b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF")
            pdf = call("pdf_inspect", {"path": "problem.pdf"})
            self.assertEqual(pdf["format"], "pdf")
            stored = call("artifact_write", {"project_id": str(uuid4()), "logical_name": "evidence.txt", "text": "verified", "kind": "log", "media_type": "text/plain"})
            recovered = call("artifact_read", {"relative_path": stored["relative_path"], "sha256": stored["sha256"]})
            self.assertEqual(recovered["content"], "verified")
            recovered_by_id = call("artifact_read", {"artifact_id": stored["artifact_id"], "sha256": stored["sha256"]})
            self.assertEqual(recovered_by_id["content"], "verified")
            profile = call("data_profile", {"path": "input/data.csv"})
            self.assertEqual(profile["rows"], 2)
            self.assertEqual(profile["stats"]["x"]["mean"], 2.0)
            execution = call("python_execute", {"code": "print(2 + 3)"})
            self.assertEqual(execution["stdout"].strip(), "5")
            blocked = call("python_execute", {"code": "import socket; print('no')"}, expect_ok=False)
            self.assertEqual(blocked, {})
            dynamic_import = call("python_execute", {"code": "__import__('socket')"}, expect_ok=False)
            self.assertEqual(dynamic_import, {})
            validation = call("validation_run", {"code": "import json; print(json.dumps({'ok': True}))"})
            self.assertTrue(validation["passed"])
            report = call("report_render", {"markdown": "# Hello\n\nClaim", "path": "reports/out.md"})
            self.assertTrue((root / "workspace" / report["markdown_path"]).is_file())
            self.assertTrue((root / "workspace" / report["html_path"]).is_file())
            latex = call("report_render", {"markdown": "# Hello\n\nClaim", "latex": "\\documentclass{article}\n\\title{Hello}\n\\begin{document}\n\\begin{abstract}Claim.\\end{abstract}\n\\section*{Result}\nClaim.\\end{document}\n", "path": "reports/paper.md"})
            self.assertTrue(latex["latex_validated"])
            validated = call("latex_validate", {"path": latex["latex_path"]})
            self.assertTrue(validated["valid"])
            dag = call("dag_task_table", canonical_single_question_dag().model_dump(mode="json"))
            self.assertEqual(dag["terminal_task_id"], "publish-paper")
            self.assertEqual(dag["topological_order"][-1], "publish-paper")
            self.assertTrue(dag["publication_terminal"])
            self.assertGreaterEqual(bundle.tool_audit_store.verify(), 8)
            self.assertEqual(bundle.artifact_registry.verify(bundle.tool_environment.artifact_store), 1)

    def test_local_tools_reject_escape_and_web_search_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_local_runtime(workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts", database_path=Path(temp) / "runtime.db")
            required = bundle.capabilities.resolve([CapabilityRequirement(capability=bundle.tools.get("workspace_read").required_capability), CapabilityRequirement(capability=bundle.tools.get("web_search").required_capability)])

            def call(name: str, arguments: dict):
                definition = bundle.tools.get(name)
                assert definition is not None
                return bundle.tool_runtime.execute(ToolCall(call_id=uuid4(), tool_name=name, tool_version=definition.version, activity_id=uuid4(), session_id=uuid4(), idempotency_key=str(uuid4()), arguments=arguments, requested_at=datetime.now(UTC)), required)

            escaped = call("workspace_read", {"path": "../outside"})
            self.assertFalse(escaped.ok)
            self.assertEqual(escaped.error_code, "tool_execution_error")
            disabled = call("web_search", {"query": "test"})
            self.assertFalse(disabled.ok)
            self.assertIn("disabled", disabled.error_message or "")

    def test_local_tool_write_and_execution_budgets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_local_runtime(workspace_root=Path(temp) / "workspace", artifact_root=Path(temp) / "artifacts", database_path=Path(temp) / "runtime.db", allow_host_sandbox=True)
            requirement = CapabilityRequirement(capability=bundle.tools.get("artifact_write").required_capability)
            resolution = bundle.capabilities.resolve([requirement])
            definition = bundle.tools.get("artifact_write")
            assert definition is not None
            result = bundle.tool_runtime.execute(ToolCall(call_id=uuid4(), tool_name=definition.name, tool_version=definition.version, activity_id=uuid4(), session_id=uuid4(), idempotency_key=str(uuid4()), arguments={"project_id": str(uuid4()), "logical_name": "large.bin", "base64": "A" * 25_000_000}, requested_at=datetime.now(UTC)), resolution)
            self.assertFalse(result.ok)
            self.assertIn("budget", result.error_message or "")


if __name__ == "__main__":
    unittest.main()
