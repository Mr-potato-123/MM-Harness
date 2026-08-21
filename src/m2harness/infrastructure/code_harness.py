"""PI-inspired bounded local Code Harness.

The Code Harness owns realization, not modeling decisions.  A provider supplies
one ``CodeProposal`` from a ``UnifiedModelingReport``; this adapter validates,
materializes, executes, captures bounded output, and returns evidence to the
Model Agent.  It never marks a run successful solely because source code was
generated.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from m2harness.domain.code import CodeProposal
from m2harness.domain.solve_problem import CodingHarnessReport, SolveProblemContext, SolveProblemTask, UnifiedModelingReport
from m2harness.infrastructure.local_sandbox import LocalSandboxClient
from m2harness.models import ArtifactKind, ProducedArtifact, ReportPayload


class CodeProposalProvider(Protocol):
    def propose(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int) -> CodeProposal: ...


def _validate_python_policy(source: str) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"generated Python is invalid: {exc}") from exc
    blocked_modules = {"socket", "subprocess", "ctypes", "multiprocessing", "requests", "httpx", "urllib", "ftplib"}
    blocked_calls = {"system", "popen", "spawn", "fork", "execv", "execve", "__import__", "eval", "exec", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0] in blocked_modules for alias in node.names):
            raise PermissionError("generated code imports a blocked network/process module")
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in blocked_modules:
            raise PermissionError("generated code imports a blocked network/process module")
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
            if name in blocked_calls:
                raise PermissionError(f"generated code uses a blocked escape call: {name}")


class LocalPythonCodeHarness:
    def __init__(self, provider: CodeProposalProvider, sandbox: LocalSandboxClient, workspace_root: Path, *, max_output_bytes: int = 1_000_000) -> None:
        self.provider = provider
        self.sandbox = sandbox
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max_output_bytes

    def execute(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int) -> CodingHarnessReport:
        try:
            proposal = self.provider.propose(task, context, modeling, iteration=iteration)
            _validate_python_policy(proposal.source)
            script_path = self._write_script(task, proposal, iteration=iteration)
            output = self.sandbox.run(
                (sys.executable, "-I", str(script_path)), timeout_seconds=proposal.timeout_seconds,
                env={"PYTHONIOENCODING": "utf-8", "M2HARNESS_NETWORK": "deny"},
            )
            stdout = output.stdout[:self.max_output_bytes].decode("utf-8", errors="replace")
            stderr = output.stderr[:self.max_output_bytes].decode("utf-8", errors="replace")
            parsed = self._parse_json(stdout)
            required = tuple(dict.fromkeys((*modeling.required_validations, *proposal.expected_validations)))
            validations = {name: bool(parsed.get("validations", {}).get(name, False)) for name in required} if isinstance(parsed, dict) else {name: False for name in required}
            succeeded = output.exit_code == 0 and not output.timed_out and bool(parsed is not None) and all(validations.values())
            issues: list[str] = []
            if output.timed_out:
                issues.append("execution timed out")
            if output.exit_code not in (0, None):
                issues.append(f"execution exited with code {output.exit_code}")
            if parsed is None:
                issues.append("stdout did not contain one JSON evidence object")
            if not all(validations.values()):
                issues.append("one or more required validations failed or were absent")
            metrics = parsed.get("metrics", {}) if isinstance(parsed, dict) and isinstance(parsed.get("metrics", {}), dict) else {}
            log = json.dumps({"exit_code": output.exit_code, "timed_out": output.timed_out, "stdout": stdout, "stderr": stderr}, ensure_ascii=False, indent=2)
            return CodingHarnessReport(
                report=ReportPayload(title="Local Code Harness execution", summary="Execution evidence captured locally.", markdown="# Code Harness\n\nExecution evidence was captured locally.", claims=["Source was parsed and executed by the configured local sandbox."], limitations=issues),
                execution_succeeded=succeeded, validations=validations, metrics={key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool))},
                issues=tuple(issues), artifacts=[
                    ProducedArtifact(logical_name=proposal.logical_name, kind=ArtifactKind.OUTPUT, media_type="text/x-python", text=proposal.source, metadata={"workspace_script": str(script_path.relative_to(self.workspace_root).as_posix()), "task_id": task.task_id, "iteration": iteration}),
                    ProducedArtifact(logical_name=proposal.logical_name.removesuffix(".py") + ".execution.json", kind=ArtifactKind.LOG, media_type="application/json", text=log),
                ],
            )
        except Exception as exc:
            return CodingHarnessReport(
                report=ReportPayload(title="Local Code Harness failure", summary="Code execution did not produce accepted evidence.", markdown="# Code Harness\n\nExecution failed before acceptance.", claims=[], limitations=[str(exc)[:2_000]]),
                execution_succeeded=False, validations={}, issues=(str(exc)[:2_000],), artifacts=[],
            )

    def _write_script(self, task: SolveProblemTask, proposal: CodeProposal, *, iteration: int) -> Path:
        # Separate task/iteration lanes prevent parallel root DAG nodes from
        # overwriting one another while preserving the shared workspace for
        # staged input data and generated output files.
        target = (self.workspace_root / ".m2harness-code" / task.task_id / f"iteration-{iteration}" / proposal.logical_name).resolve()
        if self.workspace_root not in target.parents:
            raise ValueError("generated script escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".m2h-code-", suffix=".py", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(proposal.source)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @staticmethod
    def _parse_json(stdout: str) -> dict | None:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
