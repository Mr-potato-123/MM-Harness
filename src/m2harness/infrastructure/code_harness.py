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
import hashlib
import mimetypes
import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from m2harness.domain.code import CodeProposal
from m2harness.domain.solve_problem import (
    CodingHarnessReport, ReadOnlyFileReference, ReadOnlyFileRole,
    SolveProblemContext, SolveProblemTask, UnifiedModelingReport,
)
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
            evidence_payload = parsed.get("validation_evidence", {}) if isinstance(parsed, dict) else {}
            validation_evidence = {
                name: str(evidence_payload.get(name, "")).strip()[:8_000]
                for name in required
            } if isinstance(evidence_payload, dict) else {name: "" for name in required}
            succeeded = (
                output.exit_code == 0 and not output.timed_out and bool(parsed is not None)
                and all(validations.values())
                and all(validation_evidence.get(name) for name in required)
            )
            issues: list[str] = []
            if output.timed_out:
                issues.append("execution timed out")
            if output.exit_code not in (0, None):
                issues.append(f"execution exited with code {output.exit_code}")
            if parsed is None:
                issues.append("stdout did not contain one JSON evidence object")
            if not all(validations.values()):
                issues.append("one or more required validations failed or were absent")
            if not all(validation_evidence.get(name) for name in required):
                issues.append("one or more required validations lacked reproducible evidence")
            metrics = parsed.get("metrics", {}) if isinstance(parsed, dict) and isinstance(parsed.get("metrics", {}), dict) else {}
            log = json.dumps({"exit_code": output.exit_code, "timed_out": output.timed_out, "stdout": stdout, "stderr": stderr}, ensure_ascii=False, indent=2)
            generated_files = self._generated_files(
                task, iteration, script_path=script_path,
                metadata=proposal.metadata,
            )
            return CodingHarnessReport(
                report=ReportPayload(title="Local Code Harness execution", summary="Execution evidence captured locally.", markdown="# Code Harness\n\nExecution evidence was captured locally.", claims=["Source was parsed and executed by the configured local sandbox."], limitations=issues),
                execution_succeeded=succeeded, validations=validations,
                validation_evidence=validation_evidence,
                metrics={key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool))},
                issues=tuple(issues), artifacts=[
                    ProducedArtifact(logical_name=proposal.logical_name, kind=ArtifactKind.OUTPUT, media_type="text/x-python", text=proposal.source, metadata={"workspace_script": str(script_path.relative_to(self.workspace_root).as_posix()), "task_id": task.task_id, "iteration": iteration, **proposal.metadata}),
                    ProducedArtifact(logical_name=proposal.logical_name.removesuffix(".py") + ".execution.json", kind=ArtifactKind.LOG, media_type="application/json", text=log),
                ], generated_files=generated_files,
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

    def _generated_files(self, task: SolveProblemTask, iteration: int, *, script_path: Path | None = None, metadata: dict[str, str] | None = None) -> tuple[ReadOnlyFileReference, ...]:
        result: list[ReadOnlyFileReference] = []
        if script_path is not None and script_path.is_file() and not script_path.is_symlink():
            data = script_path.read_bytes()
            result.append(ReadOnlyFileReference(
                relative_path=script_path.relative_to(self.workspace_root).as_posix(),
                purpose=f"Code Agent source generated for {task.task_id} iteration {iteration}; read-only review evidence.",
                role=ReadOnlyFileRole.GENERATED, owner_task_id=task.task_id,
                media_type="text/x-python", sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            ))
        output_dir = (self.workspace_root / ".m2harness-code" / task.task_id / f"iteration-{iteration}" / "outputs").resolve()
        candidates: list[Path] = []
        if output_dir.is_dir():
            candidates.extend(sorted(output_dir.rglob("*")))
        # Mature Code Agent runtimes expose their durable event stream as an
        # ordinary review artifact.  Only provider-declared relative paths are
        # admitted; arbitrary metadata cannot grant filesystem access.
        event_path = (metadata or {}).get("event_log")
        if event_path:
            candidate = (self.workspace_root / event_path).resolve()
            if self.workspace_root == candidate or self.workspace_root in candidate.parents:
                candidates.append(candidate)
        if not candidates:
            return tuple(result)
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file() or path.is_symlink():
                continue
            data = path.read_bytes()
            result.append(ReadOnlyFileReference(
                relative_path=path.relative_to(self.workspace_root).as_posix(),
                purpose=f"Output generated by {task.task_id} iteration {iteration}: {path.name}",
                role=ReadOnlyFileRole.GENERATED, owner_task_id=task.task_id,
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            ))
            if len(result) >= 100:
                break
        return tuple(result)

    @staticmethod
    def _parse_json(stdout: str) -> dict | None:
        text = stdout.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

        # Generated scripts commonly pretty-print their final JSON object.
        # The old implementation parsed only the last physical line (often
        # just ``}``), incorrectly converting valid evidence into a failure.
        decoder = json.JSONDecoder()
        for index in range(len(text) - 1, -1, -1):
            if text[index] != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None
