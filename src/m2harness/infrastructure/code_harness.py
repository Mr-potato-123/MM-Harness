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
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in blocked_modules:
                    raise PermissionError(f"generated code imports blocked module: {root}")
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in blocked_modules:
                raise PermissionError(f"generated code imports blocked module: {root}")
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
                # ``-I`` deliberately ignores PYTHON* environment variables;
                # add ``-X utf8`` explicitly so Chinese Markdown reports stay
                # UTF-8 on the Windows host as well as in Docker.
                (sys.executable, "-I", "-X", "utf8", str(script_path)), timeout_seconds=proposal.timeout_seconds,
                env={"PYTHONIOENCODING": "utf-8", "M2HARNESS_NETWORK": "deny"},
            )
            stdout = output.stdout[:self.max_output_bytes].decode("utf-8", errors="replace")
            stderr = output.stderr[:self.max_output_bytes].decode("utf-8", errors="replace")
            parsed = self._parse_json(stdout)
            required = tuple(dict.fromkeys((*modeling.required_validations, *proposal.expected_validations)))
            # The executable is allowed to return a human-readable Markdown
            # report.  JSON ``validations: false`` is only an agent hint now;
            # it is not a Harness failure signal.  Review receives the exact
            # stdout in ``report.markdown`` and decides whether the claims are
            # supported by the reproducible local run.
            reported_validations = parsed.get("validations", {}) if isinstance(parsed, dict) else {}
            validations = {
                str(name): value for name, value in reported_validations.items()
                if isinstance(name, str) and isinstance(value, bool)
            } if isinstance(reported_validations, dict) else {}
            evidence_payload = parsed.get("validation_evidence", {}) if isinstance(parsed, dict) else {}
            validation_evidence = {
                name: str(evidence_payload.get(name, "")).strip()[:8_000]
                for name in required
                if isinstance(evidence_payload, dict) and str(evidence_payload.get(name, "")).strip()
            }
            stdout_report = stdout.strip()
            succeeded = output.exit_code == 0 and not output.timed_out and bool(stdout_report)
            issues: list[str] = []
            if output.timed_out:
                issues.append("execution timed out")
            if output.exit_code not in (0, None):
                issues.append(f"execution exited with code {output.exit_code}")
            if stdout_report and parsed is None:
                issues.append("stdout was retained as a Markdown execution report; no JSON validation map was supplied")
            if any(value is False for value in validations.values()):
                issues.append("the executable reported one or more false validation hints; Review Agent must inspect the Markdown evidence")
            if parsed is not None and not validation_evidence:
                issues.append("no per-item validation_evidence field was supplied; Review Agent must use the Markdown report")
            metrics = parsed.get("metrics", {}) if isinstance(parsed, dict) and isinstance(parsed.get("metrics", {}), dict) else {}
            log = json.dumps({"exit_code": output.exit_code, "timed_out": output.timed_out, "stdout": stdout, "stderr": stderr}, ensure_ascii=False, indent=2)
            report_markdown = "\n".join([
                "# Code Harness 执行报告",
                "",
                "本报告由本地沙箱生成。stdout 不要求是 JSON；若不是 JSON，将按 Markdown 原文交给 Review Agent 审查。",
                "",
                f"- exit_code: `{output.exit_code}`",
                f"- timed_out: `{output.timed_out}`",
                f"- stdout_chars: `{len(stdout_report)}`",
                "",
                "## 程序原始报告",
                "",
                "````text",
                stdout_report or "（程序没有输出报告）",
                "````",
                "",
                "## stderr",
                "",
                "````text",
                stderr.strip() or "（无）",
                "````",
            ])
            generated_files = self._generated_files(
                task, iteration, script_path=script_path,
                metadata=proposal.metadata,
            )
            return CodingHarnessReport(
                report=ReportPayload(title="Local Code Harness execution", summary="Execution report captured locally.", markdown=report_markdown, claims=["Source was parsed and executed by the configured local sandbox."], limitations=issues),
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
        for metadata_key in ("event_log", "prompt_file"):
            declared_path = (metadata or {}).get(metadata_key)
            if not declared_path:
                continue
            candidate = (self.workspace_root / declared_path).resolve()
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
