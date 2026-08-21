"""Activity execution boundary for independently deployable agent workers."""

from __future__ import annotations

import json
import os
import subprocess
import base64
import sys
import hashlib
import mimetypes
import shutil
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from pydantic import ValidationError

from m2harness.errors import ActivityExecutionError, ConfigurationError
from m2harness.artifacts import ArtifactStore
from m2harness.models import ActivityRequest, ActivityResponse, ArtifactKind, CodingStageOutput, ProducedArtifact, StageKind, utc_now
from m2harness.infrastructure.local_sandbox import LocalSandboxClient


class ActivityExecutor(Protocol):
    def execute(self, request: ActivityRequest) -> ActivityResponse: ...


class CommandActivityExecutor:
    """Runs an agent adapter without a shell, exchanging one JSON document on stdio.

    This isolates the Harness protocol from agent implementation. It is not a code
    sandbox; production adapters should submit generated code to a container/VM
    execution service and return only its evidence.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int = 1800,
        cwd: Path | None = None,
        pass_env: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not command or not command[0].strip():
            raise ConfigurationError("executor command cannot be empty")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.pass_env = tuple(pass_env)
        self.environment = dict(environment or {})
        self.max_output_bytes = max_output_bytes

    def execute(self, request: ActivityRequest) -> ActivityResponse:
        inherited = ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        env = {name: os.environ[name] for name in (*inherited, *self.pass_env) if name in os.environ}
        env.update(self.environment)
        try:
            process = subprocess.run(
                self.command,
                input=request.model_dump_json(),
                text=True,
                encoding="utf-8",
                errors="strict",
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=self.cwd,
                env=env,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ActivityExecutionError(f"executor process failed: {exc}") from exc
        stdout = process.stdout.encode("utf-8")
        stderr = process.stderr[-4000:]
        if len(stdout) > self.max_output_bytes:
            raise ActivityExecutionError("executor response exceeded configured byte limit")
        if process.returncode != 0:
            raise ActivityExecutionError(f"executor exited with code {process.returncode}: {stderr}")
        try:
            response = ActivityResponse.model_validate_json(stdout)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise ActivityExecutionError(f"executor returned invalid protocol JSON: {exc}") from exc
        if response.idempotency_key != request.idempotency_key:
            raise ActivityExecutionError("executor response idempotency key does not match request")
        if response.output.stage != request.stage:
            raise ActivityExecutionError("executor response stage does not match request")
        return response


class SandboxedActivityExecutor:
    """Execute model-produced coding artifacts in the configured local Sandbox.

    The provider proposes source code; this wrapper is the evidence boundary.
    It ignores any claimed execution flag, runs the selected source in the
    configured local execution backend (trusted host by default, Docker/VM
    when selected), and appends bounded stdout/stderr logs before persistence.
    """

    def __init__(self, inner: ActivityExecutor, sandbox: LocalSandboxClient, artifact_store: ArtifactStore, *, timeout_seconds: int = 600, image_digest: str | None = None) -> None:
        self.inner = inner
        self.sandbox = sandbox
        self.artifact_store = artifact_store
        self.timeout_seconds = timeout_seconds
        self.image_digest = image_digest or getattr(sandbox, "image_digest", "unknown")

    def execute(self, request: ActivityRequest) -> ActivityResponse:
        response = self.inner.execute(request)
        if request.stage != StageKind.CODING or not isinstance(response.output, CodingStageOutput):
            return response
        output = response.output
        source = self._source_artifact(output)
        if source is None:
            return self._with_execution_result(response, output, False, {"sandbox_error": "coding response did not include a Python source artifact"})
        try:
            code = self._artifact_bytes(source)
            if len(code) > 2 * 1024 * 1024:
                raise ValueError("coding source exceeds the 2 MiB Sandbox input budget")
            activity_workspace = self.sandbox.workspace_root / "activities" / str(request.activity_id)
            activity_workspace.mkdir(parents=True, exist_ok=True)
            self._materialize_inputs(request, activity_workspace)
            script = activity_workspace / "model.py"
            script.write_bytes(code)
            try:
                result = self.sandbox.run(
                    (sys.executable, "-I", str(script)),
                    timeout_seconds=min(self.timeout_seconds, max(1, int((request.deadline - utc_now()).total_seconds()))),
                    env={"PYTHONIOENCODING": "utf-8", "M2HARNESS_NETWORK": "deny"},
                    cwd=activity_workspace,
                )
            finally:
                script.unlink(missing_ok=True)
            succeeded = result.exit_code == 0 and not result.timed_out
            stdout_text = result.stdout.decode("utf-8", errors="replace")
            stderr_text = result.stderr.decode("utf-8", errors="replace")
            structured: dict[str, object] = {}
            try:
                decoded = json.loads(stdout_text) if stdout_text.strip() else {}
                if isinstance(decoded, dict):
                    structured = decoded
            except json.JSONDecodeError:
                structured = {}
            reported_validations = structured.get("validations") if isinstance(structured.get("validations"), dict) else {}
            validations = {name: reported_validations.get(name) is True for name in output.validations}
            reported_metrics = structured.get("metrics") if isinstance(structured.get("metrics"), dict) else {}
            metrics = {key: value for key, value in reported_metrics.items() if isinstance(key, str) and isinstance(value, (str, int, float, bool))}
            artifacts = list(output.artifacts)
            artifacts.extend([
                ProducedArtifact(logical_name="execution.stdout.log", kind=ArtifactKind.LOG, media_type="text/plain", text=stdout_text),
                ProducedArtifact(logical_name="execution.stderr.log", kind=ArtifactKind.LOG, media_type="text/plain", text=stderr_text),
                ProducedArtifact(logical_name="execution.result.json", kind=ArtifactKind.OUTPUT, media_type="application/json", text=json.dumps({"validations": validations, "metrics": metrics, "structured": bool(structured)}, ensure_ascii=False, sort_keys=True)),
            ])
            artifacts.extend(self._collect_outputs(activity_workspace))
            updated = output.model_copy(update={
                "execution_succeeded": succeeded,
                "validations": validations,
                "metrics": {**output.metrics, **metrics, "sandbox_exit_code": result.exit_code, "sandbox_timed_out": result.timed_out},
                "artifacts": artifacts,
            })
            return response.model_copy(update={"output": updated, "executor_metadata": {**response.executor_metadata, "execution_backend": "local-sandbox", "sandbox_image": self.image_digest, "sandbox_exit_code": result.exit_code, "sandbox_timed_out": result.timed_out}})
        except Exception as exc:
            return self._with_execution_result(response, output, False, {"sandbox_error": str(exc)[:1000]})
        finally:
            shutil.rmtree(self.sandbox.workspace_root / "activities" / str(request.activity_id), ignore_errors=True)

    @staticmethod
    def _source_artifact(output: CodingStageOutput) -> ProducedArtifact | None:
        for artifact in output.artifacts:
            if artifact.logical_name.lower().endswith(".py") or artifact.media_type in {"text/x-python", "text/python"} or artifact.metadata.get("role") == "source_code":
                return artifact
        return None

    def _materialize_inputs(self, request: ActivityRequest, activity_workspace: Path) -> None:
        input_dir = activity_workspace / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, str | int]] = []
        total_input = 0
        for artifact in request.inputs:
            data = self.artifact_store.read(artifact)
            total_input += len(data)
            if total_input > 200 * 1024 * 1024:
                raise ValueError("materialized Activity inputs exceed the 200 MiB Sandbox budget")
            name = Path(artifact.logical_name).name or "input.bin"
            target = input_dir / f"{artifact.sha256[:12]}-{name}"
            target.write_bytes(data)
            manifest.append({"logical_name": artifact.logical_name, "path": f"inputs/{target.name}", "sha256": artifact.sha256, "size_bytes": len(data)})
        (activity_workspace / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _artifact_bytes(artifact: ProducedArtifact) -> bytes:
        if artifact.text is not None:
            return artifact.text.encode("utf-8")
        return base64.b64decode(artifact.base64 or "", validate=True)

    @staticmethod
    def _collect_outputs(activity_workspace: Path) -> list[ProducedArtifact]:
        collected: list[ProducedArtifact] = []
        total = 0
        for path in sorted(activity_workspace.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.is_symlink() or path.name == "model.py" or "inputs" in path.relative_to(activity_workspace).parts or path.name == "input_manifest.json":
                continue
            resolved = path.resolve()
            if activity_workspace not in resolved.parents:
                continue
            size = path.stat().st_size
            if size > 10 * 1024 * 1024 or total + size > 50 * 1024 * 1024 or len(collected) >= 100:
                continue
            data = path.read_bytes()
            total += len(data)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            try:
                text = data.decode("utf-8")
                collected.append(ProducedArtifact(logical_name=f"execution/{path.relative_to(activity_workspace).as_posix()}", kind=ArtifactKind.OUTPUT, media_type=media_type, text=text, metadata={"sha256": hashlib.sha256(data).hexdigest(), "source": "local-sandbox"}))
            except UnicodeDecodeError:
                collected.append(ProducedArtifact(logical_name=f"execution/{path.relative_to(activity_workspace).as_posix()}", kind=ArtifactKind.OUTPUT, media_type=media_type, base64=base64.b64encode(data).decode("ascii"), metadata={"sha256": hashlib.sha256(data).hexdigest(), "source": "local-sandbox"}))
        return collected

    @staticmethod
    def _with_execution_result(response: ActivityResponse, output: CodingStageOutput, succeeded: bool, metadata: dict[str, object]) -> ActivityResponse:
        updated = output.model_copy(update={"execution_succeeded": succeeded, "validations": {name: False for name in output.validations}, "artifacts": [*output.artifacts, ProducedArtifact(logical_name="execution.error.log", kind=ArtifactKind.LOG, media_type="text/plain", text=json.dumps(metadata, ensure_ascii=False, sort_keys=True))]})
        return response.model_copy(update={"output": updated, "executor_metadata": {**response.executor_metadata, "execution_backend": "local-sandbox", **metadata}})
