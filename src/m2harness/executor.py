"""Activity execution boundary for independently deployable agent workers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from pydantic import ValidationError

from m2harness.errors import ActivityExecutionError, ConfigurationError
from m2harness.models import ActivityRequest, ActivityResponse


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
