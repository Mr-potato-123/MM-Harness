"""Trusted local subprocess execution adapter.

This adapter is intentionally explicit about its security level: it provides
fixed-argv execution, scrubbed environment, workspace cwd, and output
budgets. It is the default execution boundary for the trusted single-user
local Harness. A hostile multi-tenant deployment must replace it with a
container/VM implementation of the same ``SandboxClient`` port.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from m2harness.ports.sandbox import ExecResult, ExecSpec, SandboxSpec


@dataclass(frozen=True)
class LocalExecOutput:
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    cancelled: bool = False


def _run_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int | None,
    max_output_bytes: int,
    cancel_check: Callable[[], bool] | None,
) -> LocalExecOutput:
    """Run a child while remaining responsive to both deadline and operator stop."""

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    stdout = bytearray()
    stderr = bytearray()

    def drain(stream, target: bytearray) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                if len(target) < max_output_bytes:
                    target.extend(chunk[: max_output_bytes - len(target)])
        except (OSError, ValueError):
            return

    readers = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    timed_out = False
    cancelled = False
    try:
        while process.poll() is None:
            if cancel_check is not None:
                try:
                    cancelled = bool(cancel_check())
                except Exception:
                    cancelled = False
            if cancelled:
                process.terminate()
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.05)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    finally:
        for reader in readers:
            reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return LocalExecOutput(
        tuple(argv), process.returncode, timed_out, bytes(stdout), bytes(stderr), cancelled=cancelled,
    )


class LocalSandboxClient:
    def __init__(self, workspace_root: Path, *, max_output_bytes: int = 1_000_000, allow_host_processes: bool = False) -> None:
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_output_bytes = max_output_bytes
        self.allow_host_processes = allow_host_processes
        # This is deliberately not presented as a reproducible container
        # identity. It records that the first-mile run used the trusted host.
        self.image_digest = "local-host-process"
        self._sandboxes: dict[uuid.UUID, SandboxSpec] = {}

    @property
    def available(self) -> bool:
        return self.allow_host_processes

    def create(self, spec: SandboxSpec) -> uuid.UUID:
        if not self.available:
            raise RuntimeError("host_execution_disabled: enable the trusted local backend or configure Docker/VM")
        if spec.network not in {"deny", "allowlisted"}:
            raise PermissionError("local sandbox refuses unrestricted network")
        sandbox_id = uuid.uuid4()
        self._sandboxes[sandbox_id] = spec
        return sandbox_id

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int | None = None, env: Mapping[str, str] | None = None, cwd: Path | None = None, cancel_check: Callable[[], bool] | None = None) -> LocalExecOutput:
        if not self.available:
            raise RuntimeError("host_execution_disabled: enable the trusted local backend or configure Docker/VM")
        if not argv or any(not item or "\x00" in item for item in argv):
            raise ValueError("sandbox argv must be non-empty and NUL-free")
        inherited = ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        safe_env = {name: os.environ[name] for name in inherited if name in os.environ}
        # The host backend is only for the trusted single-user scope. Keep the
        # environment narrow so a model cannot smuggle dynamic
        # loader, proxy, or provider-secret variables into the child process.
        allowed_overrides = {"PATH", "PYTHONIOENCODING", "M2HARNESS_NETWORK"}
        safe_env.update({key: value for key, value in (env or {}).items() if key in allowed_overrides})
        working_directory = self.workspace_root if cwd is None else cwd.resolve()
        if self.workspace_root != working_directory and self.workspace_root not in working_directory.parents:
            raise ValueError("sandbox working directory escapes workspace")
        working_directory.mkdir(parents=True, exist_ok=True)
        return _run_process(
            tuple(argv), cwd=working_directory, env=safe_env,
            timeout_seconds=timeout_seconds, max_output_bytes=self.max_output_bytes,
            cancel_check=cancel_check,
        )

    def execute(self, sandbox_id: uuid.UUID, spec: ExecSpec) -> ExecResult:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            raise KeyError(f"sandbox not found: {sandbox_id}")
        if spec.timeout_seconds > sandbox.timeout_seconds:
            raise ValueError("execution timeout exceeds sandbox timeout")
        output = self.run(spec.argv, timeout_seconds=spec.timeout_seconds, env=spec.env)
        return ExecResult(sandbox_id=sandbox_id, exit_code=output.exit_code, timed_out=output.timed_out, image_digest=sandbox.image_digest)

    def destroy(self, sandbox_id: uuid.UUID) -> None:
        self._sandboxes.pop(sandbox_id, None)


class DockerSandboxClient(LocalSandboxClient):
    """Docker-backed implementation used when Docker Desktop/Engine exists."""

    def __init__(self, workspace_root: Path, *, image: str = "python:3.12-slim", max_output_bytes: int = 1_000_000) -> None:
        super().__init__(workspace_root, max_output_bytes=max_output_bytes, allow_host_processes=False)
        self.image = image
        self.docker = shutil.which("docker")
        self._available = bool(self.docker and self._docker_ready() and self._image_ready())
        if self._available:
            self.image_digest = self._image_identity() or image

    @property
    def available(self) -> bool:
        return self._available

    def _docker_ready(self) -> bool:
        try:
            result = subprocess.run([self.docker or "docker", "info"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, shell=False, check=False)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _image_ready(self) -> bool:
        try:
            result = subprocess.run([self.docker or "docker", "image", "inspect", self.image], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, shell=False, check=False)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _image_identity(self) -> str | None:
        """Return Docker's immutable local image ID for audit evidence."""
        try:
            result = subprocess.run(
                [self.docker or "docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=10, shell=False, check=False, text=True,
            )
            identity = result.stdout.strip()
            return identity if result.returncode == 0 and identity.startswith("sha256:") else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int | None = None, env: Mapping[str, str] | None = None, cwd: Path | None = None, cancel_check: Callable[[], bool] | None = None) -> LocalExecOutput:
        if not self.available:
            raise RuntimeError("sandbox_unavailable: Docker daemon is not ready")
        if not argv:
            raise ValueError("sandbox argv must be non-empty")
        mapped: list[str] = []
        for item in argv:
            if item == sys.executable or Path(item).name.lower().startswith("python"):
                mapped.append("python")
                continue
            path = Path(item)
            if path.is_absolute():
                resolved = path.resolve()
                if self.workspace_root not in resolved.parents:
                    raise ValueError("sandbox argv path escapes workspace")
                mapped.append("/workspace/" + resolved.relative_to(self.workspace_root).as_posix())
            else:
                mapped.append(item)
        working_directory = self.workspace_root if cwd is None else cwd.resolve()
        if self.workspace_root != working_directory and self.workspace_root not in working_directory.parents:
            raise ValueError("sandbox working directory escapes workspace")
        working_directory.mkdir(parents=True, exist_ok=True)
        mapped_cwd = "/workspace" if working_directory == self.workspace_root else "/workspace/" + working_directory.relative_to(self.workspace_root).as_posix()
        command = [
            self.docker or "docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--cpus", "1", "--memory", "2g", "--pids-limit", "128", "--ulimit", "nofile=1024:1024",
            "-v", f"{self.workspace_root}:/workspace:rw", "-w", mapped_cwd, self.image, *mapped,
        ]
        safe_env = {key: value for key, value in (env or {}).items() if key in {"PYTHONIOENCODING", "M2HARNESS_NETWORK"}}
        image_index = command.index(self.image)
        for key, value in safe_env.items():
            command[image_index:image_index] = ["-e", f"{key}={value}"]
            image_index += 2
        return _run_process(
            tuple(command), cwd=self.workspace_root, env={"PATH": os.environ.get("PATH", "")},
            timeout_seconds=timeout_seconds, max_output_bytes=self.max_output_bytes,
            cancel_check=cancel_check,
        )
