"""Small, local-only human control plane for a running solve tool.

The control plane is deliberately outside the model/provider protocol. A toy
UI can append a suggestion or interrupt command while the Main Harness remains
the owner of workflow state. The solve tool consumes commands at safe stage
boundaries; the sandbox also polls the interrupt bit while a generated script
is running.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


ControlKind = Literal["suggestion", "interrupt"]


class HumanInterruptRequested(RuntimeError):
    """Raised internally when an operator asks the current solve to stop."""

    def __init__(self, run_id: str, task_id: str, iteration: int | None, reason: str, *, context: Any = None) -> None:
        super().__init__(reason)
        self.run_id = run_id
        self.task_id = task_id
        self.iteration = iteration
        self.reason = reason
        self.context = context


@dataclass(frozen=True)
class HumanCommand:
    sequence: int
    command_id: str
    run_id: str
    kind: ControlKind
    message: str
    created_at: str
    task_id: str | None = None


class HumanControlStore:
    """File-backed control/status store suitable for one local operator."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.root = self.workspace_root / ".m2harness-control"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def submit_suggestion(self, run_id: str, message: str, *, task_id: str | None = None) -> HumanCommand:
        return self._append(run_id, "suggestion", message, task_id=task_id)

    def request_interrupt(self, run_id: str, reason: str = "operator requested interrupt", *, task_id: str | None = None) -> HumanCommand:
        return self._append(run_id, "interrupt", reason, task_id=task_id)

    def pending(self, run_id: str, *, after_sequence: int = 0) -> tuple[HumanCommand, ...]:
        result: list[HumanCommand] = []
        for payload in self._read_commands(run_id):
            try:
                sequence = int(payload.get("sequence", 0))
                if sequence <= after_sequence:
                    continue
                kind = payload.get("kind")
                if kind not in {"suggestion", "interrupt"}:
                    continue
                message = str(payload.get("message", "")).strip()
                if not message:
                    continue
                result.append(HumanCommand(
                    sequence=sequence,
                    command_id=str(payload.get("command_id", "")),
                    run_id=run_id,
                    kind=kind,
                    message=message[:8_000],
                    created_at=str(payload.get("created_at", "")),
                    task_id=str(payload["task_id"]) if payload.get("task_id") else None,
                ))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(result, key=lambda item: item.sequence))

    def is_interrupted(self, run_id: str) -> bool:
        return any(item.kind == "interrupt" for item in self.pending(run_id))

    def publish_status(self, run_id: str, **fields: Any) -> None:
        payload = {"run_id": run_id, "updated_at": datetime.now(UTC).isoformat(), **fields}
        target = self._status_path(run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".status-", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def read_status(self, run_id: str) -> dict[str, Any] | None:
        target = self._status_path(run_id)
        if not target.is_file() or target.is_symlink():
            return None
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def list_status(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.status.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            run_id = path.name.removesuffix(".status.json")
            value = self.read_status(run_id)
            if value is not None:
                result.append(value)
        return result

    def probe_events(self, run_id: str, *, limit: int = 250) -> list[dict[str, Any]]:
        normalized = self._safe_run_id(run_id)
        target = (self.workspace_root / "reports" / "runs" / normalized / "probe.ndjson").resolve()
        allowed_root = (self.workspace_root / "reports" / "runs").resolve()
        if allowed_root not in target.parents or not target.is_file() or target.is_symlink():
            return []
        try:
            lines = target.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 1_000)):]
        except OSError:
            return []
        result: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def _append(self, run_id: str, kind: ControlKind, message: str, *, task_id: str | None) -> HumanCommand:
        normalized = self._safe_run_id(run_id)
        text = message.strip()
        if not text:
            raise ValueError("human control message cannot be empty")
        if len(text) > 8_000:
            raise ValueError("human control message exceeds 8,000 characters")
        with self._lock:
            existing = self._read_commands(normalized)
            sequence = max((int(item.get("sequence", 0)) for item in existing), default=0) + 1
            created_at = datetime.now(UTC).isoformat()
            command_id = f"{normalized}-{sequence}"
            payload = {
                "sequence": sequence,
                "command_id": command_id,
                "run_id": normalized,
                "kind": kind,
                "message": text,
                "created_at": created_at,
                "task_id": task_id,
            }
            target = self._command_path(normalized)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return HumanCommand(sequence, command_id, normalized, kind, text, created_at, task_id)

    def _read_commands(self, run_id: str) -> list[dict[str, Any]]:
        target = self._command_path(run_id)
        if not target.is_file() or target.is_symlink():
            return []
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        result: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def _command_path(self, run_id: str) -> Path:
        return self.root / f"{self._safe_run_id(run_id)}.ndjson"

    def _status_path(self, run_id: str) -> Path:
        return self.root / f"{self._safe_run_id(run_id)}.status.json"

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        normalized = str(run_id).strip()
        if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", normalized):
            raise ValueError("invalid human-control run id")
        return normalized
