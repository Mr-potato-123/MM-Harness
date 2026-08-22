"""DeepSeek Harness (DSH) Code Agent adapter.

This module is a narrow process boundary, not a second orchestration engine.
DSH owns the Code Agent's durable session, tool loop, compaction and model
conversation.  M2Harness remains the authority for the modeling contract,
the two-revision policy, local execution, evidence gates and publication.

The adapter speaks the DSH SDK JSON-RPC shape directly so the Python project
does not need to copy DSH's implementation or depend on its TypeScript
package layout.  The configured command must be a DSH JSON-RPC runtime (the
official runtime entry point is commonly named ``dsh-jsonrpc-agent``).  A
runtime is started once per provider instance and one DSH session is reused
for all iterations of a task.  Notifications are append-only NDJSON artifacts
for monitoring and review; this adapter never kills or interrupts a running
runtime because a local observation window elapsed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from queue import Empty, Queue
from typing import Any, Callable, Mapping
from uuid import uuid4

from m2harness.domain.code import CodeProposal
from m2harness.application.code_agent_prompts import CODE_AGENT_SYSTEM_PROMPT, build_code_agent_task_prompt
from m2harness.domain.solve_problem import (
    SolveProblemContext,
    SolveProblemTask,
    UnifiedModelingReport,
)
from m2harness.errors import ActivityExecutionError
from m2harness.infrastructure.code_harness import CodeProposalProvider


class DshRuntimeError(ActivityExecutionError):
    """The DSH runtime or its wire protocol failed."""


@dataclass(frozen=True)
class DshRuntimeConfig:
    """Explicit launch and routing configuration for a DSH runtime."""

    command: tuple[str, ...] = ("dsh-jsonrpc-agent",)
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None

    @classmethod
    def from_environment(cls, *, cwd: Path, model: str | None = None) -> "DshRuntimeConfig":
        """Build config without silently installing or downloading anything.

        ``M2HARNESS_DSH_RUNTIME_COMMAND_JSON`` is preferred because it is
        unambiguous on Windows.  The string variant is retained for operators
        launching ``node .../bin.ts`` during local development.
        """

        raw_json = os.environ.get("M2HARNESS_DSH_RUNTIME_COMMAND_JSON", "").strip()
        command: tuple[str, ...]
        if raw_json:
            try:
                value = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise ValueError("M2HARNESS_DSH_RUNTIME_COMMAND_JSON must be a JSON string array") from exc
            if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                raise ValueError("M2HARNESS_DSH_RUNTIME_COMMAND_JSON must be a non-empty string array")
            command = tuple(value)
        else:
            raw = os.environ.get("M2HARNESS_DSH_RUNTIME_COMMAND", "dsh-jsonrpc-agent").strip()
            command = tuple(shlex.split(raw, posix=os.name != "nt"))
        if not command:
            raise ValueError("DSH runtime command cannot be empty")
        max_tokens_raw = os.environ.get("M2HARNESS_DSH_MAX_TOKENS", "").strip()
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("M2HARNESS_DSH_MAX_TOKENS must be positive")
        env = dict(os.environ)
        # Explicit overrides are copied; the API key is inherited by the child
        # but is never written into event records or handoff files.
        for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DSH_CORDIS_CONFIG", "DSH_HOME"):
            if key in os.environ:
                env[key] = os.environ[key]
        return cls(
            command=command,
            cwd=cwd.resolve(),
            env=env,
            provider=os.environ.get("M2HARNESS_DSH_PROVIDER", "deepseek-official"),
            model=model or os.environ.get("M2HARNESS_DSH_MODEL", "deepseek-v4-flash"),
            max_tokens=max_tokens,
        )


@dataclass(frozen=True)
class DshRun:
    session_id: str
    final_response: str
    events: tuple[dict[str, Any], ...]
    event_log: Path


class _JsonRpcRuntime:
    """Small, typed-enough JSON-RPC peer for the DSH SDK wire contract."""

    def __init__(self, config: DshRuntimeConfig, event_sink: Callable[[dict[str, Any]], None]) -> None:
        self.config = config
        self.event_sink = event_sink
        self.process: subprocess.Popen[str] | None = None
        self._responses: dict[str, Queue[object]] = {}
        self._notifications: Queue[dict[str, Any] | BaseException] = Queue()
        self._write_lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._counter = 0
        self._reader: threading.Thread | None = None
        self._stderr: threading.Thread | None = None
        self._stderr_tail: list[str] = []

    def start(self) -> None:
        if self.process is not None:
            if self.process.poll() is not None:
                raise DshRuntimeError(self._diagnostic("DSH runtime has exited"))
            return
        command = list(self.config.command)
        executable = command[0]
        if not Path(executable).is_absolute() and shutil.which(executable) is None:
            raise DshRuntimeError(
                f"DSH runtime executable not found: {executable}; install the DSH runtime or set "
                "M2HARNESS_DSH_RUNTIME_COMMAND_JSON"
            )
        try:
            process = subprocess.Popen(
                command,
                cwd=None if self.config.cwd is None else str(self.config.cwd),
                env=dict(self.config.env) if self.config.env is not None else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
        except OSError as exc:
            raise DshRuntimeError(f"unable to launch DSH runtime: {exc}") from exc
        self.process = process
        self._reader = threading.Thread(target=self._read_stdout, name="m2h-dsh-reader", daemon=True)
        self._stderr = threading.Thread(target=self._read_stderr, name="m2h-dsh-stderr", daemon=True)
        self._reader.start()
        self._stderr.start()

    def initialize(self, *, cwd: Path) -> None:
        params: dict[str, Any] = {
            "cwd": str(cwd.resolve()),
            "provider": self.config.provider,
            "model": self.config.model,
        }
        if self.config.max_tokens is not None:
            params["maxTokens"] = self.config.max_tokens
        response = self.request("initialize", params)
        info = response.get("serverInfo") if isinstance(response, dict) else None
        if not isinstance(info, dict) or not isinstance(info.get("name"), str):
            raise DshRuntimeError(f"DSH initialize returned invalid server identity: {response!r}")

    def run(self, session_id: str, content: str) -> DshRun:
        result = self.request("session/prompt", {
            "sessionId": session_id,
            "contentBlocks": [{"type": "text", "text": content}],
        })
        message_id = result.get("messageId") if isinstance(result, dict) else None
        if not isinstance(message_id, str) or not message_id:
            raise DshRuntimeError(f"DSH session/prompt returned no messageId: {result!r}")
        events: list[dict[str, Any]] = []
        received = False
        while True:
            item = self._notifications.get()
            if isinstance(item, BaseException):
                raise DshRuntimeError(self._diagnostic("DSH notification stream closed")) from item
            method = item.get("method")
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            if method == "session.event" and params.get("sessionId") == session_id:
                event = params.get("event")
                if isinstance(event, dict):
                    events.append(event)
                    if self._is_receipt(event, message_id):
                        received = True
            if received and method == "session.status" and params.get("sessionId") == session_id and params.get("status") == "idle":
                break
        final_response = _final_response(events)
        return DshRun(session_id=session_id, final_response=final_response, events=tuple(events), event_log=Path())

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.start()
        process = self.process
        if process is None or process.stdin is None:
            raise DshRuntimeError("DSH runtime is not running")
        with self._response_lock:
            self._counter += 1
            request_id = str(self._counter)
            waiter: Queue[object] = Queue(maxsize=1)
            self._responses[request_id] = waiter
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
                process.stdin.flush()
        except Exception as exc:
            with self._response_lock:
                self._responses.pop(request_id, None)
            raise DshRuntimeError(self._diagnostic(f"failed to write DSH request {method}")) from exc
        try:
            item = waiter.get(timeout=self.config.request_timeout_seconds)
        except Empty as exc:
            # Observation timeout is deliberately non-destructive.  The
            # runtime remains alive and the caller may continue monitoring it.
            raise DshRuntimeError(self._diagnostic(f"observation window elapsed waiting for DSH {method}; runtime was not interrupted")) from exc
        if isinstance(item, BaseException):
            raise DshRuntimeError(self._diagnostic(f"DSH request {method} failed")) from item
        if not isinstance(item, dict):
            raise DshRuntimeError(f"DSH request {method} returned a non-object response")
        if isinstance(item.get("error"), dict):
            raise DshRuntimeError(f"DSH request {method} failed: {item['error']}")
        value = item.get("result")
        if not isinstance(value, dict):
            raise DshRuntimeError(f"DSH request {method} returned invalid result: {value!r}")
        return value

    @staticmethod
    def _is_receipt(event: dict[str, Any], message_id: str) -> bool:
        if event.get("type") != "agent/inbox/spliced":
            return False
        data = event.get("data")
        inserted = data.get("inserted") if isinstance(data, dict) else None
        return isinstance(inserted, list) and any(isinstance(item, dict) and item.get("id") == message_id for item in inserted)

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    self.event_sink({"stream": "stdout", "line": line.rstrip(), "protocol": "invalid-json"})
                    continue
                if not isinstance(frame, dict):
                    continue
                self.event_sink({"stream": "protocol", "frame": frame})
                if isinstance(frame.get("id"), (str, int)) and "method" not in frame:
                    key = str(frame["id"])
                    with self._response_lock:
                        waiter = self._responses.pop(key, None)
                    if waiter is not None:
                        waiter.put(frame)
                elif isinstance(frame.get("method"), str):
                    self._notifications.put(frame)
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._fail(DshRuntimeError("DSH runtime stdout closed"))

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            for line in process.stderr:
                cleaned = line.rstrip()
                if cleaned:
                    self._stderr_tail.append(cleaned)
                    self._stderr_tail[:] = self._stderr_tail[-100:]
                    self.event_sink({"stream": "stderr", "line": cleaned})
        except BaseException as exc:
            self._fail(exc)

    def _fail(self, error: BaseException) -> None:
        with self._response_lock:
            waiters = list(self._responses.values())
            self._responses.clear()
        for waiter in waiters:
            waiter.put(error)
        self._notifications.put(error)

    def _diagnostic(self, message: str) -> str:
        process = self.process
        suffix = ""
        if process is not None and process.poll() is not None:
            suffix = f"; exit_code={process.returncode}"
        if self._stderr_tail:
            suffix += "\nstderr_tail:\n" + "\n".join(self._stderr_tail[-20:])
        return message + suffix


class DshCodeProposalProvider(CodeProposalProvider):
    """Use a persistent DSH Code Agent session to materialize a proposal."""

    def __init__(self, workspace_root: Path, *, config: DshRuntimeConfig | None = None, event_dir: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.config = config or DshRuntimeConfig.from_environment(cwd=self.workspace_root)
        self.event_dir = (event_dir or (self.workspace_root / ".m2harness-code" / "dsh-events")).resolve()
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, _JsonRpcRuntime] = {}
        self._initialized: set[str] = set()
        self._event_files: dict[str, Path] = {}
        self._event_sequence: dict[str, int] = {}
        self._lock = threading.Lock()

    def propose(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int) -> CodeProposal:
        raw_run_id = str(context.metadata.get("run_id", "unscoped")) if isinstance(context.metadata, dict) else "unscoped"
        raw_run_name = str(context.metadata.get("run_name", raw_run_id)) if isinstance(context.metadata, dict) else raw_run_id
        run_path_key = str(context.metadata.get("run_path_key", raw_run_id)) if isinstance(context.metadata, dict) else raw_run_id
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_path_key)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_run_id)
        attempt = int(context.metadata.get("task_attempt", 1)) if isinstance(context.metadata, dict) else 1
        session_key = f"{safe_run_id}:{task.task_id}:attempt-{attempt}"
        event_file = self.event_dir / safe_run_name / f"task-{task.task_id}__attempt-{attempt}.ndjson"
        event_file.parent.mkdir(parents=True, exist_ok=True)
        self._event_files[session_key] = event_file

        def sink(event: dict[str, Any]) -> None:
            with self._lock:
                sequence = self._event_sequence.get(session_key, 0) + 1
                self._event_sequence[session_key] = sequence
            record = {
                "seq": sequence,
                "occurred_at": datetime.now(UTC).isoformat(),
                "task_id": task.task_id,
                "iteration": iteration,
                **event,
            }
            # Credentials never enter frames, but redact common accidental
            # fields before durable materialization as a defense in depth.
            safe = _redact(record)
            with event_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n")

        runtime = self._sessions.get(session_key)
        if runtime is None:
            runtime = _JsonRpcRuntime(self.config, sink)
            self._sessions[session_key] = runtime
        if session_key not in self._initialized:
            runtime.initialize(cwd=self.workspace_root)
            self._initialized.add(session_key)

        logical_name = "solve_" + "".join(char if char.isalnum() or char in "_-" else "_" for char in task.task_id) + ".py"
        source_path = PurePosixPath(".m2harness-code") / "runs" / safe_run_name / f"task-{task.task_id}" / f"attempt-{attempt}" / f"iteration-{iteration}" / logical_name
        prompt = self._prompt(task, context, modeling, iteration=iteration, logical_name=logical_name, source_path=source_path.as_posix())
        prompt_path = self.workspace_root / Path(*source_path.parts).parent / "code-agent-prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        external_session_id = f"m2h-code-{safe_run_id}-{task.task_id}-attempt-{attempt}"
        session_manifest_path, prompt_index_path = self._record_session_context(
            prompt_path=prompt_path,
            prompt=prompt,
            event_file=event_file,
            session_id=external_session_id,
            run_id=raw_run_id,
            run_name=raw_run_name,
            task_id=task.task_id,
            attempt=attempt,
            iteration=iteration,
        )
        run = runtime.run(external_session_id, prompt)
        value = _parse_json_object(run.final_response)
        handoff = value.get("m2harness_handoff", value)
        if not isinstance(handoff, dict):
            raise DshRuntimeError("DSH Code Agent final response did not contain a handoff object")
        returned_path = handoff.get("source_path")
        if not isinstance(returned_path, str) or not returned_path.strip():
            raise DshRuntimeError("DSH Code Agent handoff omitted source_path")
        normalized = returned_path.replace("\\", "/")
        target = (self.workspace_root / normalized).resolve()
        if self.workspace_root != target and self.workspace_root not in target.parents:
            raise DshRuntimeError("DSH Code Agent source_path escapes the workspace")
        if target.suffix.lower() != ".py" or not target.is_file() or target.is_symlink():
            raise DshRuntimeError(f"DSH Code Agent source_path is not a regular Python file: {returned_path}")
        source = target.read_text(encoding="utf-8")
        expected = handoff.get("expected_validations", modeling.required_validations)
        if not isinstance(expected, list) or not all(isinstance(item, str) and item for item in expected):
            raise DshRuntimeError("DSH Code Agent expected_validations must be a string array")
        timeout_seconds = handoff.get("timeout_seconds")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 86_400:
            raise DshRuntimeError("DSH Code Agent handoff must include timeout_seconds between 1 and 86400")
        return CodeProposal(
            source=source,
            logical_name=target.name,
            timeout_seconds=timeout_seconds,
            expected_validations=tuple(dict.fromkeys(expected)),
            metadata={
                "agent_runtime": "dsh",
                "session_id": external_session_id,
                "source_path": target.relative_to(self.workspace_root).as_posix(),
                "event_log": str(event_file.relative_to(self.workspace_root).as_posix()),
                "prompt_file": prompt_path.relative_to(self.workspace_root).as_posix(),
                "session_manifest": session_manifest_path.relative_to(self.workspace_root).as_posix(),
                "prompt_index": prompt_index_path.relative_to(self.workspace_root).as_posix(),
            },
        )

    def _record_session_context(
        self,
        *,
        prompt_path: Path,
        prompt: str,
        event_file: Path,
        session_id: str,
        run_id: str,
        run_name: str,
        task_id: str,
        attempt: int,
        iteration: int,
    ) -> tuple[Path, Path]:
        """Persist local pointers without becoming a second conversation store.

        DSH remains the canonical owner of the session event log. These two
        files are deliberately small M2Harness audit artifacts: one immutable
        manifest per session and one append-only index of prompt snapshots.
        """

        prompt_path = prompt_path.resolve()
        event_file = event_file.resolve()
        session_dir = prompt_path.parent.parent
        manifest_path = session_dir / "session.json"
        prompt_index_path = session_dir / "prompt-index.ndjson"
        session_dir.mkdir(parents=True, exist_ok=True)

        if not manifest_path.exists():
            manifest = {
                "schema_version": 1,
                "kind": "dsh-session-manifest",
                "context_owner": "dsh-runtime",
                "context_model": "append-only session event log; messages are derived by DSH",
                "session_id": session_id,
                "run_id": run_id,
                "run_name": run_name,
                "task_id": task_id,
                "attempt": attempt,
                "provider": self.config.provider,
                "model": self.config.model,
                "cwd": self.workspace_root.as_posix(),
                "wire_observation_log": event_file.resolve().relative_to(self.workspace_root).as_posix(),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DshRuntimeError(f"DSH session manifest is unreadable: {manifest_path}") from exc
            if not isinstance(existing_manifest, dict) or (
                existing_manifest.get("run_id") != run_id
                or existing_manifest.get("session_id") != session_id
            ):
                raise DshRuntimeError(
                    "refusing to reuse a DSH session directory with a different run_id or session_id: "
                    f"{session_dir}"
                )

        prompt_record = {
            "schema_version": 1,
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "attempt": attempt,
            "iteration": iteration,
            "kind": "bootstrap" if iteration == 1 else "delta",
            "prompt_file": prompt_path.relative_to(self.workspace_root).as_posix(),
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        with prompt_index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(prompt_record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return manifest_path, prompt_index_path

    def _prompt(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int, logical_name: str, source_path: str) -> str:
        return CODE_AGENT_SYSTEM_PROMPT + "\n\n" + build_code_agent_task_prompt(
            task,
            context,
            modeling,
            iteration=iteration,
            target_relative=source_path,
            output_relative=source_path.rsplit("/", 1)[0] + "/outputs",
            session_managed=True,
        ) + "\n\n最终只返回 JSON 交接对象：source_path、logical_name、expected_validations、timeout_seconds。"
        allowlisted = [
            {"path": item.relative_path, "purpose": item.purpose, "role": item.role.value, "sha256": item.sha256}
            for item in context.readonly_files
        ]
        exchange_paths = sorted(
            item.relative_path for item in context.readonly_files
            if "/exchanges/" in item.relative_path.replace("\\", "/")
        )
        if iteration == 1:
            modeling_contract: dict[str, Any] = modeling.model_dump(mode="json")
        else:
            initial_contract = next((path for path in exchange_paths if path.endswith("model-to-code.md")), "首轮 model-to-code.md（按白名单路径读取）")
            repair_contract = next((path for path in reversed(exchange_paths) if path.endswith("model-to-code-revision.md")), "最新 model-to-code-revision.md（按白名单路径读取）")
            modeling_contract = {
                "continuation": "沿用首轮已锁定的建模契约；本轮只执行 Code 返修，不重新建模。",
                "initial_contract_path": initial_contract,
                "latest_repair_handoff_path": repair_contract,
                "required_validations": list(modeling.required_validations),
                "expected_outputs": list(modeling.expected_outputs),
                "expected_figures": list(modeling.expected_figures),
            }
        disclosed = [
            {"path": item.relative_path, "purpose": item.purpose, "content": item.content, "truncated": item.truncated}
            for item in context.disclosed_text_files
        ]
        payload = {
            "task": {"task_id": task.task_id, "title": task.title, "problem": task.problem, "requested_outputs": task.requested_outputs},
            "iteration": iteration,
            "modeling_contract": modeling_contract,
            "allowlisted_readonly_paths": allowlisted,
            "already_disclosed_text": disclosed,
            "model_to_code_repair_instructions": list(context.instructions[-32:]),
            "target_source_path": source_path,
            "output_contract": {
                "logical_name": logical_name,
                "timeout_seconds": "required integer between 1 and 86400; every generated script must have a finite execution budget",
                "required_validations": list(modeling.required_validations),
                "required_evidence": "脚本 stdout 必须包含可审查的中文 Markdown 执行报告；JSON validations/validation_evidence 仅作辅助索引，false 不得单独作为失败依据。",
            },
        }
        header = (
            "MANDATORY: every generated script execution must use a finite timeout_seconds integer from 1 through 86400, and the final handoff must include that field. Never omit it or use null.\n"
            "你是 M2Harness 的 Code Agent，运行在 DeepSeek Harness（DSH）持久 Session 中。全流程使用中文。\n"
            "Model Agent 已锁定建模契约；你只负责把本题当前范围实现为可执行、可复现的 Python 源文件，不得重新定义问题、替换主方案或处理后续题目。\n"
            f"这是第 {iteration} 轮；继续使用本 DSH Session 的既有上下文。Model Agent 上下文与你隔离，只通过本次结构化交接和下方明确列出的只读路径传入。\n"
            "只读渐进披露：只能打开 allowlisted_readonly_paths 中列出的路径；路径本身不等于授权其他目录。需要更多内容时先读取清单中的路径，不能扫描工作区或凭空声称文件内容。\n"
            "使用 DSH 自带文件/命令工具写入和检查目标源文件。所有生成输出只能写到目标源文件所在 iteration 目录的 outputs 子目录。不得访问网络、密钥、主 Harness 数据库或发布工具。\n"
            "完成一次完整实现后，只做必需验证；如果验证失败，基于实际错误定点修复，最多遵守上游返修预算，不进行无界的求解器重写。\n"
            "源文件写入并验证后，最终回答必须只包含一个 JSON 对象（可用 m2harness_handoff 字段包裹），字段为 source_path、logical_name、expected_validations；默认不设置执行时限，不要把源代码放进最终回答。\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        return header


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else candidate
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                value = parsed
        if value is None:
            raise DshRuntimeError("DSH final response did not contain valid JSON")
    if not isinstance(value, dict):
        raise DshRuntimeError("DSH final response must be one JSON object")
    return value


def _final_response(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        message = data.get("message") if isinstance(data, dict) else None
        owner = message if isinstance(message, dict) else data if isinstance(data, dict) else {}
        content = owner.get("content") if isinstance(owner, dict) else None
        if not isinstance(content, list):
            continue
        return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text")
    return ""


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if any(marker in key.lower() for marker in ("api_key", "access_token", "secret", "authorization")) else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
