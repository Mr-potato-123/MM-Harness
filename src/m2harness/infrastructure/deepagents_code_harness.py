"""DeepAgents-backed Code Harness.

DeepAgents is the default Code Agent runtime for M2Harness.  It supplies the
agent loop, ``write_todos``, filesystem tools, context summarization and the
LangGraph checkpoint boundary.  M2Harness supplies the domain contract around
it: progressive-disclosure paths, local execution/evidence validation,
iteration policy and review handoffs.

The adapter deliberately consumes the framework through public APIs.  It does
not reimplement a model/tool loop and it never kills a running invocation when
an operator's observation window expires.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from m2harness.domain.code import CodeProposal
from m2harness.domain.solve_problem import SolveProblemContext, SolveProblemTask, UnifiedModelingReport
from m2harness.errors import ActivityExecutionError
from m2harness.infrastructure.code_harness import CodeProposalProvider, _validate_python_policy
from m2harness.infrastructure.local_sandbox import LocalSandboxClient


class DeepAgentsUnavailable(ActivityExecutionError):
    """The maintained DeepAgents/LangGraph runtime is not installed or configured."""


class CodeAgentHandoff(BaseModel):
    """Structured final contract emitted by the DeepAgents Code Agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_path: str = Field(min_length=1, max_length=1_000)
    logical_name: str = Field(pattern=r"^[A-Za-z0-9._-]+\.py$", max_length=200)
    timeout_seconds: int = Field(default=300, ge=1, le=600)
    expected_validations: tuple[str, ...] = ()


class DeepAgentsCodeProposalProvider(CodeProposalProvider):
    """Persistent, framework-owned Code Agent session per task."""

    def __init__(
        self,
        workspace_root: Path,
        sandbox: LocalSandboxClient,
        *,
        model: Any | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        checkpoint_path: Path | None = None,
        event_root: Path | None = None,
        skills: list[str] | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self.model = model or self._build_model(base_url=base_url, model_name=model_name, api_key=api_key)
        self.checkpoint_path = (checkpoint_path or (self.workspace_root / ".m2harness" / "deepagents-checkpoints.sqlite")).resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_root = (event_root or (self.workspace_root / ".m2harness-code" / "deepagents-events")).resolve()
        self.event_root.mkdir(parents=True, exist_ok=True)
        self.skills = skills or []
        self._agent: Any | None = None
        self._agent_target: str | None = None
        self._checkpoint_context: Any | None = None
        self._checkpointer: Any | None = None

    def propose(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int) -> CodeProposal:
        target_relative = self._target_path(task, iteration)
        prompt_path = self.workspace_root / ".m2harness-code" / task.task_id / f"iteration-{iteration}" / "deepagents-prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = self._build_prompt(task, context, modeling, iteration=iteration, target_relative=target_relative)
        prompt_path.write_text(prompt, encoding="utf-8")
        run_id = str(context.metadata.get("run_id", "unscoped")) if isinstance(context.metadata, dict) else "unscoped"
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        event_path = self.event_root / f"{safe_run_id}-{task.task_id}.ndjson"
        final_state = self._run_agent(task, context, modeling, iteration=iteration, prompt=prompt, event_path=event_path)
        handoff = self._handoff_from_state(final_state)
        target = self._resolve_agent_path(handoff.source_path)
        if not target.is_file() or target.is_symlink() or target.suffix.lower() != ".py":
            raise DeepAgentsUnavailable(f"DeepAgents handoff source_path is not a regular Python file: {handoff.source_path}")
        source = target.read_text(encoding="utf-8")
        if not source.strip():
            raise DeepAgentsUnavailable("DeepAgents wrote an empty source file")
        return CodeProposal(
            source=source,
            logical_name=target.name,
            timeout_seconds=handoff.timeout_seconds,
            expected_validations=tuple(dict.fromkeys(handoff.expected_validations or modeling.required_validations)),
            metadata={
                "agent_runtime": "deepagents",
                "session_id": self._session_id(task, context),
                "event_log": str(event_path.relative_to(self.workspace_root).as_posix()),
                "prompt_file": str(prompt_path.relative_to(self.workspace_root).as_posix()),
            },
        )

    def close(self) -> None:
        """Release the checkpoint connection during normal application shutdown."""

        if self._checkpoint_context is not None:
            self._checkpoint_context.__exit__(None, None, None)
            self._checkpoint_context = None
            self._checkpointer = None
            self._agent = None
            self._agent_target = None

    def _ensure_agent(self, context: SolveProblemContext, target_relative: str) -> Any:
        if self._agent is not None and self._agent_target == target_relative:
            return self._agent
        try:
            from deepagents import (
                GeneralPurposeSubagentProfile,
                HarnessProfile,
                create_deep_agent,
                register_harness_profile,
            )
            from deepagents.backends import FilesystemBackend
            from deepagents.middleware.filesystem import FilesystemPermission
            from langgraph.checkpoint.sqlite import SqliteSaver
            from m2harness.infrastructure.agent_middleware import M2AgentAuditMiddleware
        except ImportError as exc:
            raise DeepAgentsUnavailable("install the agents extra: deepagents, langchain, langgraph and langgraph-checkpoint-sqlite") from exc

        backend = FilesystemBackend(root_dir=self.workspace_root, virtual_mode=True)
        permissions = self._permissions(context, target_relative, FilesystemPermission)
        if self._checkpointer is None:
            self._checkpoint_context = SqliteSaver.from_conn_string(str(self.checkpoint_path))
            self._checkpointer = self._checkpoint_context.__enter__()
            self._checkpointer.setup()
        system_prompt = (
            "你是 M2Harness 的 Code Agent。全流程使用中文。\n"
            "你只负责当前题目当前迭代的代码实现，不负责建模、不负责审查、不负责发布，也不能改变 Main Harness 的 DAG/TODO。\n"
            "必须先用 write_todos 建立实现清单；每次只允许一个 in_progress，完成一项就更新为 completed。\n"
            "只能通过文件工具访问已授权路径；授权路径是逐项清单，不等于整个工作区。禁止网络、密钥、主 Harness 数据库和后续题目。\n"
            "源代码必须写入交接中指定的 target_source_path，验证输出只能写入同一 iteration 目录的 outputs 子目录。\n"
            "交接信息已经包含建模契约和必要路径，不要循环读取报告清单。第一步直接调用 write_code_source 写入完整源代码（尽量不超过 350 行）；若工具返回 complete=false，必须用 append=true 继续写后续源码块，直到 complete=true，再调用 python_execute 验证。必要时用 read_file 读取一次源文件，不要在写源代码前连续调用 ls/read_file。"
            "完成后必须使用结构化输出字段 source_path、logical_name、timeout_seconds、expected_validations；不要把源代码放进最终回答。"
        )
        model_identifier = getattr(self.model, "model_name", None) or getattr(self.model, "model", None)
        if model_identifier:
            register_harness_profile(
                f"openai:{model_identifier}",
                HarnessProfile(
                    excluded_tools=frozenset({"execute"}),
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
        try:
            self._agent = create_deep_agent(
                model=self.model,
                tools=[self._write_code_source_tool(target_relative), self._python_execute_tool()],
                backend=backend,
                permissions=permissions,
                skills=self.skills or None,
                system_prompt=system_prompt,
                response_format=CodeAgentHandoff,
                checkpointer=self._checkpointer,
                middleware=[M2AgentAuditMiddleware(event_root=self.event_root)],
                # The outer M2 Harness owns delegation.  Keeping the Code Agent
                # on one durable thread avoids a hidden second orchestration
                # tree while retaining DeepAgents' built-in Todo and compact.
                subagents=[],
                name="m2h-code-agent",
            )
            self._agent_target = target_relative
        except Exception as exc:
            self.close()
            raise DeepAgentsUnavailable(f"failed to compose DeepAgents Code Harness: {exc}") from exc
        return self._agent

    def _write_code_source_tool(self, target_relative: str) -> Any:
        """Write the one source artifact through a fixed, auditable boundary."""
        from langchain_core.tools import tool
        import tempfile

        target = self._resolve_agent_path(target_relative)
        workspace_root = self.workspace_root

        @tool("write_code_source")
        def write_code_source(source: str, append: bool = False) -> str:
            """将 Python 源码首块或后续块写入本轮交接指定的固定路径。"""
            if not isinstance(source, str) or not source.strip():
                return json.dumps({"ok": False, "error": "source must be non-empty"}, ensure_ascii=False)
            if len(source.encode("utf-8")) > 2_000_000:
                return json.dumps({"ok": False, "error": "source exceeds 2MB"}, ensure_ascii=False)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if append:
                    if not target.is_file() or target.is_symlink():
                        return json.dumps({"ok": False, "error": "append requires an existing first source chunk"}, ensure_ascii=False)
                    with target.open("a", encoding="utf-8") as stream:
                        stream.write(source)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if target.stat().st_size > 2_000_000:
                        return json.dumps({"ok": False, "error": "assembled source exceeds 2MB"}, ensure_ascii=False)
                else:
                    descriptor, temporary_name = tempfile.mkstemp(prefix=".m2h-source-", suffix=".py", dir=target.parent)
                    temporary = Path(temporary_name)
                    try:
                        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                            stream.write(source)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                complete = True
                syntax_error = ""
                try:
                    _validate_python_policy(target.read_text(encoding="utf-8"))
                except Exception as exc:
                    complete = False
                    syntax_error = str(exc)[:1_000]
                return json.dumps(
                    {
                        "ok": True,
                        "complete": complete,
                        "path": str(target.relative_to(workspace_root).as_posix()),
                        "bytes": target.stat().st_size,
                        "syntax_error": syntax_error,
                        "next_action": "continue with write_code_source(append=true)" if not complete else "run python_execute",
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)[:4_000]}, ensure_ascii=False)

        return write_code_source

    def _python_execute_tool(self) -> Any:
        """Expose only the fixed local Python boundary to DeepAgents."""
        from langchain_core.tools import tool

        sandbox = self.sandbox
        workspace_root = self.workspace_root

        @tool("python_execute")
        def python_execute(code: str, timeout_seconds: int = 120) -> str:
            """在本地受控沙箱中执行一段 Python 验证代码；不得联网或运行 shell。"""
            if not isinstance(code, str) or not code.strip():
                return json.dumps({"ok": False, "error": "code must be non-empty"}, ensure_ascii=False)
            if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 600:
                return json.dumps({"ok": False, "error": "timeout_seconds must be 1..600"}, ensure_ascii=False)
            try:
                _validate_python_policy(code)
                result = sandbox.run(
                    (sys.executable, "-I", "-c", code),
                    timeout_seconds=timeout_seconds,
                    cwd=workspace_root,
                    env={"PYTHONIOENCODING": "utf-8", "M2HARNESS_NETWORK": "deny"},
                )
                return json.dumps(
                    {
                        "ok": result.exit_code == 0 and not result.timed_out,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                        "stdout": result.stdout.decode("utf-8", errors="replace")[:200_000],
                        "stderr": result.stderr.decode("utf-8", errors="replace")[:200_000],
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)[:4_000]}, ensure_ascii=False)

        return python_execute

    def _run_agent(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int, prompt: str, event_path: Path) -> dict[str, Any]:
        target_relative = self._target_path(task, iteration)
        agent = self._ensure_agent(context, target_relative)
        config = {
            "configurable": {"thread_id": self._session_id(task, context)},
            "metadata": {
                "m2h_task_id": task.task_id,
                "m2h_iteration": iteration,
                "m2h_run_id": str(context.metadata.get("run_id", "unscoped")) if isinstance(context.metadata, dict) else "unscoped",
                "m2h_runtime": "deepagents",
            },
            # This is an agent-loop safety ceiling, not a model output-token
            # cap.  It prevents a malformed tool loop from running forever;
            # an observation timeout never terminates the active process.
            "recursion_limit": int(os.environ.get("M2HARNESS_DEEPAGENTS_RECURSION_LIMIT", "200")),
        }
        event_path.parent.mkdir(parents=True, exist_ok=True)
        final_state: dict[str, Any] | None = None
        sequence = 0
        try:
            for state in agent.stream({"messages": [{"role": "user", "content": prompt}]}, config=config, stream_mode="values"):
                if not isinstance(state, dict):
                    continue
                final_state = state
                sequence += 1
                self._append_event(event_path, {
                    "seq": sequence,
                    "occurred_at": datetime.now(UTC).isoformat(),
                    "task_id": task.task_id,
                    "iteration": iteration,
                    "kind": "state",
                    "state": _summarize_state(state),
                })
        except Exception as exc:
            self._append_event(event_path, {
                "seq": sequence + 1,
                "occurred_at": datetime.now(UTC).isoformat(),
                "task_id": task.task_id,
                "iteration": iteration,
                "kind": "runtime_error",
                "error": str(exc)[:4_000],
            })
            raise ActivityExecutionError(f"DeepAgents Code Agent failed: {exc}") from exc
        if final_state is None:
            raise DeepAgentsUnavailable("DeepAgents returned no state snapshots")
        return final_state

    def _handoff_from_state(self, state: dict[str, Any]) -> CodeAgentHandoff:
        structured = state.get("structured_response")
        if isinstance(structured, CodeAgentHandoff):
            return structured
        if isinstance(structured, dict):
            return CodeAgentHandoff.model_validate(structured, strict=False)
        messages = state.get("messages")
        text = _last_text(messages)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = _find_json_object(text)
        if isinstance(value, dict) and isinstance(value.get("m2harness_handoff"), dict):
            value = value["m2harness_handoff"]
        if not isinstance(value, dict):
            raise DeepAgentsUnavailable("DeepAgents final state did not contain a structured handoff")
        return CodeAgentHandoff.model_validate(value, strict=False)

    def _build_model(self, *, base_url: str | None, model_name: str | None, api_key: str | None) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise DeepAgentsUnavailable("langchain-openai is required for the DeepAgents Code Agent") from exc
        endpoint = base_url or os.environ.get("QWEN_BASE_URL", "https://api.deepseek.com")
        name = model_name or os.environ.get("M2HARNESS_CODE_MODEL", os.environ.get("QWEN_MODEL", "deepseek-v4-flash"))
        key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise DeepAgentsUnavailable("DEEPSEEK_API_KEY or DASHSCOPE_API_KEY is required for DeepAgents Code Agent")
        model_options: dict[str, Any] = {}
        if "deepseek.com" in endpoint.lower():
            # DeepSeek currently rejects tool_choice/structured-output calls
            # while thinking mode is enabled.  The Model Agent keeps its own
            # reasoning setting; Code Agent tool execution must be callable.
            model_options["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(
            model=name,
            base_url=endpoint,
            api_key=key,
            temperature=0,
            max_retries=2,
            streaming=True,
            **model_options,
        )

    def _permissions(self, context: SolveProblemContext, target_relative: str, permission_type: Any) -> list[Any]:
        allow_read = ["/" + item.relative_path.replace("\\", "/").lstrip("/") for item in context.readonly_files]
        # Previously generated source and event/prompt files are review input
        # on later iterations; the Code Agent must be able to inspect them.
        allow_read.extend([
            "/" + target_relative.replace("\\", "/").lstrip("/"),
            "/" + str(Path(target_relative).parent / "**").replace("\\", "/").lstrip("/"),
        ])
        output_glob = "/" + str(Path(target_relative).parent / "outputs" / "**").replace("\\", "/").lstrip("/")
        allow_write = ["/" + target_relative.replace("\\", "/").lstrip("/"), output_glob]
        rules = [
            permission_type(["read"], sorted(set(allow_read)), mode="allow"),
            permission_type(["write"], sorted(set(allow_write)), mode="allow"),
            permission_type(["read", "write"], ["/**"], mode="deny"),
        ]
        return rules

    def _resolve_agent_path(self, source_path: str) -> Path:
        normalized = source_path.replace("\\", "/")
        if normalized.startswith("/"):
            normalized = normalized[1:]
        path = PurePosixPath(normalized)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise DeepAgentsUnavailable("DeepAgents source_path must be workspace-relative")
        target = (self.workspace_root / Path(*path.parts)).resolve()
        if self.workspace_root != target and self.workspace_root not in target.parents:
            raise DeepAgentsUnavailable("DeepAgents source_path escapes workspace")
        return target

    @staticmethod
    def _target_path(task: SolveProblemTask, iteration: int) -> str:
        logical = "solve_" + re.sub(r"[^A-Za-z0-9_-]+", "_", task.task_id) + ".py"
        return str(Path(".m2harness-code") / task.task_id / f"iteration-{iteration}" / logical).replace("\\", "/")

    @staticmethod
    def _session_id(task: SolveProblemTask, context: SolveProblemContext | None = None) -> str:
        run_id = "unscoped"
        if context is not None and isinstance(context.metadata, dict):
            run_id = str(context.metadata.get("run_id", run_id))
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        return f"m2h-deepagents-code-{safe_run_id}-{task.task_id}"

    def _build_prompt(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int, target_relative: str) -> str:
        payload = {
            "task": task.model_dump(mode="json"),
            "iteration": iteration,
            "target_source_path": "/" + target_relative,
            "modeling_contract": modeling.model_dump(mode="json"),
            "readonly_allowlist": [
                {"path": item.relative_path, "purpose": item.purpose, "role": item.role.value, "sha256": item.sha256}
                for item in context.readonly_files
            ],
            "disclosed_text_files": [
                {"path": item.relative_path, "purpose": item.purpose, "content": item.content, "truncated": item.truncated}
                for item in context.disclosed_text_files
            ],
            "required_validations": list(modeling.required_validations),
            "expected_outputs": list(modeling.expected_outputs),
        }
        return "当前为第 %d 轮 Code Agent 实现。继续复用本任务的 LangGraph thread/session；只读清单之外的路径不得打开。以下是本轮 Model→Code 结构化交接：\n\n%s" % (iteration, json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")


def _last_text(messages: Any) -> str:
    if not isinstance(messages, (list, tuple)):
        return ""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
            if text.strip():
                return text
    return ""


def _find_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found = value
    return found


def _summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages")
    summary: dict[str, Any] = {"keys": sorted(str(key) for key in state.keys())}
    if isinstance(state.get("todos"), list):
        summary["todos"] = state["todos"]
    if isinstance(messages, (list, tuple)):
        items: list[dict[str, Any]] = []
        for message in messages[-8:]:
            role = getattr(message, "type", None) or (message.get("role") if isinstance(message, dict) else None)
            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
            items.append({"role": str(role or "unknown"), "content_preview": str(content)[:2_000]})
        summary["messages"] = items
    if "structured_response" in state:
        value = state["structured_response"]
        summary["structured_response"] = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return summary
