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
import hashlib
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
    # No default wall-clock limit is part of the Code→Model report contract.
    # An explicit value is accepted only as an operator emergency stop.
    timeout_seconds: int | None = Field(default=None, ge=1, le=86_400)
    # JSON structured-output providers emit arrays, not Python tuples.  Keep
    # the wire contract JSON-native here and normalize to a tuple at the
    # domain boundary in ``propose``.  ``strict=True`` is intentional for the
    # scalar fields, but a tuple would make an otherwise valid handoff fail
    # Pydantic validation with ``input_type=list``.
    expected_validations: list[str] = Field(default_factory=list)


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
        self._forced_handoff_targets: set[str] = set()

    def propose(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int) -> CodeProposal:
        target_relative = self._target_path(task, context, iteration)
        prompt_path = self.workspace_root / Path(target_relative).parent / "deepagents-prompt.md"
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
            "write_todos 只在初始化或状态真正变化时调用；相同清单不要重复写入。python_execute 超时或重写源码后，禁止再次调用 write_todos，直接运行 python_execute；若工具返回 next_action，严格按其下一步执行。"
            "完成后必须使用结构化输出字段 source_path、logical_name、expected_validations；不要把源代码放进最终回答。默认不设置执行时限。"
        )
        system_prompt += (
            "\n\nM2Harness 运行约束：本 Code Agent 只可使用 write_todos、write_code_source 和 python_execute。"
            "不要调用 ls、read_file、write_file、edit_file、glob、grep、execute 或 delete；这些文件工具已被运行时移除。"
            "Model→Code 交接契约已经完整放在本次用户消息中，不需要再次读取文件。只能生成一个目标 Python 文件，禁止创建辅助源码文件。"
            "python_execute 的当前工作目录就是目标源码所在的 iteration 目录；使用相对路径检查该文件和 outputs，不要把 Windows 绝对路径传给任何工具。"
            "源码写入后持续执行必要的语法/运行验证；若失败，依据探针和实际报告修复后再运行，不设置固定次数或墙钟时限。"
            "如果任何工具结果包含 forced_stop=true 或 next_action 要求返回 CodeAgentHandoff，必须立刻返回结构化交接，绝对不能再次调用 write_code_source、python_execute、Todo 或其他工具。"
            "python_execute 只负责可观测执行；Windows 没有 signal.SIGALRM，验证代码禁止自行安装 SIGALRM 或假设 solve_dp 等不存在的函数。验证源码时只运行 import runpy; runpy.run_path('solve_q1.py', run_name='__main__')，并读取其 stdout Markdown。"
            "资源补给不得对三种资源做无界三重数量枚举；必须使用候选边界、需求驱动枚举或支配剪枝，并在探针显示资源异常或操作员显式停止后改变算法复杂度。"
            "建立 MILP 后必须先做小规模可行性烟雾检查；到达终点后的活动标志约束不得与 a_0=1、a_T=0 和终点位置约束互相矛盾，禁止使用会令到达终点后 a_t<0 的不等式。"
            "路线实现硬性自检：候选路线的节点序列必须显式包含需要停留的补给平台（不能只把补给需求按总量加到初始资源上）；允许重复访问作业点，或给出可检查的上界/支配证明说明为何不会漏掉重复访问。若 bfs_path(a,b) 返回含起点和终点的节点列表，移动日必须遍历 path[1:]（第1天到达第一个相邻节点），不得用 path[0:distance] 导致终点和补给平台被跳过。到达终点当天立即结算并停止，不得用 END 填充后续消耗日。"
            "在正式求解前必须运行最小路径烟雾测试：验证 START→END 的首个移动位置是其相邻节点、最后一个移动位置是 END；验证一条经过 SUPPLY 的候选路线确实产生补给记录且补给前后 O/H/F、M 和 O+H+F<=CAP 均满足。禁止使用名为 solve_simplified 的总量估算作为主求解器，也不得在未逐日模拟时声称 V1-V3 已通过。"
            "最终脚本必须在 stdout 返回一份可审查的中文 Markdown 执行报告，内容包括完整路径、每日行动与 O/H/F/M/Z、目标值、算法偏差、验证过程和限制；可以附带 JSON，但 JSON 中的 false 只是待审查提示，不是 Harness 失败信号。"
            "必须把同一份可审查结果保存到当前 iteration 的 outputs/result.md（必要时另存 outputs/result.json）；不要只返回 all_valid 或孤立的 true/false。"
        )
        model_identifier = getattr(self.model, "model_name", None) or getattr(self.model, "model", None)
        if model_identifier:
            register_harness_profile(
                f"openai:{model_identifier}",
                HarnessProfile(
                    excluded_tools=frozenset({
                        "execute", "ls", "read_file", "write_file", "edit_file",
                        "delete", "glob", "grep",
                    }),
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
        try:
            todo_write, todo_read = self._todo_tools(target_relative)
            self._agent = create_deep_agent(
                model=self.model,
                tools=[todo_write, todo_read, self._write_code_source_tool(target_relative), self._python_execute_tool(target_relative)],
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

        def previous_revision() -> Path | None:
            """Return the immediately previous iteration's source, if any."""

            match = re.fullmatch(r"iteration-(\d+)", target.parent.name)
            if match is None:
                return None
            number = int(match.group(1))
            if number <= 1:
                return None
            return target.parent.parent / f"iteration-{number - 1}" / target.name

        @tool("write_code_source")
        def write_code_source(source: str, append: bool = False) -> str:
            """将 Python 源码首块或后续块写入本轮交接指定的固定路径。"""
            if target_relative in self._forced_handoff_targets:
                return json.dumps({
                    "ok": False,
                    "forced_stop": True,
                    "path": target_relative,
                    "error": "validation budget exhausted; source is frozen for handoff",
                    "next_action": "return the structured CodeAgentHandoff now; do not call another tool",
                }, ensure_ascii=False)
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
                    previous = previous_revision()
                    if (
                        previous is not None
                        and previous.is_file()
                        and previous.read_text(encoding="utf-8").replace("\r\n", "\n")
                        == source.replace("\r\n", "\n")
                    ):
                        return json.dumps(
                            {
                                "ok": False,
                                "error": "source is identical to the previous iteration after newline normalization; apply the Review instructions and write a changed source before validating",
                                "previous_path": previous.relative_to(workspace_root).as_posix(),
                                "next_action": "rewrite the complete source with an observable algorithm or output change",
                            },
                            ensure_ascii=False,
                        )
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
                payload = {"ok": False, "error": str(exc)[:4_000]}
                return json.dumps(payload, ensure_ascii=False)

        return write_code_source

    def _todo_tools(self, target_relative: str) -> tuple[Any, Any]:
        """Provide the small persistent TODO lane even when no stock middleware is installed."""

        from langchain_core.tools import tool

        todo_path = self._resolve_agent_path(target_relative).parent.parent / "todo.json"
        todo_write_count = 0
        last_todo_payload: str | None = None

        def _read() -> list[dict[str, Any]]:
            if not todo_path.is_file() or todo_path.is_symlink():
                return []
            try:
                value = json.loads(todo_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
            return value if isinstance(value, list) else []

        @tool("write_todos")
        def write_todos(todos: list[dict[str, Any]]) -> str:
            """写入本轮 Code Agent 的最小 TODO 清单；只能有一个 in_progress 项。"""

            nonlocal todo_write_count, last_todo_payload

            if not isinstance(todos, list) or len(todos) > 20:
                return json.dumps({"ok": False, "error": "todos must be a list of at most 20 items"}, ensure_ascii=False)
            normalized: list[dict[str, Any]] = []
            in_progress = 0
            for item in todos:
                if not isinstance(item, dict):
                    return json.dumps({"ok": False, "error": "each todo must be an object"}, ensure_ascii=False)
                status = str(item.get("status", "pending"))
                if status not in {"pending", "in_progress", "completed", "cancelled"}:
                    return json.dumps({"ok": False, "error": f"invalid todo status: {status}"}, ensure_ascii=False)
                in_progress += status == "in_progress"
                normalized.append({
                    "id": str(item.get("id", len(normalized) + 1))[:120],
                    "content": str(item.get("content", item.get("task", "")))[:500],
                    "status": status,
                })
            if in_progress > 1:
                return json.dumps({"ok": False, "error": "only one todo may be in_progress"}, ensure_ascii=False)
            normalized_json = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            todo_write_count += 1
            if normalized_json == last_todo_payload:
                return json.dumps({
                    "ok": True, "unchanged": True, "call_count": todo_write_count,
                    "path": todo_path.relative_to(self.workspace_root).as_posix(),
                    "todos": normalized,
                    "next_action": "do not call write_todos again; run python_execute or return the structured CodeAgentHandoff",
                }, ensure_ascii=False)
            todo_path.parent.mkdir(parents=True, exist_ok=True)
            todo_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            last_todo_payload = normalized_json
            return json.dumps({
                "ok": True, "call_count": todo_write_count,
                "path": todo_path.relative_to(self.workspace_root).as_posix(),
                "todos": normalized,
                "next_action": "call python_execute after the source is complete",
            }, ensure_ascii=False)

        @tool("read_todos")
        def read_todos() -> str:
            """读取本轮 Code Agent 的 TODO 清单。"""

            return json.dumps({"ok": True, "path": todo_path.relative_to(self.workspace_root).as_posix(), "todos": _read()}, ensure_ascii=False)

        return write_todos, read_todos

    def _python_execute_tool(self, target_relative: str) -> Any:
        """Expose only the fixed local Python boundary to DeepAgents."""
        from langchain_core.tools import tool

        sandbox = self.sandbox
        workspace_root = self.workspace_root
        target_path = self._resolve_agent_path(target_relative)
        target_directory = target_path.parent
        call_count = 0
        last_source_hash: str | None = None
        rewrite_required = False

        @tool("python_execute")
        def python_execute(code: str, timeout_seconds: int | None = None) -> str:
            """在本地受控沙箱中执行一段 Python 验证代码；不得联网或运行 shell。"""
            if not isinstance(code, str) or not code.strip():
                return json.dumps({"ok": False, "error": "code must be non-empty"}, ensure_ascii=False)
            if timeout_seconds is not None and (not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 180):
                return json.dumps({"ok": False, "error": "显式紧急停止值必须为 1..180；默认省略 timeout_seconds"}, ensure_ascii=False)
            nonlocal call_count
            nonlocal last_source_hash, rewrite_required
            call_count += 1
            try:
                _validate_python_policy(code)
                source_hash = hashlib.sha256(target_path.read_bytes()).hexdigest() if target_path.is_file() else "missing"
                if rewrite_required and source_hash == last_source_hash:
                    return json.dumps({
                        "ok": False,
                        "error": "rewrite the target source before retrying；上一次显式紧急停止后必须先修改目标源码，再次运行验证",
                        "call_count": call_count,
                        "next_action": "call write_code_source with a materially simpler algorithm, then run python_execute",
                    }, ensure_ascii=False)
                result = sandbox.run(
                    # ``-I`` ignores PYTHON* environment variables, so make
                    # UTF-8 explicit for Chinese diagnostics from the inner
                    # validation probe as well as the outer execution lane.
                    (sys.executable, "-I", "-X", "utf8", "-c", code),
                    timeout_seconds=timeout_seconds,
                    cwd=target_directory,
                    env={"PYTHONIOENCODING": "utf-8", "M2HARNESS_NETWORK": "deny"},
                )
                payload = {
                        "ok": result.exit_code == 0 and not result.timed_out,
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                        "stdout": result.stdout.decode("utf-8", errors="replace")[:200_000],
                        "stderr": result.stderr.decode("utf-8", errors="replace")[:200_000],
                        "call_count": call_count,
                    }
                last_source_hash = source_hash
                rewrite_required = bool(timeout_seconds is not None and result.timed_out)
                return json.dumps(payload, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({
                    "ok": False,
                    "error": str(exc)[:4_000],
                    "next_action": "rewrite source and retry",
                }, ensure_ascii=False)

        return python_execute

    def _run_agent(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int, prompt: str, event_path: Path) -> dict[str, Any]:
        target_relative = self._target_path(task, context, iteration)
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
            "recursion_limit": int(os.environ.get("M2HARNESS_DEEPAGENTS_RECURSION_LIMIT", "64")),
        }
        event_path.parent.mkdir(parents=True, exist_ok=True)
        final_state: dict[str, Any] | None = None
        sequence = 0
        seen_message_ids: set[str] = set()
        repeated_tool_results: dict[tuple[str, str], int] = {}
        baseline_initialized = False
        try:
            for state in agent.stream({"messages": [{"role": "user", "content": prompt}]}, config=config, stream_mode="values"):
                if not isinstance(state, dict):
                    continue
                final_state = state
                sequence += 1
                messages = state.get("messages", ()) if isinstance(state.get("messages"), (list, tuple)) else ()
                # A durable LangGraph thread includes all previous iterations.
                # Seed the identity set from the first snapshot so historical
                # tool failures cannot trip this iteration's loop guard.
                if not baseline_initialized:
                    for index, message in enumerate(messages):
                        role = getattr(message, "type", None) or (message.get("role") if isinstance(message, dict) else None)
                        if role != "tool":
                            continue
                        message_id = getattr(message, "id", None) or (message.get("id") if isinstance(message, dict) else None)
                        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
                        if isinstance(content, list):
                            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                        seen_message_ids.add(str(message_id or f"{index}:{str(content)[:400]}"))
                    baseline_initialized = True
                for index, message in enumerate(messages):
                    role = getattr(message, "type", None) or (message.get("role") if isinstance(message, dict) else None)
                    if role != "tool":
                        continue
                    message_id = getattr(message, "id", None) or (message.get("id") if isinstance(message, dict) else None)
                    content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
                    if isinstance(content, list):
                        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                    identity = str(message_id or f"{index}:{str(content)[:400]}")
                    if identity in seen_message_ids:
                        continue
                    seen_message_ids.add(identity)
                    tool_name = str(getattr(message, "name", None) or (message.get("name") if isinstance(message, dict) else "tool"))
                    signature = (tool_name, str(content)[:800])
                    repeated_tool_results[signature] = repeated_tool_results.get(signature, 0) + 1
                    if repeated_tool_results[signature] >= 3:
                        error = f"repeated tool result loop detected: {tool_name} (three identical results)"
                        self._append_event(event_path, {
                            "seq": sequence + 1,
                            "occurred_at": datetime.now(UTC).isoformat(),
                            "task_id": task.task_id,
                            "iteration": iteration,
                            "kind": "loop_guard",
                            "error": error,
                        })
                        raise ActivityExecutionError(error)
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

    def _target_path(self, task: SolveProblemTask, context: SolveProblemContext, iteration: int) -> str:
        logical = "solve_" + re.sub(r"[^A-Za-z0-9_-]+", "_", task.task_id) + ".py"
        run_id = "unscoped"
        if isinstance(context.metadata, dict):
            run_id = str(context.metadata.get("run_id", run_id))
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        return str(Path(".m2harness-code") / "runs" / safe_run_id / task.task_id / f"iteration-{iteration}" / logical).replace("\\", "/")

    @staticmethod
    def _session_id(task: SolveProblemTask, context: SolveProblemContext | None = None) -> str:
        run_id = "unscoped"
        if context is not None and isinstance(context.metadata, dict):
            run_id = str(context.metadata.get("run_id", run_id))
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        return f"m2h-deepagents-code-{safe_run_id}-{task.task_id}"

    def _build_prompt(self, task: SolveProblemTask, context: SolveProblemContext, modeling: UnifiedModelingReport, *, iteration: int, target_relative: str) -> str:
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
            }
        payload = {
            "task": task.model_dump(mode="json"),
            "iteration": iteration,
            "target_source_path": "/" + target_relative,
            "modeling_contract": modeling_contract,
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
            # Revision rounds keep the Code session but receive the latest
            # Model→Code instructions explicitly; no re-modeling is needed.
            "model_to_code_repair_instructions": list(context.instructions[-32:]),
            "model_conversation_tail": list(context.metadata.get("model_conversation", ())) [-12:]
            if isinstance(context.metadata.get("model_conversation", ()), (list, tuple)) else [],
        }
        return (
            "当前为第 %d 轮 Code Agent 实现。继续复用本任务的 LangGraph thread/session；只读清单之外的路径不得打开。"
            "以下是本轮 Model→Code 结构化交接：\n\n%s\n\n"
            "验收硬契约：脚本应返回可读 Markdown 报告；同一个 Model Agent 在 Code→Model 阶段依据报告、源代码和本地探针给出返修意见。"
            "JSON validations/validation_evidence 若存在只作辅助索引，不能因为某个 false 字段直接判定失败；必须把完整 Markdown 结果写入当前 iteration 的 outputs/result.md。"
        ) % (iteration, json.dumps(payload, ensure_ascii=False, indent=2))

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
