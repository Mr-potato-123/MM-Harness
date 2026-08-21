"""Operations CLI for the single-question Harness."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import shutil
import sys
import tempfile
from urllib.parse import urlsplit
from pathlib import Path
from uuid import UUID, uuid4

from m2harness.artifacts import ArtifactStore
from m2harness.executor import CommandActivityExecutor, SandboxedActivityExecutor
from m2harness.models import ArtifactKind, HarnessSettings, utc_now
from m2harness.store import HarnessStore
from m2harness.workflow import SingleQuestionWorkflow
from m2harness.application.runtime_bundle import build_local_runtime, default_skill_root
from m2harness.infrastructure.db.tool_runs import SQLiteToolAuditStore
from m2harness.infrastructure.db.sqlite_artifacts import SQLiteArtifactRegistry
from m2harness.domain.dag import canonical_single_question_dag
from m2harness.domain.dag import canonical_main_harness_dag
from m2harness.domain.media import MultimodalInput
from m2harness.domain.solve_problem import ReadOnlyFileReference, ReadOnlyFileRole, SolveProblemContext
from m2harness.domain.capability import CapabilityRequirement
from m2harness.domain.tool import ToolCall

MAX_INPUT_FILE_BYTES = 200 * 1024 * 1024


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".m2h-preflight-", dir=path, delete=True):
            pass
        return True
    except OSError:
        return False


def _resolve_adapter_path(value: Path) -> Path:
    """Resolve a CLI adapter from cwd, then from the source checkout.

    The second lookup keeps an installed/packaged CLI usable when an operator
    invokes it outside the repository root, while preserving an explicit cwd
    override for deployment-specific adapters.
    """
    candidate = value if value.is_absolute() else (Path.cwd() / value)
    if candidate.is_file():
        return candidate.resolve()
    checkout_candidate = Path(__file__).resolve().parents[2] / value
    if checkout_candidate.is_file():
        return checkout_candidate.resolve()
    if value.as_posix().replace("\\", "/") == "adapters/qwen38_max.py":
        spec = importlib.util.find_spec("adapters.qwen38_max")
        if spec is not None and spec.origin and spec.origin not in {"built-in", "frozen"}:
            bundled = Path(spec.origin)
            if bundled.is_file():
                return bundled.resolve()
    return checkout_candidate.resolve()


def _read_input_file(path: Path) -> bytes:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    if size > MAX_INPUT_FILE_BYTES:
        raise ValueError(f"input file exceeds the {MAX_INPUT_FILE_BYTES} byte Harness budget")
    data = path.read_bytes()
    if len(data) != size:
        raise OSError(f"input file changed while reading: {path}")
    return data


def _settings(args: argparse.Namespace) -> HarnessSettings:
    return HarnessSettings(
        database_path=args.database,
        artifact_root=args.artifacts,
        lease_seconds=args.lease_seconds,
        activity_timeout_seconds=args.timeout,
        max_activity_attempts=args.max_attempts,
        max_revisions=args.max_revisions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m2harness", description="M2Harness operations CLI")
    parser.add_argument("--database", type=Path, default=Path(".m2harness/state.db"))
    parser.add_argument("--artifacts", type=Path, default=Path(".m2harness/artifacts"))
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-revisions", type=int, default=2, help="最多允许两轮审查返修")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    project = commands.add_parser("create-project")
    project.add_argument("name")
    question = commands.add_parser("add-question")
    question.add_argument("project_id", type=UUID)
    question.add_argument("key")
    question.add_argument("title")
    question.add_argument("problem_file", type=Path)
    attachment = commands.add_parser("add-input")
    attachment.add_argument("question_id", type=UUID)
    attachment.add_argument("input_file", type=Path)
    attachment.add_argument("--kind", choices=["input", "source", "data"], default="input")
    run = commands.add_parser("run-question")
    run.add_argument("question_id", type=UUID)
    run.add_argument("--worker-id", required=True)
    run.add_argument("--executor", nargs=argparse.REMAINDER, required=True)
    qwen = commands.add_parser("run-qwen")
    qwen.add_argument("question_id", type=UUID)
    qwen.add_argument("--worker-id", required=True)
    qwen.add_argument("--adapter", type=Path, default=Path("adapters/qwen38_max.py"))
    qwen.add_argument("--model", default=None)
    qwen.add_argument("--allow-remote-code-interpreter", action="store_true", help="preliminary provider-only run; remote execution can never satisfy the production evidence gate")
    main_qwen = commands.add_parser("run-main-qwen", help="run the target Main Harness -> solve_problem -> paper path")
    main_qwen.add_argument("input", type=Path, help="PDF/image/video input selected for the problem")
    main_qwen.add_argument("--problem", default="请解决所附数学建模题并产出可审计报告；各编号问题必须按 TODO 串行处理。")
    main_qwen.add_argument("--scope", default="question-1", help="scope label used by a one-question graph; multi-question runs use question-1..question-N")
    main_qwen.add_argument("--question-count", type=int, default=4, help="number of numbered questions to solve serially (default: 4)")
    main_qwen.add_argument("--model", default=None)
    main_qwen.add_argument("--code-agent", choices=("dsh", "qwen"), default=os.environ.get("M2HARNESS_CODE_AGENT", "dsh"), help="Code Agent 运行时；默认使用持久 DSH Session")
    main_qwen.add_argument("--max-iterations", type=int, default=3, help="总迭代上限：初始实现轮 + 最多两轮返修")
    resume_qwen = commands.add_parser("resume-main-qwen", help="从已归档的 Model→Code 交接恢复主 Harness")
    resume_qwen.add_argument("input", type=Path, help="原始 PDF/图片输入")
    resume_qwen.add_argument("--run-root", type=Path, required=True, help="已有运行根目录")
    resume_qwen.add_argument("--source-run-id", default=None, help="历史归档的 run_id；省略时取运行目录中最近的归档")
    resume_qwen.add_argument("--resume-payload", type=Path, required=True, help="序列化的建模契约与初步路线")
    resume_qwen.add_argument("--resume-iteration", type=int, default=2, help="恢复开始的 solve_problem 迭代号")
    resume_qwen.add_argument("--problem", default="请继续完成 2026B 数学建模题；从已有 q1 返修交接恢复，之后按 q1→q2→q3→q4 串行处理。")
    resume_qwen.add_argument("--model", default=None)
    resume_qwen.add_argument("--code-agent", choices=("dsh", "qwen"), default=os.environ.get("M2HARNESS_CODE_AGENT", "dsh"), help="Code Agent 运行时；默认使用持久 DSH Session")
    resume_qwen.add_argument("--max-iterations", type=int, default=3, help="总迭代上限：初始实现轮 + 最多两轮返修")
    commands.add_parser("catalog")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--input", type=Path, default=None)
    preflight.add_argument("--qwen", action="store_true", help="require the Qwen provider secret for run-qwen")
    status = commands.add_parser("status")
    status.add_argument("question_id", type=UUID)
    verify = commands.add_parser("verify")
    verify.add_argument("--verify-artifacts", action="store_true")
    export = commands.add_parser("export-audit")
    export.add_argument("question_id", type=UUID)
    export.add_argument("output", type=Path)
    export.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    store = HarnessStore(settings.database_path)
    store.initialize()
    if args.command == "init":
        _emit({"database": str(settings.database_path.resolve()), "artifacts": str(settings.artifact_root.resolve()), "schema": 1})
        return 0
    if args.command == "create-project":
        _emit(store.create_project(args.name))
        return 0
    if args.command == "add-question":
        data = _read_input_file(args.problem_file)
        artifact_store = ArtifactStore(settings.artifact_root)
        media_type = mimetypes.guess_type(args.problem_file.name)[0] or "application/octet-stream"
        problem = artifact_store.put_bytes(
            data, project_id=args.project_id, question_id=None, activity_id=None,
            kind=ArtifactKind.PROBLEM, logical_name=args.problem_file.name,
            media_type=media_type,
        )
        question = store.create_question(args.project_id, args.key, args.title, problem)
        registry = SQLiteArtifactRegistry(settings.database_path)
        registry.register(store.get_artifact(question.problem_artifact_id))
        table = canonical_single_question_dag()
        dag = artifact_store.put_bytes(
            table.model_dump_json(indent=2).encode("utf-8"), project_id=question.project_id,
            question_id=question.id, activity_id=None, kind=ArtifactKind.DAG_TASK_TABLE,
            logical_name=f"{question.key}-dag-task-table.json", media_type="application/json",
            metadata={"schema_version": table.schema_version, "terminal_task_id": table.terminal_task_id, "topological_order": table.topological_order()},
        )
        registry.register(store.register_artifact(dag))
        _emit(question)
        return 0
    if args.command == "add-input":
        question = store.get_question(args.question_id)
        artifact_store = ArtifactStore(settings.artifact_root)
        media_type = mimetypes.guess_type(args.input_file.name)[0] or "application/octet-stream"
        artifact = artifact_store.put_bytes(
            _read_input_file(args.input_file), project_id=question.project_id,
            question_id=question.id, activity_id=None, kind=ArtifactKind(args.kind),
            logical_name=args.input_file.name, media_type=media_type,
        )
        registered = store.register_artifact(artifact)
        SQLiteArtifactRegistry(settings.database_path).register(registered)
        _emit(registered)
        return 0
    if args.command == "run-question":
        if not args.executor:
            raise SystemExit("--executor must be followed by an executable and arguments")
        executor = CommandActivityExecutor(
            args.executor, timeout_seconds=settings.activity_timeout_seconds,
            pass_env=("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MAX_OUTPUT_TOKENS", "M2HARNESS_MODEL_HOSTS", "QWEN_ENABLE_REMOTE_CODE_INTERPRETER"),
            environment={"M2HARNESS_ARTIFACT_ROOT": str(settings.artifact_root.resolve()), "M2HARNESS_SKILL_ROOT": str(default_skill_root())},
        )
        workflow = SingleQuestionWorkflow(settings, executor, artifact_registry=SQLiteArtifactRegistry(settings.database_path))
        _emit(workflow.run_until_terminal(args.question_id, args.worker_id))
        return 0
    if args.command == "run-qwen":
        runtime = build_local_runtime(database_path=settings.database_path, artifact_root=settings.artifact_root, allow_host_sandbox=True)
        if not runtime.sandbox.available and not args.allow_remote_code_interpreter:
            raise RuntimeError("local_execution_disabled: set M2HARNESS_SANDBOX_BACKEND=host or configure Docker/VM; --allow-remote-code-interpreter is provider-only and cannot complete a report")
        if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
            raise RuntimeError("DASHSCOPE_API_KEY or DEEPSEEK_API_KEY is required for run-qwen; inject it through the process environment or a secret provider")
        adapter = _resolve_adapter_path(args.adapter)
        if not adapter.is_file():
            raise FileNotFoundError(f"Qwen adapter not found: {adapter}")
        environment = {"M2HARNESS_ARTIFACT_ROOT": str(settings.artifact_root.resolve()), "M2HARNESS_SKILL_ROOT": str(default_skill_root())}
        if args.model:
            environment["QWEN_MODEL"] = args.model
        if args.allow_remote_code_interpreter and not runtime.sandbox.available:
            # This is an explicit preliminary-provider mode.  The workflow's
            # Qwen evidence gate still prevents a remote result from reaching
            # APPROVED/COMPLETED.
            environment["QWEN_ENABLE_REMOTE_CODE_INTERPRETER"] = "1"
        executor = CommandActivityExecutor(
            [sys.executable, str(adapter)], timeout_seconds=settings.activity_timeout_seconds,
            pass_env=("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MAX_OUTPUT_TOKENS", "M2HARNESS_MODEL_HOSTS", "QWEN_ENABLE_REMOTE_CODE_INTERPRETER"), environment=environment,
        )
        if runtime.sandbox.available:
            executor = SandboxedActivityExecutor(
                executor, runtime.sandbox, ArtifactStore(settings.artifact_root),
                timeout_seconds=min(settings.activity_timeout_seconds, 600),
                image_digest=getattr(runtime.sandbox, "image_digest", None),
            )
        workflow = SingleQuestionWorkflow(settings, executor, artifact_registry=runtime.artifact_registry)
        _emit(workflow.run_until_terminal(args.question_id, args.worker_id))
        return 0
    if args.command == "run-main-qwen":
        from adapters.qwen_solve_problem import (
            QwenChatClient,
            QwenPaperComposer,
            build_qwen_solve_problem_service,
            build_dsh_solve_problem_service,
        )

        data = _read_input_file(args.input)
        if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
            raise RuntimeError("DASHSCOPE_API_KEY or DEEPSEEK_API_KEY is required for run-main-qwen; inject it through the process environment or a secret provider")
        if args.max_iterations < 1 or args.max_iterations > 20:
            raise ValueError("--max-iterations must be between 1 and 20")
        if args.question_count < 1 or args.question_count > 4:
            raise ValueError("--question-count must be between 1 and 4")
        media_type = mimetypes.guess_type(args.input.name)[0] or "application/octet-stream"
        if media_type == "application/pdf" and len(data) > 150 * 1024 * 1024:
            raise ValueError("Qwen PDF input exceeds the 150MB provider limit")
        multimodal = MultimodalInput(
            logical_name=args.input.name, media_type=media_type,
            data_base64=base64.b64encode(data).decode("ascii"),
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        )
        runtime = build_local_runtime(database_path=settings.database_path, artifact_root=settings.artifact_root, allow_host_sandbox=True)
        if not runtime.sandbox.available:
            raise RuntimeError("local_execution_disabled: run-main-qwen requires the trusted host or Docker/VM sandbox")
        staged_name = Path(args.input.name).name
        staged_path = runtime.tool_environment.resolve_workspace(Path("inputs") / staged_name)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_bytes(data)
        client = QwenChatClient(
            base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=args.model or os.environ.get("QWEN_MODEL", "qwen3.8-max"),
        )
        service_builder = build_dsh_solve_problem_service if args.code_agent == "dsh" else build_qwen_solve_problem_service
        service = service_builder(
            sandbox=runtime.sandbox, workspace_root=runtime.tool_environment.workspace_root,
            research_agent=runtime.research_service, skills=runtime.skills,
            tool_runtime=runtime.tool_runtime, capabilities=runtime.capabilities,
            archive_writer=runtime.main_harness.report_store,
            client=client, max_iterations=args.max_iterations,
        )
        runtime = runtime.attach_solve_problem_service(service)
        scope_prompt = (
            f"{args.problem}\n\nThe Main Harness owns a serial TODO list of {args.question_count} numbered questions. "
            "Each solve_problem call must analyze exactly its own question scope and must not solve, summarize, "
            "formulate, code, validate, or report results for later questions. Previous-question results arrive "
            "only through the compressed dependency context."
        )
        base_context = SolveProblemContext(
            multimodal_inputs=(multimodal,),
            readonly_files=(ReadOnlyFileReference(
                relative_path=str((Path("inputs") / staged_name).as_posix()),
                purpose="Original problem statement or source file selected by the user; read-only input for this solve.",
                role=ReadOnlyFileRole.PROBLEM, media_type=media_type,
                sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
            ),),
            metadata={
                "staged_input_relative_path": str(Path("inputs") / staged_name),
                "scope": args.scope,
            },
        )
        from m2harness.application.langgraph_main_harness import LangGraphMainHarness
        graph_runner = LangGraphMainHarness(
            runtime.main_harness,
            checkpoint_path=runtime.tool_environment.workspace_root.parent / "main-harness-checkpoints.sqlite",
            context_for_task=lambda _task_id, _state: base_context,
        )
        state = graph_runner.run(
            scope_prompt,
            canonical_main_harness_dag(scope=args.scope, task_problem=scope_prompt, question_count=args.question_count),
            composer=QwenPaperComposer(client, skills=runtime.skills),
            max_iterations=args.max_iterations,
        )
        if state.final_report is None or state.final_latex_paper is None:
            run_report_root = runtime.tool_environment.workspace_root / "reports" / "runs" / str(state.run_id)
            raise RuntimeError(
                "Main Harness paper composer returned no final publication; "
                f"run_id={state.run_id}; probe={run_report_root / 'probe.md'}; "
                f"probe_ndjson={run_report_root / 'probe.ndjson'}; exchanges={run_report_root / 'tasks'}"
            )
        render_definition = runtime.tools.get("report_render")
        if render_definition is None:
            raise RuntimeError("report_render tool is not registered")
        render_resolution = runtime.capabilities.resolve([
            CapabilityRequirement(capability=render_definition.required_capability, reason="Main Harness final publication"),
        ])
        render_call = ToolCall(
            call_id=uuid4(), tool_name=render_definition.name, tool_version=render_definition.version,
            activity_id=uuid4(), session_id=state.run_id,
            idempotency_key=f"main-harness:{state.run_id}:publish-paper",
            arguments={"markdown": state.final_report.markdown, "title": state.final_report.title, "path": "reports/final-question-report.md", "latex": state.final_latex_paper.text, "latex_path": "reports/final-question-paper.tex", "overwrite": True},
            requested_at=utc_now(),
        )
        rendered = runtime.tool_runtime.execute(render_call, render_resolution)
        if not rendered.ok:
            raise RuntimeError(f"final report rendering failed: {rendered.error_message}")
        _emit({"todo": runtime.main_harness.todo_view(state), "published_files": rendered.output})
        return 0
    if args.command == "resume-main-qwen":
        from adapters.qwen_solve_problem import (
            QwenChatClient,
            QwenPaperComposer,
            build_qwen_solve_problem_service,
            build_dsh_solve_problem_service,
        )

        if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
            raise RuntimeError("DASHSCOPE_API_KEY or DEEPSEEK_API_KEY is required for resume-main-qwen; inject it through the process environment or a secret provider")
        if args.resume_iteration < 1 or args.resume_iteration > 3:
            raise ValueError("--resume-iteration must be between 1 and 3")
        if args.max_iterations < args.resume_iteration or args.max_iterations > 20:
            raise ValueError("--max-iterations must cover the resume iteration and be at most 20")
        payload = json.loads(args.resume_payload.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("modeling"), dict) or not isinstance(payload.get("preliminary"), list):
            raise ValueError("resume payload must contain modeling and preliminary objects")
        data = _read_input_file(args.input)
        media_type = mimetypes.guess_type(args.input.name)[0] or "application/octet-stream"
        run_root = args.run_root.resolve()
        runtime = build_local_runtime(
            workspace_root=run_root / ".m2harness" / "workspace",
            artifact_root=run_root / ".m2harness" / "artifacts",
            database_path=run_root / ".m2harness" / "state.db",
            allow_host_sandbox=True,
        )
        if not runtime.sandbox.available:
            raise RuntimeError("local_execution_disabled: resume-main-qwen requires the trusted local host sandbox")
        staged_name = Path(args.input.name).name
        staged_path = runtime.tool_environment.resolve_workspace(Path("inputs") / staged_name)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_bytes(data)
        client = QwenChatClient(
            base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=args.model or os.environ.get("QWEN_MODEL", "qwen3.8-max"),
        )
        service_builder = build_dsh_solve_problem_service if args.code_agent == "dsh" else build_qwen_solve_problem_service
        service = service_builder(
            sandbox=runtime.sandbox, workspace_root=runtime.tool_environment.workspace_root,
            research_agent=runtime.research_service, skills=runtime.skills,
            tool_runtime=runtime.tool_runtime, capabilities=runtime.capabilities,
            archive_writer=runtime.main_harness.report_store,
            client=client, max_iterations=args.max_iterations,
        )
        runtime = runtime.attach_solve_problem_service(service)
        workspace_root = runtime.tool_environment.workspace_root
        source_run_id = getattr(args, "source_run_id", None)
        if not source_run_id:
            candidates = sorted((workspace_root / "reports" / "runs").glob("*"), key=lambda item: item.stat().st_mtime if item.exists() else 0)
            source_run_id = candidates[-1].name if candidates else ""
        q1_prefix = f"reports/runs/{source_run_id}/tasks/q1/attempt-1/exchanges"

        def resume_reference(relative_path: str, role: ReadOnlyFileRole, purpose: str) -> ReadOnlyFileReference | None:
            target = (workspace_root / relative_path).resolve()
            if not target.is_file() or target.is_symlink() or workspace_root not in target.parents:
                return None
            raw = target.read_bytes()
            return ReadOnlyFileReference(
                relative_path=relative_path, purpose=purpose, role=role, owner_task_id="q1",
                media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
                sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
            )

        readonly: list[ReadOnlyFileReference] = []
        original_ref = ReadOnlyFileReference(
            relative_path=f"inputs/{staged_name}", purpose="原始题目输入；恢复流程只读使用。",
            role=ReadOnlyFileRole.PROBLEM, media_type=media_type,
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        )
        readonly.append(original_ref)
        for relative_path, role, purpose in (
            (f"{q1_prefix}/iteration-1/modeling/modeling_report.md", ReadOnlyFileRole.REFERENCE, "上一轮已接受的完整建模报告。"),
            (f"{q1_prefix}/iteration-1/modeling/preliminary-q1-mip-baseline.md", ReadOnlyFileRole.REFERENCE, "上一轮初步路线报告。"),
            (f"{q1_prefix}/iteration-1/review/review_report.md", ReadOnlyFileRole.REFERENCE, "上一轮 Review 决定与理由。"),
            (f"{q1_prefix}/iteration-1/review/revision_instructions.md", ReadOnlyFileRole.REFERENCE, "上一轮 Review 的具体返修指令。"),
            (f"{q1_prefix}/iteration-1/handoff/review-to-next-stage.md", ReadOnlyFileRole.REFERENCE, "上一轮 Review→返修交接。"),
            (f"{q1_prefix}/iteration-2/handoff/model-to-code.md", ReadOnlyFileRole.REFERENCE, "已保存的第 2 轮 Model→Code 交接。"),
            (".m2harness-code/q1/iteration-2/solve_q1.py", ReadOnlyFileRole.GENERATED, "第 2 轮已生成的待返修源文件。"),
        ):
            reference = resume_reference(relative_path, role, purpose)
            if reference is not None:
                readonly.append(reference)
        multimodal = MultimodalInput(
            logical_name=staged_name, media_type=media_type,
            data_base64=base64.b64encode(data).decode("ascii"),
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        )
        scope_prompt = (
            f"{args.problem}\n\nMain Harness 维护 q1→q2→q3→q4 串行 TODO。"
            "当前只恢复 q1 第 2 轮 Code 阶段；不得重新建模，不得处理后续问题。"
        )
        resume_context = SolveProblemContext(
            multimodal_inputs=(multimodal,), readonly_files=tuple(readonly),
            instructions=("Review 已接受建模，仅执行 code 返修；完成后交给 Review 验收。",),
            metadata={
                "scope": "question-1", "resume_iteration": args.resume_iteration,
                "resume_modeling": payload["modeling"], "resume_preliminary": payload["preliminary"],
                "resume_revision_target": payload.get("revision_target", "code"),
            },
        )
        normal_context = resume_context.model_copy(update={
            "metadata": {"scope": "question-1"},
            "instructions": (),
        })
        from m2harness.application.langgraph_main_harness import LangGraphMainHarness
        graph_runner = LangGraphMainHarness(
            runtime.main_harness,
            checkpoint_path=runtime.tool_environment.workspace_root.parent / "main-harness-checkpoints.sqlite",
            context_for_task=lambda task_id, _state: resume_context if task_id == "q1" else normal_context,
        )
        state = graph_runner.run(
            scope_prompt,
            canonical_main_harness_dag(scope="question-1", task_problem=scope_prompt, question_count=4),
            composer=QwenPaperComposer(client, skills=runtime.skills),
            max_iterations=args.max_iterations,
        )
        if state.final_report is None or state.final_latex_paper is None:
            run_report_root = runtime.tool_environment.workspace_root / "reports" / "runs" / str(state.run_id)
            raise RuntimeError(
                "Main Harness paper composer returned no final publication; "
                f"run_id={state.run_id}; probe={run_report_root / 'probe.md'}; "
                f"probe_ndjson={run_report_root / 'probe.ndjson'}; exchanges={run_report_root / 'tasks'}"
            )
        render_definition = runtime.tools.get("report_render")
        if render_definition is None:
            raise RuntimeError("report_render tool is not registered")
        render_resolution = runtime.capabilities.resolve([
            CapabilityRequirement(capability=render_definition.required_capability, reason="Main Harness final publication"),
        ])
        render_call = ToolCall(
            call_id=uuid4(), tool_name=render_definition.name, tool_version=render_definition.version,
            activity_id=uuid4(), session_id=state.run_id,
            idempotency_key=f"main-harness:{state.run_id}:publish-paper",
            arguments={"markdown": state.final_report.markdown, "title": state.final_report.title, "path": "reports/final-question-report.md", "latex": state.final_latex_paper.text, "latex_path": "reports/final-question-paper.tex", "overwrite": True},
            requested_at=utc_now(),
        )
        rendered = runtime.tool_runtime.execute(render_call, render_resolution)
        if not rendered.ok:
            raise RuntimeError(f"final report rendering failed: {rendered.error_message}")
        _emit({"todo": runtime.main_harness.todo_view(state), "published_files": rendered.output})
        return 0
    if args.command == "catalog":
        runtime = build_local_runtime(database_path=settings.database_path, artifact_root=settings.artifact_root)
        _emit({
            "skills": [item.model_dump(mode="json") for item in runtime.skills.list()],
            "tools": [item.model_dump(mode="json") for item in runtime.tools.catalog()],
            "capabilities": [item.model_dump(mode="json") for item in runtime.capabilities.list()],
        })
        return 0
    if args.command == "preflight":
        runtime = build_local_runtime(database_path=settings.database_path, artifact_root=settings.artifact_root)
        adapter = _resolve_adapter_path(Path("adapters/qwen38_max.py"))
        provider_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        provider_parts = urlsplit(provider_url)
        provider_host = (provider_parts.hostname or "").lower()
        configured_hosts = {item.strip().lower() for item in os.environ.get("M2HARNESS_MODEL_HOSTS", "dashscope.aliyuncs.com,api.deepseek.com").split(",") if item.strip()}
        provider_is_deepseek = provider_host.endswith("deepseek.com")
        checks = {
            "database_parent_writable": _writable_directory(settings.database_path.parent.resolve()),
            "artifact_root_ready": _writable_directory(settings.artifact_root.resolve()),
            "skills_loaded": len(runtime.skills.list()) > 0,
            "tools_registered": len(runtime.tools.catalog()) >= 10,
            "workflow_events_ready": runtime.events.verify() >= 0,
            "sandbox_available": runtime.sandbox.available,
            "docker_cli_present": bool(shutil.which("docker")),
            "qwen_adapter_present": adapter.is_file(),
            "model_api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY") if provider_is_deepseek else os.environ.get("DASHSCOPE_API_KEY")),
            "model_provider_https": provider_parts.scheme == "https" and bool(provider_host),
            "model_provider_allowlisted": provider_host in configured_hosts,
        }
        if args.input is not None:
            checks["input_present"] = args.input.is_file()
            if args.input.is_file():
                checks["input_nonempty"] = args.input.stat().st_size > 0
                if args.input.suffix.lower() == ".pdf":
                    checks["input_within_qwen_pdf_limit"] = args.input.stat().st_size <= 150 * 1024 * 1024
                    with args.input.open("rb") as stream:
                        checks["input_pdf_magic"] = stream.read(4) == b"%PDF"
        # Docker is optional in the trusted single-machine scope; it becomes
        # relevant only when the operator explicitly selects the docker
        # backend. The provider secret is likewise conditional on --qwen.
        required_checks = {key: value for key, value in checks.items() if key not in {"model_api_key_present", "docker_cli_present"}}
        if os.environ.get("M2HARNESS_SANDBOX_BACKEND", "host").lower() == "docker":
            required_checks["docker_cli_present"] = checks["docker_cli_present"]
        structural = all(required_checks.values()) and (not args.qwen or checks["model_api_key_present"])
        _emit({"status": "ready" if structural else "blocked", "checks": checks, "note": "the first-mile Harness uses the trusted local host backend by default; set M2HARNESS_SANDBOX_BACKEND=docker for isolation or =none to fail closed; --qwen makes the provider secret mandatory"})
        return 0 if structural else 1
    if args.command == "status":
        _emit({
            "question": store.get_question(args.question_id).model_dump(mode="json"),
            "activities": [item.model_dump(mode="json") for item in store.list_activities(args.question_id)],
            "artifacts": [item.model_dump(mode="json") for item in store.list_artifacts(args.question_id)],
            "claim_evidence": [item.model_dump(mode="json") for item in store.list_claim_evidence(args.question_id)],
            "events": [item.model_dump(mode="json") for item in store.list_events(args.question_id)],
        })
        return 0
    if args.command == "verify":
        events = store.verify_event_chain()
        artifacts = 0
        if args.verify_artifacts:
            artifact_store = ArtifactStore(settings.artifact_root)
            for artifact in store.list_all_artifacts():
                artifact_store.read(artifact); artifacts += 1
        audit = SQLiteToolAuditStore(settings.database_path)
        runtime_artifacts = SQLiteArtifactRegistry(settings.database_path).verify(ArtifactStore(settings.artifact_root))
        claims = store.verify_all_claim_evidence()
        tool_audit_events = audit.verify()
        _emit({"event_chain": "valid", "events": events, "artifacts_verified": artifacts, "runtime_artifacts_verified": runtime_artifacts, "claim_evidence": "valid", "claim_evidence_records": claims, "tool_audit_chain": "valid", "tool_audit_events": tool_audit_events})
        return 0
    if args.command == "export-audit":
        question = store.get_question(args.question_id)
        claims = store.list_claim_evidence(args.question_id)
        claim_count = store.verify_claim_evidence(args.question_id)
        for claim in claims:
            for artifact_id in claim.evidence_artifact_ids:
                ArtifactStore(settings.artifact_root).read(store.get_artifact(artifact_id))
        payload = {
            "schema": "m2harness/audit-bundle/v1",
            "question": question.model_dump(mode="json"),
            "activities": [item.model_dump(mode="json") for item in store.list_activities(args.question_id)],
            "artifacts": [item.model_dump(mode="json") for item in store.list_artifacts(args.question_id)],
            "claim_evidence": [item.model_dump(mode="json") for item in claims],
            "events": [item.model_dump(mode="json") for item in store.list_events(args.question_id)],
            "integrity": {"event_chain_events": store.verify_event_chain(), "claim_evidence_records": claim_count},
        }
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"audit output exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
        _emit({"output": str(args.output.resolve()), "schema": payload["schema"], "claims": len(claims), "events": len(payload["events"])})
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
