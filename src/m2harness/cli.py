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
from m2harness.domain.solve_problem import SolveProblemContext
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
    parser.add_argument("--max-revisions", type=int, default=3)
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
    main_qwen.add_argument("--problem", default="Solve the attached single mathematical-modeling problem and produce an auditable report.")
    main_qwen.add_argument("--model", default=None)
    main_qwen.add_argument("--max-iterations", type=int, default=3)
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
            pass_env=("DASHSCOPE_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MAX_OUTPUT_TOKENS", "M2HARNESS_MODEL_HOSTS", "QWEN_ENABLE_REMOTE_CODE_INTERPRETER"),
            environment={"M2HARNESS_ARTIFACT_ROOT": str(settings.artifact_root.resolve()), "M2HARNESS_SKILL_ROOT": str(default_skill_root())},
        )
        workflow = SingleQuestionWorkflow(settings, executor, artifact_registry=SQLiteArtifactRegistry(settings.database_path))
        _emit(workflow.run_until_terminal(args.question_id, args.worker_id))
        return 0
    if args.command == "run-qwen":
        runtime = build_local_runtime(database_path=settings.database_path, artifact_root=settings.artifact_root, allow_host_sandbox=True)
        if not runtime.sandbox.available and not args.allow_remote_code_interpreter:
            raise RuntimeError("local_execution_disabled: set M2HARNESS_SANDBOX_BACKEND=host or configure Docker/VM; --allow-remote-code-interpreter is provider-only and cannot complete a report")
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise RuntimeError("DASHSCOPE_API_KEY is required for run-qwen; inject it through the process environment or a secret provider")
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
            pass_env=("DASHSCOPE_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MAX_OUTPUT_TOKENS", "M2HARNESS_MODEL_HOSTS", "QWEN_ENABLE_REMOTE_CODE_INTERPRETER"), environment=environment,
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
        from adapters.qwen_solve_problem import QwenChatClient, QwenPaperComposer, build_qwen_solve_problem_service

        data = _read_input_file(args.input)
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise RuntimeError("DASHSCOPE_API_KEY is required for run-main-qwen; inject it through the process environment or a secret provider")
        if args.max_iterations < 1 or args.max_iterations > 20:
            raise ValueError("--max-iterations must be between 1 and 20")
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
        client = QwenChatClient(model=args.model or os.environ.get("QWEN_MODEL", "qwen3.8-max"))
        service = build_qwen_solve_problem_service(
            sandbox=runtime.sandbox, workspace_root=runtime.tool_environment.workspace_root,
            research_agent=runtime.research_service, client=client, max_iterations=args.max_iterations,
        )
        runtime = runtime.attach_solve_problem_service(service)
        state = runtime.main_harness.start(args.problem, canonical_main_harness_dag())
        state = runtime.main_harness.dispatch(
            state, "q1",
            context=SolveProblemContext(
                multimodal_inputs=(multimodal,),
                metadata={"staged_input_relative_path": str(Path("inputs") / staged_name)},
            ),
            max_iterations=args.max_iterations,
        )
        if not state.reports or state.reports[-1].status.value != "completed":
            _emit(state)
            return 1
        state = runtime.main_harness.generate_paper(state, QwenPaperComposer(client))
        if state.final_report is None or state.final_latex_paper is None:
            raise RuntimeError("Main Harness paper composer returned no final publication")
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
        _emit({"state": state, "published_files": rendered.output})
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
        configured_hosts = {item.strip().lower() for item in os.environ.get("M2HARNESS_MODEL_HOSTS", "dashscope.aliyuncs.com").split(",") if item.strip()}
        checks = {
            "database_parent_writable": _writable_directory(settings.database_path.parent.resolve()),
            "artifact_root_ready": _writable_directory(settings.artifact_root.resolve()),
            "skills_loaded": len(runtime.skills.list()) > 0,
            "tools_registered": len(runtime.tools.catalog()) >= 10,
            "workflow_events_ready": runtime.events.verify() >= 0,
            "sandbox_available": runtime.sandbox.available,
            "docker_cli_present": bool(shutil.which("docker")),
            "qwen_adapter_present": adapter.is_file(),
            "qwen_api_key_present": bool(os.environ.get("DASHSCOPE_API_KEY")),
            "qwen_provider_https": provider_parts.scheme == "https" and bool(provider_host),
            "qwen_provider_allowlisted": provider_host in configured_hosts,
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
        required_checks = {key: value for key, value in checks.items() if key not in {"qwen_api_key_present", "docker_cli_present"}}
        if os.environ.get("M2HARNESS_SANDBOX_BACKEND", "host").lower() == "docker":
            required_checks["docker_cli_present"] = checks["docker_cli_present"]
        structural = all(required_checks.values()) and (not args.qwen or checks["qwen_api_key_present"])
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
