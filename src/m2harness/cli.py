"""Operations CLI for the single-question Harness."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from uuid import UUID

from m2harness.artifacts import ArtifactStore
from m2harness.executor import CommandActivityExecutor
from m2harness.models import ArtifactKind, HarnessSettings
from m2harness.store import HarnessStore
from m2harness.workflow import SingleQuestionWorkflow


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    print(json.dumps(value, ensure_ascii=False, indent=2))


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
    status = commands.add_parser("status")
    status.add_argument("question_id", type=UUID)
    verify = commands.add_parser("verify")
    verify.add_argument("--verify-artifacts", action="store_true")
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
        data = args.problem_file.read_bytes()
        artifact_store = ArtifactStore(settings.artifact_root)
        media_type = mimetypes.guess_type(args.problem_file.name)[0] or "application/octet-stream"
        problem = artifact_store.put_bytes(
            data, project_id=args.project_id, question_id=None, activity_id=None,
            kind=ArtifactKind.PROBLEM, logical_name=args.problem_file.name,
            media_type=media_type,
        )
        _emit(store.create_question(args.project_id, args.key, args.title, problem))
        return 0
    if args.command == "add-input":
        question = store.get_question(args.question_id)
        artifact_store = ArtifactStore(settings.artifact_root)
        media_type = mimetypes.guess_type(args.input_file.name)[0] or "application/octet-stream"
        artifact = artifact_store.put_bytes(
            args.input_file.read_bytes(), project_id=question.project_id,
            question_id=question.id, activity_id=None, kind=ArtifactKind(args.kind),
            logical_name=args.input_file.name, media_type=media_type,
        )
        _emit(store.register_artifact(artifact))
        return 0
    if args.command == "run-question":
        if not args.executor:
            raise SystemExit("--executor must be followed by an executable and arguments")
        executor = CommandActivityExecutor(
            args.executor, timeout_seconds=settings.activity_timeout_seconds,
            pass_env=("DASHSCOPE_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL"),
            environment={"M2HARNESS_ARTIFACT_ROOT": str(settings.artifact_root.resolve())},
        )
        workflow = SingleQuestionWorkflow(settings, executor)
        _emit(workflow.run_until_terminal(args.question_id, args.worker_id))
        return 0
    if args.command == "status":
        _emit({
            "question": store.get_question(args.question_id).model_dump(mode="json"),
            "activities": [item.model_dump(mode="json") for item in store.list_activities(args.question_id)],
            "artifacts": [item.model_dump(mode="json") for item in store.list_artifacts(args.question_id)],
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
        _emit({"event_chain": "valid", "events": events, "artifacts_verified": artifacts})
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
