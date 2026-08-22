"""Durable report hierarchy and path index for Main Harness runs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from pydantic import Field

from m2harness.domain.solve_problem import ReadOnlyFileReference, ReadOnlyFileRole, SolveProblemReport
from m2harness.models import ProducedArtifact, ReportPayload, StrictModel


class ReportFileRecord(StrictModel):
    relative_path: str
    purpose: str
    role: ReadOnlyFileRole
    task_id: str
    attempt: int = Field(ge=1)
    iteration: int | None = Field(default=None, ge=1)
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    def as_readonly(self) -> ReadOnlyFileReference:
        return ReadOnlyFileReference(
            relative_path=self.relative_path, purpose=self.purpose, role=self.role,
            owner_task_id=self.task_id, media_type=self.media_type,
            sha256=self.sha256, size_bytes=self.size_bytes,
        )


class RunReportStore:
    """Atomically materialize every meaningful solve exchange under workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._run_names: dict[str, str] = {}
        self._probe_locks: dict[str, threading.Lock] = {}
        self._probe_locks_guard = threading.Lock()

    def register_run(self, run_id: UUID, run_name: str, *, metadata: Mapping[str, Any] | None = None) -> Path:
        """Register a traceable name for one UUID run."""

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip(".-")[:180]
        if not safe_name:
            raise ValueError("run_name must contain at least one safe character")
        key = str(run_id)
        existing = self._run_names.get(key)
        if existing is not None and existing != safe_name:
            raise ValueError(f"run {key} is already registered as {existing}")
        self._run_names[key] = safe_name
        base = self._run_base(run_id)
        payload = {
            "schema_version": 1,
            "run_id": key,
            "run_name": safe_name,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": _probe_value(metadata or {}),
        }
        self._write(
            base / "run.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            purpose="Human-readable identity manifest for one isolated run.",
            role=ReadOnlyFileRole.REFERENCE, task_id="run", attempt=1,
            iteration=None, media_type="application/json",
        )
        return base

    def _run_base(self, run_id: UUID) -> Path:
        return Path("reports") / "runs" / self._run_names.get(str(run_id), str(run_id))

    def archive_markdown(
        self,
        run_id: UUID,
        *,
        task_id: str,
        attempt: int,
        iteration: int,
        stage: str,
        name: str,
        markdown: str,
        purpose: str,
        role: ReadOnlyFileRole,
    ) -> ReadOnlyFileReference:
        """Persist one live Agent exchange before the next stage is called."""

        if stage == "modeling":
            relative = self._run_base(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "modeling" / (self._safe_name(name) + ".md")
        elif stage == "coding":
            relative = self._run_base(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "coding" / (self._safe_name(name) + ".md")
        elif stage == "review":
            relative = self._run_base(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "review" / (self._safe_name(name) + ".md")
        elif stage == "handoff":
            relative = self._run_base(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "handoff" / (self._safe_name(name) + ".md")
        else:
            raise ValueError(f"unknown solve_problem archive stage: {stage}")
        record = self._write(
            relative, markdown, purpose=purpose, role=role, task_id=task_id,
            attempt=attempt, iteration=iteration, media_type="text/markdown",
        )
        return record.as_readonly()

    def record_probe(
        self,
        run_id: UUID,
        *,
        task_id: str,
        attempt: int,
        iteration: int | None,
        event: str,
        actor: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one bounded stage probe for live operator inspection.

        ``probe.ndjson`` is machine-readable and append-only for tailing while
        a run is active.  ``probe.md`` is the same stream rendered as a short
        Markdown ledger.  Details are deliberately summarized: source code,
        multimodal bytes, prompts and credentials never belong in a probe.
        """

        if attempt < 1 or not event.strip() or not actor.strip() or not status.strip():
            raise ValueError("invalid solve probe identity")
        run_key = self._run_names.get(str(run_id), str(run_id))
        base = self._run_base(run_id)
        ndjson = (base / "probe.ndjson").as_posix()
        markdown = (base / "probe.md").as_posix()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_key,
            "task_id": task_id,
            "attempt": attempt,
            "iteration": iteration,
            "event": event,
            "actor": actor,
            "status": status,
            "details": _probe_value(details or {}),
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        md_line = (
            f"- `{payload['timestamp']}` · `{task_id}` · 迭代 `{iteration or '-'}` · "
            f"**{actor}** · `{event}` · `{status}` · "
            f"`{json.dumps(payload['details'], ensure_ascii=False, separators=(',', ':'))}`\n"
        )
        with self._probe_lock(run_key):
            self._append_text(Path(ndjson), line, encoding="utf-8")
            target = (self.workspace_root / markdown).resolve()
            if not target.exists():
                self._write(Path(markdown), "# Solve Problem 探针\n\n" +
                            "本文件记录 Model、Code、Code→Model 返修与文件披露边界；完整交接内容见同一 run 下各题目的 `exchanges/`。\n\n",
                            purpose="Live operator probe index for solve_problem.", role=ReadOnlyFileRole.REFERENCE,
                            task_id="probe", attempt=1, iteration=None, media_type="text/markdown")
            self._append_text(Path(markdown), md_line, encoding="utf-8")

    def _probe_lock(self, run_key: str) -> threading.Lock:
        with self._probe_locks_guard:
            return self._probe_locks.setdefault(run_key, threading.Lock())

    def _append_text(self, relative: Path, text: str, *, encoding: str) -> None:
        target = (self.workspace_root / relative).resolve()
        if self.workspace_root not in target.parents:
            raise ValueError("probe path escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def persist(self, run_id: UUID, report: SolveProblemReport, *, attempt: int) -> tuple[ReportFileRecord, ...]:
        base = self._run_base(run_id) / "tasks" / report.task_id / f"attempt-{attempt}"
        records: list[ReportFileRecord] = []
        for snapshot in report.iterations:
            model_dir = base / "modeling" / f"iteration-{snapshot.iteration}"
            for preliminary in snapshot.preliminary_reports:
                name = "preliminary-" + self._safe_name(preliminary.branch_id) + ".md"
                canonical = base / "exchanges" / f"iteration-{snapshot.iteration}" / "modeling" / name
                content = _pointer_or_fallback(
                    self.workspace_root,
                    canonical,
                    title=f"建模路线 `{preliminary.branch_id}` 索引",
                    fallback=preliminary.report.markdown,
                )
                records.append(self._write(
                    model_dir / name, content,
                    purpose=f"题目 {report.task_id} 的初步建模路线 {preliminary.branch_id} 正文索引。",
                    role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                    attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
                ))
            canonical_model = base / "exchanges" / f"iteration-{snapshot.iteration}" / "modeling" / "modeling_report.md"
            records.append(self._write(
                model_dir / "modeling_report.md",
                _pointer_or_fallback(
                    self.workspace_root,
                    canonical_model,
                    title="统一建模报告索引",
                    fallback=snapshot.modeling_report.report.markdown,
                ),
                purpose=f"题目 {report.task_id} 第 {snapshot.iteration} 轮统一建模报告正文索引。",
                role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
            ))
            coding_dir = base / "coding" / f"run-{snapshot.iteration:03d}"
            # The canonical Code→Model report lives in the exchange handoff
            # written by solve_problem.  Keep this legacy index tiny so old
            # readers can discover it without creating a second full report.
            code_handoff = base / "exchanges" / f"iteration-{snapshot.iteration}" / "handoff" / "code-to-model.md"
            records.append(self._write(
                coding_dir / "coding_report.md",
                "# Code→Model 报告索引\n\n"
                f"本文件不复制执行报告正文；请读取 `{code_handoff.as_posix()}`。\n",
                purpose=f"题目 {report.task_id} 第 {snapshot.iteration} 轮 Code→Model 正文索引。",
                role=ReadOnlyFileRole.DEPENDENCY_OUTPUT, task_id=report.task_id,
                attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
            ))
            for artifact in snapshot.coding_report.artifacts:
                name = self._safe_name(artifact.logical_name)
                data = artifact.text.encode("utf-8") if artifact.text is not None else base64.b64decode(artifact.base64 or "", validate=True)
                records.append(self._write_bytes(
                    coding_dir / "artifacts" / name, data,
                    purpose=f"Generated solve file {artifact.logical_name} from {report.task_id}, run {snapshot.iteration}.",
                    role=ReadOnlyFileRole.GENERATED, task_id=report.task_id,
                    attempt=attempt, iteration=snapshot.iteration, media_type=artifact.media_type,
                ))
            for generated in snapshot.coding_report.generated_files:
                records.append(self._index_existing(
                    generated, task_id=report.task_id, attempt=attempt,
                    iteration=snapshot.iteration,
                ))
            review_dir = base / "review" / f"iteration-{snapshot.iteration}"
            if snapshot.review.decision.value == "approve":
                transfer_reason = "审查已批准且证据满足门槛；交给最终报告阶段，只能使用已审查声明。"
            else:
                transfer_reason = (
                    f"审查未批准当前产物；按最窄目标 `{snapshot.review.revision_target.value}` "
                    "转回返修，并执行下方具体指令。"
                )
            revision_handoff = base / "exchanges" / f"iteration-{snapshot.iteration}" / "handoff" / "model-to-code-revision.md"
            records.append(self._write(
                review_dir / "review_report.md",
                "# Model Agent 返修意见索引\n\n"
                f"本文件不复制返修报告正文；请读取 `{revision_handoff.as_posix()}`。\n",
                purpose=f"题目 {report.task_id} 第 {snapshot.iteration} 轮 Model→Code 返修交接正文索引。",
                role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
            ))
            if snapshot.review.revision_instructions:
                records.append(self._write(
                    review_dir / "revision_instructions.md",
                    "# Model Agent → Code Agent 返修指令索引\n\n"
                    f"正文已合并到 `{revision_handoff.as_posix()}`，此处不重复展开。\n",
                    purpose=f"题目 {report.task_id} 第 {snapshot.iteration} 轮 Code 返修指令正文索引。",
                    role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                    attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
                ))
        for archive in report.archive_files:
            records.append(self._index_existing(
                archive, task_id=report.task_id, attempt=attempt,
                iteration=_archive_iteration(archive.relative_path),
            ))
        if report.final_report is not None:
            records.append(self._write(
                base / "final" / "solution_report.md", report.final_report.markdown,
                purpose=f"上游题目 {report.task_id} 的最终单题报告。",
                role=ReadOnlyFileRole.DEPENDENCY_SOLUTION, task_id=report.task_id,
                attempt=attempt, iteration=report.iteration_count or None, media_type="text/markdown",
            ))
        records.append(self._write(
            base / "solve_problem_report.json", report.model_dump_json(indent=2),
            purpose=f"题目 {report.task_id} 的 solve_problem 完整结构化审计快照。",
            role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
            attempt=attempt, iteration=None, media_type="application/json",
        ))
        # This is the single upstream package index for the question.  It is
        # deliberately an index, not another copy of any report: Main Harness
        # discloses it to the next serial question, and the next Model Agent
        # opens only the listed read-only paths it actually needs.
        package_path = base / "question-package.md"
        package_records = tuple(dict.fromkeys(item.relative_path for item in records))
        records.append(self._write(
            package_path,
            _question_package_markdown(report, package_records, base=base),
            purpose=f"Read-only question output package index for upstream task {report.task_id}.",
            role=ReadOnlyFileRole.DEPENDENCY_SOLUTION, task_id=report.task_id,
            attempt=attempt, iteration=report.iteration_count or None, media_type="text/markdown",
        ))
        run_base = self._run_base(run_id)
        for relative, media_type in ((run_base / "probe.md", "text/markdown"), (run_base / "probe.ndjson", "application/x-ndjson")):
            target = (self.workspace_root / relative).resolve()
            if target.is_file() and not target.is_symlink():
                data = target.read_bytes()
                records.append(ReportFileRecord(
                    relative_path=relative.as_posix(),
                    purpose="Live solve_problem probe ledger for operator inspection.",
                    role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                    attempt=attempt, iteration=None, media_type=media_type,
                    sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                ))
        return tuple(records)

    def persist_publication(self, run_id: UUID, final_report: ReportPayload, final_latex: ProducedArtifact) -> tuple[ReportFileRecord, ...]:
        base = self._run_base(run_id) / "final"
        latex = final_latex.text.encode("utf-8") if final_latex.text is not None else base64.b64decode(final_latex.base64 or "", validate=True)
        return (
            self._write(
                base / "final-question-report.md", final_report.markdown,
                purpose="Main Harness final report synthesized from accepted problem reports.",
                role=ReadOnlyFileRole.DEPENDENCY_SOLUTION, task_id="publish-paper",
                attempt=1, iteration=None, media_type="text/markdown",
            ),
            self._write_bytes(
                base / "final-question-paper.tex", latex,
                purpose="Main Harness final LaTeX publication source.",
                role=ReadOnlyFileRole.GENERATED, task_id="publish-paper",
                attempt=1, iteration=None, media_type=final_latex.media_type,
            ),
        )

    def _write(self, relative: Path, text: str, **metadata) -> ReportFileRecord:
        return self._write_bytes(relative, text.encode("utf-8"), **metadata)

    def _write_bytes(self, relative: Path, data: bytes, **metadata) -> ReportFileRecord:
        target = (self.workspace_root / relative).resolve()
        if self.workspace_root not in target.parents:
            raise ValueError("report path escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".m2h-report-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ReportFileRecord(
            relative_path=target.relative_to(self.workspace_root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), **metadata,
        )

    def _index_existing(self, reference: ReadOnlyFileReference, *, task_id: str, attempt: int, iteration: int | None) -> ReportFileRecord:
        target = (self.workspace_root / reference.relative_path).resolve()
        if self.workspace_root not in target.parents or not target.is_file() or target.is_symlink():
            raise ValueError(f"generated output is not a safe workspace file: {reference.relative_path}")
        data = target.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if reference.sha256 is not None and digest != reference.sha256:
            raise ValueError(f"generated output digest changed: {reference.relative_path}")
        return ReportFileRecord(
            relative_path=reference.relative_path, purpose=reference.purpose,
            role=reference.role, task_id=task_id, attempt=attempt, iteration=iteration,
            media_type=reference.media_type, sha256=digest, size_bytes=len(data),
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        name = Path(value).name
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
        return sanitized[:200] or "artifact"


def _pointer_or_fallback(workspace_root: Path, canonical: Path, *, title: str, fallback: str) -> str:
    """Avoid copying a canonical exchange into a second legacy directory."""

    target = (workspace_root / canonical).resolve()
    if target.is_file() and not target.is_symlink():
        relative = canonical.as_posix()
        return f"# {title}\n\n正文已归档于 `{relative}`；本文件仅作索引，不复制正文。\n"
    return fallback if fallback.endswith("\n") else fallback + "\n"


def _question_package_markdown(
    report: SolveProblemReport,
    record_paths: tuple[str, ...],
    *,
    base: Path,
) -> str:
    """Render the compact, read-only output index returned between questions."""

    prefix = base.as_posix().rstrip("/") + "/"
    paths = [path for path in record_paths if path.startswith(prefix)]
    handoffs = [path for path in paths if "/exchanges/" in path]
    generated = [path for path in paths if "/coding/" in path and "/artifacts/" in path]
    source_files = [path for path in paths if ".m2harness-code/" in path or path.endswith(".py")]
    final_path = f"{base.as_posix()}/final/solution_report.md" if report.final_report is not None else None
    lines = [
        "# 单题输出包（只读索引）",
        "",
        f"- 题目：`{report.task_id}`",
        f"- 状态：`{report.status.value}`",
        f"- 总迭代数：`{report.iteration_count}`（初始实现 + 最多两轮 Code 返修）",
        f"- 总题目报告：`{final_path or '未生成'}`",
        "- 使用规则：Main Harness 只把本索引和下方路径作为下一题的只读输入；不要把整套历史正文复制进下一题上下文。",
        "- 交接规则：首轮 Model→Code 契约完整保存；后续 Model→Code 文档只保存本轮增量返修指令；每轮仅有一个 Code→Model 完整执行报告。",
        "",
        "## 关键交接路径",
        "",
        *(f"- `{path}`" for path in handoffs if path.endswith(("model-to-code.md", "model-to-code-revision.md", "code-to-model.md"))),
        "",
        "## 代码与结果",
        "",
        *(f"- `{path}`" for path in sorted(dict.fromkeys((*source_files, *generated)))),
        "",
        "## 可按需读取的其他文件",
        "",
        *(f"- `{path}`" for path in paths if path not in set(handoffs) and path not in set(source_files) and path not in set(generated) and not path.endswith("question-package.md")),
        "",
        "## 下游交接说明",
        "",
        "- 下一题只接收本题总报告的摘要、声明的 downstream_outputs，以及本索引列出的只读路径。",
        "- 图片、数据和源代码不嵌入本文件；需要时按路径只读打开并核对 sha256。",
    ]
    return "\n".join(lines).strip() + "\n"


def _archive_iteration(relative_path: str) -> int | None:
    match = re.search(r"/exchanges/iteration-(\d+)/", "/" + relative_path.replace("\\", "/"))
    return int(match.group(1)) if match else None


def _probe_value(value: Any, *, depth: int = 0) -> Any:
    """Bound probe details and remove fields that could carry secrets/bytes."""

    if depth > 3:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:32]:
            name = str(key)
            lowered = name.lower()
            if any(token in lowered for token in ("api_key", "access_token", "secret", "password", "base64", "credential")):
                result[name] = "<redacted>"
            else:
                result[name] = _probe_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_probe_value(item, depth=depth + 1) for item in list(value)[:64]]
    if isinstance(value, str):
        return value[:2_000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]
