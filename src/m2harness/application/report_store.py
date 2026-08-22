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
        self._probe_locks: dict[str, threading.Lock] = {}
        self._probe_locks_guard = threading.Lock()

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
            relative = Path("reports") / "runs" / str(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "modeling" / (self._safe_name(name) + ".md")
        elif stage == "coding":
            relative = Path("reports") / "runs" / str(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "coding" / (self._safe_name(name) + ".md")
        elif stage == "review":
            relative = Path("reports") / "runs" / str(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "review" / (self._safe_name(name) + ".md")
        elif stage == "handoff":
            relative = Path("reports") / "runs" / str(run_id) / "tasks" / task_id / f"attempt-{attempt}" / "exchanges" / f"iteration-{iteration}" / "handoff" / (self._safe_name(name) + ".md")
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
        run_key = str(run_id)
        base = Path("reports") / "runs" / run_key
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
        base = Path("reports") / "runs" / str(run_id) / "tasks" / report.task_id / f"attempt-{attempt}"
        records: list[ReportFileRecord] = []
        for snapshot in report.iterations:
            model_dir = base / "modeling" / f"iteration-{snapshot.iteration}"
            for preliminary in snapshot.preliminary_reports:
                name = "preliminary-" + self._safe_name(preliminary.branch_id) + ".md"
                records.append(self._write(
                    model_dir / name, preliminary.report.markdown,
                    purpose=f"Preliminary modeling route {preliminary.branch_id} for {report.task_id}.",
                    role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                    attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
                ))
            records.append(self._write(
                model_dir / "modeling_report.md", snapshot.modeling_report.report.markdown,
                purpose=f"Unified Modeling Report for {report.task_id}, iteration {snapshot.iteration}.",
                role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
            ))
            coding_dir = base / "coding" / f"run-{snapshot.iteration:03d}"
            records.append(self._write(
                coding_dir / "coding_report.md", snapshot.coding_report.report.markdown,
                purpose=f"Coding Report and execution findings for {report.task_id}, run {snapshot.iteration}.",
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
            records.append(self._write(
                review_dir / "review_report.md", "\n".join([
                    f"# Model Agent 代码返修意见：{snapshot.review.decision.value}", "",
                    "本报告来自同一 Model Agent 读取 Code→Model 交接后的返修阶段；不重新建模。",
                    "## 转接理由", "", transfer_reason,
                    "## 理由", "", snapshot.review.rationale,
                    "", "## 返修目标", "", f"`{snapshot.review.revision_target.value}`",
                    "", "## 已接受声明", "", *(f"- {item}" for item in snapshot.review.accepted_claims),
                    "", "## 返修指令", "", *(f"- {item}" for item in snapshot.review.revision_instructions),
                    "", "## 请求读取的只读文件", "", *(f"- `{item}`" for item in snapshot.review.requested_file_paths),
                ]).strip() + "\n",
                purpose=f"Model Agent Code repair decision for {report.task_id}, iteration {snapshot.iteration}.",
                role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
                attempt=attempt, iteration=snapshot.iteration, media_type="text/markdown",
            ))
            if snapshot.review.revision_instructions:
                records.append(self._write(
                    review_dir / "revision_instructions.md",
                    "# Model Agent → Code Agent 返修指令\n\n" + "\n".join(f"- {item}" for item in snapshot.review.revision_instructions) + "\n",
                    purpose=f"Model Agent 在 {report.task_id} 第 {snapshot.iteration} 轮后的 Code 返修指令。",
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
                purpose=f"Accepted final solution report for upstream task {report.task_id}.",
                role=ReadOnlyFileRole.DEPENDENCY_SOLUTION, task_id=report.task_id,
                attempt=attempt, iteration=report.iteration_count or None, media_type="text/markdown",
            ))
        records.append(self._write(
            base / "solve_problem_report.json", report.model_dump_json(indent=2),
            purpose=f"Complete typed solve_problem audit snapshot for {report.task_id}.",
            role=ReadOnlyFileRole.REFERENCE, task_id=report.task_id,
            attempt=attempt, iteration=None, media_type="application/json",
        ))
        run_base = Path("reports") / "runs" / str(run_id)
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
        base = Path("reports") / "runs" / str(run_id) / "final"
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
