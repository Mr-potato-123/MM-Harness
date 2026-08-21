"""Durable report hierarchy and path index for Main Harness runs."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path
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

    def _index_existing(self, reference: ReadOnlyFileReference, *, task_id: str, attempt: int, iteration: int) -> ReportFileRecord:
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
