"""Deterministic protocol adapter used for CLI/process-boundary regression tests."""

from __future__ import annotations

import json
import sys

from m2harness.models import (
    ActivityRequest,
    ActivityResponse,
    ArtifactKind,
    CodingStageOutput,
    FinalizeStageOutput,
    ModelingStageOutput,
    ProducedArtifact,
    ReportPayload,
    ReviewStageOutput,
    ReviewVerdict,
    StageKind,
)


def main() -> int:
    request = ActivityRequest.model_validate_json(sys.stdin.buffer.read())
    report = ReportPayload(title=f"{request.stage.value} report", markdown=f"# {request.stage.value}\n\nAuditable process output.", summary="process test", claims=["claim"], limitations=["test adapter"])
    if request.stage == StageKind.MODELING:
        output = ModelingStageOutput(stage=request.stage, report=report, required_validations=["residual"], expected_outputs=["answer"])
    elif request.stage == StageKind.CODING:
        output = CodingStageOutput(stage=request.stage, report=report, execution_succeeded=True, validations={"residual": True}, metrics={"value": 1}, artifacts=[ProducedArtifact(logical_name="execution.log", kind=ArtifactKind.LOG, media_type="text/plain", text="residual=0")])
    elif request.stage == StageKind.REVIEW:
        output = ReviewStageOutput(stage=request.stage, report=report, verdict=ReviewVerdict.APPROVE, accepted_claims=["claim"])
    else:
        output = FinalizeStageOutput(
            stage=request.stage,
            report=report,
            artifacts=[ProducedArtifact(
                logical_name="final-question-report.tex",
                kind=ArtifactKind.FINAL_LATEX_PAPER,
                media_type="text/x-tex",
                text="\\documentclass{article}\n\\title{Final Question Report}\n\\begin{document}\n\\begin{abstract}Auditable process output.\\end{abstract}\n\\section*{Final Question Report}\nAuditable process output.\\end{document}\n",
            )],
        )
    sys.stdout.write(ActivityResponse(idempotency_key=request.idempotency_key, output=output).model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
