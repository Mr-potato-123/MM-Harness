"""Qwen3.8-Max adapters for the target Main Harness semantics.

This module is deliberately separate from the legacy stage adapter.  It maps
the report-driven ``ModelAgentPort`` and ``CodeProposalProvider`` contracts to
one OpenAI-compatible Qwen endpoint, while the actual code execution remains
inside ``LocalPythonCodeHarness`` and the configured local sandbox.

The API key is read only from ``DASHSCOPE_API_KEY`` (or an explicitly supplied
client in tests); it is never embedded in source or reports.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from m2harness.application.solve_problem import ModelAgentPort
from m2harness.application.solve_problem import SolveProblemService, WorkspaceReadOnlyFileReader
from m2harness.domain.code import CodeProposal
from m2harness.domain.media import MultimodalInput
from m2harness.domain.solve_problem import (
    CodingHarnessReport,
    PreliminaryModelingReport,
    SolveProblemContext,
    SolveProblemReview,
    SolveProblemTask,
    UnifiedModelingReport,
)
from m2harness.errors import ActivityExecutionError
from m2harness.infrastructure.code_harness import CodeProposalProvider
from m2harness.models import ReportPayload
from m2harness.models import ArtifactKind, ProducedArtifact
from m2harness.application.compact import compact_text, estimate_tokens
from m2harness.application.agent_prompts import AgentRole, build_agent_system_prompt
from m2harness.application.skill_context import assemble_skill_context


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be one JSON object")
    return value


def _content_parts(context: SolveProblemContext, prompt: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    binary_total = 0
    for item in context.multimodal_inputs:
        # MultimodalInput validates base64, byte count and digest at the
        # boundary.  The adapter still enforces the combined provider budget.
        raw_size = item.size_bytes
        if item.media_type.lower() == "application/pdf" and raw_size > 150 * 1024 * 1024:
            raise ValueError("Qwen PDF input exceeds the 150MB provider limit")
        binary_total += raw_size
        if binary_total > 200 * 1024 * 1024:
            raise ValueError("combined solve_problem multimodal input exceeds 200MB")
        if item.media_type.lower() == "application/pdf":
            parts.append({"type": "file", "file": {"file_data": "data:application/pdf;base64," + item.data_base64, "filename": item.logical_name}})
        elif item.media_type.lower().startswith("image/"):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{item.media_type};base64," + item.data_base64}})
        elif item.media_type.lower().startswith("video/"):
            parts.append({"type": "video_url", "video_url": {"url": f"data:{item.media_type};base64," + item.data_base64}})
        else:
            parts.append({"type": "text", "text": f"Binary input available as {item.logical_name}; media_type={item.media_type}; sha256={item.sha256}; use only as supplied evidence."})
    parts.append({"type": "text", "text": prompt})
    return parts


def _context_json(context: SolveProblemContext) -> dict[str, Any]:
    """Serialize context without duplicating binary multimodal bytes in text."""
    value = context.model_dump(mode="json")
    value["multimodal_inputs"] = [
        {key: item[key] for key in ("logical_name", "media_type", "sha256", "size_bytes")}
        for item in value.get("multimodal_inputs", [])
    ]
    return value


def _compact_provider_payload(payload: dict[str, Any], *, budget_tokens: int = 100_000) -> dict[str, Any]:
    """Compact oversized structured prompts without losing continuation state."""

    if estimate_tokens(payload) <= budget_tokens:
        return payload
    value = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    context = value.get("context")
    if isinstance(context, dict):
        research = context.get("research_report")
        if isinstance(research, dict):
            research["sources"] = research.get("sources", [])[:12]
            research["findings"] = research.get("findings", [])[:12]
            research["gaps"] = research.get("gaps", [])[:12]
        disclosed = context.get("disclosed_text_files", [])
        if isinstance(disclosed, list) and disclosed:
            per_file = max(500, 20_000 // len(disclosed))
            for item in disclosed:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    item["content"] = compact_text(item["content"], per_file)
        context["compression"] = {
            **(context.get("compression") if isinstance(context.get("compression"), dict) else {}),
            "provider_compact": True,
            "provider_budget_tokens": budget_tokens,
        }
    evidence = value.get("evidence")
    if evidence is not None:
        value["evidence"] = _compact_evidence(evidence)
    # Skill context is already distilled and independently budgeted. Preserve
    # the complete cross-role catalog while compacting volatile task evidence.
    return value


def _compact_evidence(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact_evidence(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"text", "base64"}:
            continue
        if key == "markdown" and isinstance(item, str):
            result[key] = compact_text(item, 2_000)
        elif key == "artifacts" and isinstance(item, list):
            result[key] = [
                {field: artifact.get(field) for field in ("logical_name", "kind", "media_type", "metadata") if field in artifact}
                for artifact in item if isinstance(artifact, dict)
            ]
        else:
            result[key] = _compact_evidence(item)
    return result


@dataclass
class QwenChatClient:
    """Small synchronous JSON client; transport is injectable for tests."""

    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-max"
    api_key: str | None = None
    timeout_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Qwen solve_problem adapter")
        self.api_key = key
        parsed = urlsplit(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise PermissionError("Qwen base URL must use HTTPS")
        allowlist = {item.strip().lower() for item in os.environ.get("M2HARNESS_MODEL_HOSTS", "dashscope.aliyuncs.com").split(",") if item.strip()}
        if parsed.hostname.lower() not in allowlist:
            raise PermissionError(f"Qwen provider host is not allowlisted: {parsed.hostname}")

    def json(self, *, system: str, content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "stream": False,
            "enable_thinking": True,
            "response_format": {"type": "json_object"},
            "max_tokens": max(256, min(int(os.environ.get("QWEN_MAX_OUTPUT_TOKENS", "12000")), 64_000)),
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
                    response = client.post(self.base_url.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload)
                    if response.status_code >= 400:
                        response.raise_for_status()
                    decoded = response.json()
                choices = decoded.get("choices") or []
                message = (choices[0].get("message") if choices else None) or {}
                value = _json_object(message.get("content", ""))
                # Validate against the requested model schema at the caller. The
                # schema is included in the prompt as a provider compatibility
                # fallback; some OpenAI-compatible endpoints ignore response_schema.
                return value
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {408, 409, 425, 429} and exc.response.status_code < 500:
                    break
            except (httpx.HTTPError, ValueError, KeyError, IndexError, AttributeError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                if isinstance(exc, (ValueError, KeyError, IndexError, AttributeError, TypeError, json.JSONDecodeError)):
                    break
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise ActivityExecutionError(f"Qwen solve_problem request failed: {last_error}") from last_error


def _skill_context(registry, names: Sequence[str], *, budget_tokens: int = 96_000) -> str:
    """All Agents see the full distilled catalog; names only select focus resources."""

    return assemble_skill_context(registry, names, budget_tokens=budget_tokens)


class QwenModelAgent(ModelAgentPort):
    def __init__(self, client: QwenChatClient, skills=None) -> None:
        self.client = client
        self.skills = skills

    def _request(self, task: SolveProblemTask, context: SolveProblemContext, instruction: str, schema: dict[str, Any], *, evidence: dict[str, Any] | None = None, skill_names: Sequence[str] = (), role: AgentRole = AgentRole.MODEL) -> dict[str, Any]:
        system = build_agent_system_prompt(
            role,
            output_contract="Return one JSON object matching the supplied schema. Use requested_file_paths for allowlisted files whose content is required.",
        )
        payload = {
            "task": task.model_dump(mode="json"), "context": _context_json(context),
            "instruction": instruction, "skill_context": _skill_context(self.skills, skill_names),
            "output_schema": schema,
        }
        if evidence:
            payload["evidence"] = evidence
        prompt = json.dumps(_compact_provider_payload(payload), ensure_ascii=False)
        return self.client.json(system=system, content=_content_parts(context, prompt), schema=schema)

    def explore(self, task, context, *, branch_count, iteration):
        schema = PreliminaryModelingReport.model_json_schema()
        reports: list[PreliminaryModelingReport] = []
        for index in range(branch_count):
            value = self._request(task, context, f"Produce independent preliminary modeling route {index + 1}/{branch_count} for iteration {iteration}. Include assumptions, risks and expected outputs. If an allowlisted UTF-8 file is required beyond the compact context, list its path in requested_file_paths.", schema, skill_names=("problem-intake", "modeling-project-orchestration", "problem-decomposition", "research-planning", "deep-research", "modeling-knowledge", "modeling-core", "exploratory-data-analysis"))
            value.setdefault("branch_id", f"route-{index + 1}")
            reports.append(PreliminaryModelingReport.model_validate(value, strict=False))
        return tuple(reports)

    def synthesize(self, task, context, preliminary, *, iteration):
        value = self._request(task, context, f"Select and unify the best preliminary routes for iteration {iteration}. Produce one executable main scheme and required validations.", UnifiedModelingReport.model_json_schema(), evidence={"preliminary_reports": [item.model_dump(mode="json") for item in preliminary]}, skill_names=("model-selection", "modeling-core", "dimensional-analysis", "uncertainty-quantification", "sensitivity-analysis", "numerical-validation"))
        return UnifiedModelingReport.model_validate(value, strict=False)

    def review(self, task, context, modeling, coding, *, iteration):
        value = self._request(task, context, f"Review the Unified Modeling Report and Coding Harness Report for iteration {iteration}. Approve only when every required validation has reproducible evidence; otherwise choose the narrowest revision_target and give concrete instructions.", SolveProblemReview.model_json_schema(), evidence={"modeling_report": modeling.model_dump(mode="json"), "coding_report": coding.model_dump(mode="json")}, skill_names=("report-review", "claim-evidence", "numerical-validation", "model-diagnostics"), role=AgentRole.REVIEW)
        return SolveProblemReview.model_validate(value, strict=False)

    def compose_final_report(self, task, context, modeling, coding, review, *, iteration):
        value = self._request(task, context, f"Compose the final single-question solution report after approved iteration {iteration}. Use only verified Coding Report evidence, preserve limitations, and put machine-readable values needed by later tasks under structured.downstream_outputs.", ReportPayload.model_json_schema(), evidence={"modeling_report": modeling.model_dump(mode="json"), "coding_report": coding.model_dump(mode="json"), "review": review.model_dump(mode="json")}, skill_names=("report-rendering", "scientific-writing", "claim-evidence", "report-review"), role=AgentRole.PAPER)
        return ReportPayload.model_validate(value, strict=False)


class QwenCodeProposalProvider(CodeProposalProvider):
    def __init__(self, client: QwenChatClient, skills=None) -> None:
        self.client = client
        self.skills = skills

    def propose(self, task, context, modeling, *, iteration):
        system = build_agent_system_prompt(
            AgentRole.CODE,
            output_contract=(
                "Return one CodeProposal JSON object containing a complete deterministic Python script. The script must "
                "print one final JSON object with validations, non-empty validation_evidence for every required ID, and "
                f"optional metrics; write files only under .m2harness-code/{task.task_id}/iteration-{iteration}/outputs."
            ),
        )
        prompt = json.dumps(_compact_provider_payload({"task": task.model_dump(mode="json"), "context": _context_json(context), "modeling": modeling.model_dump(mode="json"), "iteration": iteration, "skill_context": _skill_context(self.skills, ("coding-contract", "code-debugging", "dimensional-analysis", "visualization", "scientific-figure-design", "numerical-validation")), "schema": CodeProposal.model_json_schema()}), ensure_ascii=False)
        try:
            value = self.client.json(system=system, content=_content_parts(context, prompt), schema=CodeProposal.model_json_schema())
            source = value.get("source", "")
            if isinstance(source, str) and source.strip().startswith("```"):
                source = re.sub(r"^```(?:python)?\s*|\s*```$", "", source.strip(), flags=re.IGNORECASE)
                value["source"] = source
            return CodeProposal.model_validate(value, strict=False)
        except Exception as exc:
            raise ActivityExecutionError(f"Qwen Code Proposal is invalid: {exc}") from exc


class _PaperComposition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    final_report: ReportPayload
    latex: str = Field(min_length=1, max_length=2 * 1024 * 1024)


class QwenPaperComposer:
    """Global-context final composer used by ``MainHarness.generate_paper``."""

    def __init__(self, client: QwenChatClient, skills=None) -> None:
        self.client = client
        self.skills = skills

    def compose(self, problem: str, reports: tuple[Any, ...]) -> tuple[ReportPayload, ProducedArtifact]:
        schema = _PaperComposition.model_json_schema()
        prompt = json.dumps({
            "problem": problem,
            "approved_solve_reports": [_compact_solve_report(report) for report in reports],
            "schema": schema,
            "skill_context": _skill_context(self.skills, ("report-rendering", "latex-publication", "scientific-writing", "scientific-figure-design", "publication-verification", "report-review")),
            "requirements": [
                "Synthesize only reviewed claims and preserve limitations.",
                "Return a polished single-question report and one self-contained compile-ready LaTeX article.",
                "The LaTeX must begin with documentclass, include title/abstract/sections, and contain no input/include/write18/external reads.",
                "Use a clean academic layout with geometry, booktabs and mathematical notation only when supported by evidence.",
            ],
        }, ensure_ascii=False)
        value = self.client.json(
            system=build_agent_system_prompt(
                AgentRole.PAPER,
                output_contract="Return exactly one JSON object containing a reviewed final_report and one self-contained compile-ready LaTeX article.",
            ),
            content=[{"type": "text", "text": prompt}], schema=schema,
        )
        composition = _PaperComposition.model_validate(value, strict=False)
        artifact = ProducedArtifact(
            logical_name="final-question-paper.tex", kind=ArtifactKind.FINAL_LATEX_PAPER,
            media_type="text/x-tex", text=composition.latex,
        )
        return composition.final_report, artifact


def build_qwen_solve_problem_service(*, sandbox, workspace_root, research_agent=None, skills=None, client: QwenChatClient | None = None, max_iterations: int = 3) -> SolveProblemService:
    """Compose the target solve tool after ``build_local_runtime()`` created its sandbox."""
    qwen = client or QwenChatClient()
    from m2harness.infrastructure.code_harness import LocalPythonCodeHarness

    model_agent = QwenModelAgent(qwen, skills=skills)
    code_harness = LocalPythonCodeHarness(QwenCodeProposalProvider(qwen, skills=skills), sandbox, workspace_root)
    return SolveProblemService(
        model_agent, code_harness, max_iterations=max_iterations,
        research_agent=research_agent,
        file_reader=WorkspaceReadOnlyFileReader(workspace_root),
    )


def _compact_solve_report(report: Any) -> dict[str, Any]:
    """Paper composition sees accepted results, not every historical turn."""

    final = getattr(report, "final_report", None)
    artifacts = getattr(report, "artifacts", ())
    return {
        "task_id": getattr(report, "task_id", ""),
        "status": getattr(getattr(report, "status", None), "value", str(getattr(report, "status", ""))),
        "iteration_count": getattr(report, "iteration_count", 0),
        "final_report": final.model_dump(mode="json") if final is not None else None,
        "artifacts": [
            {
                "logical_name": item.logical_name,
                "kind": item.kind.value,
                "media_type": item.media_type,
                "metadata": item.metadata,
            }
            for item in artifacts
        ],
        "research_summary": (
            {
                "query": report.research_report.query,
                "findings": [item.model_dump(mode="json") for item in report.research_report.findings[:12]],
                "gaps": list(report.research_report.gaps[:12]),
            }
            if getattr(report, "research_report", None) is not None else None
        ),
    }
