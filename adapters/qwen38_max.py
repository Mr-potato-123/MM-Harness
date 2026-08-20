"""Qwen3.8-Max multimodal Activity adapter using OpenAI-compatible HTTP.

Secrets are read from the environment. stdout is reserved for protocol JSON.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from m2harness.models import (
    ActivityRequest,
    ActivityResponse,
    CodingStageOutput,
    FinalizeStageOutput,
    ModelingStageOutput,
    ReviewStageOutput,
    StageKind,
)


OUTPUT_TYPES = {
    StageKind.MODELING: ModelingStageOutput,
    StageKind.CODING: CodingStageOutput,
    StageKind.REVIEW: ReviewStageOutput,
    StageKind.FINALIZE: FinalizeStageOutput,
}
TEXT_TYPES = {"text/plain", "text/markdown", "text/csv", "application/json", "application/xml"}


def _safe_blob(root: Path, relative: str, digest: str, size: int) -> bytes:
    path = (root / relative).resolve()
    if root not in path.parents or path.name != digest or not path.is_file():
        raise ValueError(f"unsafe or missing artifact: {relative}")
    data = path.read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"artifact integrity failure: {relative}")
    return data


def _multimodal_content(request: ActivityRequest, root: Path, prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    total_binary = 0
    for artifact in request.inputs:
        data = _safe_blob(root, artifact.relative_path, artifact.sha256, artifact.size_bytes)
        media_type = artifact.media_type.lower()
        if media_type == "application/pdf":
            if len(data) > 150 * 1024 * 1024:
                raise ValueError(f"PDF exceeds Qwen 150MB limit: {artifact.logical_name}")
            total_binary += len(data)
            content.append({
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64," + base64.b64encode(data).decode("ascii"),
                    "filename": artifact.logical_name,
                },
            })
        elif media_type.startswith("image/"):
            total_binary += len(data)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")},
            })
        elif media_type.startswith("video/"):
            total_binary += len(data)
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")},
            })
        elif media_type in TEXT_TYPES or media_type.startswith("text/"):
            text = data.decode("utf-8", errors="strict")
            content.append({"type": "text", "text": f"\n--- Artifact: {artifact.logical_name} ({media_type}, sha256={artifact.sha256}) ---\n{text}"})
        else:
            content.append({"type": "text", "text": f"Binary artifact available but not directly ingestible: {artifact.logical_name}, {media_type}, sha256={artifact.sha256}, {len(data)} bytes."})
    if total_binary > 200 * 1024 * 1024:
        raise ValueError("combined binary multimodal input exceeds adapter 200MB policy")
    content.append({"type": "text", "text": prompt})
    return content


def _prompt(request: ActivityRequest) -> str:
    output_type = OUTPUT_TYPES[request.stage]
    schema = output_type.model_json_schema()
    role = {
        StageKind.MODELING: "You are the modeling activity. Formalize the problem, assumptions, variables, equations, validation contract, expected outputs, limitations, and auditable claims.",
        StageKind.CODING: "You are the coding and execution activity. Use the hosted code interpreter for every numerical claim. Report whether execution really succeeded and mark each required validation truthfully.",
        StageKind.REVIEW: "You are an independent reviewer. Reject unsupported claims, failed/missing validations, weak assumptions, and non-reproducible execution. Return concrete revision instructions when needed.",
        StageKind.FINALIZE: "You are the final report activity. Produce a self-contained production single-question report using only reviewed evidence; never introduce an unverified claim.",
    }[request.stage]
    return (
        f"{role}\nQuestion: {request.question.title}\nRevision: {request.revision}\n"
        f"Harness instructions: {json.dumps(request.instructions, ensure_ascii=False)}\n"
        "Return ONLY one JSON object matching this JSON Schema. Do not use Markdown fences around the JSON. "
        "The stage field must match exactly. The report.markdown field may contain Markdown.\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def _stream_chat(url: str, key: str, payload: dict[str, Any], timeout: float) -> tuple[str, dict[str, Any]]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0)) as client:
        with client.stream("POST", url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", errors="replace")[:4000]
                raise RuntimeError(f"Qwen HTTP {response.status_code}: {body}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                event = json.loads(raw)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
    return "".join(text_parts), usage


def _extract_json(text: str) -> dict[str, Any]:
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
        raise ValueError("Qwen output must be a JSON object")
    return value


def main() -> int:
    request = ActivityRequest.model_validate_json(sys.stdin.buffer.read())
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")
    root_value = os.environ.get("M2HARNESS_ARTIFACT_ROOT")
    if not root_value:
        raise RuntimeError("M2HARNESS_ARTIFACT_ROOT is required")
    root = Path(root_value).resolve()
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = os.environ.get("QWEN_MODEL", "qwen3.8-max")
    prompt = _prompt(request)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _multimodal_content(request, root, prompt)}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
        "response_format": {"type": "json_object"},
    }
    if request.stage == StageKind.CODING:
        payload["enable_code_interpreter"] = True
    answer, usage = _stream_chat(base_url + "/chat/completions", key, payload, timeout=1800.0)
    output_type = OUTPUT_TYPES[request.stage]
    output = output_type.model_validate(_extract_json(answer), strict=False)
    response = ActivityResponse(
        idempotency_key=request.idempotency_key,
        output=output,
        executor_metadata={"provider": "qwen", "model": model, "usage": usage},
    )
    sys.stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
