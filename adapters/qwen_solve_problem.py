"""OpenAI-compatible model adapters for the target Main Harness semantics.

This module is deliberately separate from the legacy stage adapter.  It maps
the report-driven ``ModelAgentPort`` and ``CodeProposalProvider`` contracts to
an OpenAI-compatible endpoint, while the actual code execution remains
inside ``LocalPythonCodeHarness`` and the configured local sandbox.

The default Qwen endpoint reads ``DASHSCOPE_API_KEY``; a DeepSeek endpoint
reads ``DEEPSEEK_API_KEY``.  Keys are never embedded in source or reports.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from m2harness.application.solve_problem import ModelAgentPort
from m2harness.application.solve_problem import SolveProblemArchivePort, SolveProblemService, WorkspaceReadOnlyFileReader
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
from m2harness.application.capabilities import CapabilityRegistry
from m2harness.application.tools import ToolRuntime
from m2harness.application.todo import TodoLedger
from m2harness.domain.capability import CapabilityRequirement
from m2harness.domain.tool import ToolCall


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


def _stream_content_text(value: Any) -> str:
    """Normalize OpenAI-compatible streamed content fragments."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in value)
    return "" if value is None else str(value)


def _stream_progress(message: str) -> None:
    """Emit operational progress without exposing prompts, responses or keys."""

    if os.environ.get("QWEN_STREAM_PROGRESS", "0") == "1":
        line = f"[qwen-stream] {message}"
        print(line, file=sys.stderr, flush=True)
        progress_path = os.environ.get("QWEN_STREAM_PROGRESS_FILE")
        if not progress_path:
            progress_path = str(Path.cwd() / ".m2harness" / "workspace" / "agent-progress.md")
        try:
            target = Path(progress_path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(f"- {datetime.now(UTC).isoformat()} {message}\n")
        except OSError:
            # Progress observability must never make a model/tool call fail.
            pass


class MainHarnessToolBridge:
    """Expose a narrow, audited Main Harness tool lane to the Code Agent."""

    # The Code Agent can inspect and mutate only its workspace lane and run
    # deterministic local checks.  It cannot recursively call solve_problem,
    # publish reports, use web search, or access the artifact registry.
    ALLOWED_TOOLS = frozenset({
        "workspace_list", "workspace_read", "workspace_search", "workspace_write",
        "workspace_edit", "artifact_inspect", "pdf_inspect", "data_profile",
        "python_execute", "validation_run",
    })
    # Implementation-level planning is deliberately separate from the Main
    # Harness DAG.  These two tools are available to the legacy provider too;
    # DSH supplies its own equivalent Todo middleware in its profile.
    TODO_TOOLS = frozenset({"todo_read", "todo_write"})

    def __init__(self, runtime: ToolRuntime, capabilities: CapabilityRegistry, *, session_id: UUID | None = None, workspace_root: Path | None = None) -> None:
        self.runtime = runtime
        self.capabilities = capabilities
        self.session_id = session_id or uuid4()
        self.workspace_root = workspace_root.resolve() if workspace_root is not None else None
        self._todo_ledgers: dict[str, TodoLedger] = {}

    def definitions(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for definition in self.runtime.registry.catalog():
            if definition.name not in self.ALLOWED_TOOLS:
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            })
        result.extend([
            {
                "type": "function",
                "function": {
                    "name": "todo_read",
                    "description": "Read the current Code Agent implementation todo list; this is not the Main Harness workflow TODO.",
                    "parameters": {"type": "object", "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "todo_write",
                    "description": "Atomically replace the Code Agent implementation todo list; at most one item may be in_progress.",
                    "parameters": {
                        "type": "object", "additionalProperties": False, "required": ["todos"],
                        "properties": {"todos": {"type": "array", "maxItems": 100, "items": {
                            "type": "object", "additionalProperties": False, "required": ["content", "status"],
                            "properties": {"content": {"type": "string", "minLength": 1, "maxLength": 500}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "active_form": {"type": "string", "maxLength": 500}},
                        }}},
                    },
                },
            },
        ])
        return result

    def execute(self, name: str, arguments: dict[str, Any], *, task_id: str, iteration: int, turn: int) -> dict[str, Any]:
        if name in self.TODO_TOOLS:
            ledger = self._todo_ledgers.get(task_id)
            if ledger is None:
                todo_root = None
                if self.workspace_root is not None:
                    todo_root = self.workspace_root / ".m2harness-code" / task_id / "todo.json"
                ledger = TodoLedger(task_id, todo_root)
                self._todo_ledgers[task_id] = ledger
            if name == "todo_read":
                return {"ok": True, "tool_name": name, "output": {"todos": ledger.render()}}
            try:
                snapshot = ledger.write(arguments.get("todos"), iteration=iteration)
            except (TypeError, ValueError) as exc:
                return {"ok": False, "tool_name": name, "error_code": "invalid_todo", "error_message": str(exc)}
            return {"ok": True, "tool_name": name, "output": snapshot.model_dump(mode="json")}
        if name not in self.ALLOWED_TOOLS:
            return {"ok": False, "error_code": "tool_not_allowed", "error_message": f"Code Agent tool is not allowed: {name}"}
        definition = self.runtime.registry.get(name)
        if definition is None:
            return {"ok": False, "error_code": "tool_not_found", "error_message": f"tool is not registered: {name}"}
        resolution = self.capabilities.resolve((CapabilityRequirement(capability=definition.required_capability, reason=f"Code Agent {name}"),))
        call_id = uuid4()
        call = ToolCall(
            call_id=call_id, tool_name=name, tool_version=definition.version,
            activity_id=uuid4(), session_id=self.session_id,
            idempotency_key=f"code-agent:{task_id}:iteration-{iteration}:turn-{turn}:{call_id}",
            arguments=arguments, requested_at=datetime.now(UTC),
        )
        result = self.runtime.execute(call, resolution)
        if result.ok:
            return {"ok": True, "tool_name": name, "output": result.output or {}}
        return {"ok": False, "tool_name": name, "error_code": result.error_code, "error_message": result.error_message}


def _content_parts(context: SolveProblemContext, prompt: str, *, deepseek: bool = False) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    binary_total = 0
    for item in context.multimodal_inputs:
        # MultimodalInput validates base64, byte count and digest at the
        # boundary.  The adapter still enforces the combined provider budget.
        raw_size = item.size_bytes
        if item.media_type.lower() == "application/pdf" and raw_size > 150 * 1024 * 1024:
            raise ValueError("PDF input exceeds the 150MB provider limit")
        binary_total += raw_size
        if binary_total > 200 * 1024 * 1024:
            raise ValueError("combined solve_problem multimodal input exceeds 200MB")
        if item.media_type.lower() == "application/pdf":
            if deepseek:
                # DeepSeek's vision experimental endpoint accepts image
                # parts, not PDF file parts.  Render each page locally so the
                # model receives the original visual evidence without giving
                # the provider filesystem access.
                try:
                    try:
                        import pymupdf as fitz  # PyMuPDF >= 1.24
                    except ImportError:
                        import fitz  # older PyMuPDF releases
                except ImportError as exc:
                    raise RuntimeError("DeepSeek PDF vision input requires PyMuPDF; install the media extras") from exc
                document = fitz.open(stream=base64.b64decode(item.data_base64), filetype="pdf")
                try:
                    for page_number, page in enumerate(document, start=1):
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                        png_base64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_base64}})
                    parts.append({"type": "text", "text": f"已将 {item.logical_name} 渲染为 {len(document)} 页图像；请以这些图像作为题面证据。"})
                finally:
                    document.close()
            else:
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
        parsed = urlsplit(self.base_url)
        provider_is_deepseek = bool(parsed.hostname and parsed.hostname.lower().endswith("deepseek.com"))
        preferred_key_name = "DEEPSEEK_API_KEY" if provider_is_deepseek else "DASHSCOPE_API_KEY"
        key = self.api_key or os.environ.get(preferred_key_name)
        if not key and provider_is_deepseek:
            key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(f"{preferred_key_name} is required for solve_problem adapter")
        self.api_key = key
        if parsed.scheme != "https" or not parsed.hostname:
            raise PermissionError("model provider base URL must use HTTPS")
        default_hosts = "dashscope.aliyuncs.com,api.deepseek.com"
        allowlist = {item.strip().lower() for item in os.environ.get("M2HARNESS_MODEL_HOSTS", default_hosts).split(",") if item.strip()}
        if parsed.hostname.lower() not in allowlist:
            raise PermissionError(f"model provider host is not allowlisted: {parsed.hostname}")

    @property
    def _deepseek_mode(self) -> bool:
        parsed = urlsplit(self.base_url)
        return bool(parsed.hostname and parsed.hostname.lower().endswith("deepseek.com"))

    def _thinking_fields(self, enabled: bool) -> dict[str, Any]:
        """Return provider-specific thinking controls at the transport boundary."""

        if self._deepseek_mode:
            return {
                "thinking": {"type": "enabled" if enabled else "disabled"},
                "reasoning_effort": "high" if enabled else "low",
            }
        return {"enable_thinking": enabled}

    def _max_output_tokens(self) -> int:
        """Honor the configured output budget up to the provider's documented cap."""

        provider_limit = 384_000 if self._deepseek_mode else 64_000
        return max(256, min(int(os.environ.get("QWEN_MAX_OUTPUT_TOKENS", str(provider_limit))), provider_limit))

    def _stream_json(self, client: httpx.Client, url: str, headers: dict[str, str], payload: dict[str, Any]) -> tuple[str, str | None]:
        """Read one OpenAI-compatible SSE response, with JSON fallback."""

        text_parts: list[str] = []
        finish_reason: str | None = None
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type and "text/event-stream" not in content_type:
                decoded = response.json()
                choices = decoded.get("choices") or []
                for choice in choices:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    message = choice.get("message") or choice.get("delta") or {}
                    text_parts.append(_stream_content_text(message.get("content")))
            else:
                last_reported = 0
                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    event = json.loads(raw)
                    for choice in event.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or choice.get("message") or {}
                        text_parts.append(_stream_content_text(delta.get("content")))
                    current = sum(len(item) for item in text_parts)
                    if current - last_reported >= 2_000:
                        _stream_progress(f"received_chars={current}")
                        last_reported = current
        answer = "".join(text_parts)
        _stream_progress(f"response_complete chars={len(answer)} finish_reason={finish_reason!r}")
        if not answer.strip():
            raise ValueError("Qwen returned an empty response")
        return answer, finish_reason

    def _stream_tool_turn(self, client: httpx.Client, url: str, headers: dict[str, str], payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
        """Read one streamed assistant turn, including fragmented tool calls."""

        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type and "text/event-stream" not in content_type:
                decoded = response.json()
                choices = decoded.get("choices") or []
                choice = choices[0] if choices else {}
                finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
                text_parts.append(_stream_content_text(message.get("content")))
                for index, item in enumerate(message.get("tool_calls") or []):
                    function = item.get("function") or {}
                    calls[index] = {
                        "id": item.get("id") or f"call-{index}",
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
            else:
                last_reported = 0
                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    event = json.loads(raw)
                    for choice in event.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}
                        text_parts.append(_stream_content_text(delta.get("content")))
                        for item in delta.get("tool_calls") or []:
                            index = int(item.get("index", 0))
                            call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            call["id"] = item.get("id") or call["id"] or f"call-{index}"
                            function = item.get("function") or {}
                            call["name"] += function.get("name", "") or ""
                            call["arguments"] += function.get("arguments", "") or ""
                    current = sum(len(item) for item in text_parts)
                    if current - last_reported >= 2_000:
                        _stream_progress(f"received_chars={current} tool_calls={len(calls)}")
                        last_reported = current
        answer = "".join(text_parts)
        normalized = [calls[index] for index in sorted(calls)]
        _stream_progress(f"turn_complete chars={len(answer)} tool_calls={len(normalized)} finish_reason={finish_reason!r}")
        return answer, normalized, finish_reason

    def tool_agent_json(
        self,
        *,
        system: str,
        content: list[dict[str, Any]],
        schema: dict[str, Any],
        tool_bridge: MainHarnessToolBridge,
        task_id: str,
        iteration: int,
        max_turns: int = 64,
    ) -> dict[str, Any]:
        """Run a bounded tool-calling Code Agent session and return final JSON."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        max_tokens = self._max_output_tokens()
        for turn in range(1, max_turns + 1):
            payload = {
                "model": self.model, "messages": messages, "stream": True,
                "tools": tool_bridge.definitions(), "tool_choice": "auto", "max_tokens": max_tokens,
            }
            payload.update(self._thinking_fields(os.environ.get("QWEN_ENABLE_THINKING", "1") != "0"))
            _stream_progress(f"tool_turn_start turn={turn} tools={len(payload['tools'])}")
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
                    answer, calls, finish_reason = self._stream_tool_turn(client, url, headers, payload)
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                raise ActivityExecutionError(f"Qwen Code Agent tool turn failed: {exc}") from exc
            if not calls:
                return _json_object(answer)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": answer or None, "tool_calls": []}
            for index, call in enumerate(calls):
                assistant_message["tool_calls"].append({
                    "id": call["id"] or f"call-{index}", "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                })
            messages.append(assistant_message)
            for call in calls:
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    result = tool_bridge.execute(call["name"], arguments, task_id=task_id, iteration=iteration, turn=turn)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    result = {"ok": False, "error_code": "invalid_tool_arguments", "error_message": str(exc)}
                _stream_progress(f"tool_call name={call['name']} ok={result.get('ok', False)}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        raise ActivityExecutionError(f"Qwen Code Agent exceeded tool turn budget ({max_turns})")

    def json(self, *, system: str, content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        requested_max_tokens = self._max_output_tokens()
        thinking_enabled = os.environ.get("QWEN_ENABLE_THINKING", "1") != "0"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": requested_max_tokens,
        }
        payload.update(self._thinking_fields(thinking_enabled))
        last_error: Exception | None = None
        for attempt in range(3):
            # A long multimodal request can exhaust the first JSON response
            # while the provider is still reasoning.  One bounded recovery
            # attempt asks for a compact, no-thinking JSON response and gives
            # it a larger output budget.  We never attempt to synthetically
            # repair malformed JSON: the typed caller must validate the whole
            # object before it can advance workflow state.
            attempt_payload = payload
            if attempt > 0 and isinstance(last_error, json.JSONDecodeError):
                retry_max = min(384_000 if self._deepseek_mode else 64_000, max(requested_max_tokens, requested_max_tokens * 2))
                retry_content = [*content, {"type": "text", "text": (
                    "The previous response was truncated or invalid JSON. Retry now with exactly one compact "
                    "JSON object matching the supplied schema; do not include markdown fences, reasoning, or "
                    "extra prose. Keep every free-text field concise enough to finish before the token limit."
                )}]
                attempt_payload = {
                    **payload,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": retry_content},
                    ],
                    "max_tokens": retry_max,
                }
                attempt_payload.update(self._thinking_fields(False))
            use_stream = os.environ.get("QWEN_STREAM", "1") != "0"
            attempt_payload = {**attempt_payload, "stream": use_stream}
            _stream_progress(f"request_start attempt={attempt + 1} stream={use_stream} max_tokens={attempt_payload['max_tokens']}")
            answer = ""
            finish_reason: str | None = None
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
                    url = self.base_url.rstrip("/") + "/chat/completions"
                    headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                    if use_stream:
                        answer, finish_reason = self._stream_json(client, url, headers, attempt_payload)
                    else:
                        response = client.post(url, headers=headers, json=attempt_payload)
                        if response.status_code >= 400:
                            response.raise_for_status()
                        decoded = response.json()
                        choices = decoded.get("choices") or []
                        finish_reason = (choices[0].get("finish_reason") if choices else None)
                        message = (choices[0].get("message") if choices else None) or {}
                        answer = message.get("content", "")
                value = _json_object(answer)
                # Validate against the requested model schema at the caller. The
                # schema is included in the prompt as a provider compatibility
                # fallback; some OpenAI-compatible endpoints ignore response_schema.
                return value
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {408, 409, 425, 429} and exc.response.status_code < 500:
                    break
            except json.JSONDecodeError as exc:
                raw_length = len(answer) if isinstance(locals().get("answer"), str) else 0
                last_error = json.JSONDecodeError(
                    f"{exc.msg} (provider_finish_reason={finish_reason!r}, content_length={raw_length})",
                    exc.doc,
                    exc.pos,
                )
                if attempt < 2:
                    continue
            except (httpx.HTTPError, ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
                last_error = exc
                if isinstance(exc, (ValueError, KeyError, IndexError, AttributeError, TypeError)):
                    break
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise ActivityExecutionError(f"Qwen solve_problem request failed: {last_error}") from last_error


def _skill_context(registry, names: Sequence[str], *, budget_tokens: int = 96_000) -> str:
    """All Agents see the full distilled catalog; names only select focus resources."""

    return assemble_skill_context(registry, names, budget_tokens=budget_tokens)


def _scope_instruction(task: SolveProblemTask) -> str:
    scope = str(task.metadata.get("scope", "")).strip()
    language = (
        "\n语言约束：这是中文长程建模流程。所有 Agent 的说明、报告、交接、审查理由和验证证据使用中文；"
        "代码标识符可以使用英文，但代码注释和面向人的输出使用中文。"
    )
    ledger = (
        "\n上下文约束：Model Agent 与 Code Agent 使用彼此隔离的上下文；Model Agent 只延续 "
        "context.metadata.model_conversation，Code Agent 只延续 context.metadata.code_conversation。"
        "返修迭代不得重置各自上下文；两者只通过精简、结构化的交接报告交流，详细文件按清单路径按需读取。"
    )
    if not scope:
        return language + ledger
    return (
        language + ledger +
        f"\n范围锁定：本次 solve_problem 只处理 {scope}。附件可能包含其他问题；完全忽略它们，"
        "不得求解、总结、比较、建模、编码、验证或报告范围外问题。若某个细节只属于其他问题，标记为超出范围并继续当前问题。"
    )


class QwenModelAgent(ModelAgentPort):
    def __init__(self, client: QwenChatClient, skills=None) -> None:
        self.client = client
        self.skills = skills

    def _request(self, task: SolveProblemTask, context: SolveProblemContext, instruction: str, schema: dict[str, Any], *, evidence: dict[str, Any] | None = None, skill_names: Sequence[str] = (), role: AgentRole = AgentRole.MODEL) -> dict[str, Any]:
        system = build_agent_system_prompt(
            role,
            output_contract="Return one JSON object matching the supplied schema. Use requested_file_paths for allowlisted files whose content is required.",
        ) + _scope_instruction(task)
        payload = {
            "task": task.model_dump(mode="json"), "context": _context_json(context),
            "instruction": instruction, "skill_context": _skill_context(self.skills, skill_names),
            "output_schema": schema,
        }
        if evidence:
            payload["evidence"] = evidence
        prompt = json.dumps(_compact_provider_payload(payload), ensure_ascii=False)
        deepseek = bool(getattr(self.client, "_deepseek_mode", False))
        return self.client.json(system=system, content=_content_parts(context, prompt, deepseek=deepseek), schema=schema)

    def explore(self, task, context, *, branch_count, iteration):
        schema = PreliminaryModelingReport.model_json_schema()
        reports: list[PreliminaryModelingReport] = []
        for index in range(branch_count):
            value = self._request(task, context, f"仅用中文产生第 {index + 1}/{branch_count} 条独立初步建模路线（迭代 {iteration}），只覆盖范围锁定的问题。列出假设、风险和该问题的预期输出，不得讨论其他问题。若需要白名单 UTF-8 文件，请在 requested_file_paths 中返回路径。", schema, skill_names=("problem-intake", "modeling-project-orchestration", "problem-decomposition", "research-planning", "deep-research", "modeling-knowledge", "modeling-core", "exploratory-data-analysis"))
            value.setdefault("branch_id", f"route-{index + 1}")
            reports.append(PreliminaryModelingReport.model_validate(value, strict=False))
        return tuple(reports)

    def synthesize(self, task, context, preliminary, *, iteration):
        value = self._request(task, context, f"用中文选择并统一第 {iteration} 轮的最佳路线，只针对当前范围。给出一个可执行主方案和必需验证，不得扩展到其他问题。", UnifiedModelingReport.model_json_schema(), evidence={"preliminary_reports": [item.model_dump(mode="json") for item in preliminary]}, skill_names=("model-selection", "modeling-core", "dimensional-analysis", "uncertainty-quantification", "sensitivity-analysis", "numerical-validation"))
        return UnifiedModelingReport.model_validate(value, strict=False)

    def review(self, task, context, modeling, coding, *, iteration):
        handoff_paths = sorted(
            item.relative_path for item in context.readonly_files
            if "/exchanges/" in item.relative_path.replace("\\", "/")
        )
        value = self._request(
            task, context,
            f"用中文审查第 {iteration} 轮的统一建模报告和代码执行报告，只针对当前范围。 "
            "精确的 Code→Review 交接已存于白名单路径；如需内容请使用 "
            "requested_file_paths 请求路径。以 Coding Report 的 execution_succeeded、validations 和 validation_evidence 为执行权威；输出目录中的文件不能替代 stdout/exit_code 证据。"
            "如果主方案、算法或搜索覆盖范围偏离已接受建模契约，必须列为 deviation 并要求返修或明确降级声明。"
            "只有所有验证都有可复现证据且声明范围与实际算法一致时才批准，否则选择最窄的返修目标并给出中文具体指令。",
            SolveProblemReview.model_json_schema(),
            evidence={
                "modeling_report": modeling.model_dump(mode="json"),
                "coding_report": coding.model_dump(mode="json"),
                "handoff_paths": handoff_paths,
            },
            skill_names=("report-review", "claim-evidence", "numerical-validation", "model-diagnostics"),
            role=AgentRole.REVIEW,
        )
        return SolveProblemReview.model_validate(value, strict=False)

    def compose_final_report(self, task, context, modeling, coding, review, *, iteration):
        value = self._request(task, context, f"用中文编写当前范围在第 {iteration} 轮通过审查后的最终单题报告。只使用已验证的 Coding Report 证据，保留限制，并把后续题目需要的机器值放入 structured.downstream_outputs。", ReportPayload.model_json_schema(), evidence={"modeling_report": modeling.model_dump(mode="json"), "coding_report": coding.model_dump(mode="json"), "review": review.model_dump(mode="json")}, skill_names=("report-rendering", "scientific-writing", "claim-evidence", "report-review"), role=AgentRole.PAPER)
        return ReportPayload.model_validate(value, strict=False)


class QwenCodeProposalProvider(CodeProposalProvider):
    def __init__(self, client: QwenChatClient, skills=None, *, tool_runtime: ToolRuntime | None = None, capabilities: CapabilityRegistry | None = None, workspace_root: Path | None = None) -> None:
        self.client = client
        self.skills = skills
        self.tool_runtime = tool_runtime
        self.capabilities = capabilities
        self.workspace_root = workspace_root.resolve() if workspace_root is not None else None

    def propose(self, task, context, modeling, *, iteration):
        system = build_agent_system_prompt(
            AgentRole.CODE,
            output_contract=(
                "Return one CodeProposal JSON object containing a complete deterministic Python script. The script must "
                "print one final JSON object with validations, non-empty validation_evidence for every required ID, and "
                f"optional metrics; write files only under .m2harness-code/{task.task_id}/iteration-{iteration}/outputs."
            ),
        ) + _scope_instruction(task)
        handoff_paths = sorted(
            item.relative_path for item in context.readonly_files
            if "/exchanges/" in item.relative_path.replace("\\", "/")
        )
        prompt = json.dumps(_compact_provider_payload({"task": task.model_dump(mode="json"), "context": _context_json(context), "modeling": modeling.model_dump(mode="json"), "iteration": iteration, "handoff_paths": handoff_paths, "skill_context": _skill_context(self.skills, ("coding-contract", "code-debugging", "dimensional-analysis", "visualization", "scientific-figure-design", "numerical-validation")), "schema": CodeProposal.model_json_schema()}), ensure_ascii=False)
        try:
            if self.tool_runtime is not None and self.capabilities is not None:
                logical_name = "solve_" + re.sub(r"[^A-Za-z0-9_-]+", "_", task.task_id) + ".py"
                source_path = f".m2harness-code/{task.task_id}/iteration-{iteration}/{logical_name}"
                tool_prompt = prompt + json.dumps({
                    "tool_protocol": [
                        "You are a small Code Agent Harness. Use the supplied workspace tools for all file inspection and edits.",
                        "全过程使用中文交流、注释和验证说明；只在代码标识符中保留必要英文。",
                        "Continue only the isolated Code Agent context from context.metadata.code_conversation; do not reset it on a revision iteration. Model Agent context is separate and arrives only through the handoff contract.",
                        "First read the latest model-to-code handoff under the supplied allowlisted exchange paths with workspace_read; it is the auditable contract for this implementation.",
                        f"Write the complete deterministic Python source with workspace_write to {source_path}.",
                        "If the source is too large for one tool call, write it in ordered chunks: first workspace_write with overwrite=true, then workspace_write with append=true; never omit or summarize source chunks.",
                        "Use workspace_read/workspace_search to inspect staged inputs and existing files, and python_execute or validation_run for checks.",
                        "Do not replace the accepted modeling main scheme with a materially different algorithm or search space. If an implementation deviation is unavoidable, record it explicitly and do not claim stronger optimality than the evidence supports.",
                        "After the first complete implementation, run only the checks needed for the required validations; do not spend unbounded turns on exploratory solver surgery. Once a reproducible feasible result and its optimality/time-limit evidence are available, write the final script and return the handoff JSON.",
                        "Do not put source code in the final JSON. Return only logical_name, source_path, timeout_seconds, and expected_validations after the file is written and checked.",
                    ],
                    "final_schema": {
                        "type": "object", "additionalProperties": False,
                        "required": ["logical_name", "source_path", "timeout_seconds", "expected_validations"],
                        "properties": {
                            "logical_name": {"type": "string", "pattern": r"^[A-Za-z0-9._-]+\\.py$"},
                            "source_path": {"type": "string"}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                            "expected_validations": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                }, ensure_ascii=False)
                bridge = MainHarnessToolBridge(self.tool_runtime, self.capabilities, workspace_root=self.workspace_root)
                value = self.client.tool_agent_json(
                    system=system + "\nThis Code Agent has a Main Harness tool lane. Never claim a file was written unless a workspace_write tool result confirms it.",
                    content=_content_parts(context, tool_prompt, deepseek=bool(getattr(self.client, "_deepseek_mode", False))), schema=CodeProposal.model_json_schema(),
                    tool_bridge=bridge, task_id=task.task_id, iteration=iteration,
                )
                if "source_path" in value:
                    read_result = bridge.execute(str("workspace_read"), {"path": str(value["source_path"]), "max_bytes": 2_000_000}, task_id=task.task_id, iteration=iteration, turn=0)
                    if not read_result.get("ok"):
                        raise ValueError(f"Code Agent source_path could not be read: {read_result.get('error_message', 'unknown error')}")
                    value["source"] = (read_result.get("output") or {}).get("content", "")
                    value.pop("source_path", None)
                value.pop("source_path", None)
            else:
                    value = self.client.json(system=system, content=_content_parts(context, prompt, deepseek=bool(getattr(self.client, "_deepseek_mode", False))), schema=CodeProposal.model_json_schema())
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
                "全流程使用中文，最终报告、图表标题、验证说明和 LaTeX 正文使用中文（数学符号和代码标识符除外）。",
                "Synthesize only reviewed claims and preserve limitations.",
                "Return a polished single-question report and one self-contained compile-ready LaTeX article.",
                "The LaTeX must begin with documentclass, include title/abstract/sections, and contain no input/include/write18/external reads.",
                "Use a clean academic layout with geometry, booktabs and mathematical notation only when supported by evidence.",
            ],
        }, ensure_ascii=False)
        value = self.client.json(
            system=build_agent_system_prompt(
                AgentRole.PAPER,
                output_contract="Return exactly one JSON object containing a reviewed final_report and one self-contained compile-ready LaTeX article. Use Chinese throughout.",
            ),
            content=[{"type": "text", "text": prompt}], schema=schema,
        )
        composition = _PaperComposition.model_validate(value, strict=False)
        artifact = ProducedArtifact(
            logical_name="final-question-paper.tex", kind=ArtifactKind.FINAL_LATEX_PAPER,
            media_type="text/x-tex", text=composition.latex,
        )
        return composition.final_report, artifact


def build_qwen_solve_problem_service(*, sandbox, workspace_root, research_agent=None, skills=None, client: QwenChatClient | None = None, max_iterations: int = 3, tool_runtime: ToolRuntime | None = None, capabilities: CapabilityRegistry | None = None, archive_writer: SolveProblemArchivePort | None = None) -> SolveProblemService:
    """Compose the target solve tool after ``build_local_runtime()`` created its sandbox."""
    qwen = client or QwenChatClient()
    from m2harness.infrastructure.code_harness import LocalPythonCodeHarness

    model_agent = QwenModelAgent(qwen, skills=skills)
    code_harness = LocalPythonCodeHarness(QwenCodeProposalProvider(qwen, skills=skills, tool_runtime=tool_runtime, capabilities=capabilities, workspace_root=Path(workspace_root)), sandbox, workspace_root)
    return SolveProblemService(
        model_agent, code_harness, max_iterations=max_iterations,
        research_agent=research_agent,
        file_reader=WorkspaceReadOnlyFileReader(workspace_root),
        archive_writer=archive_writer,
    )


def build_dsh_solve_problem_service(*, sandbox, workspace_root, research_agent=None, skills=None, client: QwenChatClient | None = None, max_iterations: int = 3, archive_writer: SolveProblemArchivePort | None = None, dsh_config=None) -> SolveProblemService:
    """Compose the production LangGraph/DeepAgents Code Harness.

    ``dsh`` remains the stable CLI selector for compatibility with existing
    runs, but the default implementation is now the maintained DeepAgents
    runtime.  The old JSON-RPC adapter is retained only as an explicit legacy
    integration and is never selected implicitly.
    """
    qwen = client or QwenChatClient()
    from m2harness.infrastructure.code_harness import LocalPythonCodeHarness
    from m2harness.infrastructure.deepagents_code_harness import DeepAgentsCodeProposalProvider

    model_agent = QwenModelAgent(qwen, skills=skills)
    provider = DeepAgentsCodeProposalProvider(
        Path(workspace_root), sandbox,
        base_url=qwen.base_url, model_name=qwen.model, api_key=qwen.api_key,
        skills=_deepagents_skill_paths(skills, Path(workspace_root)),
    )
    code_harness = LocalPythonCodeHarness(provider, sandbox, Path(workspace_root))
    return SolveProblemService(
        model_agent, code_harness, max_iterations=max_iterations,
        research_agent=research_agent,
        file_reader=WorkspaceReadOnlyFileReader(Path(workspace_root)),
        archive_writer=archive_writer,
    )


def build_deepagents_solve_problem_service(**kwargs: Any) -> SolveProblemService:
    """Named alias for callers that want to make the runtime explicit."""
    return build_dsh_solve_problem_service(**kwargs)


def _deepagents_skill_paths(skills: Any, workspace_root: Path) -> list[str]:
    """Stage registry skills inside the Code Agent's virtual workspace.

    DeepAgents intentionally resolves skills relative to its backend root.  We
    therefore copy the immutable, digest-checked skill bodies into a dedicated
    read-only-by-contract lane instead of handing the agent arbitrary host
    paths.  This preserves progressive disclosure while keeping the Code
    Agent's filesystem permission boundary meaningful.
    """
    staged_root = workspace_root.resolve() / ".m2harness-code" / "skills"
    staged_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if hasattr(skills, "list") and hasattr(skills, "get"):
        summaries = skills.list(model_invocable=True)
        for summary in summaries:
            definition = skills.get(summary.name)
            if definition is None:
                continue
            source = Path(definition.source).resolve()
            target = staged_root / summary.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / "SKILL.md")
            paths.append("/.m2harness-code/skills/" + summary.name + "/")
        return paths
    for item in skills or ():
        path = getattr(item, "path", None) or getattr(item, "source_path", None)
        if path:
            paths.append(str(path).replace("\\", "/"))
    return paths


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
