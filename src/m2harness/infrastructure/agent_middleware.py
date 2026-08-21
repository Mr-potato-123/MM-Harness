"""LangChain middleware shared by the M2Harness agent runtimes.

The middleware is deliberately observational.  It records lifecycle and tool
boundaries without owning orchestration, retry policy, or cancellation.  The
outer harness remains the authority for task transitions; this file only makes
the Code Agent's internal handoff visible in the same NDJSON stream as its
LangGraph state snapshots.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware


class M2AgentAuditMiddleware(AgentMiddleware):
    """Audit hooks for a single DeepAgents/LangChain graph invocation."""

    def __init__(self, *, event_root: Path) -> None:
        self.event_root = event_root.resolve()
        self.event_root.mkdir(parents=True, exist_ok=True)
        self.tools = []

    @property
    def name(self) -> str:
        return "M2AgentAuditMiddleware"

    def _metadata(self, runtime: Any) -> dict[str, Any]:
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            metadata = config.get("metadata", {})
            return metadata if isinstance(metadata, dict) else {}
        return {}

    def _emit(self, kind: str, runtime: Any, state: Any = None, **extra: Any) -> None:
        metadata = self._metadata(runtime)
        task_id = str(metadata.get("m2h_task_id", "unscoped"))
        run_id = str(metadata.get("m2h_run_id", "unscoped"))
        safe_run_id = "".join(char if char.isalnum() or char in "_.-" else "_" for char in run_id)
        path = self.event_root / f"{safe_run_id}-{task_id}.ndjson"
        payload: dict[str, Any] = {
            "occurred_at": datetime.now(UTC).isoformat(),
            "kind": f"middleware.{kind}",
            "task_id": task_id,
            "iteration": metadata.get("m2h_iteration"),
            "runtime": metadata.get("m2h_runtime", "langchain"),
            **extra,
        }
        if isinstance(state, dict):
            payload["state_keys"] = sorted(str(key) for key in state.keys())
            messages = state.get("messages")
            if isinstance(messages, (list, tuple)):
                payload["message_count"] = len(messages)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")

    def before_agent(self, state: Any, runtime: Any) -> None:
        self._emit("before_agent", runtime, state)

    def before_model(self, state: Any, runtime: Any) -> None:
        self._emit("before_model", runtime, state)

    def after_model(self, state: Any, runtime: Any) -> None:
        self._emit("after_model", runtime, state)

    def after_agent(self, state: Any, runtime: Any) -> None:
        self._emit("after_agent", runtime, state)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        runtime = getattr(request, "runtime", None)
        tool_call = getattr(request, "tool_call", {})
        name = tool_call.get("name") if isinstance(tool_call, dict) else None
        self._emit("before_tool", runtime, getattr(request, "state", None), tool_name=name)
        try:
            result = handler(request)
        except Exception as exc:
            self._emit("tool_error", runtime, getattr(request, "state", None), tool_name=name, error=str(exc)[:2_000])
            raise
        result_content = getattr(result, "content", None)
        self._emit(
            "after_tool",
            runtime,
            getattr(request, "state", None),
            tool_name=name,
            result_preview=str(result_content)[:1_000] if result_content is not None else None,
        )
        return result
