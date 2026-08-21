"""Provider-neutral Agent turn loop with bounded Tool dispatch."""

from __future__ import annotations

from typing import Any

from m2harness.application.tools import ToolRuntime, ToolRegistry
from m2harness.application.compact import ContextCompactor, ContextCompactorPort
from m2harness.domain.agent import AgentSession, ModelRequest, ModelResponse
from m2harness.domain.capability import CapabilityResolution
from m2harness.domain.tool import ToolCall
from m2harness.ports.provider import ModelProvider


class AgentRuntime:
    def __init__(self, provider: ModelProvider, tool_runtime: ToolRuntime, tool_registry: ToolRegistry, *, compactor: ContextCompactorPort | None = None) -> None:
        self.provider = provider
        self.tool_runtime = tool_runtime
        self.tool_registry = tool_registry
        self.compactor = compactor or ContextCompactor()

    async def run(self, request: ModelRequest, resolution: CapabilityResolution) -> ModelResponse:
        session = request.session
        current = request.model_copy(update={"tools": self._project_tools(resolution)})
        for _ in range(session.max_turns):
            current = await self.compactor.compact(current)
            response = await self.provider.complete(current)
            if not response.tool_calls:
                return response
            tool_outputs: list[dict[str, Any]] = []
            for model_call in response.tool_calls:
                definition = self.tool_registry.get(model_call.name)
                if definition is None:
                    tool_outputs.append({"tool": model_call.name, "ok": False, "error": "unknown_tool"})
                    continue
                call = ToolCall(
                    call_id=model_call.call_id, tool_name=definition.name, tool_version=definition.version,
                    activity_id=session.activity_id, session_id=session.session_id,
                    idempotency_key=f"{session.session_id}:{model_call.call_id}", arguments=model_call.arguments,
                    requested_at=current.context.created_at,
                )
                result = self.tool_runtime.execute(call, resolution)
                tool_outputs.append({"tool": result.tool_name, "ok": result.ok, "output": result.output, "error": result.error_message})
            current = current.model_copy(update={
                "messages": current.messages + ({"role": "tool", "content": tool_outputs},),
                "session": session.model_copy(update={"turn_count": session.turn_count + 1}),
                "tools": self._project_tools(resolution),
            })
        raise RuntimeError(f"agent exceeded max turns: {session.max_turns}")

    def _project_tools(self, resolution: CapabilityResolution) -> tuple[dict[str, Any], ...]:
        """Project only authorized descriptors into provider-neutral JSON.

        Providers can translate this small OpenAI-compatible shape into their
        native request format.  The registry remains the source of truth and
        no handler implementation or secret-bearing policy is exposed.
        """
        projected = []
        for definition in self.tool_registry.list(resolution):
            projected.append({
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            })
        return tuple(projected)
