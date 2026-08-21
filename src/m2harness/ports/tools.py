"""Tool registry and executor ports."""

from __future__ import annotations

from typing import Protocol

from m2harness.domain.capability import CapabilityResolution
from m2harness.domain.tool import ToolCall, ToolDefinition, ToolResult


class ToolRegistryPort(Protocol):
    def register(self, definition: ToolDefinition, handler) -> None: ...
    def list(self, resolution: CapabilityResolution | None = None) -> tuple[ToolDefinition, ...]: ...
    def get(self, name: str, version: str | None = None) -> ToolDefinition | None: ...


class ToolExecutor(Protocol):
    def execute(self, call: ToolCall, definition: ToolDefinition) -> ToolResult: ...
