"""Capability-scoped Tool Registry and middleware execution pipeline."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from m2harness.domain.capability import CapabilityResolution
from m2harness.domain.tool import ToolCall, ToolDefinition, ToolResult


Handler = Callable[[dict[str, Any]], dict[str, Any]]


class ToolExecutionStore(Protocol):
    """Durable idempotency boundary for tool calls.

    ``reserve`` must be atomic across workers.  It returns ``True`` for a new
    (or expired) reservation and ``False`` when another completed/running call
    owns the key.  ``lookup`` returns a completed result when available.
    """

    def lookup(self, idempotency_key: str) -> ToolResult | None: ...
    def reserve(self, idempotency_key: str, lease_seconds: int, lease_token: str | None = None) -> bool: ...
    def complete(self, idempotency_key: str, result: ToolResult, lease_token: str | None = None) -> None: ...


class ToolAuditSink(Protocol):
    def append_called(self, call: ToolCall, definition: ToolDefinition) -> None: ...
    def append_completed(self, call: ToolCall, result: ToolResult) -> None: ...


class ToolMiddleware(Protocol):
    def before(self, call: ToolCall, definition: ToolDefinition, resolution: CapabilityResolution) -> None: ...
    def after(self, call: ToolCall, definition: ToolDefinition, result: ToolResult) -> ToolResult: ...


class ResultBudgetMiddleware:
    """Rejects oversized model-visible output before it reaches the session."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes

    def before(self, call: ToolCall, definition: ToolDefinition, resolution: CapabilityResolution) -> None:
        return None

    def after(self, call: ToolCall, definition: ToolDefinition, result: ToolResult) -> ToolResult:
        if result.ok and result.output is not None:
            encoded = json.dumps(result.output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > self.max_bytes:
                return ToolResult(
                    call_id=result.call_id, tool_name=result.tool_name, ok=False,
                    error_code="result_budget_exceeded", error_message="tool result exceeds middleware budget",
                    completed_at=result.completed_at,
                )
        return result


class ResultRedactionMiddleware:
    """Redacts configured keys from model-visible results while retaining the fact of redaction."""

    def __init__(self, keys: set[str] | frozenset[str]) -> None:
        self.keys = frozenset(keys)

    def before(self, call: ToolCall, definition: ToolDefinition, resolution: CapabilityResolution) -> None:
        return None

    def after(self, call: ToolCall, definition: ToolDefinition, result: ToolResult) -> ToolResult:
        if not result.ok or result.output is None:
            return result
        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: ("[REDACTED]" if key in self.keys else redact(item)) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, tuple):
                return tuple(redact(item) for item in value)
            return value
        output = redact(result.output)
        return result.model_copy(update={"output": output, "redacted": output != result.output})


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], ToolDefinition] = {}
        self._handlers: dict[tuple[str, str], Handler] = {}

    def register(self, definition: ToolDefinition, handler: Handler) -> None:
        key = (definition.name, definition.version)
        if key in self._definitions:
            raise ValueError(f"duplicate tool: {definition.name}@{definition.version}")
        self._definitions[key] = definition
        self._handlers[key] = handler

    def get(self, name: str, version: str | None = None) -> ToolDefinition | None:
        if version:
            return self._definitions.get((name, version))
        matches = [definition for (tool_name, _), definition in self._definitions.items() if tool_name == name]
        return sorted(matches, key=lambda item: item.version)[-1] if matches else None

    def list(self, resolution: CapabilityResolution | None = None) -> tuple[ToolDefinition, ...]:
        """Return deterministic tool catalog entries.

        A capability resolution is optional for catalog/diagnostic callers.  The
        execution path still passes a resolution and therefore only exposes
        tools whose required capability is granted.  Returning the full catalog
        when no resolution is supplied makes the registry inspectable during
        startup and keeps model-tool projection separate from authorization.
        """
        if resolution is None:
            return tuple(sorted(self._definitions.values(), key=lambda item: (item.name, item.version)))
        granted = {(item.name, item.version) for item in resolution.granted}
        return tuple(sorted((definition for definition in self._definitions.values() if (definition.required_capability.name, definition.required_capability.version) in granted), key=lambda item: (item.name, item.version)))

    def catalog(self) -> tuple[ToolDefinition, ...]:
        """Alias used by startup checks and operator-facing catalog endpoints."""
        return self.list()

    def handler(self, definition: ToolDefinition) -> Handler:
        return self._handlers[(definition.name, definition.version)]


class ToolRuntime:
    """Fixed-order middleware: authorization → input → idempotency → handler → budget."""

    def __init__(self, registry: ToolRegistry, *, max_result_bytes: int = 1_048_576, middlewares: tuple[ToolMiddleware, ...] = (), execution_store: ToolExecutionStore | None = None, audit_sink: ToolAuditSink | None = None, idempotency_lease_seconds: int = 3_600) -> None:
        self.registry = registry
        self.max_result_bytes = max_result_bytes
        self.middlewares = middlewares
        self.execution_store = execution_store
        self.audit_sink = audit_sink
        self.idempotency_lease_seconds = idempotency_lease_seconds
        self._completed: dict[str, ToolResult] = {}
        self._running: set[str] = set()
        self._lock = threading.RLock()

    def _audit_completed(self, call: ToolCall, result: ToolResult) -> None:
        if self.audit_sink is not None:
            try:
                self.audit_sink.append_completed(call, result)
            except Exception:
                pass

    def execute(self, call: ToolCall, resolution: CapabilityResolution) -> ToolResult:
        definition = self.registry.get(call.tool_name, call.tool_version)
        if definition is None:
            return self._error(call, "tool_not_found", "tool is not registered")
        if self.audit_sink is not None:
            try:
                self.audit_sink.append_called(call, definition)
            except Exception:
                # Auditing must not make an otherwise safe read-only call
                # disappear; the sink itself exposes verify/health to operators.
                pass
        if not any(item.name == definition.required_capability.name and item.version == definition.required_capability.version for item in resolution.granted):
            result = self._error(call, "capability_denied", "activity has no grant for this tool")
            self._audit_completed(call, result)
            return result
        if definition.policy.network == "unrestricted":
            result = self._error(call, "policy_denied", "unrestricted network tools are not executable")
            self._audit_completed(call, result)
            return result
        reserved = False
        reservation_token: str | None = None
        durable_key = f"{definition.name}@{definition.version}:{call.idempotency_key}"
        if definition.idempotency_required:
            with self._lock:
                cached = self._completed.get(durable_key)
                if cached is None and self.execution_store is not None:
                    try:
                        cached = self.execution_store.lookup(durable_key)
                    except Exception as exc:
                        result = self._error(call, "idempotency_store_error", str(exc)[:2000])
                        self._audit_completed(call, result)
                        return result
                if cached is not None:
                    return cached
                if self.execution_store is not None:
                    reservation_token = uuid4().hex
                    try:
                        reserved = self.execution_store.reserve(durable_key, self.idempotency_lease_seconds, reservation_token)
                    except Exception as exc:
                        result = self._error(call, "idempotency_store_error", str(exc)[:2000])
                        self._audit_completed(call, result)
                        return result
                    if not reserved:
                        result = self._error(call, "idempotency_in_progress", "another worker owns this idempotency key")
                        self._audit_completed(call, result)
                        return result
                elif durable_key in self._running:
                    result = self._error(call, "idempotency_in_progress", "another call owns this idempotency key")
                    self._audit_completed(call, result)
                    return result
                else:
                    self._running.add(durable_key)
        try:
            for middleware in self.middlewares:
                middleware.before(call, definition, resolution)
            self._validate_object(definition.input_schema, call.arguments)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="m2h-tool")
            future = executor.submit(self.registry.handler(definition), call.arguments)
            try:
                value = future.result(timeout=definition.timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"tool exceeded timeout of {definition.timeout_seconds}s") from exc
            finally:
                # Cancellation is cooperative for Python handlers; subprocess
                # based tools enforce their own hard timeout.  Do not block the
                # caller while a misbehaving handler unwinds.
                executor.shutdown(wait=False, cancel_futures=True)
            if not isinstance(value, dict):
                raise ValueError("tool handler must return an object")
            self._validate_object(definition.output_schema, value)
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > min(definition.output_limit_bytes, self.max_result_bytes):
                raise ValueError("tool result exceeds output budget")
            result = ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=True, output=value, completed_at=datetime.now(UTC))
        except Exception as exc:
            result = self._error(call, "tool_execution_error", str(exc)[:2000])
        for middleware in reversed(self.middlewares):
            try:
                result = middleware.after(call, definition, result)
            except Exception as exc:
                result = self._error(call, "tool_middleware_error", str(exc)[:2000])
        if definition.idempotency_required:
            with self._lock:
                self._completed[durable_key] = result
                self._running.discard(durable_key)
            if self.execution_store is not None and reserved:
                try:
                    self.execution_store.complete(durable_key, result, reservation_token)
                except Exception as exc:
                    # The in-process result remains available; operators can
                    # observe the persistence failure through infrastructure
                    # health checks and retry the call after the lease expires.
                    with self._lock:
                        self._completed.pop(durable_key, None)
                    result = self._error(call, "idempotency_lease_lost", str(exc)[:2000])
        self._audit_completed(call, result)
        return result

    @staticmethod
    def _validate_object(schema: dict[str, Any], value: dict[str, Any]) -> None:
        ToolRuntime._validate_schema(schema, value, "arguments")

    @staticmethod
    def _validate_schema(schema: dict[str, Any], value: Any, path: str) -> None:
        expected = schema.get("type")
        valid = {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }
        if expected in valid and not valid[expected]:
            raise ValueError(f"{path} must be {expected}")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} must be one of: {', '.join(map(str, schema['enum']))}")
        if isinstance(value, dict):
            required = schema.get("required", [])
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(f"{path} missing required properties: " + ", ".join(missing))
            properties = schema.get("properties", {})
            if schema.get("additionalProperties") is False:
                unknown = sorted(set(value) - set(properties))
                if unknown:
                    raise ValueError(f"{path} has unknown properties: " + ", ".join(unknown))
            for key, item in value.items():
                if key in properties:
                    ToolRuntime._validate_schema(properties[key], item, f"{path}.{key}")
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                raise ValueError(f"{path} has fewer than {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                raise ValueError(f"{path} has more than {schema['maxItems']} items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    ToolRuntime._validate_schema(item_schema, item, f"{path}[{index}]")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                raise ValueError(f"{path} is shorter than {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                raise ValueError(f"{path} exceeds {schema['maxLength']} characters")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"{path} is below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"{path} exceeds maximum {schema['maximum']}")

    @staticmethod
    def _error(call: ToolCall, code: str, message: str) -> ToolResult:
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error_code=code, error_message=message, completed_at=datetime.now(UTC))
