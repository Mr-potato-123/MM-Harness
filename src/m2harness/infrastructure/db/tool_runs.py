"""SQLite-backed tool idempotency reservations and completed results."""

from __future__ import annotations

import sqlite3
import hashlib
import json
from uuid import uuid4
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from m2harness.domain.tool import ToolResult
from m2harness.domain.tool import ToolCall, ToolDefinition


class SQLiteToolExecutionStore:
    """Crash-tolerant idempotency store shared by local workers.

    A running reservation expires after the configured lease, allowing a
    crashed worker's call to be retried.  Completed results are immutable for
    the lifetime of the key and are returned byte-for-byte through Pydantic's
    JSON representation.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS m2_tool_runs (
                    idempotency_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('running','completed')),
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_json TEXT,
                    lease_token TEXT
                )"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(m2_tool_runs)")}
            if "lease_token" not in columns:
                connection.execute("ALTER TABLE m2_tool_runs ADD COLUMN lease_token TEXT")

    def lookup(self, idempotency_key: str) -> ToolResult | None:
        with self._connection() as connection:
            row = connection.execute("SELECT status,result_json FROM m2_tool_runs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        if not row or row[0] != "completed" or not row[1]:
            return None
        return ToolResult.model_validate_json(row[1], strict=False)

    def reserve(self, idempotency_key: str, lease_seconds: int, lease_token: str | None = None) -> bool:
        now = datetime.now(UTC)
        lease_token = lease_token or uuid4().hex
        expiry = now - timedelta(seconds=max(1, lease_seconds))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT status,started_at FROM m2_tool_runs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row is None:
                    connection.execute("INSERT INTO m2_tool_runs(idempotency_key,status,started_at,lease_token) VALUES(?,?,?,?)", (idempotency_key, "running", now.isoformat(), lease_token))
                    connection.commit()
                    return True
                if row[0] == "completed":
                    connection.rollback()
                    return False
                try:
                    started = datetime.fromisoformat(row[1])
                except (TypeError, ValueError):
                    started = now
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                if started < expiry:
                    connection.execute("UPDATE m2_tool_runs SET status='running',started_at=?,completed_at=NULL,result_json=NULL,lease_token=? WHERE idempotency_key=?", (now.isoformat(), lease_token, idempotency_key))
                    connection.commit()
                    return True
                connection.rollback()
                return False
            except BaseException:
                connection.rollback()
                raise

    def complete(self, idempotency_key: str, result: ToolResult, lease_token: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            if lease_token is None:
                cursor = connection.execute("UPDATE m2_tool_runs SET status='completed',completed_at=?,result_json=? WHERE idempotency_key=? AND status='running'", (now, result.model_dump_json(), idempotency_key))
            else:
                cursor = connection.execute("UPDATE m2_tool_runs SET status='completed',completed_at=?,result_json=? WHERE idempotency_key=? AND status='running' AND lease_token=?", (now, result.model_dump_json(), idempotency_key, lease_token))
            if cursor.rowcount != 1:
                raise RuntimeError("tool idempotency lease was lost before completion")


ZERO_HASH = "0" * 64


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class SQLiteToolAuditStore:
    """Append-only, hash-chained tool call audit records."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS m2_tool_audit (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )"""
            )

    def _append(self, event_type: str, call: ToolCall, payload: dict) -> None:
        from uuid import uuid4
        event_id = uuid4()
        occurred_at = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous = connection.execute("SELECT event_hash FROM m2_tool_audit ORDER BY seq DESC LIMIT 1").fetchone()
                previous_hash = previous[0] if previous else ZERO_HASH
                body = {"event_id": str(event_id), "event_type": event_type, "idempotency_key": call.idempotency_key, "call_id": str(call.call_id), "tool_name": call.tool_name, "occurred_at": occurred_at, "payload": payload, "previous_hash": previous_hash}
                event_hash = hashlib.sha256((previous_hash + _canonical(body)).encode("utf-8")).hexdigest()
                connection.execute("INSERT INTO m2_tool_audit(event_id,event_type,idempotency_key,call_id,tool_name,occurred_at,payload,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?)", (str(event_id), event_type, call.idempotency_key, str(call.call_id), call.tool_name, occurred_at, _canonical(payload), previous_hash, event_hash))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def append_called(self, call: ToolCall, definition: ToolDefinition) -> None:
        arguments_digest = hashlib.sha256(_canonical(call.arguments).encode("utf-8")).hexdigest()
        self._append("tool.called.v1", call, {"tool_version": definition.version, "activity_id": str(call.activity_id), "session_id": str(call.session_id), "argument_keys": sorted(call.arguments), "arguments_sha256": arguments_digest})

    def append_completed(self, call: ToolCall, result: ToolResult) -> None:
        self._append("tool.completed.v1", call, {"ok": result.ok, "error_code": result.error_code, "artifact_ids": [str(item) for item in result.artifact_ids], "redacted": result.redacted})

    def list(self, *, idempotency_key: str | None = None) -> list[dict]:
        with self._connection() as connection:
            if idempotency_key:
                rows = connection.execute("SELECT * FROM m2_tool_audit WHERE idempotency_key=? ORDER BY seq", (idempotency_key,)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM m2_tool_audit ORDER BY seq").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def verify(self) -> int:
        previous = ZERO_HASH
        rows = self.list()
        for row in rows:
            body = {"event_id": row["event_id"], "event_type": row["event_type"], "idempotency_key": row["idempotency_key"], "call_id": row["call_id"], "tool_name": row["tool_name"], "occurred_at": row["occurred_at"], "payload": row["payload"], "previous_hash": row["previous_hash"]}
            expected = hashlib.sha256((previous + _canonical(body)).encode("utf-8")).hexdigest()
            if row["previous_hash"] != previous or row["event_hash"] != expected:
                raise ValueError(f"tool audit chain verification failed at seq {row['seq']}")
            previous = row["event_hash"]
        return len(rows)
