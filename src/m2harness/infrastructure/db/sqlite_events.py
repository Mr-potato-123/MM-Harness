"""SQLite event adapter with optimistic aggregate versions and hash chaining."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from m2harness.domain.events import DomainEvent, EventEnvelope
from m2harness.errors import ConflictError

ZERO_HASH = "0" * 64


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteEventStore:
    """A standalone event store; it deliberately uses m2_ tables beside legacy tables."""

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS m2_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    aggregate_version INTEGER NOT NULL,
                    correlation_id TEXT NOT NULL,
                    causation_id TEXT,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    payload_schema TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_m2_event_aggregate_version
                    ON m2_events(aggregate_id, aggregate_version);
                CREATE TABLE IF NOT EXISTS m2_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    topic TEXT NOT NULL,
                    body TEXT NOT NULL,
                    published_at TEXT
                );
                """
            )

    def _last(self, connection: sqlite3.Connection, aggregate_id: UUID) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM m2_events WHERE aggregate_id=? ORDER BY aggregate_version DESC LIMIT 1",
            (str(aggregate_id),),
        ).fetchone()

    def append_domain(
        self,
        aggregate_id: UUID,
        expected_version: int,
        event: DomainEvent,
        *,
        aggregate_type: str = "workflow",
        correlation_id: UUID | None = None,
        payload_schema: str = "m2harness/domain-event/v1",
    ) -> EventEnvelope:
        occurred_at = datetime.now(UTC)
        event_id = uuid4()
        correlation_id = correlation_id or aggregate_id
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous_row = self._last(connection, aggregate_id)
                actual = int(previous_row["aggregate_version"]) if previous_row else 0
                previous_hash = previous_row["event_hash"] if previous_row else ZERO_HASH
                if actual != expected_version:
                    raise ConflictError(f"aggregate version conflict: expected {expected_version}, actual {actual}")
                version = actual + 1
                envelope_data = {
                    "event_id": str(event_id), "event_type": event.event_type.value,
                    "aggregate_type": aggregate_type, "aggregate_id": str(aggregate_id),
                    "aggregate_version": version, "correlation_id": str(correlation_id),
                    "causation_id": str(event.causation_id) if event.causation_id else None,
                    "actor_type": event.actor_type, "actor_id": event.actor_id,
                    "occurred_at": occurred_at.isoformat(), "payload": event.payload,
                    "payload_schema": payload_schema, "previous_hash": previous_hash,
                }
                event_hash = hashlib.sha256((previous_hash + _canonical(envelope_data)).encode("utf-8")).hexdigest()
                envelope = EventEnvelope.model_validate({**envelope_data, "event_hash": event_hash}, strict=False)
                connection.execute(
                    "INSERT INTO m2_events(event_id,event_type,aggregate_type,aggregate_id,aggregate_version,correlation_id,causation_id,actor_type,actor_id,occurred_at,payload,payload_schema,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(event_id), event.event_type.value, aggregate_type, str(aggregate_id), version,
                     str(correlation_id), str(event.causation_id) if event.causation_id else None,
                     event.actor_type, event.actor_id, occurred_at.isoformat(), _canonical(event.payload),
                     payload_schema, previous_hash, event_hash),
                )
                connection.execute(
                    "INSERT INTO m2_outbox(event_id,topic,body) VALUES(?,?,?)",
                    (str(event_id), event.event_type.value, envelope.model_dump_json()),
                )
                connection.commit()
                return envelope
            except BaseException:
                connection.rollback()
                raise

    def append(self, aggregate_id: UUID, expected_version: int, event: DomainEvent) -> EventEnvelope:
        return self.append_domain(aggregate_id, expected_version, event)

    def list(self, aggregate_id: UUID) -> list[EventEnvelope]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM m2_events WHERE aggregate_id=? ORDER BY aggregate_version", (str(aggregate_id),)).fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data.pop("seq", None)
            data["payload"] = json.loads(data["payload"])
            result.append(EventEnvelope.model_validate(data, strict=False))
        return result

    def verify(self) -> int:
        previous_by_aggregate: dict[str, str] = {}
        version_by_aggregate: dict[str, int] = {}
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM m2_events ORDER BY seq").fetchall()
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data["payload"])
            aggregate_id = data["aggregate_id"]
            previous = previous_by_aggregate.get(aggregate_id, ZERO_HASH)
            expected_version = version_by_aggregate.get(aggregate_id, 0) + 1
            if int(data["aggregate_version"]) != expected_version:
                raise ConflictError(f"m2 event aggregate version gap at seq {row['seq']}: expected {expected_version}, got {data['aggregate_version']}")
            expected = hashlib.sha256((previous + _canonical({
                "event_id": data["event_id"], "event_type": data["event_type"],
                "aggregate_type": data["aggregate_type"], "aggregate_id": data["aggregate_id"],
                "aggregate_version": data["aggregate_version"], "correlation_id": data["correlation_id"],
                "causation_id": data["causation_id"], "actor_type": data["actor_type"],
                "actor_id": data["actor_id"], "occurred_at": data["occurred_at"],
                "payload": data["payload"], "payload_schema": data["payload_schema"],
                "previous_hash": data["previous_hash"],
            })).encode("utf-8")).hexdigest()
            if data["previous_hash"] != previous or data["event_hash"] != expected:
                raise ConflictError(f"m2 event chain verification failed at seq {row['seq']}")
            previous_by_aggregate[aggregate_id] = data["event_hash"]
            version_by_aggregate[aggregate_id] = int(data["aggregate_version"])
        return len(rows)

    def unpublished(self, limit: int = 100) -> list[EventEnvelope]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT e.* FROM m2_events e JOIN m2_outbox o ON o.event_id=e.event_id WHERE o.published_at IS NULL ORDER BY e.seq LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            data = dict(row); data["payload"] = json.loads(data["payload"])
            data.pop("seq", None)
            result.append(EventEnvelope.model_validate(data, strict=False))
        return result

    def mark_published(self, event_id: UUID) -> None:
        with self._connection() as connection:
            connection.execute("UPDATE m2_outbox SET published_at=? WHERE event_id=?", (datetime.now(UTC).isoformat(), str(event_id)))
