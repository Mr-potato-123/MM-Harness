"""Transactional SQLite state, leases, activities, and hash-chained events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence, TypeVar
from uuid import UUID

from pydantic import BaseModel

from m2harness.errors import ConfigurationError, ConflictError, InvalidTransitionError, LeaseLostError, NotFoundError
from m2harness.models import (
    ActivityRecord,
    ActivityStatus,
    ArtifactRecord,
    EventRecord,
    ProjectRecord,
    QuestionRecord,
    QuestionState,
    StageKind,
    new_uuid,
    utc_now,
)

ZERO_HASH = "0" * 64
T = TypeVar("T", bound=BaseModel)


def _json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_model(model: type[T], row: sqlite3.Row) -> T:
    return model.model_validate(dict(row), strict=False)


class HarnessStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    version INTEGER NOT NULL
                );
                INSERT INTO schema_meta(version)
                    SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_meta);

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                    question_id TEXT, activity_id TEXT,
                    kind TEXT NOT NULL, logical_name TEXT NOT NULL,
                    media_type TEXT NOT NULL, sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, relative_path TEXT NOT NULL,
                    metadata TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                    key TEXT NOT NULL, title TEXT NOT NULL, state TEXT NOT NULL,
                    problem_artifact_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    failure_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL, lease_owner TEXT, lease_expires_at TEXT,
                    UNIQUE(project_id, key),
                    FOREIGN KEY(project_id) REFERENCES projects(id),
                    FOREIGN KEY(problem_artifact_id) REFERENCES artifacts(id)
                );
                CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                    question_id TEXT NOT NULL, stage TEXT NOT NULL,
                    revision INTEGER NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, attempt_count INTEGER NOT NULL,
                    request_json TEXT, result_json TEXT, error TEXT,
                    worker_id TEXT, lease_expires_at TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(question_id, stage, revision),
                    FOREIGN KEY(project_id) REFERENCES projects(id),
                    FOREIGN KEY(question_id) REFERENCES questions(id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE, project_id TEXT NOT NULL,
                    question_id TEXT, activity_id TEXT, event_type TEXT NOT NULL,
                    payload TEXT NOT NULL, occurred_at TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_question ON artifacts(question_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_question ON events(question_id, seq);
                CREATE INDEX IF NOT EXISTS idx_activities_question ON activities(question_id, created_at);
                """
            )
            versions = [row["version"] for row in connection.execute("SELECT version FROM schema_meta")]
            if versions != [1]:
                raise ConfigurationError(f"unsupported database schema metadata: {versions}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _append_event(
        self, connection: sqlite3.Connection, *, project_id: UUID, event_type: str,
        payload: dict[str, Any], question_id: UUID | None = None,
        activity_id: UUID | None = None,
    ) -> None:
        last = connection.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last["event_hash"] if last else ZERO_HASH
        event_id, occurred_at = new_uuid(), utc_now()
        envelope = {
            "event_id": str(event_id), "project_id": str(project_id),
            "question_id": str(question_id) if question_id else None,
            "activity_id": str(activity_id) if activity_id else None,
            "event_type": event_type, "payload": payload,
            "occurred_at": occurred_at.isoformat(), "previous_hash": previous,
        }
        event_hash = hashlib.sha256((previous + _json(envelope)).encode()).hexdigest()
        connection.execute(
            "INSERT INTO events(event_id,project_id,question_id,activity_id,event_type,payload,occurred_at,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (str(event_id), str(project_id), str(question_id) if question_id else None,
             str(activity_id) if activity_id else None, event_type, _json(payload),
             occurred_at.isoformat(), previous, event_hash),
        )

    def create_project(self, name: str) -> ProjectRecord:
        now = utc_now()
        project = ProjectRecord(id=new_uuid(), name=name, created_at=now, updated_at=now, version=1)
        with self._transaction() as connection:
            connection.execute("INSERT INTO projects VALUES(?,?,?,?,?)", (
                str(project.id), project.name, now.isoformat(), now.isoformat(), 1))
            self._append_event(connection, project_id=project.id, event_type="project.created", payload={"name": name})
        return project

    def get_project(self, project_id: UUID) -> ProjectRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id=?", (str(project_id),)).fetchone()
        if not row:
            raise NotFoundError(f"project not found: {project_id}")
        return _load_model(ProjectRecord, row)

    def create_question(self, project_id: UUID, key: str, title: str, problem: ArtifactRecord) -> QuestionRecord:
        if problem.project_id != project_id or problem.kind.value != "problem":
            raise ValueError("problem artifact must belong to the project and have problem kind")
        now, question_id = utc_now(), new_uuid()
        question = QuestionRecord(
            id=question_id, project_id=project_id, key=key, title=title,
            state=QuestionState.PENDING, problem_artifact_id=problem.id, revision=0,
            created_at=now, updated_at=now, version=1,
        )
        problem = problem.model_copy(update={"question_id": question_id})
        with self._transaction() as connection:
            self._insert_artifact(connection, problem)
            connection.execute(
                "INSERT INTO questions(id,project_id,key,title,state,problem_artifact_id,revision,failure_reason,created_at,updated_at,version,lease_owner,lease_expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(question.id), str(project_id), key, title, question.state.value,
                 str(problem.id), 0, None, now.isoformat(), now.isoformat(), 1, None, None),
            )
            self._append_event(connection, project_id=project_id, question_id=question.id,
                               event_type="question.created", payload={"key": key, "title": title, "problem_artifact_id": str(problem.id)})
        return question

    def get_question(self, question_id: UUID) -> QuestionRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM questions WHERE id=?", (str(question_id),)).fetchone()
        if not row:
            raise NotFoundError(f"question not found: {question_id}")
        return _load_model(QuestionRecord, row)

    def list_questions(self, project_id: UUID) -> list[QuestionRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM questions WHERE project_id=? ORDER BY created_at", (str(project_id),)).fetchall()
        return [_load_model(QuestionRecord, row) for row in rows]

    def _insert_artifact(self, connection: sqlite3.Connection, artifact: ArtifactRecord) -> None:
        connection.execute(
            "INSERT INTO artifacts(id,project_id,question_id,activity_id,kind,logical_name,media_type,sha256,size_bytes,relative_path,metadata,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(artifact.id), str(artifact.project_id), str(artifact.question_id) if artifact.question_id else None,
             str(artifact.activity_id) if artifact.activity_id else None, artifact.kind.value,
             artifact.logical_name, artifact.media_type, artifact.sha256, artifact.size_bytes,
             artifact.relative_path, _json(artifact.metadata), artifact.created_at.isoformat()),
        )

    def get_artifact(self, artifact_id: UUID) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE id=?", (str(artifact_id),)).fetchone()
        if not row:
            raise NotFoundError(f"artifact not found: {artifact_id}")
        data = dict(row); data["metadata"] = json.loads(data["metadata"])
        return ArtifactRecord.model_validate(data, strict=False)

    def register_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        if artifact.question_id is None:
            raise ValueError("a registered question input must have question provenance")
        with self._transaction() as connection:
            question = connection.execute(
                "SELECT project_id FROM questions WHERE id=?", (str(artifact.question_id),)
            ).fetchone()
            if not question:
                raise NotFoundError(f"question not found: {artifact.question_id}")
            if question["project_id"] != str(artifact.project_id):
                raise ValueError("artifact project does not match its question")
            self._insert_artifact(connection, artifact)
            self._append_event(
                connection, project_id=artifact.project_id, question_id=artifact.question_id,
                activity_id=artifact.activity_id, event_type="artifact.registered",
                payload={
                    "artifact_id": str(artifact.id), "kind": artifact.kind.value,
                    "media_type": artifact.media_type, "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                },
            )
        return artifact

    def list_artifacts(self, question_id: UUID) -> list[ArtifactRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifacts WHERE question_id=? ORDER BY created_at,id", (str(question_id),)).fetchall()
        result = []
        for row in rows:
            data = dict(row); data["metadata"] = json.loads(data["metadata"])
            result.append(ArtifactRecord.model_validate(data, strict=False))
        return result

    def list_all_artifacts(self) -> list[ArtifactRecord]:
        with self._connect() as connection:
            ids = [UUID(row["id"]) for row in connection.execute("SELECT id FROM artifacts ORDER BY created_at,id")]
        return [self.get_artifact(artifact_id) for artifact_id in ids]

    def acquire_question_lease(self, question_id: UUID, worker_id: str, lease_seconds: int) -> QuestionRecord:
        now, expires = utc_now(), utc_now() + timedelta(seconds=lease_seconds)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM questions WHERE id=?", (str(question_id),)).fetchone()
            if not row:
                raise NotFoundError(f"question not found: {question_id}")
            if row["lease_owner"] not in (None, worker_id) and row["lease_expires_at"] and row["lease_expires_at"] > now.isoformat():
                raise ConflictError(f"question is leased by {row['lease_owner']}")
            connection.execute("UPDATE questions SET lease_owner=?,lease_expires_at=? WHERE id=?", (worker_id, expires.isoformat(), str(question_id)))
            self._append_event(connection, project_id=UUID(row["project_id"]), question_id=question_id,
                               event_type="question.lease_acquired", payload={"worker_id": worker_id, "expires_at": expires.isoformat()})
        return self.get_question(question_id)

    def renew_question_lease(self, question_id: UUID, worker_id: str, lease_seconds: int) -> None:
        now, expires = utc_now(), utc_now() + timedelta(seconds=lease_seconds)
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE questions SET lease_expires_at=? WHERE id=? AND lease_owner=? AND lease_expires_at>=?",
                (expires.isoformat(), str(question_id), worker_id, now.isoformat()))
            if cursor.rowcount != 1:
                raise LeaseLostError("question lease was lost")

    def release_question_lease(self, question_id: UUID, worker_id: str) -> None:
        with self._transaction() as connection:
            row = connection.execute("SELECT project_id FROM questions WHERE id=? AND lease_owner=?", (str(question_id), worker_id)).fetchone()
            if row:
                connection.execute("UPDATE questions SET lease_owner=NULL,lease_expires_at=NULL WHERE id=? AND lease_owner=?", (str(question_id), worker_id))
                self._append_event(connection, project_id=UUID(row["project_id"]), question_id=question_id,
                                   event_type="question.lease_released", payload={"worker_id": worker_id})

    def transition(self, question_id: UUID, worker_id: str, expected: Sequence[QuestionState], new_state: QuestionState, *, revision: int | None = None, failure_reason: str | None = None) -> QuestionRecord:
        now = utc_now()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM questions WHERE id=?", (str(question_id),)).fetchone()
            if not row:
                raise NotFoundError(f"question not found: {question_id}")
            if row["lease_owner"] != worker_id or not row["lease_expires_at"] or row["lease_expires_at"] < now.isoformat():
                raise LeaseLostError("worker does not own a live question lease")
            old = QuestionState(row["state"])
            if old not in expected:
                raise InvalidTransitionError(f"cannot transition {old.value} to {new_state.value}")
            next_revision = row["revision"] if revision is None else revision
            connection.execute("UPDATE questions SET state=?,revision=?,failure_reason=?,updated_at=?,version=version+1 WHERE id=?", (new_state.value, next_revision, failure_reason, now.isoformat(), str(question_id)))
            self._append_event(connection, project_id=UUID(row["project_id"]), question_id=question_id,
                               event_type="question.transitioned", payload={"from": old.value, "to": new_state.value, "revision": next_revision, "failure_reason": failure_reason})
        return self.get_question(question_id)

    def get_or_create_activity(self, question: QuestionRecord, stage: StageKind) -> ActivityRecord:
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM activities WHERE question_id=? AND stage=? AND revision=?", (str(question.id), stage.value, question.revision)).fetchone()
            if not row:
                now, activity_id = utc_now(), new_uuid()
                key = f"m2h:{question.id}:{stage.value}:r{question.revision}"
                connection.execute(
                    "INSERT INTO activities(id,project_id,question_id,stage,revision,idempotency_key,status,attempt_count,request_json,result_json,error,worker_id,lease_expires_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(activity_id), str(question.project_id), str(question.id), stage.value, question.revision,
                     key, ActivityStatus.PENDING.value, 0, None, None, None, None, None, now.isoformat(), now.isoformat()))
                self._append_event(connection, project_id=question.project_id, question_id=question.id, activity_id=activity_id,
                                   event_type="activity.created", payload={"stage": stage.value, "revision": question.revision, "idempotency_key": key})
                row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
        return self._activity(row)

    def claim_activity(self, activity_id: UUID, worker_id: str, lease_seconds: int, max_attempts: int, request_json: dict[str, Any]) -> ActivityRecord:
        now, expires = utc_now(), utc_now() + timedelta(seconds=lease_seconds)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
            if not row:
                raise NotFoundError(f"activity not found: {activity_id}")
            if row["status"] == ActivityStatus.SUCCEEDED.value:
                return self._activity(row)
            live = row["status"] == ActivityStatus.RUNNING.value and row["lease_expires_at"] and row["lease_expires_at"] >= now.isoformat()
            if live and row["worker_id"] != worker_id:
                raise ConflictError(f"activity is leased by {row['worker_id']}")
            if row["attempt_count"] >= max_attempts:
                raise ConflictError(f"activity exhausted {max_attempts} attempts")
            attempt = row["attempt_count"] + 1
            connection.execute("UPDATE activities SET status=?,attempt_count=?,request_json=?,error=NULL,worker_id=?,lease_expires_at=?,updated_at=? WHERE id=?", (ActivityStatus.RUNNING.value, attempt, _json(request_json), worker_id, expires.isoformat(), now.isoformat(), str(activity_id)))
            self._append_event(connection, project_id=UUID(row["project_id"]), question_id=UUID(row["question_id"]), activity_id=activity_id,
                               event_type="activity.claimed", payload={"worker_id": worker_id, "attempt": attempt, "expires_at": expires.isoformat()})
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
        return self._activity(row)

    def renew_activity_lease(self, activity_id: UUID, worker_id: str, lease_seconds: int) -> None:
        now, expires = utc_now(), utc_now() + timedelta(seconds=lease_seconds)
        with self._transaction() as connection:
            cursor = connection.execute("UPDATE activities SET lease_expires_at=? WHERE id=? AND worker_id=? AND status=? AND lease_expires_at>=?", (expires.isoformat(), str(activity_id), worker_id, ActivityStatus.RUNNING.value, now.isoformat()))
            if cursor.rowcount != 1:
                raise LeaseLostError("activity lease was lost")

    def complete_activity(self, activity_id: UUID, worker_id: str, result_json: dict[str, Any], artifacts: list[ArtifactRecord]) -> ActivityRecord:
        now = utc_now()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
            if not row:
                raise NotFoundError(f"activity not found: {activity_id}")
            if row["status"] == ActivityStatus.SUCCEEDED.value:
                return self._activity(row)
            if row["worker_id"] != worker_id or row["status"] != ActivityStatus.RUNNING.value or not row["lease_expires_at"] or row["lease_expires_at"] < now.isoformat():
                raise LeaseLostError("cannot complete an activity without its live lease")
            for artifact in artifacts:
                if artifact.activity_id != activity_id or str(artifact.question_id) != row["question_id"]:
                    raise ValueError("artifact provenance does not match activity")
                self._insert_artifact(connection, artifact)
            connection.execute("UPDATE activities SET status=?,result_json=?,worker_id=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?", (ActivityStatus.SUCCEEDED.value, _json(result_json), now.isoformat(), str(activity_id)))
            self._append_event(connection, project_id=UUID(row["project_id"]), question_id=UUID(row["question_id"]), activity_id=activity_id,
                               event_type="activity.succeeded", payload={"artifact_ids": [str(item.id) for item in artifacts]})
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
        return self._activity(row)

    def fail_activity(self, activity_id: UUID, worker_id: str, error: str) -> ActivityRecord:
        now = utc_now(); error = error[:4000]
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
            if not row:
                raise NotFoundError(f"activity not found: {activity_id}")
            if row["worker_id"] != worker_id or row["status"] != ActivityStatus.RUNNING.value:
                raise LeaseLostError("cannot fail an activity owned by another worker")
            connection.execute("UPDATE activities SET status=?,error=?,worker_id=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?", (ActivityStatus.FAILED.value, error, now.isoformat(), str(activity_id)))
            self._append_event(connection, project_id=UUID(row["project_id"]), question_id=UUID(row["question_id"]), activity_id=activity_id,
                               event_type="activity.failed", payload={"error": error})
            row = connection.execute("SELECT * FROM activities WHERE id=?", (str(activity_id),)).fetchone()
        return self._activity(row)

    def list_activities(self, question_id: UUID) -> list[ActivityRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM activities WHERE question_id=? ORDER BY created_at", (str(question_id),)).fetchall()
        return [self._activity(row) for row in rows]

    def _activity(self, row: sqlite3.Row) -> ActivityRecord:
        data = dict(row)
        for key in ("request_json", "result_json"):
            data[key] = json.loads(data[key]) if data[key] else None
        return ActivityRecord.model_validate(data, strict=False)

    def list_events(self, question_id: UUID | None = None) -> list[EventRecord]:
        query, params = "SELECT * FROM events ORDER BY seq", ()
        if question_id:
            query, params = "SELECT * FROM events WHERE question_id=? ORDER BY seq", (str(question_id),)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            data = dict(row); data["payload"] = json.loads(data["payload"])
            result.append(EventRecord.model_validate(data, strict=False))
        return result

    def verify_event_chain(self) -> int:
        previous, count = ZERO_HASH, 0
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY seq").fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            envelope = {
                "event_id": row["event_id"], "project_id": row["project_id"],
                "question_id": row["question_id"], "activity_id": row["activity_id"],
                "event_type": row["event_type"], "payload": payload,
                "occurred_at": row["occurred_at"], "previous_hash": row["previous_hash"],
            }
            expected = hashlib.sha256((previous + _json(envelope)).encode()).hexdigest()
            if row["previous_hash"] != previous or row["event_hash"] != expected:
                raise ConflictError(f"event chain verification failed at seq {row['seq']}")
            previous, count = row["event_hash"], count + 1
        return count
