"""Persistent metadata index for content-addressed runtime artifacts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from m2harness.artifacts import ArtifactStore
from m2harness.errors import ConflictError, NotFoundError
from m2harness.models import ArtifactRecord


class SQLiteArtifactRegistry:
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
                """CREATE TABLE IF NOT EXISTS m2_artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    question_id TEXT,
                    activity_id TEXT,
                    kind TEXT NOT NULL,
                    logical_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    relative_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_m2_artifacts_sha ON m2_artifacts(sha256)")

    def register(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO m2_artifacts(id,project_id,question_id,activity_id,kind,logical_name,media_type,sha256,size_bytes,relative_path,metadata,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(artifact.id), str(artifact.project_id), str(artifact.question_id) if artifact.question_id else None, str(artifact.activity_id) if artifact.activity_id else None, artifact.kind.value, artifact.logical_name, artifact.media_type, artifact.sha256, artifact.size_bytes, artifact.relative_path, json.dumps(artifact.metadata, ensure_ascii=False, sort_keys=True), artifact.created_at.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                existing = self.get(artifact.id)
                if existing != artifact:
                    raise ConflictError(f"artifact id already registered with different metadata: {artifact.id}") from exc
        return artifact

    def get(self, artifact_id: UUID) -> ArtifactRecord:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM m2_artifacts WHERE id=?", (str(artifact_id),)).fetchone()
        if row is None:
            raise NotFoundError(f"runtime artifact not found: {artifact_id}")
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"])
        return ArtifactRecord.model_validate(data, strict=False)

    def list(self) -> list[ArtifactRecord]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM m2_artifacts ORDER BY created_at,id").fetchall()
        result = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data["metadata"])
            result.append(ArtifactRecord.model_validate(data, strict=False))
        return result

    def verify(self, store: ArtifactStore) -> int:
        artifacts = self.list()
        for artifact in artifacts:
            store.read(artifact)
        return len(artifacts)
