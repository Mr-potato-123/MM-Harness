"""SQLite workflow-state repository for restart-safe runtime composition."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from m2harness.domain.workflow import WorkflowState
from m2harness.errors import ConflictError, NotFoundError


class SQLiteWorkflowRepository:
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
            connection.execute("CREATE TABLE IF NOT EXISTS m2_workflow_states (workflow_id TEXT PRIMARY KEY, version INTEGER NOT NULL, state_json TEXT NOT NULL)")

    def create(self, state: WorkflowState) -> WorkflowState:
        with self._connection() as connection:
            try:
                connection.execute("INSERT INTO m2_workflow_states(workflow_id,version,state_json) VALUES(?,?,?)", (str(state.workflow_id), state.version, state.model_dump_json()))
            except sqlite3.IntegrityError as exc:
                raise ConflictError(f"workflow already exists: {state.workflow_id}") from exc
        return state

    def get(self, workflow_id: UUID) -> WorkflowState:
        with self._connection() as connection:
            row = connection.execute("SELECT state_json FROM m2_workflow_states WHERE workflow_id=?", (str(workflow_id),)).fetchone()
        if row is None:
            raise NotFoundError(f"workflow not found: {workflow_id}")
        return WorkflowState.model_validate_json(row[0], strict=False)

    def save(self, state: WorkflowState, expected_version: int) -> WorkflowState:
        if state.version != expected_version + 1:
            raise ConflictError(f"workflow state must advance exactly one version: {state.version} != {expected_version + 1}")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT version FROM m2_workflow_states WHERE workflow_id=?", (str(state.workflow_id),)).fetchone()
                if row is None:
                    raise NotFoundError(f"workflow not found: {state.workflow_id}")
                if int(row[0]) != expected_version:
                    raise ConflictError(f"workflow version conflict: {expected_version} != {row[0]}")
                connection.execute("UPDATE m2_workflow_states SET version=?,state_json=? WHERE workflow_id=? AND version=?", (state.version, state.model_dump_json(), str(state.workflow_id), expected_version))
                connection.commit()
                return state
            except BaseException:
                connection.rollback()
                raise
