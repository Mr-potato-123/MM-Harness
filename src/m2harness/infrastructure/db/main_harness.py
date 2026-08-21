"""SQLite persistence for Main Harness DAG/TODO runs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import UUID

from m2harness.application.main_harness import MainHarnessState
from m2harness.errors import ConflictError, NotFoundError


class SQLiteMainHarnessRepository:
    """Optimistic, restart-safe repository for ``MainHarnessState``.

    Reports remain embedded in the state snapshot for the first delivery; the
    referenced immutable ArtifactStore is still the source of large bytes.
    A later migration can normalize report snapshots without changing the
    MainHarnessRepository port.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS main_harness_runs (
                    run_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, state: MainHarnessState) -> MainHarnessState:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO main_harness_runs(run_id,version,state_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (str(state.run_id), state.version, state.model_dump_json(), state.created_at.isoformat(), state.updated_at.isoformat()),
                )
                connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"Main Harness run already exists: {state.run_id}") from exc
        return state

    def get(self, run_id: UUID) -> MainHarnessState:
        with self._connect() as connection:
            row = connection.execute("SELECT state_json FROM main_harness_runs WHERE run_id=?", (str(run_id),)).fetchone()
        if row is None:
            raise NotFoundError(f"Main Harness run not found: {run_id}")
        return MainHarnessState.model_validate_json(row["state_json"], strict=False)

    def save(self, state: MainHarnessState, expected_version: int) -> MainHarnessState:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                "UPDATE main_harness_runs SET version=?, state_json=?, updated_at=? WHERE run_id=? AND version=?",
                (state.version, state.model_dump_json(), state.updated_at.isoformat(), str(state.run_id), expected_version),
            )
            if result.rowcount != 1:
                connection.execute("ROLLBACK")
                raise ConflictError(f"Main Harness run version conflict: {state.run_id}@{expected_version}")
            connection.execute("COMMIT")
        return state

