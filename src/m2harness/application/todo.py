"""Durable, bounded Todo ledger shared by Code Agent adapters.

The ledger is intentionally smaller than the Main Harness DAG.  Main Harness
TODOs describe *workflow ownership* (which question/stage may run); this ledger
describes *implementation steps inside one Code Agent session*.  It mirrors
the useful Claude Code/DeerFlow invariants without allowing a model to change
the outer workflow: bounded items, deterministic order, one in-progress item,
atomic snapshots, and an append-only audit trail.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1, max_length=500)
    status: TodoStatus = "pending"
    active_form: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def active_form_for_progress(self) -> "TodoItem":
        if self.status == "in_progress" and self.active_form is None:
            object.__setattr__(self, "active_form", self.content)
        return self


class TodoSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sequence: int = Field(ge=1)
    task_id: str = Field(min_length=1, max_length=120)
    iteration: int = Field(ge=1)
    items: tuple[TodoItem, ...] = ()
    updated_at: datetime


@dataclass
class TodoLedger:
    """In-memory ledger with optional atomic JSON/NDJSON persistence."""

    task_id: str
    root: Path | None = None
    max_items: int = 100
    _sequence: int = 0
    _items: tuple[TodoItem, ...] = ()

    def read(self) -> tuple[TodoItem, ...]:
        return self._items

    def write(self, raw_items: Any, *, iteration: int) -> TodoSnapshot:
        if not isinstance(raw_items, list):
            raise ValueError("todos must be an array")
        if len(raw_items) > self.max_items:
            raise ValueError(f"todo list exceeds {self.max_items} items")
        items = tuple(TodoItem.model_validate(item, strict=True) for item in raw_items)
        contents = [item.content.casefold() for item in items]
        if len(contents) != len(set(contents)):
            raise ValueError("todo content must be unique")
        in_progress = sum(item.status == "in_progress" for item in items)
        if in_progress > 1:
            raise ValueError("at most one todo may be in_progress")
        self._sequence += 1
        self._items = items
        snapshot = TodoSnapshot(
            sequence=self._sequence, task_id=self.task_id, iteration=iteration,
            items=items, updated_at=datetime.now(UTC),
        )
        self._persist(snapshot)
        return snapshot

    def render(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self._items]

    def _persist(self, snapshot: TodoSnapshot) -> None:
        if self.root is None:
            return
        target = self.root.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # A snapshot is the recovery source; replace it atomically so a crash
        # cannot leave a partially written JSON document.
        descriptor, temporary = tempfile.mkstemp(prefix=".todo-", suffix=".json", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                json.dump(snapshot.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        events = target.with_suffix(".ndjson")
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n")

