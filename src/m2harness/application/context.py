"""Deterministic progressive-disclosure context planning."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from m2harness.domain.agent import ContextItem, ContextSnapshot


class ContextEngine:
    def build(self, session_id: UUID, items: list[ContextItem], *, budget_tokens: int, prompt_template: str) -> ContextSnapshot:
        if budget_tokens < 1:
            raise ValueError("context budget must be positive")
        selected: list[ContextItem] = []
        omitted: list[str] = []
        used = 0
        for item in sorted(items, key=lambda value: (-value.priority, value.source_id)):
            if used + item.estimated_tokens <= budget_tokens:
                selected.append(item)
                used += item.estimated_tokens
            else:
                omitted.append(item.source_id)
        digest = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
        return ContextSnapshot(
            snapshot_id=uuid4(), session_id=session_id, items=tuple(selected),
            omitted_source_ids=tuple(omitted), budget_tokens=budget_tokens,
            used_tokens=used, prompt_template_digest=digest, created_at=datetime.now(UTC),
        )
