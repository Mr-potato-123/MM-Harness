"""Context planning port."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from m2harness.domain.agent import ContextItem, ContextSnapshot


class ContextEnginePort(Protocol):
    def build(self, session_id: UUID, items: list[ContextItem], *, budget_tokens: int, prompt_template: str) -> ContextSnapshot: ...
