"""Provider-boundary context compaction for long-running agent sessions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from m2harness.domain.agent import ContextCompactionSnapshot, ModelRequest


def estimate_tokens(value: Any) -> int:
    """Stable tokenizer-free estimate suitable for budget triggering."""

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if not text:
        return 0
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    other = len(text) - chinese
    return max(1, int(chinese / 1.6 + other / 3.6))


def compact_text(text: str, budget_tokens: int) -> str:
    """Keep the high-value beginning and recent tail within a token budget."""

    if estimate_tokens(text) <= budget_tokens:
        return text
    characters = max(1_000, int(budget_tokens * 3.2))
    half = characters // 2
    return text[:half].rstrip() + "\n\n... [compacted] ...\n\n" + text[-half:].lstrip()


class ContextCompactorPort(Protocol):
    async def compact(self, request: ModelRequest) -> ModelRequest: ...


SummaryFactory = Callable[[str, int], Awaitable[str]]


class ContextCompactor:
    """Compact old turns into one continuation state before provider calls.

    System messages and the recent working window are retained verbatim. Older
    turns become a summary carrying goals, accepted facts, decisions, open
    issues, file paths and next actions. A deployment may inject an LLM-backed
    ``summary_factory``; the deterministic fallback remains safe and testable.
    """

    def __init__(self, *, trigger_ratio: float = 0.8, keep_recent_messages: int = 6, summary_factory: SummaryFactory | None = None) -> None:
        if trigger_ratio <= 0 or trigger_ratio > 1:
            raise ValueError("compact trigger_ratio must be in (0, 1]")
        self.trigger_ratio = trigger_ratio
        self.keep_recent_messages = max(1, keep_recent_messages)
        self.summary_factory = summary_factory

    async def compact(self, request: ModelRequest) -> ModelRequest:
        messages = list(request.messages)
        before = estimate_tokens(messages)
        budget = request.context.budget_tokens
        if before <= int(budget * self.trigger_ratio) or len(messages) <= self.keep_recent_messages:
            return request

        system = [message for message in messages if message.get("role") == "system"]
        conversation = [message for message in messages if message.get("role") != "system"]
        recent = conversation[-self.keep_recent_messages:]
        old = conversation[:-self.keep_recent_messages]
        transcript = json.dumps(old, ensure_ascii=False, indent=2, default=str)
        summary_budget = max(1_000, budget - estimate_tokens(system) - estimate_tokens(recent) - 1_000)
        if self.summary_factory is not None:
            summary = (await self.summary_factory(transcript, summary_budget)).strip()
        else:
            summary = self._extractive_summary(old, summary_budget)
        compact_message = {
            "role": "system",
            "content": "# COMPACTED CONTINUATION STATE\n\n" + summary,
        }
        compacted = tuple((*system, compact_message, *recent))
        after = estimate_tokens(compacted)
        if after > budget:
            compact_message["content"] = self._clip(compact_message["content"], max(500, summary_budget // 2))
            compacted = tuple((*system, compact_message, *recent))
            after = estimate_tokens(compacted)
        if after > budget:
            # A single recent tool result can itself exceed the window. Preserve
            # role/order while compacting those payloads from oldest to newest.
            available = max(500, budget - estimate_tokens(system) - estimate_tokens(compact_message))
            per_recent = max(250, available // max(1, len(recent)))
            fitted_recent = []
            for message in recent:
                content = message.get("content", "")
                text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
                fitted = dict(message)
                fitted["content"] = compact_text(text, per_recent)
                fitted_recent.append(fitted)
            compacted = tuple((*system, compact_message, *fitted_recent))
            after = estimate_tokens(compacted)
        digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        snapshot = ContextCompactionSnapshot(
            compaction_id=uuid4(), summary=compact_message["content"],
            source_message_count=len(old), retained_message_count=len(recent),
            estimated_tokens_before=before, estimated_tokens_after=after,
            source_digest=digest, created_at=datetime.now(UTC),
        )
        return request.model_copy(update={
            "messages": compacted,
            "compactions": (*request.compactions, snapshot),
        })

    def _extractive_summary(self, messages: list[dict[str, Any]], budget: int) -> str:
        sections = [
            "## Objective and user requirements",
            "## Accepted facts and decisions",
            "## Work completed and evidence",
            "## Open issues and risks",
            "## Files and artifacts",
            "## Next action",
        ]
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = message.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
            text = " ".join(text.split())
            if text:
                lines.append(f"- [{role}] {text}")
        body = "\n".join(lines)
        # The headings define the continuation contract even when the fallback
        # cannot semantically classify every historical sentence.
        summary = sections[0] + "\n" + body + "\n\n" + "\n\n".join(section + "\n- See retained evidence above." for section in sections[1:])
        return self._clip(summary, budget)

    @staticmethod
    def _clip(text: str, budget: int) -> str:
        return compact_text(text, budget)
