"""Contracts for the local mathematical-modeling knowledge base.

The knowledge base is deliberately a read-only provider.  It is not an Agent,
does not execute code, and never returns text that can change Harness policy.
The default implementation is a deterministic lexical index over MM-Agent's
HMML JSON export; a vector index can be added behind the same port later.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from m2harness.models import StrictModel


class KnowledgeSourceKind(StrEnum):
    HMML = "hmml"
    SKILL = "skill"
    ARTIFACT = "artifact"


class KnowledgeQuery(StrictModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=8, ge=1, le=50)
    source_kinds: tuple[KnowledgeSourceKind, ...] = ()


class KnowledgeEntry(StrictModel):
    entry_id: str = Field(min_length=1, max_length=300)
    method: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=20_000)
    hierarchy: tuple[str, ...] = ()
    source: str = Field(min_length=1, max_length=1_000)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeHit(StrictModel):
    entry: KnowledgeEntry
    score: float = Field(ge=0.0)
    matched_terms: tuple[str, ...] = ()
    snippet: str = Field(min_length=1, max_length=8_000)


class KnowledgeSearchResult(StrictModel):
    query: str
    hits: tuple[KnowledgeHit, ...] = ()
    source_count: int = Field(ge=0)
    index_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    truncated: bool = False
