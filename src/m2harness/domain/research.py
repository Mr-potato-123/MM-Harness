"""Durable DeepResearch-style reports used inside ``solve_problem``."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from m2harness.models import StrictModel


class ResearchSourceKind(StrEnum):
    LOCAL_KNOWLEDGE = "local_knowledge"
    LOCAL_ARTIFACT = "local_artifact"
    WEB = "web"


class ResearchSource(StrictModel):
    source_id: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=500)
    uri: str = Field(min_length=1, max_length=2_000)
    kind: ResearchSourceKind
    trust: str = Field(min_length=1, max_length=100)
    excerpt: str = Field(min_length=1, max_length=12_000)
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchFinding(StrictModel):
    finding_id: str = Field(min_length=1, max_length=300)
    claim: str = Field(min_length=1, max_length=8_000)
    evidence_source_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchReport(StrictModel):
    protocol_version: int = Field(default=1, ge=1, le=1)
    query: str = Field(min_length=1, max_length=20_000)
    plan: tuple[str, ...] = Field(min_length=1, max_length=12)
    sources: tuple[ResearchSource, ...] = Field(max_length=100)
    findings: tuple[ResearchFinding, ...] = Field(max_length=100)
    gaps: tuple[str, ...] = ()
    local_only: bool = True
    completed: bool = True
    index_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def findings_reference_sources(self) -> "ResearchReport":
        source_ids = {source.source_id for source in self.sources}
        unknown = sorted({sid for finding in self.findings for sid in finding.evidence_source_ids} - source_ids)
        if unknown:
            raise ValueError("research findings reference unknown sources: " + ", ".join(unknown))
        return self
