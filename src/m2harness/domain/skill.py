"""Skill descriptors and immutable loaded definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .capability import CapabilityRef


class SkillModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InvocationPolicy(SkillModel):
    model_invocable: bool = True
    user_invocable: bool = True


class SkillManifest(SkillModel):
    api_version: Literal["m2harness/v1"]
    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=1000)
    entrypoint: str = "SKILL.md"
    requires_capabilities: tuple[CapabilityRef, ...] = ()
    resources: tuple[str, ...] = ()
    context_tokens: int = Field(default=1000, ge=0, le=1_000_000)
    invocation: InvocationPolicy = InvocationPolicy()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillSummary(SkillModel):
    name: str
    version: str
    description: str
    source: str
    provider: str
    rank: int
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: InvocationPolicy


class SkillDefinition(SkillSummary):
    content: str
    manifest: SkillManifest
    resource_base: str | None = None
    resource_digests: dict[str, str] = Field(default_factory=dict)


class SkillResource(SkillModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
