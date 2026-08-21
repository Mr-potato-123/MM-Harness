"""Remote sandbox port; local subprocess is not its production implementation."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SandboxSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    image_digest: str = Field(min_length=1)
    network: str = "deny"
    cpu_millis: int = Field(default=1000, ge=1)
    memory_bytes: int = Field(default=2_147_483_648, ge=1)
    timeout_seconds: int = Field(default=600, ge=1)


class ExecSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=600, ge=1)
    env: dict[str, str] = Field(default_factory=dict)


class ExecResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sandbox_id: UUID
    exit_code: int | None
    timed_out: bool = False
    stdout_artifact_id: UUID | None = None
    stderr_artifact_id: UUID | None = None
    output_artifact_ids: tuple[UUID, ...] = ()
    cpu_millis: int = Field(default=0, ge=0)
    memory_bytes: int = Field(default=0, ge=0)
    image_digest: str


class SandboxClient(Protocol):
    def create(self, spec: SandboxSpec) -> UUID: ...
    def execute(self, sandbox_id: UUID, spec: ExecSpec) -> ExecResult: ...
    def destroy(self, sandbox_id: UUID) -> None: ...
