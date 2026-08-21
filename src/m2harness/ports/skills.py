"""Skill discovery and loading ports."""

from __future__ import annotations

from typing import Protocol

from m2harness.domain.skill import SkillDefinition, SkillSummary


class SkillProvider(Protocol):
    name: str

    def list(self, *, cwd: str | None = None) -> tuple[SkillSummary, ...]: ...
    def get(self, summary: SkillSummary, *, cwd: str | None = None) -> SkillDefinition | None: ...


class SkillRegistryPort(Protocol):
    def list(self, *, cwd: str | None = None, model_invocable: bool | None = None) -> tuple[SkillSummary, ...]: ...
    def get(self, name: str, *, cwd: str | None = None) -> SkillDefinition | None: ...
