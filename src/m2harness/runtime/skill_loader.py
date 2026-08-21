"""Descriptor-first Skill loader with immutable per-session snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from m2harness.application.skills import SkillRegistry
from m2harness.domain.skill import SkillDefinition, SkillSummary
from m2harness.domain.capability import CapabilityResolution


@dataclass(frozen=True)
class SkillSnapshot:
    snapshot_id: UUID
    summaries: tuple[SkillSummary, ...]
    loaded: tuple[SkillDefinition, ...]


class SkillLoader:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def catalog(self, *, cwd: str | None = None) -> tuple[SkillSummary, ...]:
        return self.registry.list(cwd=cwd, model_invocable=True)

    def load(self, names: list[str], *, cwd: str | None = None, resolution: CapabilityResolution | None = None) -> SkillSnapshot:
        summaries = self.registry.list(cwd=cwd, model_invocable=True)
        by_name = {item.name: item for item in summaries}
        loaded = []
        for name in names:
            summary = by_name.get(name)
            if summary is None:
                raise KeyError(f"skill is not model-invocable or unavailable: {name}")
            definition = self.registry.get(name, cwd=cwd)
            if definition is None:
                raise KeyError(f"skill disappeared while loading: {name}")
            if resolution is not None:
                missing = [capability for capability in definition.manifest.requires_capabilities if not any(grant.name == capability.name and grant.version == capability.version for grant in resolution.granted)]
                if missing:
                    raise PermissionError(f"skill {name} requires unavailable capabilities: {', '.join(item.name for item in missing)}")
            loaded.append(definition)
        return SkillSnapshot(snapshot_id=uuid4(), summaries=summaries, loaded=tuple(loaded))
