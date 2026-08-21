"""Capability registry and deterministic requirement resolution."""

from __future__ import annotations

from m2harness.domain.capability import CapabilityRef, CapabilityRequirement, CapabilityResolution


class CapabilityRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, tuple[CapabilityRef, str]] = {}

    def register(self, capability: CapabilityRef, provider_id: str) -> None:
        current = self._providers.get(capability.name)
        if current and current[0].version != capability.version:
            raise ValueError(f"capability already registered with another version: {capability.name}")
        self._providers[capability.name] = (capability, provider_id)

    def resolve(self, requirements: list[CapabilityRequirement] | tuple[CapabilityRequirement, ...]) -> CapabilityResolution:
        granted, missing, providers = [], [], {}
        for requirement in requirements:
            match = self._providers.get(requirement.capability.name)
            if not match or match[0].version != requirement.capability.version:
                if not requirement.optional:
                    missing.append(requirement)
                continue
            granted.append(match[0]); providers[match[0].name] = match[1]
        return CapabilityResolution(requested=tuple(requirements), granted=tuple(granted), missing=tuple(missing), provider_ids=providers)

    def list(self) -> tuple[CapabilityRef, ...]:
        return tuple(value[0] for value in sorted(self._providers.values(), key=lambda item: item[0].name))
