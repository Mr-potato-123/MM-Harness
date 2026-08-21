"""Stable capability vocabulary and resolution results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated


CapabilityName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CapabilityRef(DomainModel):
    name: CapabilityName
    version: str = "1"


class CapabilityRequirement(DomainModel):
    capability: CapabilityRef
    optional: bool = False
    reason: str = ""


class CapabilityResolution(DomainModel):
    requested: tuple[CapabilityRequirement, ...]
    granted: tuple[CapabilityRef, ...]
    missing: tuple[CapabilityRequirement, ...] = ()
    provider_ids: dict[str, str] = Field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing
