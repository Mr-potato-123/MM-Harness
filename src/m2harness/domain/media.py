"""Media inventory observations; raw bytes remain immutable Artifacts."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Modality(StrEnum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    TABULAR = "tabular"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


class MediaObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_id: UUID
    declared_media_type: str
    detected_media_type: str
    modality: Modality
    size_bytes: int = Field(ge=0)
    directly_projectable: bool
    extraction_required: bool
    warnings: tuple[str, ...] = ()
