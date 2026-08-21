"""Media inventory observations; raw bytes remain immutable Artifacts."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID
from typing import Annotated
import base64
import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class MultimodalInput(BaseModel):
    """Immutable multimodal bytes explicitly selected by Main Harness.

    The solve tool receives references selected by the process owner rather
    than recursively reading a workspace.  Qwen adapters can project these
    bytes as PDF/image/video parts while retaining digest and size evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    logical_name: Annotated[str, Field(min_length=1, max_length=500)]
    media_type: Annotated[str, Field(min_length=1, max_length=200)]
    data_base64: Annotated[str, Field(min_length=1, max_length=200 * 1024 * 1024)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: int = Field(ge=0, le=200 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_payload(self) -> "MultimodalInput":
        try:
            data = base64.b64decode(self.data_base64, validate=True)
        except Exception as exc:
            raise ValueError("multimodal input data_base64 is invalid") from exc
        if len(data) != self.size_bytes or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("multimodal input digest or size does not match payload")
        return self
