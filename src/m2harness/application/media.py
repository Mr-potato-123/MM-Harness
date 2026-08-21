"""Deterministic media inventory and provider-neutral projection hints."""

from __future__ import annotations

from pathlib import Path

from m2harness.domain.media import MediaObservation, Modality
from m2harness.models import ArtifactRecord


class MediaInventory:
    def inspect(self, artifact: ArtifactRecord, data: bytes) -> MediaObservation:
        detected = self._sniff(data, artifact.media_type)
        modality = self._modality(detected)
        warnings: list[str] = []
        if artifact.size_bytes != len(data):
            warnings.append("registered size differs from read size")
        if detected == "application/pdf" and not data.startswith(b"%PDF"):
            warnings.append("declared PDF does not have a PDF magic header")
        direct = detected in {"application/pdf", "image/png", "image/jpeg", "image/webp", "video/mp4"}
        return MediaObservation(
            artifact_id=artifact.id, declared_media_type=artifact.media_type,
            detected_media_type=detected, modality=modality, size_bytes=len(data),
            directly_projectable=direct, extraction_required=not direct,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _sniff(data: bytes, declared: str) -> str:
        if data.startswith(b"%PDF"):
            return "application/pdf"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[:4] == b"\x00\x00\x00\x18" and b"ftyp" in data[:32]:
            return "video/mp4"
        return declared.lower()

    @staticmethod
    def _modality(media_type: str) -> Modality:
        if media_type == "application/pdf" or media_type.startswith("application/") and media_type in {"application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
            return Modality.DOCUMENT
        if media_type.startswith("image/"):
            return Modality.IMAGE
        if media_type.startswith("text/"):
            return Modality.TEXT
        if media_type in {"text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/json"}:
            return Modality.TABULAR
        if media_type.startswith("audio/"):
            return Modality.AUDIO
        if media_type.startswith("video/"):
            return Modality.VIDEO
        return Modality.BINARY
