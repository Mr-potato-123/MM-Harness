"""Immutable, content-addressed artifact storage."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import tempfile
from pathlib import Path
from uuid import UUID

from m2harness.errors import ArtifactIntegrityError
from m2harness.models import ArtifactKind, ArtifactRecord, ProducedArtifact, new_uuid, utc_now


class ArtifactStore:
    """Stores immutable bytes; ownership and provenance live in HarnessStore."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def _path_for_digest(self, digest: str) -> tuple[Path, str]:
        relative = Path("blobs") / digest[:2] / digest[2:4] / digest
        absolute = (self.root / relative).resolve()
        if self.root not in absolute.parents:
            raise ArtifactIntegrityError("artifact path escaped the configured root")
        return absolute, relative.as_posix()

    def put_bytes(
        self,
        data: bytes,
        *,
        project_id: UUID,
        question_id: UUID | None,
        activity_id: UUID | None,
        kind: ArtifactKind,
        logical_name: str,
        media_type: str,
        metadata: dict | None = None,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(data).hexdigest()
        target, relative = self._path_for_digest(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self._verify_file(target, digest, len(data))
        else:
            descriptor, temporary_name = tempfile.mkstemp(prefix=".m2h-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            self._verify_file(target, digest, len(data))
        return ArtifactRecord(
            id=new_uuid(),
            project_id=project_id,
            question_id=question_id,
            activity_id=activity_id,
            kind=kind,
            logical_name=logical_name,
            media_type=media_type,
            sha256=digest,
            size_bytes=len(data),
            relative_path=relative,
            metadata=metadata or {},
            created_at=utc_now(),
        )

    def put_produced(
        self,
        produced: ProducedArtifact,
        *,
        project_id: UUID,
        question_id: UUID,
        activity_id: UUID,
    ) -> ArtifactRecord:
        if produced.text is not None:
            data = produced.text.encode("utf-8")
        else:
            try:
                data = base64.b64decode(produced.base64 or "", validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ArtifactIntegrityError("produced artifact contains invalid base64") from exc
        return self.put_bytes(
            data,
            project_id=project_id,
            question_id=question_id,
            activity_id=activity_id,
            kind=produced.kind,
            logical_name=produced.logical_name,
            media_type=produced.media_type,
            metadata=produced.metadata,
        )

    def read(self, artifact: ArtifactRecord) -> bytes:
        path = (self.root / artifact.relative_path).resolve()
        if self.root not in path.parents or path.name != artifact.sha256:
            raise ArtifactIntegrityError(f"unsafe artifact path: {artifact.relative_path}")
        self._verify_file(path, artifact.sha256, artifact.size_bytes)
        return path.read_bytes()

    @staticmethod
    def _verify_file(path: Path, expected_digest: str, expected_size: int) -> None:
        if not path.is_file():
            raise ArtifactIntegrityError(f"artifact blob is missing: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != expected_size or digest.hexdigest() != expected_digest:
            raise ArtifactIntegrityError(f"artifact integrity check failed: {path}")
