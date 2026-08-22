"""Identity and filesystem layout for one isolated Main Harness run.

The execution workspace is deliberately boring: one run gets one directory,
one manifest, one database and one checkpoint namespace.  Providers may keep
their own subdirectories, but they must never write to a process-global task
directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from pathlib import Path
from uuid import UUID, uuid4


def _slug(value: str, *, fallback: str = "run", limit: int = 48) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip()).strip("-").lower()
    return (normalized[:limit].strip("-") or fallback)


@dataclass(frozen=True)
class RunIdentity:
    """Stable human label plus the UUID used by the domain state."""

    run_id: UUID
    run_name: str
    created_at: str
    scope: str
    input_name: str
    model: str
    code_agent: str

    @classmethod
    def create(
        cls,
        *,
        scope: str,
        input_name: str,
        model: str,
        code_agent: str,
        run_id: UUID | None = None,
    ) -> "RunIdentity":
        identifier = run_id or uuid4()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_name = "__".join((
            timestamp,
            _slug(scope, fallback="scope"),
            _slug(Path(input_name).stem, fallback="input"),
            _slug(model, fallback="model"),
            _slug(code_agent, fallback="code"),
            identifier.hex[:8],
        ))
        return cls(
            run_id=identifier,
            run_name=run_name,
            created_at=datetime.now(UTC).isoformat(),
            scope=scope,
            input_name=Path(input_name).name,
            model=model,
            code_agent=code_agent,
        )


@dataclass(frozen=True)
class RunWorkspace:
    """Physical paths belonging exclusively to one run."""

    identity: RunIdentity
    root: Path

    @property
    def workspace_root(self) -> Path:
        return self.root / ".m2harness" / "workspace"

    @property
    def artifact_root(self) -> Path:
        return self.root / ".m2harness" / "artifacts"

    @property
    def database_path(self) -> Path:
        return self.root / ".m2harness" / "state.db"

    @property
    def checkpoint_root(self) -> Path:
        return self.root / ".m2harness" / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        return self.root / ".m2harness" / "run.json"

    @classmethod
    def create(cls, parent: Path, identity: RunIdentity) -> "RunWorkspace":
        # Keep the full traceable run_name in the manifest, but use a compact
        # physical directory. The full name is repeated in report paths and
        # provider workspaces; repeating it on Windows can exceed MAX_PATH.
        run_parts = identity.run_name.split("__")
        physical_name = "__".join((run_parts[0], run_parts[1], run_parts[-1])) if len(run_parts) >= 3 else f"run-{identity.run_id.hex[:8]}"
        root = (parent.resolve() / physical_name).resolve()
        if root.exists():
            raise FileExistsError(f"run workspace already exists: {root}")
        workspace = cls(identity=identity, root=root)
        workspace.workspace_root.mkdir(parents=True, exist_ok=False)
        workspace.artifact_root.mkdir(parents=True, exist_ok=True)
        workspace.checkpoint_root.mkdir(parents=True, exist_ok=True)
        workspace.write_manifest(status="created")
        return workspace

    def write_manifest(self, *, status: str, **extra: object) -> Path:
        """Write a small, secret-free run manifest for humans and tooling."""

        payload = {
            "schema_version": 1,
            "run_id": str(self.identity.run_id),
            "run_name": self.identity.run_name,
            "created_at": self.identity.created_at,
            "status": status,
            "scope": self.identity.scope,
            "input_name": self.identity.input_name,
            "model": self.identity.model,
            "code_agent": self.identity.code_agent,
            "physical_workspace": self.root.name,
            "paths": {
                "workspace": ".m2harness/workspace",
                "artifacts": ".m2harness/artifacts",
                "database": ".m2harness/state.db",
                "checkpoints": ".m2harness/checkpoints",
            },
            **extra,
        }
        target = self.manifest_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
