"""Scoped, descriptor-first skill registry inspired by DeerFlow/DSH mechanisms."""

from __future__ import annotations

import hashlib
import re
import ast
import json
from pathlib import Path

from m2harness.domain.skill import InvocationPolicy, SkillDefinition, SkillManifest, SkillSummary
from m2harness.domain.capability import CapabilityRef
from m2harness.ports.skills import SkillProvider


def _frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---"):
        return {}
    lines = content.splitlines()
    end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}
    result: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


def _manifest_file(path: Path) -> dict[str, str]:
    """Read the deliberately small top-level skill.yaml contract without a YAML dependency."""
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"\'')
    return result


class FilesystemSkillProvider:
    name = "filesystem"

    def __init__(self, roots: list[Path], *, provider_name: str = "filesystem") -> None:
        self.roots = tuple(root.resolve() for root in roots)
        self.name = provider_name

    def _entries(self, cwd: str | None) -> list[tuple[int, Path, Path]]:
        entries: list[tuple[int, Path, Path]] = []
        for rank, root in enumerate(self.roots):
            if not root.exists() or root.is_symlink():
                continue
            # PI's loader discovers skills recursively.  This matters for a
            # real repository: project-local skills can live under a package
            # or experiment directory while the root catalog remains stable.
            # Ignore VCS/build trees and symlinked paths before reading them.
            ignored = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv"}
            candidates = sorted(root.rglob("SKILL.md"), key=lambda item: item.as_posix().lower())
            for body_path in candidates:
                if body_path.is_symlink() or any(part in ignored for part in body_path.parts):
                    continue
                directory = body_path.parent
                if directory.is_symlink():
                    continue
                try:
                    directory.resolve().relative_to(root.resolve())
                except ValueError:
                    continue
                entries.append((rank, directory, body_path))
            # Backward-compatible single-file skills at the provider root.
            for child in sorted(root.iterdir(), key=lambda item: item.name):
                if child.is_symlink() or not child.is_file() or child.suffix.lower() != ".md" or child.name == "SKILL.md":
                    continue
                entries.append((rank, root, child))
        return entries

    def _manifest(self, directory: Path, body_path: Path) -> SkillManifest:
        body = body_path.read_text(encoding="utf-8")
        front = _frontmatter(body)
        declared = _manifest_file(directory / "skill.yaml")
        name = declared.get("name", front.get("name", directory.name if directory.name != "" else body_path.stem))
        description = declared.get("description", front.get("description", f"Skill: {name}"))
        version = declared.get("version", front.get("version", "0.1.0"))
        api_version = declared.get("apiVersion", "m2harness/v1")
        invocation = InvocationPolicy(
            model_invocable=declared.get("modelInvocable", front.get("disable-model-invocation", "false")).lower() not in {"true", "1"},
            user_invocable=declared.get("userInvocable", front.get("user-invocable", "true")).lower() not in {"false", "0"},
        )
        entrypoint = declared.get("entrypoint", "SKILL.md")
        if Path(entrypoint).is_absolute() or ".." in Path(entrypoint).parts:
            raise ValueError("skill entrypoint must remain inside its skill directory")
        context_tokens = int(declared.get("contextTokens", "1000"))
        raw_resources = declared.get("resources", "")
        if raw_resources.strip().startswith("["):
            try:
                parsed_resources = json.loads(raw_resources)
            except json.JSONDecodeError:
                try:
                    parsed_resources = ast.literal_eval(raw_resources)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("skill resources must be a string list") from exc
            if not isinstance(parsed_resources, list) or not all(isinstance(item, str) for item in parsed_resources):
                raise ValueError("skill resources must be a string list")
            resource_names = tuple(item.strip() for item in parsed_resources if item.strip())
        else:
            resource_names = tuple(item.strip() for item in raw_resources.split(",") if item.strip())
        if any(Path(item).is_absolute() or ".." in Path(item).parts for item in resource_names):
            raise ValueError("skill resources must remain inside their skill directory")
        raw_capabilities = declared.get("requiresCapabilities", "")
        if raw_capabilities.strip().startswith("["):
            try:
                parsed_capabilities = json.loads(raw_capabilities)
            except json.JSONDecodeError:
                try:
                    parsed_capabilities = ast.literal_eval(raw_capabilities)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("skill capabilities must be a string list") from exc
        else:
            parsed_capabilities = [item.strip() for item in raw_capabilities.split(",") if item.strip()]
        capabilities: list[CapabilityRef] = []
        if isinstance(parsed_capabilities, list):
            for item in parsed_capabilities:
                if isinstance(item, str):
                    cap_name, _, cap_version = item.partition("@")
                    capabilities.append(CapabilityRef(name=cap_name, version=cap_version or "1"))
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    capabilities.append(CapabilityRef(name=item["name"], version=str(item.get("version", "1"))))
        return SkillManifest(
            api_version=api_version, name=name, version=version,
            description=description, entrypoint=entrypoint, invocation=invocation,
            context_tokens=context_tokens, resources=tuple(item.strip() for item in resource_names), requires_capabilities=tuple(capabilities),
        )

    @staticmethod
    def _resource_digests(directory: Path, manifest: SkillManifest) -> dict[str, str]:
        resources: dict[str, str] = {}
        for relative in manifest.resources:
            path = (directory / relative).resolve()
            if directory.resolve() not in path.parents or not path.is_file() or path.is_symlink():
                raise ValueError(f"skill resource is missing or unsafe: {relative}")
            resources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return resources

    @classmethod
    def _content_digest(cls, directory: Path, body_path: Path, manifest: SkillManifest) -> str:
        digest = hashlib.sha256()
        digest.update(body_path.read_bytes())
        digest.update(b"\0")
        manifest_path = directory / "skill.yaml"
        if manifest_path.is_file():
            digest.update(manifest_path.read_bytes())
        for relative, resource_digest in sorted(cls._resource_digests(directory, manifest).items()):
            digest.update(b"\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(resource_digest.encode("ascii"))
        return digest.hexdigest()

    def list(self, *, cwd: str | None = None) -> tuple[SkillSummary, ...]:
        result = []
        for rank, directory, body_path in self._entries(cwd):
            try:
                manifest = self._manifest(directory, body_path)
                digest = self._content_digest(directory, body_path, manifest)
            except (OSError, UnicodeError, ValueError):
                continue
            result.append(SkillSummary(
                name=manifest.name, version=manifest.version, description=manifest.description,
                source=str(body_path), provider=self.name, rank=rank, digest=digest,
                invocation=manifest.invocation,
            ))
        return tuple(result)

    def get(self, summary: SkillSummary, *, cwd: str | None = None) -> SkillDefinition | None:
        path = Path(summary.source).resolve()
        if not any(root in path.parents for root in self.roots) or not path.is_file() or path.is_symlink():
            return None
        content = path.read_text(encoding="utf-8")
        manifest = self._manifest(path.parent, path)
        digest = self._content_digest(path.parent, path, manifest)
        if manifest.name != summary.name or digest != summary.digest:
            return None
        resource_digests = self._resource_digests(path.parent, manifest)
        body = content
        if body.startswith("---"):
            lines = body.splitlines()
            end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
            if end is not None:
                body = "\n".join(lines[end + 1:]).lstrip()
        return SkillDefinition(**summary.model_dump(), content=body, manifest=manifest, resource_base=str(path.parent), resource_digests=resource_digests)


class SkillRegistry:
    """Merges provider descriptors, loads bodies only on explicit get()."""

    def __init__(self) -> None:
        self._providers: list[SkillProvider] = []
        self._revision = 0
        self._cache: dict[tuple[str | None, int], tuple[SkillSummary, ...]] = {}

    def register_provider(self, provider: SkillProvider) -> None:
        self._providers.append(provider)
        self.invalidate()

    def invalidate(self) -> None:
        self._revision += 1
        self._cache.clear()

    def list(self, *, cwd: str | None = None, model_invocable: bool | None = None) -> tuple[SkillSummary, ...]:
        key = (cwd, self._revision)
        summaries = self._cache.get(key)
        if summaries is None:
            candidates = [summary for provider in self._providers for summary in provider.list(cwd=cwd)]
            winners: dict[str, SkillSummary] = {}
            for candidate in sorted(candidates, key=lambda item: (item.rank, item.name, item.provider)):
                winners.setdefault(candidate.name, candidate)
            summaries = tuple(sorted(winners.values(), key=lambda item: item.name))
            self._cache[key] = summaries
        if model_invocable is None:
            return summaries
        return tuple(item for item in summaries if item.invocation.model_invocable == model_invocable)

    def get(self, name: str, *, cwd: str | None = None) -> SkillDefinition | None:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError(f"invalid skill name: {name}")
        summary = next((item for item in self.list(cwd=cwd) if item.name == name), None)
        if summary is None:
            return None
        provider = next((item for item in self._providers if item.name == summary.provider), None)
        if provider is None:
            return None
        return provider.get(summary, cwd=cwd)
