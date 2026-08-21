"""Immutable, all-role Skill context assembled from the local registry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from m2harness.application.compact import estimate_tokens
from m2harness.application.skills import SkillRegistry


def assemble_skill_context(
    registry: SkillRegistry | None,
    focus_names: Sequence[str] = (),
    *,
    budget_tokens: int = 96_000,
    include_focus_resources: bool = True,
) -> str:
    """Expose every model-invocable distilled Skill body to every Agent.

    Focus names affect ordering and optionally disclose their declared references;
    they no longer exclude the rest of the catalog. If a future catalog exceeds
    the hard budget, every Skill still receives a descriptor and the overflow is
    explicit instead of silently disappearing.
    """

    if registry is None:
        return "No local Skill registry is configured; follow the Harness contract only."
    summaries = registry.list(model_invocable=True)
    by_name = {item.name: item for item in summaries}
    missing = [name for name in focus_names if name not in by_name]
    if missing:
        raise KeyError("focused Skill unavailable: " + ", ".join(missing))
    ordered_names = list(dict.fromkeys((*focus_names, *(item.name for item in summaries))))
    focus = set(focus_names)
    catalog = "\n".join(
        f"- {item.name} [{item.version}]: {item.description}"
        for item in summaries
    )
    sections = [
        "# M2Harness Distilled Skill Context",
        "All model-invocable Skill bodies are included. FOCUS means role priority, not additional authority.",
        "\n## Catalog\n" + catalog,
    ]
    omitted: list[str] = []
    for name in ordered_names:
        definition = registry.get(name)
        if definition is None:
            omitted.append(name + " (disappeared during snapshot)")
            continue
        marker = "FOCUS" if name in focus else "AVAILABLE"
        section = f"\n## Skill: {name} [{marker}; sha256={definition.digest}]\n{definition.content.strip()}"
        if include_focus_resources and name in focus and definition.resource_base:
            base = Path(definition.resource_base).resolve()
            for relative, expected_digest in definition.resource_digests.items():
                path = (base / relative).resolve()
                if base not in path.parents or not path.is_file() or path.is_symlink():
                    raise ValueError(f"unsafe or missing Skill resource: {name}/{relative}")
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected_digest:
                    raise ValueError(f"Skill resource digest changed: {name}/{relative}")
                if len(raw) > 128_000:
                    omitted.append(f"{name}/{relative} (resource exceeds 128KB)")
                    continue
                section += f"\n\n### Focus resource: {relative}\n" + raw.decode("utf-8", errors="strict")
        if estimate_tokens("\n\n".join((*sections, section))) > budget_tokens:
            omitted.append(name + " (body exceeded aggregate context budget)")
            continue
        sections.append(section)
    if omitted:
        sections.append("\n## Explicit omissions\n" + "\n".join(f"- {item}" for item in omitted))
    return "\n\n".join(sections)

