"""Publication contracts shared by workflow and local rendering tools.

LaTeX is treated as a generated, untrusted artifact.  The validator is
deliberately conservative: it verifies a complete document and rejects shell
escape/file-write primitives before the source can be handed to a renderer.
It does not claim that TeX has compiled; compilation remains an explicit,
operator-controlled capability.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from m2harness.models import ArtifactKind, ProducedArtifact


LATEX_MEDIA_TYPES = frozenset({"text/x-tex", "application/x-tex", "text/latex"})
_DANGEROUS_COMMANDS = re.compile(
    r"\\(?:write18|input|include|openout|closeout|write|newwrite|@@input|catcode)\b",
    re.IGNORECASE,
)
_DOCUMENT_CLASS = re.compile(r"\\documentclass(?:\[[^]]*\])?\{[^{}]+\}")


def is_latex_artifact(artifact: ProducedArtifact) -> bool:
    """Return whether an output artifact unambiguously represents TeX."""

    name = artifact.logical_name.lower()
    media = artifact.media_type.lower()
    return (
        artifact.kind == ArtifactKind.FINAL_LATEX_PAPER
        and name.endswith((".tex", ".ltx"))
        and media in LATEX_MEDIA_TYPES
        and artifact.text is not None
    )


def validate_latex_source(source: str) -> list[str]:
    """Return stable validation errors for a publication source.

    The checks intentionally avoid a fake compiler.  They catch the common
    model-output failures (Markdown fences, missing document boundaries,
    unsafe shell/file primitives, and unbalanced braces) while leaving TeX
    package/layout choices extensible.
    """

    errors: list[str] = []
    stripped = source.strip()
    if not stripped:
        return ["LaTeX source is empty"]
    if stripped.startswith("```"):
        errors.append("LaTeX source must not be wrapped in Markdown fences")
    if not _DOCUMENT_CLASS.search(source):
        errors.append("missing \\documentclass declaration")
    if "\\begin{document}" not in source:
        errors.append("missing \\begin{document}")
    if "\\end{document}" not in source:
        errors.append("missing \\end{document}")
    if not re.search(r"\\title\s*\{", source):
        errors.append("missing \\title metadata")
    if "\\begin{abstract}" not in source or "\\end{abstract}" not in source:
        errors.append("missing abstract environment")
    if not re.search(r"\\section\*?\s*\{", source):
        errors.append("missing section heading")
    dangerous = sorted({match.group(0) for match in _DANGEROUS_COMMANDS.finditer(source)})
    if dangerous:
        errors.append("unsafe TeX command(s): " + ", ".join(dangerous))
    # A lightweight brace check is useful for obvious truncation, while braces
    # in comments and verbatim environments are intentionally not interpreted.
    depth = 0
    escaped = False
    for character in source:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                errors.append("unbalanced closing brace")
                break
    if depth:
        errors.append("unbalanced braces")
    return errors


def find_latex_artifact(artifacts: Iterable[ProducedArtifact]) -> ProducedArtifact | None:
    """Select the one declared final TeX artifact, rejecting ambiguity."""

    declared = [artifact for artifact in artifacts if artifact.kind == ArtifactKind.FINAL_LATEX_PAPER]
    if len(declared) > 1:
        raise ValueError("finalization must emit exactly one final_latex_paper artifact")
    if not declared:
        return None
    if not is_latex_artifact(declared[0]):
        raise ValueError("final_latex_paper must be UTF-8 text/x-tex with a .tex or .ltx name")
    return declared[0]


def validate_publication_artifacts(artifacts: Iterable[ProducedArtifact]) -> None:
    """Validate the final task's TeX artifact and its media declaration."""

    candidate = find_latex_artifact(artifacts)
    if candidate is None:
        raise ValueError(
            "finalization must emit a text/x-tex .tex artifact with kind final_latex_paper"
        )
    errors = validate_latex_source(candidate.text or "")
    if errors:
        raise ValueError("invalid final LaTeX publication: " + "; ".join(errors))
