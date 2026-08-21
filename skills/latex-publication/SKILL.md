---
name: latex-publication
description: Generate a polished, self-contained LaTeX paper from an approved single-question report while preserving claim parity and local safety boundaries.
---

# LaTeX Publication

Emit exactly one UTF-8 `.tex` artifact with `kind=final_latex_paper` and
`media_type=text/x-tex`. It must contain `\documentclass`, a complete document
environment, title/author metadata, abstract, clear sections, equations or
tables only when supported by evidence, a limitations section, and references
or an explicit “no external references used” note.

- The TeX is a presentation of reviewed evidence, not a second reasoning pass.
  Keep quantitative values, uncertainty, units, and claim wording aligned with
  the Markdown report.
- Use conservative packages and no shell escape, external file reads,
  generated code, or network-dependent assets. Include figures only when an
  immutable figure artifact exists; otherwise use a descriptive placeholder or
  omit the figure.
- Escape user/data text before placing it in commands, labels, captions, or
  URLs. Do not emit Markdown fences around the source.
- A structural validator can establish source completeness and safety, but only
  an explicitly authorized TeX compiler can establish PDF compilation.

Read [references/layout-profile.md](references/layout-profile.md) for the
default visual hierarchy and parity checklist.
