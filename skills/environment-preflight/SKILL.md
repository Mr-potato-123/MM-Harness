---
name: environment-preflight
description: Inspect the local modeling runtime for required interpreters, libraries, compilers, renderers, and output paths without silently installing software.
---

# Environment Preflight

Derive requirements from the selected workflow and inspect them read-only before a long run: Python and packages, data readers, numerical solvers, document compiler, figure renderer, PDF inspection utilities, workspace permissions, provider credentials, and configured local capabilities.

Report detected version/path, required versus optional status, affected stages, and a reproducible check. Missing optional visualization infrastructure must not be reported as a modeling failure; missing required execution or publication infrastructure must block the dependent gate.

Do not install packages, alter system configuration, or download binaries without explicit authorization. If installation is authorized, present exact targets and recovery implications, then re-run the same checks. Keep environment evidence separate from generated model evidence.

