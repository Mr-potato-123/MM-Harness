---
name: coding-contract
description: Convert a reviewed model into deterministic, inspectable source with explicit inputs, outputs, metrics, and failure behavior.
---

# Coding Contract

The coding artifact is a reproducible implementation, not evidence by itself.

- Read only the Harness-provided `inputs/` and `input_manifest.json`; never
  download data, use ambient credentials, or invent missing values.
- Emit one JSON object on stdout with `validations` containing every required
  stable validation ID, `validation_evidence` containing a non-empty,
  reproducible explanation for each ID, and optional scalar `metrics`. Send
  diagnostics to stderr or a declared log artifact.
- Use fixed seeds/configuration where randomness is needed. Record versions,
  units, tolerances, and output paths in metadata.
- Fail closed on malformed input, missing columns, non-finite values, or failed
  checks. A generated `.py` file is not an execution result.
- Keep model translation, input handling, solving, validation, and rendering in
  named functions so a code-only revision can be localized and regression-tested.
