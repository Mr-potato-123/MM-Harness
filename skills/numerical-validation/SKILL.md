---
name: numerical-validation
description: Validate numerical model outputs with reproducible checks, residual analysis, sensitivity, and failure diagnostics.
---

# Numerical Validation

Validation must consume registered inputs and produce registered evidence. A model response is not execution evidence.

For each required validation:

- use the contract ID and exact threshold;
- record input Artifact digests;
- run through the approved Sandbox Tool;
- save code, stdout/stderr, metrics, and figures;
- report pass/fail/inconclusive explicitly;
- explain numerical stability, error, and sensitivity;
- never convert a missing result into a pass.

At minimum check dimensional consistency, boundary/constraint behavior, numerical convergence, residual or error behavior, sensitivity to material assumptions, and reproducibility from a clean workspace.
