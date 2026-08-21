---
name: code-debugging
description: Debug generated modeling code from reproducible failures while preserving the accepted model and evidence contract.
---

# Code Debugging

Start from the exact source digest, input manifest, command, environment, exit status, stderr, and failing validation. Reproduce before editing. Minimize the failing case and classify the defect as input handling, translation from equations, numerical stability, dependency, output contract, or performance.

Change the smallest surface that explains the failure. Do not silently redesign the accepted model, alter validation thresholds, or replace missing data. Add a regression check that fails before the repair and passes after it, plus boundary cases related to the defect.

Return the changed source, cause, evidence of reproduction, validation results, and any residual risk. A code-only repair should reuse the accepted Modeling Report unless the diagnosis shows a model defect.

