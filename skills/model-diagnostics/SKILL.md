---
name: model-diagnostics
description: Diagnose model failure through invariants, residuals, calibration, convergence, stability, and comparison with simple baselines.
---

# Model Diagnostics

Derive diagnostics from the model's assumptions and validation contracts before seeing results. Include dimensional and domain checks, invariants, residual structure, calibration, convergence, constraint residuals, and baseline comparison as applicable.

When a check fails, localize the failure to data, formulation, implementation, numerical method, or evaluation design. Preserve the failing input and evidence. Do not repair by weakening thresholds after observing results unless the changed contract is reviewed and justified.

Distinguish warning, failure, and inconclusive. Report the affected claims, likely causes, discriminating next checks, and narrowest revision target. Passing diagnostics increases confidence only for the tested regime.

