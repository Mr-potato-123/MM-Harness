---
name: dimensional-analysis
description: Enforce unit consistency, scaling, nondimensionalization, and physically meaningful bounds throughout a model and its code.
---

# Dimensional Analysis

Assign units and domains to every observed quantity, parameter, variable, equation term, objective component, and reported metric. Reject additions of unlike dimensions and unexplained unit conversions. Record whether rates are per time, per entity, totals, densities, or normalized scores.

Use characteristic scales or nondimensional groups when magnitudes cause numerical instability or reveal governing regimes. Carry units through preprocessing and outputs; normalized features must retain reversible metadata.

Validate equations symbolically where possible and code with hand-checkable unit cases, limiting behavior, and order-of-magnitude bounds. A numerically plausible answer with inconsistent dimensions fails validation.

