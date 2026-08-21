---
name: modeling-core
description: Formalize a modeling question into assumptions, equations, validation contracts, and expected outputs.
---

# Modeling Core

This is a planning contract, not a license to claim that a model has been executed.

Before proposing equations:

1. Identify the exact decision, prediction, or quantity requested.
2. Inventory every input Artifact and distinguish observed data from assumptions.
3. Define variables, units, domains, constraints, objective, and known uncertainty.
4. Compare at least two plausible formulations when the question permits alternatives.
5. State the chosen formulation and why rejected alternatives are weaker for this problem.
6. Produce machine-readable validation contracts with stable IDs.
7. List expected tables, figures, intermediate files, and downstream outputs.

Every important claim must name the input or future execution evidence required to support it. If evidence is not available, write `unverified` rather than filling the gap with plausible prose.
