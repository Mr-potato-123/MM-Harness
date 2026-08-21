---
name: problem-decomposition
description: Decompose a production modeling question into traceable subproblems, interfaces, and validation checkpoints before coding.
---

# Problem Decomposition

Turn the intake brief into a small dependency graph rather than a narrative
guess. Define the target, inputs, transformations, outputs, and failure modes.

- Separate data preparation, estimation/inference, numerical checks, and
  reporting. Each node gets a stable ID and a measurable acceptance condition.
- Identify the minimum sufficient computation and any alternative formulation
  that could materially change the answer.
- State interfaces for the coding worker: input paths, expected schemas,
  output files, metrics, and stdout JSON contract.
- Propagate uncertainty and limitations to downstream claims. A downstream
  claim cannot be stronger than its weakest required evidence.

Use [references/decomposition-template.md](references/decomposition-template.md)
for the handoff table.

