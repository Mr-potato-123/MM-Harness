---
name: problem-decomposition
description: Decompose a production modeling question into traceable subproblems, interfaces, and validation checkpoints before coding.
---

# Problem Decomposition

Turn the intake brief into a small dependency graph rather than a narrative
guess. Count only explicit top-level questions as task nodes; background,
attachment descriptions, submission rules, and explanatory subclauses are not
automatically separate problems. Define targets, inputs, transformations,
outputs, dependencies, and failure modes.

- Separate data preparation, estimation/inference, numerical checks, and
  reporting. Each node gets a stable ID and a measurable acceptance condition.
- Identify the minimum sufficient computation and any alternative formulation
  that could materially change the answer.
- State interfaces for the coding worker: input paths, expected schemas,
  output files, metrics, and stdout JSON contract.
- Propagate uncertainty and limitations to downstream claims. A downstream
  claim cannot be stronger than its weakest required evidence.
- Before freezing the graph, list ambiguous wording with at least two plausible
  interpretations when it could change later questions. Reject an interpretation
  that makes newly introduced downstream conditions have no meaningful effect,
  unless the problem explicitly intends that result.

Use [references/decomposition-template.md](references/decomposition-template.md)
for the handoff table.
