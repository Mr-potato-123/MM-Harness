---
name: model-selection
description: Choose and document a modeling formulation by comparing plausible alternatives against the question, data, constraints, and validation budget.
---

# Model Selection

Do not optimize for sophistication. Compare the simplest defensible baseline
with alternatives that could change the decision. For each candidate record
assumptions, identifiability, computational cost, failure modes, and the exact
validation evidence required.

Choose one formulation only after explaining why the alternatives are weaker
for this question. Keep the selection reversible: coding should implement a
named interface, not embed an unreviewed one-off derivation.

