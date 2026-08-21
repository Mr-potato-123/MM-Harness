---
name: modeling-knowledge
description: Select and explain candidate mathematical modeling methods from the local HMML knowledge base.
---

# Modeling Knowledge

Use `knowledge_search` to retrieve candidate methods. Compare methods against
the task's objective, variables, constraints, data regime, assumptions, and
required outputs. Retrieval is a candidate generator, not a final decision.

For each selected or rejected method record:

- the exact method and hierarchy path;
- why it matches or conflicts with the problem;
- assumptions that must be checked;
- the validation evidence expected from Code Harness;
- any simpler baseline that should be retained.

Do not claim that a method is executable until the unified modeling report
contains a concrete scheme, expected outputs, and required validations.
