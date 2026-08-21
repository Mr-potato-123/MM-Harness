---
name: tabular-analysis
description: Profile CSV/JSON/JSONL inputs deterministically before modeling, with schema, missingness, units, and leakage checks.
---

# Tabular Analysis

Use `data_profile` or an equivalent local tool before computing statistics.
Preserve input digests and never paste a large table into the model prompt.

- Report row count, columns, inferred types, missingness, distinctness, ranges,
  units, and obvious duplicate/key problems.
- Separate profiling from transformation. Every filter, imputation, encoding,
  and split must be named and reproducible in the coding artifact.
- Check target leakage and train/test contamination when the question involves
  prediction or comparison.
- Treat parser failures and mixed-type columns as explicit validation failures,
  not silent coercions.

See [references/profile-checklist.md](references/profile-checklist.md).

