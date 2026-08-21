---
name: data-analysis
description: Inspect tabular inputs, profile schema and quality, and produce auditable analysis artifacts.
---

# Data Analysis

Use `data_profile` before writing analysis code. Report file identity, sheet/table names, row counts, column types, nulls, duplicates, units, outliers, and suspicious parsing decisions.

Do not load an entire large table into the model context. Use profile summaries and query results as Artifacts. Every filtering, join, aggregation, imputation, and exclusion must be represented by executable code or a structured query Artifact.

Keep raw input immutable. Derived tables must reference their source digest and transformation/tool call. Never silently repair malformed data; record the repair or fail the quality gate.
